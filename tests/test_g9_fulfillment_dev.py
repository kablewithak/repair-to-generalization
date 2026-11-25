"""Offline fulfillment instrument, source replay and transfer isolation tests."""
import json
import unittest
from dataclasses import asdict

from repair_to_generalization.experience_ledger import canonical, hash_value
from repair_to_generalization.fulfillment_world import CASES, MAX_ACTIONS, Move, World, reference_policy
from repair_to_generalization.g8_closed_loop_dev import Run as ClosedLoopRun
from repair_to_generalization.g8_independent import (
    CONFIGS, MASK, MAX_BLOCK_CALLS, design, setup as source_setup, stream_summary,
    main_screen, ablation_screen, TransportMeter,
)
from repair_to_generalization.g9_development_source import replay_source, FAILURE
from repair_to_generalization.g9_fulfillment_dev import (
    ARMS, MAX_CALLS, MAX_INPUT_BYTES, Run, order, parse, request, score, setup,
)
from repair_to_generalization.savant_pilot import PilotError, InvalidModelResponse
from test_g8_closed_loop_dev import provider, response


class FulfillmentTests(unittest.TestCase):
    def test_public_reference_and_distinct_actions(self):
        for case in CASES:
            world=World(case)
            history=[]
            while not world.finished and world.actions<MAX_ACTIONS:
                move=reference_policy(world.view(),history)
                obs=world.step(move)
                history.append(dict(action=move.value,observation=obs))
            self.assertTrue(world.outcome()["safe_success"],case.case_id)
            self.assertNotIn("case_id",world.view())
            self.assertNotIn("held",world.view())
        w=World(CASES[0])
        old=w.step(Move.READ)
        self.assertEqual(old["held"],0)
        self.assertLess(old["snapshot_revision"],old["inventory_revision"])
        w.step(Move.SYNC)
        self.assertEqual(w.step(Move.READ)["held"],1)
        w.step(Move.SHIP)
        w.step(Move.FINISH)
        self.assertTrue(w.outcome()["safe_success"])
        w=World(CASES[0])
        w.step(Move.HOLD)
        w.step(Move.RELEASE)
        w.step(Move.SHIP)
        w.step(Move.FINISH)
        self.assertTrue(w.outcome()["unsafe"])
        self.assertFalse(w.outcome()["safe_success"])
        w=World(CASES[4])
        w.step(Move.SHIP)
        w.step(Move.RELEASE)
        w.step(Move.FINISH)
        self.assertEqual(w.outcome()["unauthorized_attempts"],1)
        self.assertFalse(w.outcome()["safe_success"])

    def source(self):
        events=[]
        cohort,schedule,_,_=design(0)
        seed=CONFIGS[0][0]
        def tagged(r):
            events.append(dict(r,seed=seed,study_stage="main"))
        active=[0]
        def fake(payload):
            if active[0]==17:
                return {"choices":[{"finish_reason":"length","message":{"content":None}}],
                        "usage":{"prompt_tokens":100,"completion_tokens":8192,
                                 "completion_tokens_details":{"reasoning_tokens":8192}}}
            return provider(payload)
        def emit(r):
            tagged(r)
            if r["record_type"]=="g8_cl_episode_finished":
                active[0]+=1
        manifest=source_setup("test",2.29,2.50)
        tagged(dict(record_type="g8_rep_stream_started",fresh_state=True))
        meter=TransportMeter(2.50,fake,emit,max_calls=MAX_BLOCK_CALLS,max_input_bytes=MAX_INPUT_BYTES)
        run=ClosedLoopRun(meter,emit,cohort=cohort,schedule=schedule,learning_mask=MASK)
        with self.assertRaisesRegex(PilotError,"output_limit"):
            run.run()
        run.verify()
        summary=dict(stream_summary(run,0),error=FAILURE,replay_valid=True,
                     accounted_usd=meter.budget.spent_usd,transport_recoveries=0)
        tagged(dict(record_type="g8_rep_stream_finished",**summary,
                    frozen_ledgers={a:l.public() for a,l in run.ledgers.items()},
                    frozen_claims=[asdict(c) for c in run.claims]))
        terminal=dict(record_type="g8_rep_terminal",error=FAILURE,main=main_screen([summary]),
                      streams=[summary],ablation=ablation_screen([]),ablation_status="not_run",
                      scored_main_episodes=17,replication_and_ablation_positive=False,g8_pass=False,
                      calls=meter.budget.calls,transport_recoveries=0,reported_usd=meter.reported_usd,
                      unknown_reserved_usd=meter.unknown_reserved_usd,accounted_usd=meter.budget.spent_usd)
        rows=json.loads(canonical([dict(manifest,started_utc="test")]+events+[terminal]))
        return rows

    def test_source_replay_and_tamper_rejection(self):
        rows=self.source()
        source=replay_source(rows,"test")
        self.assertEqual(len(source.claims),4)
        self.assertEqual({e.episode for e in source.ledger.events},{1,2,5,6})
        self.assertEqual(source.snapshot("A0"),[])
        self.assertEqual(source.snapshot("A3"),source.ledger.public())
        for arm in ("A4_STALE","A4"):
            self.assertEqual(source.snapshot(arm)[:-2],source.snapshot("A3"))
        self.assertTrue(all(x["body"]["version"]=="v1" for x in source.snapshot("A4_STALE")[-2:]))
        self.assertTrue(all(x["body"]["version"]=="v2" for x in source.snapshot("A4")[-2:]))
        for mutation in ("request","frozen","cost","failure"):
            altered=json.loads(canonical(rows))
            if mutation=="request":
                next(r for r in altered if r["record_type"]=="call_started")["request_sha256"]="invalid"
            elif mutation=="frozen":
                altered[-2]["frozen_claims"][0]["advice"]="fabricated"
            elif mutation=="cost":
                altered[-1]["accounted_usd"]+=.01
            else:
                altered[-1]["error"]="api_network_failure"
            with self.assertRaises((ValueError,PilotError)):
                replay_source(altered,"test")

    def test_transfer_replay_no_memory_mutation_or_checkpoint_leakage(self):
        source=replay_source(self.source(),"test")
        events=[]
        def fake(payload):
            public=json.loads(payload["messages"][1]["content"])
            return response({"action":reference_policy(public["current"],public["current_interactions"]).value})
        meter=TransportMeter(.5,fake,events.append,max_calls=MAX_CALLS,max_input_bytes=MAX_INPUT_BYTES)
        run=Run(source,meter,events.append)
        run.run()
        result=score(run.rows)
        self.assertTrue(result["complete"])
        self.assertFalse(result["development_positive"])
        self.assertFalse(result["g9_pass"])
        self.assertTrue(all(t["safe"]==6 for t in result["by_arm"].values()))
        self.assertEqual(len(order()),24)
        self.assertEqual(setup(source,"test",2.40,.5)["source_g8_pass"],False)
        arm=None
        for event in events:
            if event["record_type"]=="g9_dev_episode_started":
                arm=event["arm"]
            if event["record_type"]=="call_started":
                public=json.loads(event["request"]["messages"][1]["content"])
                self.assertEqual(public["prior_interactions"],source.snapshot(arm))
                self.assertNotIn("case_id",public["current"])
                self.assertNotIn("stratum",public["current"])
        self.assertTrue(all(r["replay_valid"] for r in run.rows))

    def test_parse_boundary_and_score(self):
        self.assertEqual(parse(response({"action":Move.HOLD.value}))[0].action,Move.HOLD)
        for body in ({},{"action":"submit_export"},{"action":Move.HOLD.value,"reason":"x"}):
            with self.assertRaises(InvalidModelResponse):
                parse(response(body))
        rows=[]
        for c in CASES:
            for a in ARMS:
                safe=c.stratum!="composition" or a=="A4"
                rows.append(dict(case_id=c.case_id,arm=a,outcome=dict(completed=True,safe_success=safe,unsafe=False),
                                 decisions=1,decision_usd=.001))
        self.assertTrue(score(rows)["development_positive"])
        rows[0]["outcome"]["safe_success"]=True
        rows[-1]["outcome"].update(safe_success=False,unsafe=True)
        self.assertFalse(score(rows)["development_positive"])


if __name__=="__main__":
    unittest.main()
