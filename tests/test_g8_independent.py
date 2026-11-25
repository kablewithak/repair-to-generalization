"""Offline qualification of frozen independent replication and fixed-history controls."""
import json
import unittest

from repair_to_generalization.experience_ledger import Access, Ledger, execute, hash_value
from repair_to_generalization.g8_closed_loop_dev import Run
from repair_to_generalization.g8_independent import (
    ARMS, ABLATION_ARMS, CONFIGS, MASK, PHASES, MAX_BLOCK_CALLS, MAX_ABLATION_CALLS,
    MAX_INPUT_BYTES, NETWORK_RECOVERIES, Ablation, TransportMeter, ablation_screen,
    ablation_snapshots, design, main_screen, manifest, setup, stream_summary, validate_design,
)
from repair_to_generalization.linked_world import reference_policy
from repair_to_generalization.savant_pilot import InvalidModelResponse, PilotError
from test_g8_closed_loop_dev import provider, response


class IndependentTests(unittest.TestCase):
    def test_manifest_and_solvability(self):
        validate_design()
        self.assertEqual(len(manifest()),3)
        self.assertEqual(sum(MASK),4)
        self.assertEqual(setup("test",2.29,2.5)["manifest_sha256"],hash_value(manifest()))
        for i in range(3):
            cohort,order,ablation,ablation_order=design(i)
            self.assertEqual(len(set(order)),36)
            self.assertEqual(len(set(ablation_order)),8)
            for episode,c in enumerate(cohort+ablation,1):
                _,record=execute(c,Ledger(),episode,False,Access.CURRENT,
                                 lambda p:reference_policy(p["current"],p["current_interactions"]))
                self.assertTrue(json.loads(record.outcome_json)["safe_success"])

    def test_independent_streams_and_ablation(self):
        summaries,blocks=[],[]
        for index in range(3):
            cohort,order,_,_=design(index)
            events=[]
            meter=TransportMeter(1.0,provider,events.append,
                                 max_calls=MAX_BLOCK_CALLS,max_input_bytes=MAX_INPUT_BYTES)
            run=Run(meter,events.append,cohort=cohort,schedule=order,learning_mask=MASK)
            run.run()
            run.verify()
            result=stream_summary(run,index)
            self.assertTrue(result["mechanism"])
            self.assertEqual(result["scored_episodes"],36)
            self.assertEqual(run.ledgers["A0"],Ledger())
            for a in ("A3","A4"):
                self.assertEqual({e.episode for e in run.ledgers[a].events},{1,2,5,6})
                self.assertTrue(all(e.body_json for e in run.ledgers[a].events))
            self.assertTrue(all(a["post"]["safe"]==6 and a["pre"]["safe"]==2
                                for a in result["by_arm"].values()))
            self.assertEqual(len(run.claims),4)
            for e in events:
                if e["record_type"]!="call_started":
                    continue
                public=json.loads(e["request"]["messages"][1]["content"])
                if "prior_interactions" not in public:
                    continue
                self.assertNotIn("case_id",public["current"])
                for item in public["prior_interactions"]:
                    if "episode" in item:
                        self.assertIn(item["episode"],(1,2,5,6))
                        self.assertLess(item["episode"],e["episode"])
            summaries.append(dict(result,error=None,replay_valid=True))
            snapshots=ablation_snapshots(run)
            self.assertEqual([x["body"]["version"] for x in snapshots["A4_STALE"]],["v1","v1"])
            self.assertEqual([x["body"]["version"] for x in snapshots["A4"]],["v2","v2"])
            self.assertEqual(snapshots["A3"],[])
            self.assertTrue(all(x["kind"]=="invalidated_claim" for x in snapshots["A4_INVALIDATE_ALL"]))
            ae=[]
            am=TransportMeter(.5,provider,ae.append,max_calls=MAX_ABLATION_CALLS,
                              max_input_bytes=MAX_INPUT_BYTES)
            ab=Ablation(run,index,am,ae.append)
            ab.run()
            self.assertEqual(len(ab.rows),8)
            self.assertTrue(all(r["replay_valid"] and r["outcome"]["safe_success"] and
                                not r["record"]["learning"] for r in ab.rows))
            public_history=run.ledgers["A4"].public()
            active=None
            for e in ae:
                if e["record_type"]=="g8_rep_ablation_episode_started":
                    active=e["arm"]
                if e["record_type"]=="call_started":
                    public=json.loads(e["request"]["messages"][1]["content"])
                    self.assertEqual(public["prior_interactions"],public_history+snapshots[active])
            blocks.append((CONFIGS[index][0],ab.rows))
        self.assertTrue(main_screen(summaries)["complete"])
        self.assertFalse(main_screen(summaries)["positive"])
        self.assertTrue(ablation_screen(blocks)["complete"])
        self.assertFalse(ablation_screen(blocks)["positive"])

    def test_replication_requires_independent_positive_and_safe_controls(self):
        def s(i,net):
            return dict(seed=CONFIGS[i][0],complete=True,replay_valid=True,error=None,mechanism=True,
                        contrasts={a:dict(net_changed_safe_wins=net,additional_unsafe=[],
                                          control_regressions=[]) for a in ("A0","A3")})
        rows=[s(0,1),s(1,1),s(2,0)]
        self.assertTrue(main_screen(rows)["positive"])
        self.assertFalse(main_screen([s(0,3),s(1,0),s(2,0)])["positive"])
        self.assertFalse(main_screen([s(0,3),s(1,3),s(2,-1)])["positive"])
        self.assertFalse(main_screen(rows[:2])["positive"])
        rows[-1]["contrasts"]["A3"]["additional_unsafe"]=[10]
        self.assertFalse(main_screen(rows)["positive"])
        rows[-1]["contrasts"]["A3"]["additional_unsafe"]=[]
        rows[-1]["mechanism"]=False
        self.assertFalse(main_screen(rows)["positive"])

    def test_ablation_requires_benefit_beyond_stale_and_invalidate_all(self):
        blocks=[]
        for cfg in CONFIGS:
            rows=[]
            for i in (1,2):
                for arm in ABLATION_ARMS:
                    safe=i==2 or arm=="A4"
                    rows.append(dict(index=i,arm=arm,outcome=dict(safe_success=safe,unsafe=False)))
            blocks.append((cfg[0],rows))
        self.assertTrue(ablation_screen(blocks)["positive"])
        for _,rows in blocks:
            for row in rows:
                if row["arm"]=="A4_INVALIDATE_ALL":
                    row["outcome"]["safe_success"]=True
        self.assertFalse(ablation_screen(blocks)["positive"])

    def test_transport_policy_retries_only_unknown_network(self):
        from repair_to_generalization.g6_live import ACTION_ONLY_REVISION, parse_linked, payload_for
        from repair_to_generalization.linked_world import LinkedWorld, CONTRACT
        from repair_to_generalization.experience_ledger import request_for
        c=design(0)[0][0]
        p=payload_for(request_for(LinkedWorld(c),[],Ledger(),Access.CURRENT),ACTION_ONLY_REVISION)
        parser=lambda d:parse_linked(d,ACTION_ONLY_REVISION)
        events,requests=[],[]
        def invoke(payload):
            requests.append(payload)
            if len(requests)%2==1:
                raise PilotError("api_network_failure")
            return provider(payload)
        meter=TransportMeter(.5,invoke,events.append,parser=parser,
                             max_calls=20,max_input_bytes=MAX_INPUT_BYTES)
        for i in range(NETWORK_RECOVERIES):
            meter.decide(p,1,i)
        self.assertEqual(meter.recoveries,NETWORK_RECOVERIES)
        self.assertGreater(meter.unknown_reserved_usd,0)
        self.assertEqual(len(requests),2*NETWORK_RECOVERIES)
        self.assertTrue(all(requests[i]==requests[i+1] for i in range(0,len(requests),2)))
        with self.assertRaisesRegex(PilotError,"api_network_failure"):
            meter.decide(p,1,4)
        self.assertEqual(len(requests),2*NETWORK_RECOVERIES+1)
        recoveries=[e for e in events if e["record_type"]=="g8_rep_transport_recovery"]
        self.assertEqual(len(recoveries),NETWORK_RECOVERIES)
        failures=[]
        bad=TransportMeter(.5,lambda _:response({}),failures.append,parser=parser,
                           max_calls=20,max_input_bytes=MAX_INPUT_BYTES)
        with self.assertRaises(InvalidModelResponse):
            bad.decide(p,1,0)
        self.assertEqual(bad.budget.calls,1)
        self.assertEqual(bad.recoveries,0)
        both=TransportMeter(.5,lambda _:(_ for _ in ()).throw(PilotError("api_network_failure")),
                            lambda _:None,parser=parser,max_calls=20,max_input_bytes=MAX_INPUT_BYTES)
        with self.assertRaisesRegex(PilotError,"api_network_failure"):
            both.decide(p,1,0)
        self.assertEqual(both.budget.calls,2)
        self.assertEqual(both.recoveries,1)


if __name__=="__main__":
    unittest.main()
