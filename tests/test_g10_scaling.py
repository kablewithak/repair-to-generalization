"""Nested scaling source, memory policy and outcome accounting tests."""
import json
import unittest
from dataclasses import asdict

from repair_to_generalization.experience_ledger import Access, Ledger, canonical, execute
from repair_to_generalization.g7_source import FrozenSource
from repair_to_generalization.g8_independent import TransportMeter
from repair_to_generalization.g10_50k_qualification import (
    BASE_SHA256, SCOPES, MAX_CALLS as EXT_CALLS, MAX_INPUT_BYTES, ExtensionRun, setup as extension_setup,
)
from repair_to_generalization.g10_scaling_source import replay_extension
from repair_to_generalization.g10_scaling import (
    ARMS, MAX_CALLS, Policies, Run, cases, order, extra_payload, parse_extra, summarize,
)
from repair_to_generalization.linked_world import reference_policy
from repair_to_generalization.savant_pilot import InvalidModelResponse
from repair_to_generalization.study_design import acquisition_cases, SERVICES
from repair_to_generalization.study_memory import CompressedLesson, compression_input, episodes
from test_g8_closed_loop_dev import provider, response


class ScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cohort=acquisition_cases()[:50]
        ledger,records=Ledger(),[]
        for i,c in enumerate(cohort,1):
            ledger,r=execute(c,ledger,i,True,Access.CURRENT,
                lambda p:reference_policy(p["current"],p["current_interactions"]))
            records.append(r)
        cls.base=FrozenSource(BASE_SHA256,ledger,cohort,tuple(records),.51,1.0)
        events=[dict(extension_setup(cls.base,"test",2.48,1.75),started_utc="test")]
        def synthetic(payload):
            result=provider(payload)
            result["usage"]["prompt_tokens"]=min(51000,len(json.dumps(payload)))
            return result
        meter=TransportMeter(1.75,synthetic,events.append,max_calls=EXT_CALLS,max_input_bytes=MAX_INPUT_BYTES)
        run=ExtensionRun(cls.base,meter,events.append)
        run.run()
        run.verify()
        events.append(dict(run.summary(),record_type="g10_50k_terminal",error=None,replay_valid=True,
            calls=meter.budget.calls,reported_usd=meter.reported_usd,accounted_usd=meter.budget.spent_usd,
            transport_recoveries=0,unknown_reserved_usd=0))
        cls.report=json.loads(canonical(events))
        cls.full=run.ledger
        lessons=[]
        for service in SERVICES:
            _,anchors,versions=compression_input(ledger,service)
            ids=tuple(next(anchors[a] for a in anchors if versions[a]==v) for v in ("v1","v2"))
            lessons.append(CompressedLesson(service,"Synthetic test-only lesson",ids))
        extra=[]
        for service in SCOPES:
            _,anchors=extra_payload(cls.full,service)
            extra.append(CompressedLesson(service,"Synthetic extra lesson",(next(iter(anchors.values())),)))
        cls.policies=Policies(ledger,cls.full,tuple(lessons),[{"kind":"synthetic_claim","body":i} for i in range(4)],tuple(extra))

    def test_exact_extension_replay_and_tamper(self):
        self.assertEqual(replay_extension(self.base,self.report,"test"),self.full)
        rows=json.loads(canonical(self.report))
        rows[-1]["accounted_usd"]+=.01
        with self.assertRaises(ValueError):
            replay_extension(self.base,rows,"test")

    def test_design_and_memory(self):
        self.assertEqual(len(cases()),6)
        self.assertEqual(len(order()),162)
        self.assertEqual(len(set(order())),162)
        self.assertEqual(self.policies.lessons[50][:5],self.policies.lessons[25])
        for tier,ledger in self.policies.ledgers.items():
            original=ledger.public()
            self.assertEqual(self.policies.memory(tier,"A3","chronological",{}),original)
            self.assertEqual(self.policies.memory(tier,"A4_partial","chronological",{}),original+self.policies.claims)
            self.assertEqual(self.policies.memory(tier,"A3","reverse_episode_blocks",{}),
                             [e for g in reversed(episodes(ledger)) for e in g])
            self.assertEqual(sorted(e["event_id"] for e in self.policies.reversed[tier]),
                             sorted(e["event_id"] for e in original))
        p,a=extra_payload(self.full,SCOPES[0])
        lesson,_=parse_extra(response({"lesson":"test","evidence":next(iter(a))}),SCOPES[0],a)
        self.assertEqual(lesson.evidence_ids,(next(iter(a.values())),))
        with self.assertRaises(InvalidModelResponse):
            parse_extra(response({"lesson":"test","evidence":"invented"}),SCOPES[0],a)

    def test_reference_comparison_and_costs(self):
        identity={}
        calls=[]
        events=[]
        def tagged(row):
            if row["record_type"]=="call_finished":
                calls.append(dict(row,**{k:v for k,v in identity.items() if k!="episode"}))
        meter=TransportMeter(5.0,provider,tagged,max_calls=MAX_CALLS,max_input_bytes=MAX_INPUT_BYTES)
        run=Run(self.policies,meter,events.append,identity)
        run.run()
        result=summarize(run.rows,calls,{25:.51,50:1.2},.01)
        self.assertTrue(result["comparison_complete"])
        self.assertFalse(result["g10_pass"])
        self.assertTrue(all(c["net_safe_wins"]==0 for c in result["scale_contrasts"].values()))
        self.assertTrue(all(c["net_safe_wins"]==0 for c in result["order_contrasts"].values()))
        self.assertTrue(all(r["replay_valid"] and not r["record"]["learning"] for r in run.rows))
        self.assertAlmostEqual(sum(c["accounted_decision_usd"] for c in result["by_condition"].values()),
                               meter.budget.spent_usd)
        self.assertEqual(result["by_condition"]["25:A3:chronological"]["safe"],18)
        self.assertEqual(result["by_condition"]["50:A3:reverse_episode_blocks"]["safe"],9)


if __name__=="__main__":
    unittest.main()
