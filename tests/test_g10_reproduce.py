"""Exercise the read-only release verifier with a synthetic truncated run."""
import json
import unittest

from repair_to_generalization.experience_ledger import canonical
from repair_to_generalization.g10_scaling import (
    SCOPES, MAX_CALLS, MAX_INPUT_BYTES, Run, Policies, extra_payload, parse_extra,
    setup, summarize,
)
from repair_to_generalization.g10_reproduce import verify
from repair_to_generalization.g8_independent import TransportMeter
from repair_to_generalization.savant_pilot import PilotError
import test_g10_scaling as scaling_fixtures
from test_g8_closed_loop_dev import provider,response


class ReleaseReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ScalingTests=scaling_fixtures.ScalingTests
        ScalingTests.setUpClass()
        cls.base,cls.full=ScalingTests.base,ScalingTests.full
        cls.lessons=ScalingTests.policies.retrievers[25].lessons
        cls.claims=ScalingTests.policies.claims

    def test_reproduce_interrupted_prefix_and_reject_tamper(self):
        cost=.7
        rows=[dict(setup(self.base,self.full,"test",3.22,5.0,cost),started_utc="test")]
        identity,calls,extras={},[],[]
        def tagged(row):
            row=dict(row,**{k:v for k,v in identity.items() if k!="episode"})
            rows.append(row)
            if row["record_type"]=="call_finished":
                calls.append(row)
        def fake(payload):
            schema=payload["response_format"]["json_schema"]
            if schema["name"]=="scale_scope_compression":
                return response({"lesson":"Synthetic extra lesson",
                                 "evidence":schema["schema"]["properties"]["evidence"]["enum"][0]})
            if len(calls)==len(SCOPES)+15:
                return {"choices":[{"finish_reason":"length","message":{"content":None}}],
                        "usage":{"prompt_tokens":100,"completion_tokens":8192,
                                 "completion_tokens_details":{"reasoning_tokens":8192}}}
            return provider(payload)
        meter=TransportMeter(5.0,fake,tagged,max_calls=MAX_CALLS,max_input_bytes=MAX_INPUT_BYTES)
        for i,service in enumerate(SCOPES,1):
            identity.clear()
            identity.update(stage="compression",service=service)
            p,a=extra_payload(self.full,service)
            meter.parser=lambda d,s=service,a=a:parse_extra(d,s,a)
            extras.append(meter.decide(p,i,0))
        policies=Policies(self.base.ledger,self.full,self.lessons,self.claims,tuple(extras))
        rows.append(dict(record_type="g10_scaling_memory_frozen",**policies.freeze()))
        run=Run(policies,meter,rows.append,identity)
        with self.assertRaisesRegex(PilotError,"output_limit"):
            run.run()
        extra=sum(c["accounted_usd"] for c in calls if c.get("stage")=="compression")
        rows.append(dict(summarize(run.rows,calls,{25:self.base.accounted_usd,
                    50:self.base.accounted_usd+cost},extra),record_type="g10_scaling_terminal",
                    error="invalid_model_response:output_limit",transport_recoveries=0,
                    calls=meter.budget.calls,accounted_usd=meter.budget.spent_usd,
                    reported_usd=meter.reported_usd,unknown_reserved_usd=0,
                    additional_compression_usd=extra))
        rows=json.loads(canonical(rows))
        result=verify(self.base,self.full,self.lessons,self.claims,cost,rows,"test")
        self.assertTrue(result["offline_replay_valid"])
        self.assertEqual(result["model_calls_made"],0)
        self.assertGreater(result["scored_episodes"],0)
        self.assertFalse(result["g10_pass"])
        for field in ("hash","tag","outcome"):
            altered=json.loads(canonical(rows))
            if field=="hash":
                next(r for r in altered if r["record_type"]=="call_started")["request_sha256"]="bad"
            elif field=="tag":
                next(r for r in altered if r["record_type"]=="call_started")["service"]="bad"
            else:
                next(r for r in altered if r["record_type"]=="g10_scaling_episode_finished")["outcome"]["safe_success"]=False
            with self.assertRaises((ValueError,PilotError)):
                verify(self.base,self.full,self.lessons,self.claims,cost,altered,"test")


if __name__=="__main__":
    unittest.main()
