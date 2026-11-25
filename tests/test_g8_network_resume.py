"""Offline network recovery regression; no paid inference."""
import json
import tempfile
import unittest
from pathlib import Path

from repair_to_generalization.experience_ledger import canonical
from repair_to_generalization.g6_qualification import Meter
from repair_to_generalization.g8_closed_loop_dev import MAX_CALLS, MAX_INPUT_BYTES, Run, setup
from repair_to_generalization.g8_network_resume import ReplayBoundary, load_prefix
from repair_to_generalization.savant_pilot import PilotError
from test_g8_closed_loop_dev import provider


def parent(fail_at):
    rows = [dict(setup("test", 2.10, .5), started_utc="test")]
    calls = []
    def invoke(payload):
        calls.append(payload)
        if len(calls) == fail_at:
            raise PilotError("api_network_failure")
        return provider(payload)
    meter = Meter(.5, invoke, rows.append, max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
    run = Run(meter, rows.append)
    try:
        run.run()
    except PilotError as exc:
        if str(exc) != "api_network_failure":
            raise
    else:
        raise AssertionError("expected network failure")
    run.verify()
    rows.append(dict(run.summary(), record_type="g8_closed_loop_terminal",
                     error="api_network_failure", replay_valid=True,
                     calls=meter.budget.calls, reported_usd=meter.reported_usd,
                     unknown_reserved_usd=meter.unknown_reserved_usd,
                     accounted_usd=meter.budget.spent_usd, report="parent.jsonl"))
    return rows, calls, run


class NetworkResumeTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.jsonl"
            path.write_text("".join(canonical(row)+"\n" for row in rows), encoding="utf-8")
            return load_prefix(path, setup, Run)

    def test_continuation_replays_prefix_and_never_resends_accepted_calls(self):
        # Interrupt after a learning episode and at least one acquired claim.
        baseline_calls, baseline_events = [], []
        def baseline_provider(payload):
            baseline_calls.append(payload)
            return provider(payload)
        meter = Meter(.5, baseline_provider, baseline_events.append,
                      max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        baseline = Run(meter, baseline_events.append)
        baseline.run()
        baseline.verify()
        first_reflection = next(i for i,p in enumerate(baseline_calls,1)
                                if p["response_format"]["json_schema"]["name"] == "closed_loop_receipt_prediction")
        fail_at = first_reflection + 3
        rows, calls, interrupted = parent(fail_at)
        self.assertEqual(len(interrupted.claims), 1)
        prefix = self.load(rows)
        fresh, out = [], []
        def invoke(payload):
            fresh.append(payload)
            return provider(payload)
        live = Meter(.5-prefix.spent_usd, invoke, out.append,
                     max_calls=MAX_CALLS-prefix.calls, max_input_bytes=MAX_INPUT_BYTES)
        adapter = ReplayBoundary(prefix, live=live, emit=out.append)
        resumed = Run(adapter, adapter.emit)
        resumed.run()
        resumed.verify()
        self.assertEqual(fresh, baseline_calls[fail_at-1:])
        self.assertEqual(resumed.summary(), baseline.summary())
        self.assertEqual(resumed.claims, baseline.claims)
        self.assertEqual(resumed.ledgers, baseline.ledgers)
        self.assertEqual(out[0]["record_type"], "g8_cl_network_resume_boundary")
        self.assertEqual(out[0]["accepted_calls_replayed"], fail_at-1)
        self.assertEqual(out[0]["parent_call_id"], fail_at)
        self.assertEqual(live.budget.calls, len(fresh))
        self.assertGreater(prefix.terminal["unknown_reserved_usd"], 0)
        self.assertLessEqual(prefix.spent_usd+live.budget.spent_usd, .5)

    def test_reflection_network_failure_can_resume_exact_request(self):
        baseline_calls = []
        def invoke(payload):
            baseline_calls.append(payload)
            return provider(payload)
        meter = Meter(.5, invoke, lambda _:None, max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        Run(meter,lambda _:None).run()
        fail_at = next(i for i,p in enumerate(baseline_calls,1)
                       if p["response_format"]["json_schema"]["name"] == "closed_loop_receipt_prediction")
        rows, _, _ = parent(fail_at)
        prefix = self.load(rows)
        fresh, out = [], []
        def live_invoke(payload):
            fresh.append(payload)
            return provider(payload)
        live = Meter(.5-prefix.spent_usd, live_invoke, out.append,
                     max_calls=MAX_CALLS-prefix.calls, max_input_bytes=MAX_INPUT_BYTES)
        adapter = ReplayBoundary(prefix, live=live, emit=out.append)
        run = Run(adapter, adapter.emit)
        run.run()
        run.verify()
        self.assertEqual(fresh, baseline_calls[fail_at-1:])
        self.assertEqual(len(run.claims),4)

    def test_reject_modified_prefix_and_non_network_failure(self):
        original, _, _ = parent(5)
        for mutation in ("request","observation","decision","cost","terminal","failure","extra","chain"):
            with self.subTest(mutation=mutation):
                rows = json.loads(canonical(original))
                if mutation == "request":
                    next(r for r in rows if r["record_type"]=="call_started")["request_sha256"]="wrong"
                elif mutation == "observation":
                    next(r for r in rows if r["record_type"]=="g8_cl_public_step")["observation"]={}
                elif mutation == "decision":
                    next(r for r in rows if r["record_type"]=="call_finished")["decision"]["reason_code"]="wrong"
                elif mutation == "cost":
                    rows[-2]["accounted_usd"]+=.001
                elif mutation == "terminal":
                    rows[-1]["episodes"]+=1
                elif mutation == "failure":
                    rows[-1]["error"]="output_limit"
                elif mutation == "extra":
                    rows.insert(-1,{"record_type":"unexpected"})
                else:
                    rows[0]["parent_report_sha256"]="prior"
                with self.assertRaises((ValueError, PilotError)):
                    self.load(rows)


if __name__ == "__main__":
    unittest.main()
