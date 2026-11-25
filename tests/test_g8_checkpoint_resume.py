"""Validate the final checkpoint recovery without live inference."""
import json
import tempfile
import unittest
from pathlib import Path

from repair_to_generalization.experience_ledger import canonical
from repair_to_generalization.g6_qualification import Meter
from repair_to_generalization.g8_closed_loop_dev import MAX_CALLS, MAX_INPUT_BYTES, Run
from repair_to_generalization.g8_checkpoint_resume import first_manifest, load_chain
from repair_to_generalization.g8_network_resume import ReplayBoundary, load_prefix
from repair_to_generalization.g8_closed_loop_dev import setup
from repair_to_generalization.savant_pilot import PilotError
from test_g8_closed_loop_dev import provider
from test_g8_network_resume import parent


def write(path, rows):
    path.write_text("".join(canonical(row)+"\n" for row in rows), encoding="utf-8")


class CheckpointRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        cls.original = root/"original.jsonl"
        cls.continuation = root/"continuation.jsonl"
        baseline_events, baseline_calls = [], []
        def baseline_provider(payload):
            baseline_calls.append(payload)
            return provider(payload)
        baseline_meter = Meter(.5, baseline_provider, baseline_events.append,
                               max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        cls.baseline = Run(baseline_meter, baseline_events.append)
        cls.baseline.run()
        cls.baseline.verify()
        cls.baseline_calls = baseline_calls
        first_checkpoint = next(i for i,e in enumerate(baseline_events)
                                if e["record_type"] == "g8_cl_episode_started" and e["checkpoint"])
        target = [e for e in baseline_events[first_checkpoint:] if e["record_type"] == "call_started"][2]
        cls.fail_at = target["call_id"]
        original_rows, _, _ = parent(5)
        write(cls.original, original_rows)
        prefix = load_prefix(cls.original, setup, Run)
        cap = .5-prefix.spent_usd
        prior = 2.10+prefix.spent_usd
        rows = [dict(first_manifest(prefix, "test2", prior, cap), started_utc="test")]
        new_calls = []
        def interrupted(payload):
            new_calls.append(payload)
            if len(new_calls) == cls.fail_at-prefix.calls+1:
                raise PilotError("api_network_failure")
            return provider(payload)
        meter = Meter(cap, interrupted, rows.append, max_calls=MAX_CALLS-prefix.calls,
                      max_input_bytes=MAX_INPUT_BYTES)
        adapter = ReplayBoundary(prefix, live=meter, emit=rows.append)
        run = Run(adapter, adapter.emit)
        try:
            run.run()
        except PilotError as exc:
            if str(exc) != "api_network_failure":
                raise
        else:
            raise AssertionError("expected second network failure")
        run.verify()
        terminal = dict(run.summary(), record_type="g8_closed_loop_terminal",
                        error="api_network_failure", replay_valid=True,
                        calls=meter.budget.calls, reported_usd=meter.reported_usd,
                        unknown_reserved_usd=meter.unknown_reserved_usd,
                        accounted_usd=meter.budget.spent_usd,
                        parent_report_sha256=prefix.sha256,
                        parent_scored_episodes=prefix.terminal["episodes"],
                        chain_calls=prefix.calls+meter.budget.calls, report="continuation.jsonl")
        for k in ("accounted_usd","reported_usd","unknown_reserved_usd"):
            terminal["chain_"+k] = prefix.terminal[k]+terminal[k]
        rows.append(terminal)
        cls.rows = rows
        write(cls.continuation, rows)

    def test_exact_recovery_without_resampling(self):
        chain = load_chain(self.original, self.continuation)
        seen, events = [], []
        def fresh(payload):
            seen.append(payload)
            return provider(payload)
        meter = Meter(.5-chain.spent_usd, fresh, events.append,
                      max_calls=MAX_CALLS-chain.calls, max_input_bytes=MAX_INPUT_BYTES)
        first, second = chain.adapter(live=meter, emit=events.append)
        run = Run(first, first.emit)
        run.run()
        run.verify()
        self.assertTrue(first.crossed and second.crossed)
        self.assertEqual(seen, self.baseline_calls[self.fail_at-1:])
        self.assertEqual(run.summary(), self.baseline.summary())
        self.assertEqual(run.claims, self.baseline.claims)
        self.assertEqual(run.ledgers, self.baseline.ledgers)
        self.assertEqual(events[0]["record_type"], "g8_cl_network_resume_boundary")
        self.assertEqual(events[0]["episode"], 5)
        self.assertEqual(events[0]["step"], 2)
        self.assertEqual(meter.budget.calls, len(seen))
        self.assertLessEqual(chain.spent_usd+meter.budget.spent_usd, .5)

    def test_reject_chain_tampering(self):
        path = Path(self.tmp.name)/"modified.jsonl"
        for mutation in ("parent_hash","boundary","cost","claim","terminal","network","third"):
            with self.subTest(mutation=mutation):
                rows = json.loads(canonical(self.rows))
                if mutation == "parent_hash":
                    rows[0]["parent_report_sha256"] = "bad"
                elif mutation == "boundary":
                    rows[1]["step"] += 1
                elif mutation == "cost":
                    rows[-1]["chain_accounted_usd"] += .01
                elif mutation == "claim":
                    next(r for r in rows if r["record_type"]=="g8_cl_reflection")["claim"]["advice"] = "bad"
                elif mutation == "terminal":
                    rows[-1]["episodes"] = 13
                elif mutation == "network":
                    rows[-1]["error"] = "output_limit"
                else:
                    rows[0]["resume_revision"] = "g8_frozen_checkpoint_recovery_v1"
                write(path, rows)
                with self.assertRaises((ValueError, PilotError)):
                    load_chain(self.original, path)


if __name__ == "__main__":
    unittest.main()
