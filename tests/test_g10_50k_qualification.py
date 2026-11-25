"""Offline nested-source and qualification regressions."""
import json
import unittest

from repair_to_generalization.experience_ledger import Access, Ledger, execute
from repair_to_generalization.g7_source import FrozenSource
from repair_to_generalization.g8_independent import TransportMeter
from repair_to_generalization.g10_50k_qualification import (
    BASE_SHA256, CONTEXT_TARGET, MIN_ADDITIONAL, MAX_ADDITIONAL,
    MAX_CALLS, MAX_INPUT_BYTES, ExtensionRun, extension_cases, probes,
    setup, should_stop, validate_design,
)
from repair_to_generalization.linked_world import reference_policy
from repair_to_generalization.savant_pilot import PilotError
from repair_to_generalization.study_design import acquisition_cases
from test_g8_closed_loop_dev import provider


class QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = acquisition_cases()[:50]
        ledger, records = Ledger(), []
        for i, c in enumerate(cases, 1):
            ledger, record = execute(c, ledger, i, True, Access.CURRENT,
                lambda p: reference_policy(p["current"],p["current_interactions"]))
            records.append(record)
        cls.source = FrozenSource(BASE_SHA256, ledger, cases, tuple(records), .51, 1.0)

    def test_frozen_design_and_public_solvability(self):
        validate_design(self.source)
        self.assertEqual(len(extension_cases()), MAX_ADDITIONAL)
        self.assertEqual(len(probes()), 3)
        self.assertFalse(should_stop(MIN_ADDITIONAL-1,CONTEXT_TARGET))
        self.assertFalse(should_stop(MIN_ADDITIONAL,CONTEXT_TARGET-1))
        self.assertTrue(should_stop(MIN_ADDITIONAL,CONTEXT_TARGET))
        for i,c in enumerate(extension_cases()+probes(),1):
            _, record = execute(c,Ledger(),i,False,Access.CURRENT,
                lambda p: reference_policy(p["current"],p["current_interactions"]))
            self.assertTrue(json.loads(record.outcome_json)["safe_success"])
        self.assertFalse(set(c.service for c in self.source.cases) &
                         set(c.service for c in extension_cases()+probes()))
        self.assertTrue(setup(self.source,"test",2.48,1.75)["base_replayed"])

    def test_nested_source_and_three_frozen_probes(self):
        events = []
        def synthetic(payload):
            result = provider(payload)
            # Synthetic token metadata only: actual provider measurements are live-only.
            result["usage"]["prompt_tokens"] = min(51_000,len(json.dumps(payload)))
            return result
        meter = TransportMeter(2.0,synthetic,events.append,max_calls=MAX_CALLS,
                               max_input_bytes=MAX_INPUT_BYTES)
        run = ExtensionRun(self.source,meter,events.append)
        run.run()
        run.verify()
        result = run.summary()
        self.assertTrue(result["tier_qualified"])
        self.assertFalse(result["g10_pass"])
        self.assertEqual(result["added_episodes"],MIN_ADDITIONAL)
        self.assertEqual(result["qualifying_50k_episodes"],3)
        self.assertEqual(run.ledger.events[:len(self.source.ledger.events)],self.source.ledger.events)
        frozen = next(r for r in events if r["record_type"]=="g10_50k_history_frozen")
        self.assertEqual(frozen["ledger_head"],run.ledger.head)
        for row in events:
            if row["record_type"]=="study_source_episode_started" and not row["learning"]:
                self.assertEqual(row["source_ledger_head"],frozen["ledger_head"])
        self.assertTrue(all(r["first_input_tokens"]>=CONTEXT_TARGET for r in result["probe_context"]))

    def test_failed_call_preserves_base_without_qualification(self):
        meter = TransportMeter(1.75,lambda _: (_ for _ in ()).throw(PilotError("api_network_failure")),
                               lambda _:None,max_calls=MAX_CALLS,max_input_bytes=MAX_INPUT_BYTES)
        run=ExtensionRun(self.source,meter,lambda _:None)
        with self.assertRaisesRegex(PilotError,"api_network_failure"):
            run.run()
        run.verify()
        self.assertEqual(run.ledger,self.source.ledger)
        self.assertFalse(run.summary()["tier_qualified"])
        self.assertEqual(meter.budget.calls,2)
        self.assertGreater(meter.unknown_reserved_usd,0)


if __name__=="__main__":
    unittest.main()
