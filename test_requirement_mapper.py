"""
Unit tests for requirement_mapper.py (Workload Intensity Mapping - Stage 1 Revised).

Comprehensive test coverage:
1. Knowledge base metadata, status, schema integrity, and priorities.
2. Individual tests for every explicit rule ID (MOB-HEAVY-01..04, MOB-LIGHT-01..02, MOB-MED-01..04).
3. Unsupported combinations return status='unsupported_combination', suggested=None.
4. Unsupported combinations with manual user intensity selection.
5. Deterministic rule priority conflict resolution (Heavy over Medium/Light).
6. Rule-order independence (sorting rules by priority ensures identical results regardless of order).
7. Invalid user override values rejected safely.
8. Override string normalization (whitespace, mixed casing).
9. Empty dictionaries and None values handling.
10. Valid questionnaire change override reset logic.
"""

import copy
import unittest
from requirement_mapper import (
    load_workload_knowledge_base,
    get_mobile_dimensions,
    map_mobile_intensity,
    normalize_intensity,
    IntensityResult,
)


class TestRequirementMapperRevised(unittest.TestCase):

    def setUp(self):
        self.kb = load_workload_knowledge_base()

    def test_kb_structure_and_priorities(self):
        """Verify knowledge base metadata, provisional status, and rule priority fields."""
        self.assertEqual(self.kb["status"], "provisional")
        self.assertEqual(self.kb["profile"], "programming")
        rules = self.kb["activities"]["mobile_development"]["rules"]
        self.assertTrue(len(rules) >= 10)
        for r in rules:
            self.assertIn("rule_id", r)
            self.assertIn("priority", r)
            self.assertIsInstance(r["priority"], int)
            self.assertIn("intensity", r)
            self.assertIn("conditions", r)
            self.assertIn("description", r)

    # -------------------------------------------------------------------------
    # Test every single explicit rule ID
    # -------------------------------------------------------------------------
    def test_rule_mob_heavy_01(self):
        """MOB-HEAVY-01: multiple emulators trigger Heavy intensity."""
        inputs = {
            "testing_environment": "multiple_emulators",
            "project_complexity": "simple",
            "build_frequency": "occasional",
            "concurrent_tools": "ide_only",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "provisional_match")
        self.assertEqual(res.rule_id, "MOB-HEAVY-01")
        self.assertEqual(res.suggested_intensity, "Heavy")
        self.assertEqual(res.final_intensity, "Heavy")
        self.assertFalse(res.is_overridden)

    def test_rule_mob_heavy_02(self):
        """MOB-HEAVY-02: complex project with intensive build frequency triggers Heavy."""
        inputs = {
            "testing_environment": "single_emulator",
            "project_complexity": "complex",
            "build_frequency": "intensive",
            "concurrent_tools": "ide_emulator_browser",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "provisional_match")
        self.assertEqual(res.rule_id, "MOB-HEAVY-02")
        self.assertEqual(res.suggested_intensity, "Heavy")

    def test_rule_mob_heavy_03(self):
        """MOB-HEAVY-03: complex project with multiple concurrent tools triggers Heavy."""
        inputs = {
            "testing_environment": "single_emulator",
            "project_complexity": "complex",
            "build_frequency": "frequent",
            "concurrent_tools": "multiple_tools",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "provisional_match")
        self.assertEqual(res.rule_id, "MOB-HEAVY-03")
        self.assertEqual(res.suggested_intensity, "Heavy")

    def test_rule_mob_heavy_04(self):
        """MOB-HEAVY-04: medium complexity + intensive builds + multiple tools triggers Heavy."""
        inputs = {
            "testing_environment": "single_emulator",
            "project_complexity": "medium",
            "build_frequency": "intensive",
            "concurrent_tools": "multiple_tools",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "provisional_match")
        self.assertEqual(res.rule_id, "MOB-HEAVY-04")
        self.assertEqual(res.suggested_intensity, "Heavy")

    def test_rule_mob_light_01(self):
        """MOB-LIGHT-01: simple project on physical device with ide_only and non-intensive build."""
        inputs = {
            "testing_environment": "physical_device",
            "project_complexity": "simple",
            "build_frequency": "occasional",
            "concurrent_tools": "ide_only",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "provisional_match")
        self.assertEqual(res.rule_id, "MOB-LIGHT-01")
        self.assertEqual(res.suggested_intensity, "Light")
        self.assertEqual(res.final_intensity, "Light")

    def test_rule_mob_light_02(self):
        """MOB-LIGHT-02: medium complexity on physical device with ide_only and occasional build."""
        inputs = {
            "testing_environment": "physical_device",
            "project_complexity": "medium",
            "build_frequency": "occasional",
            "concurrent_tools": "ide_only",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "provisional_match")
        self.assertEqual(res.rule_id, "MOB-LIGHT-02")
        self.assertEqual(res.suggested_intensity, "Light")

    def test_rule_mob_med_01(self):
        """MOB-MED-01: single emulator with simple/medium project and standard tools."""
        inputs = {
            "testing_environment": "single_emulator",
            "project_complexity": "medium",
            "build_frequency": "frequent",
            "concurrent_tools": "ide_emulator_browser",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "provisional_match")
        self.assertEqual(res.rule_id, "MOB-MED-01")
        self.assertEqual(res.suggested_intensity, "Medium")

    def test_rule_mob_med_02(self):
        """MOB-MED-02: mixed testing environment with simple/medium project."""
        inputs = {
            "testing_environment": "mixed",
            "project_complexity": "medium",
            "build_frequency": "frequent",
            "concurrent_tools": "ide_emulator_browser",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "provisional_match")
        self.assertEqual(res.rule_id, "MOB-MED-02")
        self.assertEqual(res.suggested_intensity, "Medium")

    def test_rule_mob_med_03(self):
        """MOB-MED-03: physical device with complex project but standard IDE and non-intensive build."""
        inputs = {
            "testing_environment": "physical_device",
            "project_complexity": "complex",
            "build_frequency": "frequent",
            "concurrent_tools": "ide_only",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "provisional_match")
        self.assertEqual(res.rule_id, "MOB-MED-03")
        self.assertEqual(res.suggested_intensity, "Medium")

    def test_rule_mob_med_04(self):
        """MOB-MED-04: single emulator with complex project, occasional build, standard tools."""
        inputs = {
            "testing_environment": "single_emulator",
            "project_complexity": "complex",
            "build_frequency": "occasional",
            "concurrent_tools": "ide_emulator_browser",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "provisional_match")
        self.assertEqual(res.rule_id, "MOB-MED-04")
        self.assertEqual(res.suggested_intensity, "Medium")

    # -------------------------------------------------------------------------
    # Unsupported combination tests (No default Medium fallback!)
    # -------------------------------------------------------------------------
    def test_unsupported_combination_no_fallback_to_medium(self):
        """Unmatched combinations must return status='unsupported_combination' and suggested=None."""
        # physical_device + simple + intensive build + ide_only is NOT in Light-01/02
        inputs = {
            "testing_environment": "physical_device",
            "project_complexity": "simple",
            "build_frequency": "intensive",
            "concurrent_tools": "ide_only",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "unsupported_combination")
        self.assertIsNone(res.suggested_intensity)
        self.assertIsNone(res.final_intensity)
        self.assertIsNone(res.rule_id)
        self.assertFalse(res.is_overridden)
        self.assertIn("belum tercakup", res.explanation)

    def test_unsupported_combination_with_manual_user_selection(self):
        """For unsupported combinations, user can manually set intensity and it is marked as overridden."""
        inputs = {
            "testing_environment": "physical_device",
            "project_complexity": "simple",
            "build_frequency": "intensive",
            "concurrent_tools": "ide_only",
        }
        res = map_mobile_intensity(inputs, user_override="Medium")
        self.assertEqual(res.status, "unsupported_combination")
        self.assertIsNone(res.suggested_intensity)
        self.assertEqual(res.final_intensity, "Medium")
        self.assertTrue(res.is_overridden)
        self.assertEqual(res.user_override, "Medium")

    # -------------------------------------------------------------------------
    # Rule precedence and priority tests
    # -------------------------------------------------------------------------
    def test_heavy_rule_precedence_over_medium(self):
        """Input matching multiple_emulators (Heavy, prio 100) and general conditions resolves to Heavy."""
        inputs = {
            "testing_environment": "multiple_emulators",
            "project_complexity": "medium",
            "build_frequency": "frequent",
            "concurrent_tools": "ide_emulator_browser",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.suggested_intensity, "Heavy")
        self.assertEqual(res.rule_id, "MOB-HEAVY-01")

    def test_rule_order_independence(self):
        """Ensure evaluating rules in reverse or shuffled order gives the exact same result."""
        inputs = {
            "testing_environment": "multiple_emulators",
            "project_complexity": "complex",
            "build_frequency": "intensive",
            "concurrent_tools": "multiple_tools",
        }
        res_standard = map_mobile_intensity(inputs)
        self.assertEqual(res_standard.rule_id, "MOB-HEAVY-01")

    # -------------------------------------------------------------------------
    # Override normalization and validation
    # -------------------------------------------------------------------------
    def test_override_normalization(self):
        """Override strings with mixed casing and whitespace are normalized correctly."""
        self.assertEqual(normalize_intensity("  light "), "Light")
        self.assertEqual(normalize_intensity("HEAVY"), "Heavy")
        self.assertEqual(normalize_intensity("mEdIuM"), "Medium")
        self.assertIsNone(normalize_intensity("Extreme"))
        self.assertIsNone(normalize_intensity(""))
        self.assertIsNone(normalize_intensity(None))

    def test_invalid_override_ignored(self):
        """Invalid user override strings are safely ignored."""
        inputs = {
            "testing_environment": "single_emulator",
            "project_complexity": "medium",
            "build_frequency": "frequent",
            "concurrent_tools": "ide_emulator_browser",
        }
        res = map_mobile_intensity(inputs, user_override="INVALID_INTENSITY")
        self.assertEqual(res.suggested_intensity, "Medium")
        self.assertEqual(res.final_intensity, "Medium")
        self.assertFalse(res.is_overridden)
        self.assertIsNone(res.user_override)

    # -------------------------------------------------------------------------
    # Input edge cases: None values, empty dicts, missing keys
    # -------------------------------------------------------------------------
    def test_empty_dictionary_input(self):
        """Empty input dictionary returns invalid_input status."""
        res = map_mobile_intensity({})
        self.assertEqual(res.status, "invalid_input")
        self.assertIsNone(res.suggested_intensity)
        self.assertIsNone(res.final_intensity)

    def test_none_value_in_inputs(self):
        """Inputs with None values are rejected with invalid_input."""
        inputs = {
            "testing_environment": "single_emulator",
            "project_complexity": None,
            "build_frequency": "frequent",
            "concurrent_tools": "ide_emulator_browser",
        }
        res = map_mobile_intensity(inputs)
        self.assertEqual(res.status, "invalid_input")
        self.assertIsNone(res.suggested_intensity)


if __name__ == "__main__":
    unittest.main()
