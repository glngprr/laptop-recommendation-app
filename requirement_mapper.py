"""
Requirement Mapper: Workload Intensity Mapping Module (Stage 1).

This module evaluates user activity questionnaire responses (starting with Mobile Development)
and infers a suggested workload intensity level (Light, Medium, Heavy) using explicit rule-based
priorities defined in knowledge_base/workload_profiles.json.

IMPORTANT ARCHITECTURAL NOTE:
This module deliberately does NOT translate intensity into hardware specifications yet.
Workload Intensity Mapping and Hardware Requirement Mapping are separate concerns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

KB_PATH = Path(__file__).resolve().parent / "knowledge_base" / "workload_profiles.json"

VALID_INTENSITIES = {"Light", "Medium", "Heavy"}

VALID_STATUSES = {
    "provisional_match",       # Explicit rule matched in provisional knowledge base
    "unsupported_combination", # Valid inputs but no explicit rule matches
    "invalid_input",           # Missing, null, or unknown input values
}


@dataclass
class IntensityResult:
    """Result of mapping a workload questionnaire into an intensity level."""
    activity: str
    suggested_intensity: Optional[str]
    final_intensity: Optional[str]
    is_overridden: bool
    user_override: Optional[str]
    status: str  # "provisional_match", "unsupported_combination", "invalid_input"
    rule_id: Optional[str]
    explanation: str
    inputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity": self.activity,
            "suggested_intensity": self.suggested_intensity,
            "final_intensity": self.final_intensity,
            "is_overridden": self.is_overridden,
            "user_override": self.user_override,
            "status": self.status,
            "rule_id": self.rule_id,
            "explanation": self.explanation,
            "inputs": self.inputs,
        }


def load_workload_knowledge_base(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the JSON knowledge base containing activity workload definitions and rules."""
    target_path = Path(path) if path else KB_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_mobile_dimensions(kb: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Retrieve questionnaire dimensions and choices for Mobile Development."""
    data = kb or load_workload_knowledge_base()
    return data.get("activities", {}).get("mobile_development", {}).get("dimensions", {})


def normalize_intensity(val: Optional[str]) -> Optional[str]:
    """Normalize intensity string to Title Case (Light, Medium, Heavy) if valid, else None."""
    if not val or not isinstance(val, str):
        return None
    val_clean = val.strip().capitalize()
    return val_clean if val_clean in VALID_INTENSITIES else None


def map_mobile_intensity(inputs: Dict[str, Any], user_override: Optional[str] = None) -> IntensityResult:
    """
    Map mobile development questionnaire inputs to a suggested workload intensity.

    Deterministic priority evaluation:
      - Rules are sorted by (-priority, -len(conditions), rule_id) before matching.
      - Rule selection is 100% order-independent with respect to JSON array ordering.

    Statuses:
      - 'provisional_match': An explicit rule matched in the provisional knowledge base.
      - 'unsupported_combination': Valid inputs do not match any explicit rule.
      - 'invalid_input': Missing, null, or out-of-range option values.

    Expected inputs keys:
      - testing_environment: "physical_device", "single_emulator", "multiple_emulators", "mixed"
      - project_complexity: "simple", "medium", "complex"
      - build_frequency: "occasional", "frequent", "intensive"
      - concurrent_tools: "ide_only", "ide_emulator_browser", "multiple_tools"

    User override:
      - Optional string ("Light", "Medium", "Heavy"). If supplied and valid, final_intensity
        takes this value while preserving the original suggested_intensity.
    """
    activity_name = "mobile_development"
    norm_override = normalize_intensity(user_override)

    # 1. Input dictionary validation
    if not isinstance(inputs, dict):
        return IntensityResult(
            activity=activity_name,
            suggested_intensity=None,
            final_intensity=norm_override,
            is_overridden=bool(norm_override),
            user_override=norm_override,
            status="invalid_input",
            rule_id=None,
            explanation="Input kuesioner harus berupa dictionary.",
            inputs={}
        )

    kb = load_workload_knowledge_base()
    act_data = kb.get("activities", {}).get(activity_name, {})
    dims = act_data.get("dimensions", {})
    rules = act_data.get("rules", [])

    # 2. Check for missing or None keys/values
    required_keys = ["testing_environment", "project_complexity", "build_frequency", "concurrent_tools"]
    missing_keys = [k for k in required_keys if k not in inputs or inputs[k] is None]
    if missing_keys:
        return IntensityResult(
            activity=activity_name,
            suggested_intensity=None,
            final_intensity=norm_override,
            is_overridden=bool(norm_override),
            user_override=norm_override,
            status="invalid_input",
            rule_id=None,
            explanation=f"Pertanyaan belum lengkap: {', '.join(missing_keys)} tidak ditemukan atau bernilai None.",
            inputs=inputs
        )

    for k in required_keys:
        valid_options = dims.get(k, {}).get("options", {})
        val = str(inputs[k]).strip()
        if val not in valid_options:
            return IntensityResult(
                activity=activity_name,
                suggested_intensity=None,
                final_intensity=norm_override,
                is_overridden=bool(norm_override),
                user_override=norm_override,
                status="invalid_input",
                rule_id=None,
                explanation=f"Pilihan untuk '{k}' tidak valid: '{val}'. Pilihan valid: {list(valid_options.keys())}",
                inputs=inputs
            )

    # 3. Deterministic Rule evaluation: sort by (-priority, -specificity, rule_id)
    sorted_rules = sorted(
        rules,
        key=lambda r: (
            -int(r.get("priority", 0)),
            -len(r.get("conditions", {})),
            str(r.get("rule_id", ""))
        )
    )

    matched_rule = None
    for r in sorted_rules:
        conds = r.get("conditions", {})
        matches_all = True
        for cond_key, allowed_values in conds.items():
            if inputs.get(cond_key) not in allowed_values:
                matches_all = False
                break
        if matches_all:
            matched_rule = r
            break

    # 4. Determine result based on whether a rule matched
    if matched_rule:
        suggested = matched_rule["intensity"]
        status = "provisional_match"
        rule_id = matched_rule["rule_id"]
        explanation = matched_rule["description"]
        if norm_override:
            final_intensity = norm_override
            is_overridden = (norm_override != suggested)
        else:
            final_intensity = suggested
            is_overridden = False
    else:
        # Explicit unsupported combination - NO default Medium assumption!
        suggested = None
        status = "unsupported_combination"
        rule_id = None
        explanation = (
            "Kombinasi beban kerja ini belum tercakup dalam basis pengetahuan sementara "
            "(provisional knowledge base). Silakan tentukan tingkat intensitas secara manual."
        )
        if norm_override:
            final_intensity = norm_override
            is_overridden = True  # User explicitly made the decision
        else:
            final_intensity = None
            is_overridden = False

    return IntensityResult(
        activity=activity_name,
        suggested_intensity=suggested,
        final_intensity=final_intensity,
        is_overridden=is_overridden,
        user_override=norm_override,
        status=status,
        rule_id=rule_id,
        explanation=explanation,
        inputs=inputs
    )
