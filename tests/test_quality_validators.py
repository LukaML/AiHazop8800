# tests/test_quality_validators.py
"""Tests for the AI-HAZOP-8800 safety-quality validators (parts 3-9).

These exercise the per-phase ``_check_*`` helpers directly with dict rows (the
helpers read fields via ``row_utils._row_get``, which works on plain dicts), plus
the component-level catalogue accessor and the CLI/GUI column parity invariant.
"""
import re
import pathlib

import pytest

from src.validators import (
    _check_component_level,
    _check_vru_severity,
    _check_risk_diversity,
    _check_residual_links,
    _check_triage,
    _check_measure_repetition,
    _check_measure_category,
    _check_goal_coverage,
    _check_evidence,
    _check_weak_assumptions,
    _check_accept_assumptions,
    _check_accept_justification,
    _check_odd_consistency,
    _check_measure_id_uniqueness,
    _check_requirement_wording,
    _check_evidence_repetition,
    is_safety_relevant,
    is_exportable_row,
    is_valid_l1_row,
    incomplete_reason,
)
from src.graph_full import _stamp_triage
from src.catalogue_loader import get_component_level
from src.risk_model import FACTORS, acceptance_criterion_text
from src.run_pipeline import _WORKSHEET_COLUMNS

ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# PART 3 — component-level L1
# --------------------------------------------------------------------------- #

def test_component_level_lookup_by_family():
    rule = get_component_level("Camera-ObjectDetection")
    assert rule["level"] == "perception"
    assert "braking" in rule["forbidden_terms"]
    assert get_component_level("Control-Braking")["level"] == "control"
    assert get_component_level("Unknown-Class") == {}


def test_component_level_rejects_downstream_term():
    rows = [{"row_id": "L1-0", "component_class": "Camera-ObjectDetection",
             "failure_mode": "The system brakes too late and the vehicle does not stop for the cyclist."}]
    issues = _check_component_level(rows)
    assert issues and "downstream term" in issues[0]


def test_component_level_is_forbidden_terms_only():
    """The component-level check no longer requires a perception keyword (that false-rejected
    legitimate AI-specific guidewords). A failure mode without any downstream ACTION term passes."""
    rows = [{"row_id": "L1-1", "component_class": "Camera-ObjectDetection",
             "failure_mode": "Something goes generically wrong somewhere in the pipeline overall today."}]
    assert _check_component_level(rows) == []


def test_component_level_accepts_perception_output():
    rows = [{"row_id": "L1-2", "component_class": "Camera-ObjectDetection",
             "failure_mode": "The object detector produces a missed detection for a cyclist crossing from the right."}]
    assert _check_component_level(rows) == []


def test_component_level_accepts_control_command():
    rows = [{"row_id": "L1-3", "component_class": "Control-Braking",
             "failure_mode": "The braking command is issued too late, increasing the stopping distance beyond the gap."}]
    assert _check_component_level(rows) == []


# --------------------------------------------------------------------------- #
# PART 5 — risk / acceptance
# --------------------------------------------------------------------------- #

def test_vru_severity_flags_low_severity_contact():
    rows = [{"row_id": "L3-0", "potential_harm": "A cyclist could be struck and injured.", "S": "moderate"}]
    assert _check_vru_severity(rows)


def test_vru_severity_passes_serious():
    rows = [{"row_id": "L3-1", "potential_harm": "A cyclist could be struck and injured.", "S": "serious"}]
    assert _check_vru_severity(rows) == []


def test_vru_severity_ignores_non_contact():
    rows = [{"row_id": "L3-2", "potential_harm": "Minor wheel scuff against a kerb.", "S": "minor"}]
    assert _check_vru_severity(rows) == []


def _ctx(**kw):
    base = {"component": "Cam", "aspect": "Output", "scenario": "Crossing"}
    base.update(kw)
    return base


def test_risk_diversity_flags_identical_tuples():
    rows = [_ctx(row_id=f"L3-{i}", E="low", PF="low", PND="low", PNM="low", S="serious") for i in range(4)]
    assert _check_risk_diversity(rows, list(FACTORS), "initial-risk")


def test_risk_diversity_passes_when_varied():
    pfs = ["low", "medium", "high", "very_high"]
    rows = [_ctx(row_id=f"L3-{i}", E="low", PF=pfs[i], PND="low", PNM="low", S="serious") for i in range(4)]
    assert _check_risk_diversity(rows, list(FACTORS), "initial-risk") == []


def test_residual_links_flags_unjustified_drop():
    rows = [{"row_id": "L7-0", "PF": "high", "residual_PF": "low",
             "respecifications": [], "residual_rationale": "x"}]
    assert any("no respecification" in i for i in _check_residual_links(rows))


def test_residual_links_ok_with_matching_measure():
    rows = [{"row_id": "L7-1", "PF": "high", "residual_PF": "low",
             "respecifications": ["(R1) retrain on cyclist data"],
             "residual_rationale": "R1 improves recall, lowering PF."}]
    assert _check_residual_links(rows) == []


def test_residual_links_requires_rationale():
    rows = [{"row_id": "L7-2", "PND": "high", "residual_PND": "low",
             "safety_functions": ["(SF1) staleness monitor"], "residual_rationale": ""}]
    assert any("residual_rationale is empty" in i for i in _check_residual_links(rows))


def test_triage_flags_hidden_vru():
    rows = [{"row_id": "L2-0", "guideword": "no", "failure_mode": "Missed detection of a pedestrian.",
             "potentially_dangerous": False}]
    assert _check_triage(rows)


def test_triage_ok_when_dangerous():
    rows = [{"row_id": "L2-1", "guideword": "no", "failure_mode": "Missed detection of a pedestrian.",
             "potentially_dangerous": True}]
    assert _check_triage(rows) == []


# --------------------------------------------------------------------------- #
# PART 6 — measures
# --------------------------------------------------------------------------- #

def test_measure_repetition_flags_copy_paste():
    rows = [_ctx(row_id=f"L6-{i}", respecifications=["[SG1] (R1) Retrain detector on cyclist data."])
            for i in range(4)]
    assert _check_measure_repetition(rows)


def test_measure_repetition_ok_when_distinct():
    texts = ["improve recall on occluded VRUs", "reduce detection latency below budget",
             "calibrate confidence on rare classes"]
    rows = [_ctx(row_id=f"L6-{i}", respecifications=[f"[SG1] (R{i}) {texts[i]}"]) for i in range(3)]
    assert _check_measure_repetition(rows) == []


def test_measure_category_flags_downstream_respec():
    rows = [{"row_id": "L6-0", "component_class": "Camera-ObjectDetection",
             "respecifications": ["[SG1] (R1) Add automatic emergency braking when an object is near."]}]
    assert _check_measure_category(rows)


def test_measure_category_ok_for_data_respec():
    rows = [{"row_id": "L6-1", "component_class": "Camera-ObjectDetection",
             "respecifications": ["[SG1] (R1) Retrain the detector on an augmented occluded-cyclist dataset."]}]
    assert _check_measure_category(rows) == []


# --------------------------------------------------------------------------- #
# PART 7 — evidence
# --------------------------------------------------------------------------- #

def test_evidence_flags_phantom_measure_id():
    rows = [{"row_id": "L8-0",
             "safety_functions": ["[SG1] (SF1) observability monitor"],
             "respecifications": ["[SG1] (R1) retrain"],
             "evidence": ["(SF1) fault injection test", "(SF9) nonexistent test"]}]
    assert any("not present" in i for i in _check_evidence(rows))


def test_evidence_flags_uncovered_measure():
    rows = [{"row_id": "L8-1",
             "safety_functions": ["[SG1] (SF1) monitor"],
             "respecifications": ["[SG1] (R1) retrain"],
             "evidence": ["(SF1) fault injection test"]}]
    assert any("have no evidence" in i for i in _check_evidence(rows))


def test_evidence_ok_when_all_covered():
    rows = [{"row_id": "L8-2",
             "safety_functions": ["[SG1] (SF1) monitor"],
             "respecifications": ["[SG1] (R1) retrain"],
             "evidence": ["(SF1) fault injection test on the monitor",
                          "(R1) dataset audit and recall benchmark"]}]
    assert _check_evidence(rows) == []


# --------------------------------------------------------------------------- #
# PART 8 — open assumptions
# --------------------------------------------------------------------------- #

def test_weak_assumption_flagged():
    rows = [{"row_id": "L8-0", "open_assumptions": ["The dataset audit accurately reflects performance."]}]
    assert _check_weak_assumptions(rows)


def test_concrete_assumption_passes():
    rows = [{"row_id": "L8-1", "open_assumptions": [
        "The LiDAR is recalibrated every 500 operating hours by the maintenance team."]}]
    assert _check_weak_assumptions(rows) == []


# --------------------------------------------------------------------------- #
# PART 1 / 9 — column parity and no "Dangerous"
# --------------------------------------------------------------------------- #

def _fields_js_columns():
    text = (ROOT / "src/gui/static/js/fields.js").read_text(encoding="utf-8")
    block = text.split("columns:", 1)[1].split("],", 1)[0]
    return re.findall(r"key:\s*'([^']+)'", block)


def test_cli_and_gui_columns_identical():
    cli_keys = [key for key, _ in _WORKSHEET_COLUMNS]
    assert _fields_js_columns() == cli_keys


def test_columns_contain_paper_fields_and_no_dangerous():
    cli_keys = [key for key, _ in _WORKSHEET_COLUMNS]
    for required in ("hazard_id", "acceptance_criterion", "risk_status", "residual_status",
                     "safety_decision", "evidence", "open_assumptions"):
        assert required in cli_keys
    assert "dangerous" not in cli_keys
    labels = [label.lower() for _, label in _WORKSHEET_COLUMNS]
    assert all("dangerous" not in lbl for lbl in labels)


def test_acceptance_criterion_text_format():
    txt = acceptance_criterion_text()
    assert "MEM target" in txt and "R" in txt


# --------------------------------------------------------------------------- #
# Safety-relevance gate — rows must not silently stop after L2 (architecture §6)
# --------------------------------------------------------------------------- #

def _complete_safety_row(**overrides):
    """A safety-relevant row that has its L1-L4 results filled (exportable by default)."""
    row = {
        "row_id": "L1-4",
        "component_class": "Camera-ObjectDetection",
        "failure_mode": "The object detector produces a missed detection for a cyclist crossing from the right.",
        "hazardous_behavior": "The shuttle continues into the crossing and collides with the cyclist.",
        "potential_harm": "A cyclist could be struck and suffer serious injury.",
        "potentially_dangerous": True,
        "initial_risk": 1e-6,
        "risk_status": "ACCEPTABLE",
        "safety_decision": "ACCEPT",
    }
    row.update(overrides)
    return row


def test_safety_relevant_cyclist_row_overrides_false_flag():
    """Test 1: a cyclist-collision row with potentially_dangerous=false is still safety-relevant
    and _stamp_triage flips the flag so the L2->L3 gate routes it onward."""
    row = _complete_safety_row(row_id="L1-4", potentially_dangerous=False)
    assert is_safety_relevant(row) is True
    stamped = _stamp_triage([row])[0]
    assert stamped["potentially_dangerous"] is True


def test_safety_relevant_row_missing_initial_risk_is_incomplete():
    """Test 2: safety-relevant row missing Initial risk is not exportable."""
    row = _complete_safety_row(initial_risk=None)
    assert is_safety_relevant(row) is True
    assert is_exportable_row(row) is False


def test_safety_relevant_row_missing_safety_decision_is_incomplete():
    """Test 3: safety-relevant row missing Safety decision is not exportable."""
    row = _complete_safety_row(safety_decision="")
    assert is_exportable_row(row) is False


def test_l3_empty_for_safety_relevant_row_is_incomplete():
    """Test 4: L3 produced nothing (no risk fields) for a safety-relevant row -> not exportable."""
    row = _complete_safety_row(potentially_dangerous=False, initial_risk=None,
                               risk_status="", safety_decision="")
    assert is_safety_relevant(row) is True   # caught by VRU/collision wording
    assert is_exportable_row(row) is False


def test_l4_empty_for_safety_relevant_row_is_incomplete():
    """Test 5: L3 ran but L4 produced no decision -> not exportable."""
    row = _complete_safety_row(safety_decision="")
    assert isinstance(row["initial_risk"], float)
    assert is_exportable_row(row) is False


def test_benign_row_may_stop_after_l2():
    """Regression: a genuinely non-safety-relevant row (no VRU/contact words, flag false) is not
    safety-relevant and is exportable as an L1/L2-only row (early-END is valid per architecture §6)."""
    row = {
        "row_id": "L1-9",
        "component_class": "Camera-ObjectDetection",
        "failure_mode": "The detector reports a slightly low confidence on a clearly empty road segment.",
        "hazardous_behavior": "The planner receives a marginally lower confidence on an empty area.",
        "potential_harm": "No road user is present so no harm results in this scenario.",
        "potentially_dangerous": False,
    }
    assert is_safety_relevant(row) is False
    assert is_exportable_row(row) is True


def test_check_triage_flags_collision_for_any_guideword():
    """_check_triage now fires for a collision/VRU row regardless of guideword (e.g. 'early')."""
    rows = [{"row_id": "L2-6", "guideword": "early",
             "hazardous_behavior": "The shuttle brakes prematurely then rolls into the cyclist.",
             "potential_harm": "A cyclist could be struck.", "potentially_dangerous": False}]
    assert _check_triage(rows)


def test_safety_relevant_recognises_collision_inflections():
    """Regex inflection fix: a rear-end-collision harm (plural, no VRU) is safety-relevant even when the
    LLM marked it not-dangerous (this was the L1-20 'unsafe fallback' bug)."""
    row = {"row_id": "L2-20", "guideword": "unsafe fallback",
           "hazardous_behavior": "The system's unsafe fallback causes sudden stops, risking rear-end collisions.",
           "potential_harm": "Following vehicles could be at risk of rear-end collisions due to sudden stops.",
           "potentially_dangerous": False}
    assert is_safety_relevant(row) is True
    assert _check_triage([row])          # mis-triaged dangerous=false -> flagged


def test_contact_terms_match_common_inflections():
    """collisions/injury/injured/injuries/impacts must all be recognised (previously missed)."""
    for harm in ("two cyclists suffered injuries", "the pedestrian was injured",
                 "multiple rear-end collisions", "several impacts occurred"):
        row = {"row_id": "L3-x", "hazardous_behavior": "", "potential_harm": harm,
               "potentially_dangerous": False}
        assert is_safety_relevant(row) is True, harm


def test_vru_plurals_are_safety_relevant():
    """The L1-2/9/17/20 regression: plural 'Pedestrians or cyclists' was missed by _VRU_RE."""
    row = {"row_id": "L2-2", "guideword": "more",
           "hazardous_behavior": "The shuttle makes abrupt maneuvers near the parked van.",
           "potential_harm": "Pedestrians or cyclists nearby could be startled or lose balance.",
           "potentially_dangerous": False}
    assert is_safety_relevant(row) is True


def test_safety_relevant_robust_without_keywords():
    """Content clause: a row with a real hazard AND a real harm is safety-relevant even with no VRU/
    contact keyword, so a regex miss can never silently drop it at L2."""
    row = {"row_id": "L3-y",
           "hazardous_behavior": "The system produces erratic vehicle behavior.",
           "potential_harm": "Passengers could be thrown forward by the abrupt motion.",
           "potentially_dangerous": False}
    assert is_safety_relevant(row) is True


def test_explicit_no_harm_may_still_stop_at_l2():
    """The only escape: a harm that explicitly negates harm stays non-safety-relevant (architecture §6)."""
    for harm in ("No road user is present so no harm results in this scenario.",
                 "Negligible effect, no safety impact.",
                 "Harmless degradation with no injury to anyone."):
        row = {"row_id": "L2-b",
               "hazardous_behavior": "Marginally lower confidence on an empty road segment.",
               "potential_harm": harm, "potentially_dangerous": False}
        assert is_safety_relevant(row) is False, harm


# --------------------------------------------------------------------------- #
# L1 rows must not be hard-dropped from L2 by a wording heuristic (the L1-14/20/21 bug)
# --------------------------------------------------------------------------- #

def _l1(fm, cls="Camera-ObjectDetection"):
    return {"row_id": "L1-x", "component_class": cls, "failure_mode": fm}


def test_overfitted_failure_mode_reaches_l2():
    """Test 1: 'overfitted' failure mode (mentions 'route' = the road) must pass the L1 gate."""
    fm = "Fails to detect a cyclist due to poor adaptation to new lighting conditions on the route."
    assert is_valid_l1_row(_l1(fm)) is True
    assert _check_component_level([_l1(fm)]) == []


def test_unsafe_fallback_failure_mode_reaches_l2():
    """Test 2: 'unsafe fallback' (degradation response, no detection keyword) must pass the L1 gate."""
    fm = "A sudden unsafe stop in traffic occurs as a degradation response."
    assert is_valid_l1_row(_l1(fm)) is True
    assert _check_component_level([_l1(fm)]) == []


def test_valid_but_unsafe_failure_mode_reaches_l2():
    """Test 3: 'valid but unsafe' (count-only output) must pass the L1 gate."""
    fm = "Using count alone without position, velocity, uncertainty, or freshness leads to unsafe outcomes."
    assert is_valid_l1_row(_l1(fm)) is True
    assert _check_component_level([_l1(fm)]) == []


def test_genuine_downstream_drift_still_flagged_but_not_dropped():
    """A real level drift is still flagged by the soft validator, but is_valid_l1_row no longer
    hard-drops it (wording is fixed by repair, not by deleting the row)."""
    fm = "The camera brakes too late and the vehicle does not stop for the cyclist."
    assert _check_component_level([_l1(fm)])          # soft validator flags it
    assert is_valid_l1_row(_l1(fm)) is True           # but the hard gate does not drop it


def test_incomplete_reason_names_failed_phase():
    """Test 4: an incomplete safety-relevant row reports a reason naming the failed phase; a
    complete row returns ''."""
    incomplete = _complete_safety_row(potentially_dangerous=False, initial_risk=None,
                                      risk_status="", safety_decision="")
    reason = incomplete_reason(incomplete)
    assert reason and ("L3" in reason or "L4" in reason or "L2" in reason)
    assert incomplete_reason(_complete_safety_row()) == ""

    l2_missing = _complete_safety_row(hazardous_behavior="", potential_harm="",
                                      potentially_dangerous=True, initial_risk=None,
                                      risk_status="", safety_decision="")
    assert "L2" in incomplete_reason(l2_missing)


# --------------------------------------------------------------------------- #
# Each safety goal must have >=1 Safety Function; may have multiple R/SF/P
# --------------------------------------------------------------------------- #

def _l6(goals, R=None, SF=None, P=None):
    return [{
        "row_id": "L6-0",
        "ai_safety_goals": goals,
        "respecifications": R or [],
        "safety_functions": SF or [],
        "passive_operational_measures": P or [],
    }]


def test_goal_coverage_flags_goal_without_safety_function():
    """The rigid bug: SG1 has only an R, SG3 only a P -> both flagged for missing SF."""
    rows = _l6(["SG1 a", "SG2 b", "SG3 c"],
               R=["[SG1] (R1) improve recall on occluded cyclists"],
               SF=["[SG2] (SF1) cross-sensor consistency monitor"],
               P=["[SG3] (P1) reduce speed in poor observability"])
    issues = _check_goal_coverage(rows)
    assert any("no safety function" in i for i in issues)


def test_goal_coverage_ok_when_every_goal_has_sf():
    rows = _l6(["SG1 a", "SG2 b"],
               SF=["[SG1] (SF1) absence-of-evidence monitor",
                   "[SG2] (SF1) staleness watchdog"])
    assert _check_goal_coverage(rows) == []


def test_goal_coverage_accepts_one_goal_with_sf_r_and_p():
    """A single goal may carry SF + R + P (multiple classes / multiple measures) — no false positive."""
    rows = _l6(["SG1 a"],
               R=["[SG1] (R1) retrain on occluded cyclists"],
               SF=["[SG1] (SF1) absence-of-evidence monitor",
                   "[SG1] (SF2) cross-sensor consistency check"],
               P=["[SG1] (P1) reduce speed where observability is poor"])
    assert _check_goal_coverage(rows) == []


def test_goal_coverage_still_flags_uncovered_goal_and_no_measures():
    """Regression: existing coverage behavior is intact."""
    assert any("have no measure" in i for i in _check_goal_coverage(
        _l6(["SG1 a", "SG2 b"], SF=["[SG1] (SF1) monitor"])))  # SG2 uncovered entirely
    assert any("no measures across" in i for i in _check_goal_coverage(_l6(["SG1 a"])))


# --------------------------------------------------------------------------- #
# A valid (even concise) failure mode must never be hard-dropped from L2-L8
# --------------------------------------------------------------------------- #

def test_concise_failure_mode_not_dropped():
    """The L1-10 bug: a valid 7-word failure mode was rejected by the 8-word hard gate."""
    fm = "Left/right confusion causes misinterpretation of cyclist position."   # 7 words
    assert is_valid_l1_row({"row_id": "L1-10", "component_class": "Camera-ObjectDetection",
                            "failure_mode": fm}) is True


def test_very_short_real_failure_mode_not_dropped():
    fm = "Detector misclassifies the cyclist."   # 4 words, real content
    assert is_valid_l1_row({"row_id": "L1-x", "component_class": "Camera-ObjectDetection",
                            "failure_mode": fm}) is True


def test_empty_or_meta_failure_mode_still_blocked():
    assert is_valid_l1_row({"row_id": "L1-a", "failure_mode": ""}) is False
    assert is_valid_l1_row({"row_id": "L1-b",
                            "failure_mode": "Ensure the failure mode is specific and detailed."}) is False
    assert "empty" in incomplete_reason({"row_id": "L1-c", "failure_mode": "",
                                         "potentially_dangerous": True})


def test_short_failure_mode_is_not_an_incomplete_reason():
    """A complete row with a concise failure mode is exportable -> no incomplete reason."""
    row = _complete_safety_row(failure_mode="Detector mislocates the cyclist by mirroring left and right.")
    assert is_exportable_row(row) is True
    assert incomplete_reason(row) == ""


# --------------------------------------------------------------------------- #
# Content-quality pass: ACCEPT evidence/assumptions, ODD, measure ids, wording
# --------------------------------------------------------------------------- #

_ODD = "University campus shuttle. Maximum 15 km/h. Daylight only. No dense fog."


def test_accept_evidence_must_not_cite_measures():
    # no measures (ACCEPT) but evidence cites (R2) -> flagged
    row = {"row_id": "L8-1", "safety_decision": "ACCEPT",
           "respecifications": [], "safety_functions": [], "passive_operational_measures": [],
           "evidence": ["Scenario validation confirms low exposure (R2)."]}
    assert any("ACCEPT" in i and "measure" in i.lower() for i in _check_evidence([row]))


def test_accept_evidence_must_not_name_measure_class():
    row = {"row_id": "L8-2", "safety_decision": "ACCEPT",
           "respecifications": [], "safety_functions": [], "passive_operational_measures": [],
           "evidence": ["Metamorphic testing acts as a Respecification of the model."]}
    assert any("measure class" in i for i in _check_evidence([row]))


def test_accept_evidence_clean_passes():
    row = {"row_id": "L8-3", "safety_decision": "ACCEPT",
           "respecifications": [], "safety_functions": [], "passive_operational_measures": [],
           "hazardous_behavior": "", "potential_harm": "",
           "evidence": ["Closed-loop scenario validation shows the hazard stays below the MEM target "
                        "within the daylight campus ODD and 15 km/h speed limit."]}
    assert _check_evidence([row]) == []


def test_accept_row_needs_open_assumptions():
    row = {"row_id": "L8-4", "safety_decision": "ACCEPT", "open_assumptions": []}
    assert _check_accept_assumptions([row])
    row_ok = {"row_id": "L8-5", "safety_decision": "ACCEPT",
              "open_assumptions": ["The ODD remains daylight, maximum 15 km/h, no dense fog."]}
    assert _check_accept_assumptions([row_ok]) == []


def test_serious_accept_needs_evidence_and_assumptions():
    row = {"row_id": "L8-6", "safety_decision": "ACCEPT", "risk_status": "ACCEPTABLE",
           "potential_harm": "A cyclist could be struck.", "evidence": [], "open_assumptions": []}
    issues = _check_accept_justification([row])
    assert any("no evidence" in i for i in issues)
    assert any("no open_assumptions" in i for i in issues)


def test_serious_accept_above_mem_is_flagged():
    row = {"row_id": "L8-7", "safety_decision": "ACCEPT", "risk_status": "ABOVE MEM TARGET",
           "potential_harm": "A cyclist could be struck.",
           "evidence": ["scenario validation"], "open_assumptions": ["ODD daylight only"]}
    assert any("below the MEM target" in i for i in _check_accept_justification([row]))


def test_odd_rejects_out_of_odd_failure_mode():
    row = {"row_id": "L1-15", "odd": _ODD,
           "failure_mode": "The detector performs poorly when detecting night cyclists in darkness."}
    assert any("out-of-ODD" in i for i in _check_odd_consistency([row], "failure_mode"))


def test_odd_allows_in_odd_variation():
    row = {"row_id": "L1-15b", "odd": _ODD,
           "failure_mode": "The detector performs poorly on partially occluded cyclists and cargo bikes."}
    assert _check_odd_consistency([row], "failure_mode") == []


def test_odd_allows_explicit_out_of_odd_restriction():
    row = {"row_id": "L1-15c", "odd": _ODD,
           "failure_mode": "Outside the ODD at night the detector is unsupported and operation is restricted."}
    assert _check_odd_consistency([row], "failure_mode") == []


def test_measure_id_uniqueness():
    dup = [{"row_id": "L6-1",
            "respecifications": ["[SG1] (R1) improve occluded-cyclist recall",
                                 "[SG2] (R1) reduce speed near crossings"],
            "safety_functions": [], "passive_operational_measures": []}]
    assert any("reuses measure id" in i for i in _check_measure_id_uniqueness(dup))
    uniq = [{"row_id": "L6-2",
             "respecifications": ["[SG1] (R1) improve recall", "[SG2] (R2) reduce speed"],
             "safety_functions": ["[SG1] (SF1) monitor"], "passive_operational_measures": []}]
    assert _check_measure_id_uniqueness(uniq) == []


def test_requirement_wording():
    bad = [{"row_id": "L6-3",
            "respecifications": [], "passive_operational_measures": [],
            "safety_functions": ["[SG1] (SF1) The system shall cross-sensor consistency check."]}]
    assert _check_requirement_wording(bad)
    good = [{"row_id": "L6-4",
             "respecifications": [], "passive_operational_measures": [],
             "safety_functions": ["[SG1] (SF1) The system shall perform a cross-sensor consistency "
                                  "check before publishing cyclist position."]}]
    assert _check_requirement_wording(good) == []


def test_evidence_repetition_flagged():
    same = "Dataset audit with a coverage matrix to show occluded cases are represented."
    rows = [{"row_id": f"L8-{i}", "component": "Cam", "aspect": "Out", "scenario": "Crossing",
             "evidence": [same]} for i in range(4)]
    assert any("reuses the same evidence" in i for i in _check_evidence_repetition(rows))
