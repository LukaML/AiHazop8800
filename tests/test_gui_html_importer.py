"""Regression tests for the current 21-column GUI HTML import round trip."""
from copy import deepcopy
import json
import re

import pytest

from src.gui.models import Rating
from src.gui.services.html_exporter import export_to_html
from src.gui.services.html_importer import build_states_from_rows, parse_hazop_html
from src.gui.services.state_manager import RowState, StateManager


CONTEXT = {
    "component": "Camera Object Detection",
    "component_class": "Camera-ObjectDetection",
    "aspect": "Outputs",
    "odd": "Campus shuttle in daylight.",
    "scenario": "A cyclist emerges from behind a parked van.",
}


def _complete_final(**overrides):
    final = {
        **CONTEXT,
        "hazard_id": "L1-1",
        "guideword": "uncertain_but_confident",
        "failure_mode": "The detector reports high confidence for an ambiguous cyclist.",
        "hazardous_behavior": "The shuttle proceeds despite unreliable perception.",
        "potential_harm": "The shuttle could strike the cyclist.",
        "potentially_dangerous": True,
        "E": 0.2,
        "PF": 0.3,
        "PND": 0.4,
        "PNM": 0.5,
        "S": 0.8,
        "initial_risk": 4.8e-4,
        "risk_status": "ABOVE MEM TARGET",
        "risk_rationale": "The cyclist is exposed to a collision hazard.",
        "acceptance_criterion": "MEM target R <= 1.0e-05",
        "safety_decision": "IMPROVE",
        "acceptance_rationale": "The initial risk exceeds the MEM target.",
        "ai_safety_goals": [
            "Ensure ambiguous cyclists are handled conservatively.",
            "Prevent unsafe motion when confidence is poorly calibrated.",
        ],
        "respecifications": [
            "[SG1] (R1) Retrain on ambiguous cyclist examples.",
            "[SG2] (R2) Calibrate confidence on OOD inputs.",
        ],
        "safety_functions": [
            "[SG1] (SF1) Monitor perception uncertainty.",
            "[SG2] (SF2) Override unsafe proceed commands.",
        ],
        "passive_operational_measures": [
            "[SG1] (P1) Reduce speed near occlusions.",
        ],
        "residual_E": 0.1,
        "residual_PF": 0.1,
        "residual_PND": 0.1,
        "residual_PNM": 0.1,
        "residual_S": 0.8,
        "residual_risk": 4.32e-6,
        "residual_status": "ACCEPTABLE",
        "residual_rationale": "The monitor and speed restriction reduce risk.",
        "evidence": [
            "(R1) Dataset coverage audit.",
            "(SF1) Fault-injection test.",
        ],
        "open_assumptions": [
            "The uncertainty estimate remains calibrated.",
            "The planner obeys the monitor override.",
        ],
    }
    final.update(overrides)
    return final


@pytest.fixture
def exported_round_trip(monkeypatch):
    """Export two modern rows, one deliberately marked incomplete."""
    source = StateManager()
    run_id = source.create_run(
        provider="openai",
        model="test-model",
        contexts=[CONTEXT],
        notes="round-trip regression",
    )
    run = source.get_run(run_id)

    complete = _complete_final()
    incomplete = _complete_final(
        hazard_id="L1-2",
        guideword="late",
        failure_mode="The cyclist detection arrives after the stopping deadline.",
    )
    run.rows = {
        "c0__L1-0": RowState(
            row_id="c0__L1-0",
            display_id="L1-1",
            component_index=0,
            component=CONTEXT["component"],
            guideword=complete["guideword"],
            original=deepcopy(complete),
            final=deepcopy(complete),
            rating=Rating.PARTIALLY_CORRECT,
            complete=True,
        ),
        "c0__L1-1": RowState(
            row_id="c0__L1-1",
            display_id="L1-2",
            component_index=0,
            component=CONTEXT["component"],
            guideword=incomplete["guideword"],
            original=deepcopy(incomplete),
            final=deepcopy(incomplete),
            rating=Rating.UNRATED,
            complete=False,
        ),
    }

    monkeypatch.setattr("src.gui.services.html_exporter.state_manager", source)
    return export_to_html(run_id), complete


class TestModernExporterRoundTrip:
    def test_export_parse_build_and_state_manager_preserve_final_semantics(
        self, exported_round_trip
    ):
        html, expected = exported_round_trip

        # The visible worksheet keeps the human-readable state, while the
        # versioned payload preserves fields not represented by table columns.
        assert "ai-hazop-data" in html
        assert re.search(r"<tr[^>]+class=['\"][^'\"]*dangerous[^'\"]*['\"]", html)
        assert "dangerous incomplete" in html
        assert ">Partial</td>" in html

        parsed = parse_hazop_html(html)
        assert len(parsed) == 2
        assert parsed[0]["guideword"] == "uncertain_but_confident"
        assert parsed[0]["initial_risk"] == pytest.approx(4.8e-4)
        assert parsed[0]["residual_risk"] == pytest.approx(4.32e-6)
        assert parsed[0]["ai_safety_goals"] == expected["ai_safety_goals"]
        assert parsed[0]["respecifications"] == expected["respecifications"]
        assert parsed[0]["safety_functions"] == expected["safety_functions"]
        assert (
            parsed[0]["passive_operational_measures"]
            == expected["passive_operational_measures"]
        )
        assert parsed[0]["evidence"] == expected["evidence"]
        assert parsed[0]["open_assumptions"] == expected["open_assumptions"]
        assert parsed[0]["_rating"] == "partially_correct"
        assert parsed[0]["_complete"] is True
        assert parsed[1]["_complete"] is False

        contexts, states = build_states_from_rows(parsed)
        assert contexts == [CONTEXT]
        assert len(states) == 1

        restored = StateManager()
        restored_run_id = restored.create_run(
            provider="imported",
            model="imported",
            contexts=contexts,
            notes="Imported from HTML",
        )
        restored.set_states(restored_run_id, states)
        restored_rows = restored.get_all_rows(restored_run_id)

        assert len(restored_rows) == 2
        first, second = restored_rows
        assert first.rating is Rating.PARTIALLY_CORRECT
        assert first.complete is True
        assert second.complete is False
        assert first.final["potentially_dangerous"] is True
        assert first.final["guideword"] == "uncertain_but_confident"
        assert first.final["initial_risk"] == pytest.approx(4.8e-4)
        assert first.final["residual_risk"] == pytest.approx(4.32e-6)
        for key in (
            "E",
            "PF",
            "PND",
            "PNM",
            "S",
            "risk_rationale",
            "acceptance_rationale",
            "residual_E",
            "residual_PF",
            "residual_PND",
            "residual_PNM",
            "residual_S",
            "residual_rationale",
            "ai_safety_goals",
            "respecifications",
            "safety_functions",
            "passive_operational_measures",
            "evidence",
            "open_assumptions",
        ):
            assert first.final[key] == expected[key]


HEADERS = (
    "Hazard ID",
    "Component",
    "Class",
    "Aspect",
    "ODD",
    "Scenario",
    "Guideword/Question",
    "Failure mode",
    "Hazardous behavior",
    "Potential harm",
    "Initial risk",
    "Risk Status",
    "Acceptance criterion",
    "Safety decision",
    "AI Safety Goal",
    "Measures",
    "Residual risk",
    "Residual Status",
    "Evidence",
    "Open assumptions",
    "Rating",
)


def _fallback_row(
    *,
    hazard_id="L1-1",
    component="Camera Object Detection",
    component_class="Camera-ObjectDetection",
    aspect="Outputs",
    odd="Campus shuttle in daylight.",
    scenario="A cyclist emerges from behind a parked van.",
    row_class="dangerous incomplete",
):
    cells = (
        hazard_id,
        component,
        component_class,
        aspect,
        odd,
        scenario,
        "<code>uncertain but confident</code>",
        "The detector is confidently wrong.",
        "The shuttle proceeds into the crossing.",
        "The cyclist could be struck.",
        "4.80e-04",
        "ABOVE MEM TARGET",
        "MEM target R &le; 1.0e-05",
        "IMPROVE",
        (
            "<div><b>SG1</b>Handle ambiguous cyclists conservatively.</div>"
            "<div><b>SG2</b>Prevent unsafe proceed commands.</div>"
        ),
        (
            "<div><div>SG1</div>"
            "<div><b>R1</b>Retrain on ambiguous cyclists.</div>"
            "<div><b>SF1</b>Monitor uncertainty.</div>"
            "<div><b>P1</b>Reduce speed near occlusions.</div></div>"
            "<div><div>SG2</div>"
            "<div><b>R2</b>Calibrate OOD confidence.</div>"
            "<div><b>SF2</b>Override unsafe commands.</div></div>"
        ),
        "4.32e-06",
        "ACCEPTABLE",
        (
            "<div><b>EV1</b>(R1) Dataset audit.</div>"
            "<div><b>EV2</b>(SF1) Fault-injection test.</div>"
        ),
        (
            "<div><b>A1</b>Uncertainty remains calibrated.</div>"
            "<div><b>A2</b>The planner obeys overrides.</div>"
        ),
        "Partial",
    )
    assert len(cells) == len(HEADERS) == 21
    attrs = f' class="{row_class}"' if row_class else ""
    return f"<tr{attrs}>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _fallback_html(*rows):
    header = "".join(f"<th>{label}</th>" for label in HEADERS)
    return (
        "<!doctype html><html><body><table>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></body></html>"
    )


class TestCurrentTableFallback:
    def test_parses_tr_classes_numeric_values_lists_and_measure_categories(self):
        rows = parse_hazop_html(_fallback_html(_fallback_row()))

        assert len(rows) == 1
        row = rows[0]
        assert row["potentially_dangerous"] is True
        assert row["_complete"] is False
        assert row["_rating"] == "partially_correct"
        assert row["guideword"] == "uncertain_but_confident"
        assert row["initial_risk"] == pytest.approx(4.8e-4)
        assert row["residual_risk"] == pytest.approx(4.32e-6)
        assert row["ai_safety_goals"] == [
            "Handle ambiguous cyclists conservatively.",
            "Prevent unsafe proceed commands.",
        ]
        assert row["respecifications"] == [
            "[SG1] (R1) Retrain on ambiguous cyclists.",
            "[SG2] (R2) Calibrate OOD confidence.",
        ]
        assert row["safety_functions"] == [
            "[SG1] (SF1) Monitor uncertainty.",
            "[SG2] (SF2) Override unsafe commands.",
        ]
        assert row["passive_operational_measures"] == [
            "[SG1] (P1) Reduce speed near occlusions."
        ]
        assert row["evidence"] == [
            "(R1) Dataset audit.",
            "(SF1) Fault-injection test.",
        ]
        assert row["open_assumptions"] == [
            "Uncertainty remains calibrated.",
            "The planner obeys overrides.",
        ]

    def test_same_component_with_distinct_full_contexts_stays_separate(self):
        html = _fallback_html(
            _fallback_row(
                hazard_id="L1-0",
                component="Shared Planner",
                component_class="Planning-Behavior",
                odd="Daylight campus operation.",
                scenario="A cyclist crosses.",
                row_class="dangerous",
            ),
            _fallback_row(
                hazard_id="L1-0",
                component="Shared Planner",
                component_class="Planning-Behavior",
                odd="Night operation in rain.",
                scenario="A pedestrian crosses.",
                row_class="dangerous",
            ),
        )

        contexts, states = build_states_from_rows(parse_hazop_html(html))

        assert len(contexts) == len(states) == 2
        assert contexts[0]["component"] == contexts[1]["component"] == "Shared Planner"
        assert contexts[0]["odd"] == "Daylight campus operation."
        assert contexts[1]["odd"] == "Night operation in rain."
        assert contexts[0]["scenario"] == "A cyclist crosses."
        assert contexts[1]["scenario"] == "A pedestrian crosses."
        assert len(states[0]["rows_l1"]) == len(states[1]["rows_l1"]) == 1

    def test_legacy_table_without_completeness_metadata_defers_validation(self):
        html = """<table><thead><tr>
        <th>Function</th><th>Guideword</th><th>Deviation</th><th>Cause</th>
        <th>Effect</th><th>Dangerous</th></tr></thead><tbody><tr>
        <td>Braking</td><td>no</td><td>No braking</td><td>Leak</td>
        <td>Cannot stop</td><td>Dangerous</td></tr></tbody></table>"""

        assert parse_hazop_html(html)[0]["_complete"] is None


def _payload_html(payload_text):
    return (
        "<!doctype html><html><body>"
        '<script id="ai-hazop-data" type="application/json">'
        f"{payload_text}</script>"
        "</body></html>"
    )


class TestInvalidInput:
    @pytest.mark.parametrize("payload_text", ("", "{not valid json"))
    def test_empty_or_malformed_embedded_payload_is_rejected(self, payload_text):
        with pytest.raises(ValueError, match="Invalid embedded AI-HAZOP data"):
            parse_hazop_html(_payload_html(payload_text))

    def test_valid_payload_schema_with_no_rows_is_rejected(self):
        payload = json.dumps(
            {"schema": "ai-hazop-8800", "version": 1, "contexts": [], "rows": []}
        )
        with pytest.raises(ValueError, match="No data rows"):
            parse_hazop_html(_payload_html(payload))

    def test_non_finite_payload_number_is_rejected(self):
        payload = json.dumps(
            {
                "schema": "ai-hazop-8800",
                "version": 1,
                "contexts": [CONTEXT],
                "rows": [{"final": {**_complete_final(), "initial_risk": float("nan")}}],
            }
        )
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            parse_hazop_html(_payload_html(payload))

    def test_non_finite_table_number_is_rejected(self):
        html = _fallback_html(_fallback_row()).replace("4.80e-04", "Infinity", 1)
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            parse_hazop_html(html)

    def test_empty_table_is_rejected(self):
        with pytest.raises(ValueError, match="No data rows"):
            parse_hazop_html(_fallback_html())

    def test_missing_table_and_payload_is_rejected(self):
        with pytest.raises(ValueError, match="No <tbody>"):
            parse_hazop_html("<html><body><p>No worksheet here</p></body></html>")
