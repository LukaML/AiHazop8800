# tests/conftest.py
"""Shared pytest fixtures and LLM mocks for HazopLLM tests."""
import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

from src.models import Guideword


# ============================================================================
#                            ROW FIXTURES
# ============================================================================

@pytest.fixture
def sample_l1_row() -> Dict[str, Any]:
    """A valid L1 row dict."""
    return {
        "row_id": "L1-0",
        "function": "Braking",
        "guideword": "no",
        "deviation": "No braking action occurs when brake pedal is pressed",
    }


@pytest.fixture
def sample_l1_row_empty_deviation() -> Dict[str, Any]:
    """An L1 row with empty deviation field (should fail validation)."""
    return {
        "row_id": "L1-0",
        "function": "Braking",
        "guideword": "no",
        "deviation": "",
    }


@pytest.fixture
def sample_l2_row(sample_l1_row) -> Dict[str, Any]:
    """A valid L2 row dict (extends L1 with cause)."""
    return {
        **sample_l1_row,
        "row_id": "L2-0",
        "cause": "Hydraulic fluid leak in brake line",
    }


@pytest.fixture
def sample_l2_row_empty_cause(sample_l1_row) -> Dict[str, Any]:
    """An L2 row with empty cause field (should fail validation)."""
    return {
        **sample_l1_row,
        "row_id": "L2-0",
        "cause": "",
    }


@pytest.fixture
def sample_l3_row(sample_l2_row) -> Dict[str, Any]:
    """A valid L3 row dict (extends L2 with effect and triage)."""
    return {
        **sample_l2_row,
        "row_id": "L3-0",
        "effect": "Vehicle cannot stop, leading to potential collision",
        "potentially_dangerous": True,
    }


@pytest.fixture
def sample_l3_row_empty_effect(sample_l2_row) -> Dict[str, Any]:
    """An L3 row with empty effect field (should fail validation)."""
    return {
        **sample_l2_row,
        "row_id": "L3-0",
        "effect": "",
        "potentially_dangerous": True,
    }


@pytest.fixture
def sample_l3_row_missing_triage(sample_l2_row) -> Dict[str, Any]:
    """An L3 row without the potentially_dangerous field."""
    return {
        **sample_l2_row,
        "row_id": "L3-0",
        "effect": "Vehicle cannot stop",
        # Missing: "potentially_dangerous"
    }


@pytest.fixture
def all_guidewords() -> List[str]:
    """List of all 11 HAZOP guideword string values."""
    return [g.value for g in Guideword]


@pytest.fixture
def sample_functions() -> List[str]:
    """Sample list of system functions for testing."""
    return ["Braking", "Steering"]


@pytest.fixture
def sample_guideword_map() -> Dict[str, str]:
    """Sample guideword map for testing."""
    return {g.value: g.value for g in Guideword}


# ============================================================================
#                         FULL COVERAGE ROW SETS
# ============================================================================

@pytest.fixture
def full_l1_coverage_rows(sample_functions, all_guidewords) -> List[Dict[str, Any]]:
    """Generate L1 rows with full guideword coverage for all functions."""
    rows = []
    idx = 0
    for fn in sample_functions:
        for gw in all_guidewords:
            rows.append({
                "row_id": f"L1-{idx}",
                "function": fn,
                "guideword": gw,
                "deviation": f"Deviation for {fn} with {gw}",
            })
            idx += 1
    return rows


@pytest.fixture
def full_l2_coverage_rows(full_l1_coverage_rows) -> List[Dict[str, Any]]:
    """Generate L2 rows from L1 rows with cause added."""
    return [
        {
            **r,
            "row_id": r["row_id"].replace("L1-", "L2-"),
            "cause": f"Root cause for {r['deviation']}",
        }
        for r in full_l1_coverage_rows
    ]


@pytest.fixture
def full_l3_coverage_rows(full_l2_coverage_rows) -> List[Dict[str, Any]]:
    """Generate L3 rows from L2 rows with effect and triage added."""
    return [
        {
            **r,
            "row_id": r["row_id"].replace("L2-", "L3-"),
            "effect": f"System effect: {r['cause']}",
            "potentially_dangerous": True,
        }
        for r in full_l2_coverage_rows
    ]


# ============================================================================
#                            LLM MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_chat_json(mocker):
    """Mock the chat_json function in llm_client module."""
    return mocker.patch("src.llm_client.chat_json")


@pytest.fixture
def mock_chat_raw(mocker):
    """Mock the chat_raw function in llm_client module."""
    return mocker.patch("src.llm_client.chat_raw")


@pytest.fixture
def mock_openai_client(mocker):
    """Mock OpenAI client for llm_client tests.

    Returns a mock that simulates OpenAI API responses.
    """
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='[{"row_id": "L1-0", "function": "Test", "guideword": "no", "deviation": "Test deviation"}]'))
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    return mocker.patch("src.llm_client.OpenAI", return_value=mock_client)


@pytest.fixture
def mock_llm_reviewer_ok() -> Dict[str, Any]:
    """Sample reviewer OK response."""
    return {
        "decision": "OK",
        "scope": "L1",
        "issues": [],
        "suggestions": [],
    }


@pytest.fixture
def mock_llm_reviewer_return() -> Dict[str, Any]:
    """Sample reviewer RETURN response (needs regeneration)."""
    return {
        "decision": "RETURN",
        "scope": "L1",
        "issues": ["Row L1-0 has empty deviation"],
        "suggestions": ["Provide a specific deviation description"],
    }


@pytest.fixture
def mock_llm_reviewer_escalate() -> Dict[str, Any]:
    """Sample reviewer ESCALATE response."""
    return {
        "decision": "ESCALATE",
        "scope": "L2",
        "issues": ["Cause-effect inconsistency detected"],
        "suggestions": ["Review L1 deviations"],
    }


# ============================================================================
#                         GRAPH STATE FIXTURES
# ============================================================================

@pytest.fixture
def minimal_graph_state(sample_functions, sample_guideword_map) -> Dict[str, Any]:
    """Minimal valid graph state for testing."""
    return {
        "functions": sample_functions,
        "guideword_map": sample_guideword_map,
        "max_devs_per_gw": 1,
        "notes": "Test system notes",
        "rag_enabled": False,
        "rows_l1": [],
        "rows_l2": [],
        "rows_l3": [],
        "l1_ok": False,
        "l1_issues": [],
        "l1_repair_round": 0,
        "l2_ok": False,
        "l2_issues": [],
        "l2_repair_round": 0,
        "l3_ok": False,
        "l3_issues": [],
        "l3_repair_round": 0,
        "holistic_ok": False,
        "holistic_round": 0,
    }


@pytest.fixture
def graph_state_after_l1(minimal_graph_state, full_l1_coverage_rows) -> Dict[str, Any]:
    """Graph state after L1 generation (rows_l1 populated)."""
    return {
        **minimal_graph_state,
        "rows_l1": full_l1_coverage_rows,
        "l1_ok": True,
        "l1_issues": [],
    }


@pytest.fixture
def graph_state_after_l2(graph_state_after_l1, full_l2_coverage_rows) -> Dict[str, Any]:
    """Graph state after L2 generation (rows_l1 and rows_l2 populated)."""
    return {
        **graph_state_after_l1,
        "rows_l2": full_l2_coverage_rows,
        "l2_ok": True,
        "l2_issues": [],
    }


@pytest.fixture
def graph_state_after_l3(graph_state_after_l2, full_l3_coverage_rows) -> Dict[str, Any]:
    """Graph state after L3 generation (all rows populated)."""
    return {
        **graph_state_after_l2,
        "rows_l3": full_l3_coverage_rows,
        "l3_ok": True,
        "l3_issues": [],
    }


# ============================================================================
#                         PYDANTIC MODEL FIXTURES
# ============================================================================

@pytest.fixture
def sample_l1_pydantic():
    """Sample L1 row as Pydantic model."""
    from src.models import HazopL1Row
    return HazopL1Row(
        row_id="L1-0",
        function="Braking",
        guideword=Guideword.NO,
        deviation="No braking action",
    )


@pytest.fixture
def sample_l2_pydantic():
    """Sample L2 row as Pydantic model."""
    from src.models import HazopL2Row
    return HazopL2Row(
        row_id="L2-0",
        function="Braking",
        guideword=Guideword.NO,
        deviation="No braking action",
        cause="Hydraulic failure",
    )


@pytest.fixture
def sample_l3_pydantic():
    """Sample L3 row as Pydantic model."""
    from src.models import HazopL3Row
    return HazopL3Row(
        row_id="L3-0",
        function="Braking",
        guideword=Guideword.NO,
        deviation="No braking action",
        cause="Hydraulic failure",
        effect="Vehicle cannot stop",
        potentially_dangerous=True,
    )


# ============================================================================
#                         CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def reset_llm_config():
    """Reset LLM configuration before each test."""
    import src.llm_client as llm_client
    # Store original state
    original_config = llm_client._config
    original_client = llm_client._client

    yield

    # Restore original state after test
    llm_client._config = original_config
    llm_client._client = original_client


@pytest.fixture
def configured_llm(mocker):
    """Configure LLM client with mocked OpenAI for tests."""
    # Mock environment variable
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"})

    # Mock OpenAI client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="[]"))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mocker.patch("src.llm_client.OpenAI", return_value=mock_client)

    # Configure
    import src.llm_client as llm_client
    llm_client.configure(provider="openai")

    return mock_client
