# tests/test_run_pipeline.py
"""Tests for run_pipeline.py — load_functions, merge_l1_l2_l3, write_html, write_csv.

No LLM calls are made; only testing I/O helpers and merge logic.
"""
import csv
import json
import os
import pytest

from src.run_pipeline import load_functions, merge_l1_l2_l3, write_html, write_csv


# ============================================================================
#                         load_functions
# ============================================================================

class TestLoadFunctionsTxt:
    def test_txt_file(self, tmp_path):
        f = tmp_path / "funcs.txt"
        f.write_text("Braking\nSteering\n")
        assert load_functions(str(f)) == ["Braking", "Steering"]

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "funcs.txt"
        f.write_text("  Braking  \n  Steering  \n")
        assert load_functions(str(f)) == ["Braking", "Steering"]

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "funcs.txt"
        f.write_text("\n\n")
        with pytest.raises(ValueError, match="No functions"):
            load_functions(str(f))

    def test_list_extension(self, tmp_path):
        f = tmp_path / "funcs.list"
        f.write_text("A\nB\n")
        assert load_functions(str(f)) == ["A", "B"]


class TestLoadFunctionsYaml:
    def test_yaml_file(self, tmp_path):
        f = tmp_path / "funcs.yaml"
        f.write_text("functions:\n  - Braking\n  - Steering\n")
        assert load_functions(str(f)) == ["Braking", "Steering"]

    def test_yaml_missing_key_raises(self, tmp_path):
        f = tmp_path / "funcs.yaml"
        f.write_text("items:\n  - Braking\n")
        with pytest.raises(ValueError, match="YAML must contain"):
            load_functions(str(f))


class TestLoadFunctionsJson:
    def test_json_dict(self, tmp_path):
        f = tmp_path / "funcs.json"
        f.write_text(json.dumps({"functions": ["Braking", "Steering"]}))
        assert load_functions(str(f)) == ["Braking", "Steering"]

    def test_json_array(self, tmp_path):
        f = tmp_path / "funcs.json"
        f.write_text(json.dumps(["Braking", "Steering"]))
        assert load_functions(str(f)) == ["Braking", "Steering"]

    def test_json_bad_format_raises(self, tmp_path):
        f = tmp_path / "funcs.json"
        f.write_text(json.dumps({"data": "nope"}))
        with pytest.raises(ValueError, match="Bad JSON"):
            load_functions(str(f))


class TestLoadFunctionsUnsupported:
    def test_bad_extension_raises(self, tmp_path):
        f = tmp_path / "funcs.xml"
        f.write_text("<funcs/>")
        with pytest.raises(ValueError, match="Unsupported"):
            load_functions(str(f))


# ============================================================================
#                         merge_l1_l2_l3
# ============================================================================

class TestMergeL1L2L3:
    def _make_rows(self):
        l1 = [
            {"row_id": "L1-0", "function": "Braking", "guideword": "no",
             "deviation": "No braking action"},
            {"row_id": "L1-1", "function": "Braking", "guideword": "more",
             "deviation": "Excessive braking"},
        ]
        l2 = [
            {"row_id": "L2-0", "function": "Braking", "guideword": "no",
             "deviation": "No braking action", "cause": "Hydraulic leak"},
            {"row_id": "L2-1", "function": "Braking", "guideword": "more",
             "deviation": "Excessive braking", "cause": "Sensor fault"},
        ]
        l3 = [
            {"row_id": "L3-0", "function": "Braking", "guideword": "no",
             "deviation": "No braking action", "cause": "Hydraulic leak",
             "effect": "Cannot stop", "potentially_dangerous": True},
            {"row_id": "L3-1", "function": "Braking", "guideword": "more",
             "deviation": "Excessive braking", "cause": "Sensor fault",
             "effect": "Wheel lockup", "potentially_dangerous": False},
        ]
        return l1, l2, l3

    def test_merges_by_suffix(self):
        l1, l2, l3 = self._make_rows()
        merged = merge_l1_l2_l3(l1, l2, l3)
        assert len(merged) == 2
        assert merged[0]["function"] == "Braking"
        assert merged[0]["potentially_dangerous"] is True
        assert merged[1]["potentially_dangerous"] is False

    def test_missing_l2_handled(self):
        l1, _, l3 = self._make_rows()
        merged = merge_l1_l2_l3(l1, [], l3)
        assert len(merged) == 2
        assert merged[0]["cause"] == ""  # no L2 data → empty cause

    def test_missing_l3_handled(self):
        l1, l2, _ = self._make_rows()
        merged = merge_l1_l2_l3(l1, l2, [])
        assert len(merged) == 2
        assert merged[0]["effect"] == ""
        assert merged[0]["potentially_dangerous"] is False

    def test_prettify_removes_colons(self):
        l1 = [{"row_id": "L1-0", "function": "F", "guideword": "no",
               "deviation": "Label: description text"}]
        l2 = [{"row_id": "L2-0", "cause": "Cause: root issue"}]
        l3 = [{"row_id": "L3-0", "effect": "Effect: system impact",
               "potentially_dangerous": False}]
        merged = merge_l1_l2_l3(l1, l2, l3)
        assert ":" not in merged[0]["deviation"]
        assert ":" not in merged[0]["cause"]
        assert ":" not in merged[0]["effect"]

    def test_prettify_adds_period(self):
        l1 = [{"row_id": "L1-0", "function": "F", "guideword": "no",
               "deviation": "No braking"}]
        merged = merge_l1_l2_l3(l1, [], [])
        assert merged[0]["deviation"].endswith(".")


# ============================================================================
#                         write_html / write_csv
# ============================================================================

class TestWriteHtml:
    def test_writes_valid_html(self, tmp_path):
        rows = [{"function": "Braking", "guideword": "no", "deviation": "D",
                 "cause": "C", "effect": "E", "potentially_dangerous": True}]
        path = str(tmp_path / "out.html")
        write_html(rows, path)
        content = open(path, encoding="utf-8").read()
        assert "<table>" in content
        assert "Braking" in content
        assert "Dangerous" in content

    def test_contains_expected_columns(self, tmp_path):
        rows = [{"function": "F", "guideword": "no", "deviation": "D",
                 "cause": "C", "effect": "E", "potentially_dangerous": False}]
        path = str(tmp_path / "out.html")
        write_html(rows, path)
        content = open(path, encoding="utf-8").read()
        for col in ("Function", "Guideword", "Deviation", "Cause", "Effect"):
            assert col in content


class TestWriteCsv:
    def test_writes_valid_csv(self, tmp_path):
        rows = [{"function": "Braking", "guideword": "no", "deviation": "D",
                 "cause": "C", "effect": "E", "potentially_dangerous": True}]
        path = str(tmp_path / "out.csv")
        write_csv(rows, path)
        with open(path, encoding="utf-8") as f:
            reader = list(csv.reader(f))
        assert len(reader) == 2  # header + 1 data row
        assert reader[0][0] == "function"

    def test_correct_column_count(self, tmp_path):
        rows = [{"function": "F", "guideword": "g", "deviation": "D",
                 "cause": "C", "effect": "E", "potentially_dangerous": False}]
        path = str(tmp_path / "out.csv")
        write_csv(rows, path)
        with open(path, encoding="utf-8") as f:
            reader = list(csv.reader(f))
        assert len(reader[0]) == 6  # 6 columns
        assert len(reader[1]) == 6

    def test_dangerous_text(self, tmp_path):
        rows = [
            {"function": "F", "guideword": "g", "deviation": "D",
             "cause": "C", "effect": "E", "potentially_dangerous": True},
            {"function": "F", "guideword": "g", "deviation": "D",
             "cause": "C", "effect": "E", "potentially_dangerous": False},
        ]
        path = str(tmp_path / "out.csv")
        write_csv(rows, path)
        with open(path, encoding="utf-8") as f:
            reader = list(csv.reader(f))
        assert reader[1][-1] == "Dangerous"
        assert reader[2][-1] == "Not dangerous"
