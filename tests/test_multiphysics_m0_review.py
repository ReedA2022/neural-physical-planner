"""Independent M0 review checks: source fidelity, proposal boundaries and drift.

These checks validate the integration and compatibility gate, not the proposed
multiphysics mechanisms or the literature claims retained from the attachment.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import yaml

from npp.config import load_inputs
from npp.designer import validate_design_spec
from npp.models import InputValidationError
from npp.technology import validate_technology


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/multiphysics"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
spec = importlib.util.spec_from_file_location(
    "m0_review_baseline_checker", ROOT / "scripts/check_multiphysics_baseline.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def normalized_prose(text):
    # Strip presentation syntax only; preserve actual words and punctuation.
    text = re.sub(r"\[([^\]]*)\]\(https?://[^)]*\)", r"\1", text)
    return re.sub(r"[\s*_`#>|\\]", "", text)


class MultiphysicsM0ReviewTests(unittest.TestCase):
    def test_authoritative_source_and_transcription_integrity(self):
        metadata = json.loads((DOCS / "source/metadata.json").read_text())
        source = DOCS / "source" / metadata["source_filename"]
        transcription = DOCS / "SPECIFICATION_V1.md"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), metadata["source_sha256"])
        self.assertEqual(hashlib.sha256(transcription.read_bytes()).hexdigest(), metadata["transcription_sha256"])
        with ZipFile(source) as archive:
            document = ET.fromstring(archive.read("word/document.xml"))
            relationships = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        markdown = transcription.read_text()
        self.assertEqual(len(document.findall(".//w:tbl", NS)), 10)
        self.assertEqual(len(document.findall(".//m:oMath", NS)), 7)
        self.assertEqual(len(re.findall(r"^\$\$.*\$\$$", markdown, re.MULTILINE)), 7)
        table_separators = re.findall(r"^\|(?:[ :\-]+\|)+$", markdown, re.MULTILINE)
        self.assertEqual(len(table_separators), 10)
        # All non-equation text, including the original title/subtitle and every
        # table cell, survives. Equations receive a separate manual math review.
        plain_markdown = normalized_prose(markdown)
        checked = 0
        for paragraph in document.findall(".//w:p", NS):
            if paragraph.find(".//m:oMath", NS) is not None:
                continue
            text = "".join(node.text or "" for node in paragraph.findall(".//w:t", NS))
            if text.strip():
                checked += 1
                self.assertIn(normalized_prose(text), plain_markdown, text)
        self.assertEqual(checked, 480)
        links = [r.attrib["Target"] for r in relationships if r.attrib.get("TargetMode") == "External"]
        self.assertEqual(len(links), 15)
        for link in links:
            self.assertIn(link, markdown)

    def test_every_requirement_has_one_reconciled_row_and_all_families_remain(self):
        supplied = (DOCS / "SPECIFICATION_V1.md").read_text()
        inventory = (DOCS / "REQUIREMENTS.md").read_text()
        required = [f"REQ-{i:02}" for i in range(1, 38)]
        self.assertEqual(re.findall(r"\*\*(REQ-\d{2})\.", supplied), required)
        self.assertEqual(re.findall(r"^\| \*\*(REQ-\d{2}) · (?:Partial|Missing)\*\*", inventory, re.MULTILINE), required)
        registry = json.loads((DOCS / "experiments.json").read_text())
        self.assertEqual(registry["release_scope"]["full_initial_scope"]["families"],
                         ["digital_electronics", "analog_electronics", "classical_photonics",
                          "acoustics_phononics", "quantum"])

    def test_registry_has_no_executed_results_or_resolved_scenario_defaults(self):
        registry = json.loads((DOCS / "experiments.json").read_text())
        self.assertFalse(registry["planner_input_format"])
        self.assertEqual(registry["execution_status"], "not_executable")
        self.assertEqual(registry["results"], [])
        self.assertEqual([e["id"] for e in registry["experiments"]], [f"H{i}" for i in range(1, 7)])
        for experiment in registry["experiments"]:
            self.assertEqual(experiment["status"], "proposed")
            self.assertEqual(experiment["execution_status"], "not_executable")
            self.assertEqual(experiment["results"], [])
            self.assertEqual(experiment["quality_criteria"]["status"], "unresolved")
            self.assertIsNone(experiment["quality_criteria"]["thresholds"])
            for sweep in experiment["parameter_sweeps"]:
                self.assertEqual(sweep["status"], "unresolved")
                self.assertIsNone(sweep["domain"])
                self.assertTrue(sweep["unresolved_obligation"])
            for baseline in experiment["baselines"]:
                self.assertEqual(baseline["execution_status"], "not_executed")
            for blocker in experiment["blocked_by"]:
                self.assertEqual(blocker["status"], "unmet")
        # Resolve every internal JSON reference rather than assuming duplicated
        # shared protocol names point to actual definitions.
        def walk(value):
            if isinstance(value, dict):
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
            elif isinstance(value, str) and value.startswith("#/"):
                target = registry
                for key in value[2:].split("/"):
                    target = target[key.replace("~1", "/").replace("~0", "~")]
                self.assertIsNotNone(target)
        walk(registry)

    def test_actual_proposed_examples_are_rejected_by_current_entry_points(self):
        markdown = (DOCS / "SPECIFICATION_V1.md").read_text()
        blocks = re.findall(r"(?:^>     .*\n)+", markdown, re.MULTILINE)
        self.assertEqual(len(blocks), 2)
        project, component = [yaml.safe_load("\n".join(line[6:] for line in block.splitlines()))
                              for block in blocks]
        self.assertEqual(project["schema_version"], "proposed/npp-multiphysics-1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.yaml"
            path.write_text(yaml.safe_dump(project))
            with self.assertRaises(InputValidationError):
                load_inputs(project_path=path)
        with self.assertRaises(InputValidationError):
            validate_technology(component["component"])
        with self.assertRaises(InputValidationError):
            validate_design_spec(json.loads((DOCS / "experiments.json").read_text()))

    def test_gate_detects_changed_normalization_before_legacy_replanning(self):
        original = gate.load_inputs
        def changed(**kwargs):
            result = deepcopy(original(**kwargs))
            result[0]["name"] += " changed by review fault injection"
            return result
        with patch.object(gate, "load_inputs", side_effect=changed):
            report = gate.check_baseline()
        self.assertEqual(report["status"], "failed")
        cases = {case["case"]: case for case in report["checks"]}
        for name in ("demo", "friendly-v0.2"):
            self.assertEqual(cases[f"{name}-input-meanings"]["status"], "failed")
            self.assertEqual(cases[f"{name}-macro-output"]["status"], "passed")

    def test_gate_detects_schema_only_semantic_drift(self):
        schema = gate.design_schema()
        schema["properties"]["max_evaluations"]["maximum"] = 127
        with patch.object(gate, "design_schema", return_value=schema):
            report = gate.check_baseline()
        self.assertEqual(report["status"], "failed")
        failures = [case for case in report["checks"] if case["status"] == "failed"]
        self.assertEqual([case["case"] for case in failures], ["schema-physical-design"])
        self.assertEqual(failures[0]["differences"][0]["path"], "$.properties.max_evaluations.maximum")


if __name__ == "__main__":
    unittest.main()
