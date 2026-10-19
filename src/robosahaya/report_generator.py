import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from robosahaya.evidence_models import TaskEvent, PatchProposal, TestResult, DiagnosisResponse

class ReportGenerator:
    """Generates structured, auditable Markdown and JSON engineering reports."""

    _SECRET_PATTERNS = [
        re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
        re.compile(r"(?i)((?:api[_-]?key|token|secret|password)\s*[:=]\s*)[^\s,;]+"),
        re.compile(r"(?i)(x-api-key-id\s*[:=]\s*)[^\s,;]+"),
    ]

    @classmethod
    def _redact(cls, value: Any) -> str:
        text = str(value)
        for pattern in cls._SECRET_PATTERNS:
            text = pattern.sub(r"\1[REDACTED]", text)
        return text

    @classmethod
    def _scrub(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._scrub(item) for item in value]
        if isinstance(value, str):
            return cls._redact(value)
        return value

    @staticmethod
    def generate_markdown_report(
        task_id: str,
        query: str,
        events: List[TaskEvent],
        initial_test: Optional[TestResult] = None,
        diagnosis: Optional[DiagnosisResponse] = None,
        patch_proposal: Optional[PatchProposal] = None,
        patch_approved: bool = False,
        retest_result: Optional[TestResult] = None,
        mode: str = "simulation"
    ) -> str:
        if mode in {"mock", "fixture", "replay"}:
            mode = "replay"
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        if mode == "replay":
            status_banner = "🧪 REPLAY ONLY — NO LIVE VERIFICATION"
        elif retest_result and retest_result.passed:
            status_banner = "✅ RESOLVED & VERIFIED"
        elif initial_test and initial_test.passed:
            status_banner = "✅ LIVE SENSOR TESTS PASSED"
        elif retest_result and not retest_result.passed:
            status_banner = "❌ PATCH RETEST FAILED"
        else:
            status_banner = "⚠️ IN PROGRESS / NOT VERIFIED"
        
        md_lines = [
            f"# 🤖 RoboSahaya Studio — Engineering Report",
            f"**Task ID:** `{task_id}` | **Generated:** `{timestamp_str}` | **Status:** {status_banner}",
            f"**Execution Mode:** `{mode.upper()}`",
            "",
            "## 1. User Intent & Problem Statement",
            f"> \"{ReportGenerator._redact(query)}\"",
            "",
            "## 2. Engineering Lifecycle Timeline",
            "| Time (UTC) | Stage | Severity | Action / Event |",
            "|---|---|---|---|"
        ]

        for e in events:
            if hasattr(e.timestamp, "strftime"):
                time_short = e.timestamp.strftime("%H:%M:%S")
            else:
                time_short = str(e.timestamp).split("T")[-1].split(".")[0]
            sev_icon = "🟢" if e.severity == "success" else ("🔴" if e.severity == "error" else ("🟡" if e.severity == "warning" else "ℹ️"))
            stage_str = e.stage.value.upper()
            tool_str = f" (`{e.tool_name}`)" if e.tool_name else ""
            safe_message = ReportGenerator._redact(e.user_facing_message).replace("|", "\\|")
            md_lines.append(f"| `{time_short}` | `{stage_str}` | {sev_icon} {e.severity} | {safe_message}{tool_str} |")

        md_lines.append("")
        md_lines.append("## 3. Evidence & Diagnostics")
        if diagnosis:
            md_lines.append(f"- **Case ID:** `{diagnosis.case_id}`")
            md_lines.append(f"- **Reasoning source:** `{diagnosis.reasoning_source}`")
            md_lines.append(f"- **Issue Summary:** {ReportGenerator._redact(diagnosis.issue_summary)}")
            conf_str = diagnosis.confidence_label.value if hasattr(diagnosis.confidence_label, "value") else str(diagnosis.confidence_label)
            md_lines.append(f"- **Confidence:** `{conf_str.upper()}`")
            if diagnosis.plain_language_analogy:
                md_lines.append(f"- **Plain-Language Analogy:** *\"{ReportGenerator._redact(diagnosis.plain_language_analogy)}\"*")
            if diagnosis.hypotheses:
                md_lines.append("\n### Evaluated Hypotheses:")
                for idx, h in enumerate(diagnosis.hypotheses, 1):
                    conf = h.confidence.value if hasattr(h.confidence, "value") else str(h.confidence)
                    md_lines.append(f"- **{idx}. {ReportGenerator._redact(h.label)}** (Confidence: `{conf.upper()}`): {ReportGenerator._redact(h.caveat)}")
            if diagnosis.observations:
                md_lines.append(f"\n### Evidence Observations: `{', '.join(diagnosis.observations)}`")
            if diagnosis.evidence_records:
                md_lines.append("\n### Collected Evidence")
                for record in diagnosis.evidence_records:
                    md_lines.append(f"- `{record.evidence_id}` ({record.source_mode}, {record.source.value}, {record.status.value}) `{ReportGenerator._redact(record.target)}`: `{ReportGenerator._redact(json.dumps(record.result, default=str))}`")
            if diagnosis.limitations:
                md_lines.append("\n### Limitations")
                md_lines.extend(f"- {ReportGenerator._redact(item)}" for item in diagnosis.limitations)
        else:
            md_lines.append("- *No diagnostic anomaly recorded.*")

        md_lines.append("")
        md_lines.append("## 4. Initial Sensor Test")
        if initial_test:
            test_icon = "✅ PASSED" if initial_test.passed else "❌ FAILED"
            md_lines.append(f"- **Test Name:** `{initial_test.test_id}`: **{test_icon}**")
            md_lines.append(f"- **Measured Values:** `{ReportGenerator._redact(json.dumps(initial_test.measured_values))}`")
            md_lines.append(f"- **Expected Criteria:** `{ReportGenerator._redact(json.dumps(initial_test.expected_criteria))}`")
            md_lines.append(f"- **Source:** `{initial_test.source_mode}` | **Status:** `{initial_test.status}`")
            md_lines.append(f"- **Log Excerpt:** {ReportGenerator._redact(initial_test.log_excerpt)}")
        else:
            md_lines.append("- *Initial test not performed.*")

        md_lines.append("")
        md_lines.append("## 5. Supervised Patch & Code Modifications")
        if patch_proposal:
            md_lines.append(f"- **Patch ID:** `{patch_proposal.patch_id}`")
            md_lines.append(f"- **Target File:** `{patch_proposal.target_file}`")
            md_lines.append(f"- **SHA-256 Digest:** `{patch_proposal.diff_sha256}`")
            md_lines.append(f"- **Explanation:** {ReportGenerator._redact(patch_proposal.explanation)}")
            md_lines.append(f"- **Patch Approved by User:** `{'YES' if patch_approved else 'NO'}`")
            md_lines.append(f"- **Verification Test:** `{patch_proposal.verification_test_id}`")
            md_lines.append("\n```diff")
            md_lines.append(ReportGenerator._redact(patch_proposal.unified_diff.strip()))
            md_lines.append("```")
        else:
            md_lines.append("- *No code patch proposed.*")

        md_lines.append("")
        md_lines.append("## 6. Post-Repair Verification & Payload Validation")
        if retest_result:
            retest_icon = "✅ PASSED" if retest_result.passed else "❌ FAILED"
            md_lines.append(f"- **Verification Retest:** `{retest_result.test_id}`: **{retest_icon}**")
            md_lines.append(f"- **Measured Post-Repair Values:** `{ReportGenerator._redact(json.dumps(retest_result.measured_values))}`")
            md_lines.append(f"- **Source:** `{retest_result.source_mode}` | **Status:** `{retest_result.status}`")
            md_lines.append(f"- **Log Excerpt:** {ReportGenerator._redact(retest_result.log_excerpt)}")
        else:
            md_lines.append("- *Retest not executed.*")

        md_lines.append("")
        md_lines.append("## 7. Review and Remaining Unknowns")
        md_lines.append(f"- **Patch approval recorded:** `{'YES' if patch_approved else 'NO'}`")
        md_lines.append(f"- **Patch digest:** `{patch_proposal.diff_sha256 if patch_proposal else 'none'}`")
        if mode == "replay":
            md_lines.append("- This report contains fixture/replay results only. It does not assert ROS graph or Gazebo runtime verification.")
        elif not retest_result and initial_test and initial_test.source_mode == "live" and initial_test.passed:
            md_lines.append("- Initial live sensor tests passed; no post-patch retest was needed.")
        elif not retest_result:
            md_lines.append("- Live post-change verification was not completed.")
        elif not retest_result.passed:
            md_lines.append("- The live post-change verification failed; inspect the measured values and launch logs.")
        else:
            md_lines.append("- Live tests cover the recorded observation windows and payload checks only; physical robot behavior is not verified.")

        return "\n".join(md_lines)

    @staticmethod
    def generate_json_audit(
        task_id: str,
        query: str,
        events: List[TaskEvent],
        initial_test: Optional[TestResult] = None,
        diagnosis: Optional[DiagnosisResponse] = None,
        patch_proposal: Optional[PatchProposal] = None,
        retest_result: Optional[TestResult] = None,
        patch_approved: bool = False,
        mode: str = "unknown",
    ) -> Dict[str, Any]:
        audit = {
            "task_id": task_id,
            "query": ReportGenerator._redact(query),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_mode": "replay" if mode in {"mock", "fixture", "replay"} else mode,
            "patch_approved": patch_approved,
            "events": [e.model_dump(mode="json") for e in events],
            "initial_test": initial_test.model_dump(mode="json") if initial_test else None,
            "diagnosis": diagnosis.model_dump(mode="json") if diagnosis else None,
            "patch_proposal": patch_proposal.model_dump(mode="json") if patch_proposal else None,
            "retest_result": retest_result.model_dump(mode="json") if retest_result else None
        }
        return ReportGenerator._scrub(audit)
