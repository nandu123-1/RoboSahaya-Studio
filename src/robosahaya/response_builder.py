from typing import Dict, Any
from robosahaya.evidence_models import DiagnosisResponse

class ResponseBuilder:
    """Formats diagnosis responses into rich UI markdown and audio-ready scripts."""

    @staticmethod
    def format_terminal_markdown(diag: DiagnosisResponse) -> str:
        md = []
        md.append(f"### 🔍 Diagnosis: {diag.issue_summary}")
        md.append(f"**Confidence Level:** `{diag.confidence_label.upper()}` | **Case ID:** `{diag.case_id}`\n")
        
        # Plain-language analogy for non-roboticist judges
        md.append(f"> 💡 **Plain-Language Analogy:** *{diag.plain_language_analogy}*\n")
        
        # Observations
        md.append("#### 📋 Observed Facts & Evidence")
        for obs in diag.observations:
            md.append(f"- Evidence **`[{obs}]`** validated from graph inspector.")
        md.append("")

        # Hypotheses
        md.append("#### 🧠 Primary Hypothesis")
        for h in diag.hypotheses:
            md.append(f"- **{h.label}**")
            md.append(f"  *Supporting Evidence:* `{' '.join(h.evidence_ids)}`")
            md.append(f"  *Caveat:* _{h.caveat}_\n")

        # Safe Terminal Verification Steps
        md.append("#### 🛠️ Safe Verification Commands (Run Manually)")
        for cmd in diag.terminal_card:
            md.append(f"**{cmd.purpose}:**")
            md.append(f"```bash\n{cmd.raw_command}\n```")
            md.append(f"*Expected Outcome:* {cmd.expected_observation}\n")

        # Limitations / Next question
        if diag.limitations:
            md.append("#### ⚠️ Known Inspection Limits")
            for lim in diag.limitations:
                md.append(f"- _{lim}_")
            md.append("")

        if diag.next_question:
            md.append(f"**Follow-up Inquiry:** {diag.next_question}")

        return "\n".join(md)

    @staticmethod
    def format_voice_script(diag: DiagnosisResponse) -> str:
        return diag.voice_script
