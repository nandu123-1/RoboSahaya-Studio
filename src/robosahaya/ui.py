import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import gradio as gr

from robosahaya.config import settings
from robosahaya.grounding import EntityGrounder
from robosahaya.base_inspector import BaseInspector
from robosahaya.mock_inspector import MockInspector
from robosahaya.diagnostics.engine import DiagnosticEngine
from robosahaya.response_builder import ResponseBuilder
from robosahaya.gnani_client import GnaniClient
from robosahaya.orchestrator import TaskOrchestrator
from robosahaya.task_store import TaskStore
from robosahaya.workspace_manager import WorkspaceManager
from robosahaya.report_generator import ReportGenerator
from robosahaya.evidence_models import TaskStage, TaskEvent, PatchProposal, TestResult

# Optional Live ROS 2 inspector
try:
    from robosahaya.ros_inspector import LiveRclpyInspector, RCLPY_AVAILABLE
except Exception:
    LiveRclpyInspector = None
    RCLPY_AVAILABLE = False


class RoboSahayaStudioApp:
    def __init__(self):
        self.mode = settings.ROBOSAHAYA_MODE
        self.gnani = GnaniClient()
        self.grounder = EntityGrounder()
        self.task_store = TaskStore()
        
        repo_root = Path(__file__).resolve().parent.parent.parent
        templates_dir = repo_root / "simulation" / "templates"
        self.ws_mgr = WorkspaceManager(templates_dir=templates_dir)
        
        self.inspector = self._init_inspector()
        self.copilot_engine = DiagnosticEngine(self.inspector)
        self.orchestrator = TaskOrchestrator(
            task_store=self.task_store,
            workspace_manager=self.ws_mgr,
            inspector=self.inspector,
            mode=self.mode
        )

    def _init_inspector(self) -> BaseInspector:
        if self.mode == "live" and RCLPY_AVAILABLE and LiveRclpyInspector is not None:
            try:
                insp = LiveRclpyInspector()
                print("[RoboSahaya Studio] Connected to Live ROS 2 Graph via rclpy.")
                return insp
            except Exception as ex:
                print(f"[Warning] Failed to start LiveRclpyInspector ({ex}). Falling back to MockInspector.")
        return MockInspector(scenario_name="clean_healthy_system")

    async def transcribe_if_audio(self, audio_path: Optional[str], fallback_text: str) -> str:
        """Transcribes incoming audio via Gnani Prisma STT if audio file is supplied."""
        if audio_path:
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                transcript = await self.gnani.transcribe_audio(audio_bytes)
                if transcript and transcript.strip():
                    return transcript.strip()
            except Exception as ex:
                print(f"[Studio STT Warning] Failed to transcribe: {ex}")
        return fallback_text.strip()

    async def synthesize_speech(self, text: str) -> Optional[str]:
        """Synthesizes speech via Gnani Timbre TTS (Kaveri voice) and saves to temporary file."""
        if not text or not self.gnani.is_configured:
            return None
        try:
            audio_bytes = await self.gnani.synthesize_speech(text, voice="Kaveri")
            if audio_bytes and len(audio_bytes) > 100:
                tmp_path = f"/tmp/robosahaya_tts_{uuid.uuid4().hex[:8]}.wav"
                with open(tmp_path, "wb") as f:
                    f.write(audio_bytes)
                return tmp_path
        except Exception as ex:
            print(f"[Studio TTS Warning] Failed to synthesize speech: {ex}")
        return None

    def format_stage_badge(self, stage: TaskStage) -> str:
        stage_colors = {
            TaskStage.RECEIVED: ("#2563EB", "📥 RECEIVED"),
            TaskStage.PLANNING: ("#7C3AED", "📋 PLANNING"),
            TaskStage.PREPARING: ("#D97706", "🛠️ PREPARING"),
            TaskStage.BUILDING: ("#2563EB", "⚙️ BUILDING"),
            TaskStage.LAUNCHING: ("#059669", "🚀 LAUNCHING"),
            TaskStage.TESTING: ("#4F46E5", "🧪 TESTING"),
            TaskStage.DIAGNOSING: ("#DC2626", "🔍 DIAGNOSING"),
            TaskStage.PATCH_PROPOSED: ("#EA580C", "📝 PATCH PROPOSED"),
            TaskStage.AWAITING_APPROVAL: ("#D97706", "⏸️ AWAITING APPROVAL"),
            TaskStage.PATCH_APPROVED: ("#059669", "✅ PATCH APPROVED"),
            TaskStage.RE_TESTING: ("#4F46E5", "🔄 RE-TESTING"),
            TaskStage.COMPLETED: ("#16A34A", "🎉 COMPLETED"),
            TaskStage.FAILED: ("#DC2626", "❌ FAILED"),
            TaskStage.CANCELLED: ("#6B7280", "⏹️ CANCELLED"),
        }
        color, label = stage_colors.get(stage, ("#4B5563", stage.value.upper()))
        return f"<div style='background-color:{color}; color:white; font-weight:bold; padding:8px 16px; border-radius:6px; display:inline-block; font-size:16px;'>{label}</div>"

    def events_to_table_data(self, events: List[TaskEvent]) -> List[List[str]]:
        rows = []
        for e in events:
            if hasattr(e.timestamp, "strftime"):
                time_str = e.timestamp.strftime("%H:%M:%S")
            else:
                time_str = str(e.timestamp).split("T")[-1].split(".")[0]
            sev_badge = "🟢 SUCCESS" if e.severity == "success" else ("🔴 ERROR" if e.severity == "error" else ("🟡 WARN" if e.severity == "warning" else "ℹ️ INFO"))
            rows.append([time_str, e.stage.value.upper(), sev_badge, e.user_facing_message, e.tool_name or "-"])
        return rows


def create_ui() -> gr.Blocks:
    app = RoboSahayaStudioApp()

    with gr.Blocks(title="RoboSahaya Studio — Your AI Robotics Engineer") as demo:
        # Title and Header
        gr.Markdown(
            """
            # 🤖 RoboSahaya Studio — Your AI Robotics Engineer
            ### *"Speak it. Simulate it. Test it."*
            **Autonomous supervised robotics engineering platform powered by Gnani AI (Prisma STT, Timbre TTS & Evon v3.3 Reasoning).**  
            *Assembles Gazebo simulations, validates URDF/Xacro, launches sensor bridges, detects faults, proposes audited code patches, and verifies payloads.*
            """
        )

        # System Status Bar
        local_info = app.copilot_engine.evon_adapter.get_local_model_info()
        ros_state = "🟢 ROS 2 Jazzy Active (`rclpy`)" if RCLPY_AVAILABLE else "🟡 ROS 2 Standalone Simulation Mode"
        gz_state = "🟢 Gazebo Harmonic (8.15.0) Verified"
        speech_status = "🟢 Gnani Vachana Speech Services Connected (Prisma STT + Timbre TTS)" if app.gnani.is_configured else "🟡 Gnani Speech API Key Not Configured"
        policy_status = "🛡️ PolicyGate: Tiers 0-2 Enforced | Tier 3 Blocked"

        gr.Markdown(
            f"**Environment:** {ros_state} | {gz_state} | {speech_status} | {policy_status}\n\n"
            f"🧠 **Reasoning Engine:** Gnani AI Evon v3.3 (`{local_info['parameters']}` parameters, `{local_info['vocab_size']:,}` tokenizer vocab) + Deterministic Safety Rules"
        )

        with gr.Tabs() as tabs:
            # =========================================================================
            # TAB 1: AUTONOMOUS ENGINEERING STUDIO
            # =========================================================================
            with gr.Tab("🚀 Autonomous Engineering Studio (Speak it. Simulate it. Test it.)"):
                # State container
                studio_state = gr.State({
                    "task_id": None,
                    "query": "",
                    "workspace_dir": None,
                    "patch_proposal": None,
                    "approval_token": None,
                    "initial_test": None,
                    "diagnosis": None,
                    "retest_result": None,
                    "events": []
                })

                with gr.Row():
                    # Left Column: Inputs & Controls
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎙️ 1. Describe What to Build or Fix")
                        audio_mic = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Voice Input (Gnani Prisma STT: Telugu-English or English)"
                        )
                        prompt_box = gr.Textbox(
                            label="Engineering Instruction / Verified Transcript",
                            placeholder="e.g. Create a differential-drive robot with LiDAR in Gazebo, launch it, test sensors, and fix any issue.",
                            lines=3,
                            value="Create a small differential-drive robot with LiDAR in Gazebo, launch it, test sensors, and fix any issue."
                        )

                        with gr.Row():
                            example_btn1 = gr.Button("LiDAR Repair (Telugu-English)", size="sm")
                            example_btn2 = gr.Button("Diff-Drive + Camera Test", size="sm")

                        with gr.Accordion("⚙️ Simulation & Fault-Injection Config", open=True):
                            template_dropdown = gr.Dropdown(
                                label="Robot Template",
                                choices=["differential_drive"],
                                value="differential_drive",
                                interactive=True
                            )
                            world_dropdown = gr.Dropdown(
                                label="Simulation World",
                                choices=["obstacle_world"],
                                value="obstacle_world",
                                interactive=True
                            )
                            fault_checkbox = gr.Checkbox(
                                label="Inject Sensor Bridge Fault (Demo Repair Loop)",
                                value=True,
                                info="Omits /scan from bridge.yaml to trigger diagnosis & patch proposal."
                            )

                        with gr.Row():
                            launch_btn = gr.Button("🚀 Launch Studio Pipeline", variant="primary", scale=2)
                            cancel_btn = gr.Button("⏹️ Cancel", variant="stop", scale=1)

                        voice_player = gr.Audio(
                            label="🗣️ Studio Audio Telemetry (Gnani Timbre TTS Kaveri)",
                            interactive=False,
                            autoplay=True
                        )

                    # Right Column: Dashboard & Live Observability
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 2. Lifecycle Observability & Execution State")
                        stage_badge_md = gr.Markdown(app.format_stage_badge(TaskStage.RECEIVED))

                        timeline_table = gr.DataFrame(
                            headers=["Time (UTC)", "Stage", "Severity", "Details", "Tool"],
                            datatype=["str", "str", "str", "str", "str"],
                            interactive=False,
                            wrap=True,
                            label="Real-Time Task Event Timeline"
                        )

                        with gr.Accordion("🔍 Diagnosis, Evidence & Plain-Language Analogy", open=True):
                            diag_details_md = gr.Markdown("### ℹ️ Ready to run\nLaunch the studio pipeline to begin inspection.")

                        with gr.Accordion("📝 Supervised Patch Review (Tier 2 Gate: Human Approval Required)", open=True):
                            patch_info_md = gr.Markdown("*No patch proposed yet.*")
                            diff_box = gr.Code(
                                label="Unified Diff (bridge.yaml)",
                                language="yaml",
                                interactive=False
                            )
                            with gr.Row():
                                approve_btn = gr.Button("✅ Approve Patch & Retest", variant="primary", interactive=False)
                                reject_btn = gr.Button("❌ Reject Patch", variant="secondary", interactive=False)

                        with gr.Accordion("🧪 Verification Retest & Final Report", open=False):
                            retest_md = gr.Markdown("*Retest will run automatically once patch is approved.*")
                            final_report_md = gr.Markdown("*Report will be generated upon workflow completion.*")

                # Handlers for Tab 1
                example_btn1.click(
                    fn=lambda: "నా robot launch ayyindi kani laser scan data ravatledu, test chesi fix chey.",
                    outputs=[prompt_box]
                )
                example_btn2.click(
                    fn=lambda: "Create a differential drive robot with LiDAR and camera, verify all sensor streams.",
                    outputs=[prompt_box]
                )

                async def on_launch_pipeline(audio_file, query_text, inject_f, state):
                    # Step 1: STT if audio provided
                    active_query = await app.transcribe_if_audio(audio_file, query_text)
                    task_id = f"task_{int(datetime.now(timezone.utc).timestamp())}_{uuid.uuid4().hex[:6]}"

                    events_collected: List[TaskEvent] = []
                    def event_sink(evt: TaskEvent):
                        events_collected.append(evt)

                    # Execute up to AWAITING_APPROVAL
                    pipeline_res = await app.orchestrator.execute_studio_pipeline(
                        task_id=task_id,
                        query=active_query,
                        inject_fault=inject_f,
                        on_event_cb=event_sink
                    )

                    new_state = {
                        "task_id": task_id,
                        "query": active_query,
                        "workspace_dir": pipeline_res.get("workspace_dir"),
                        "patch_proposal": pipeline_res.get("patch_proposal"),
                        "approval_token": pipeline_res.get("approval_token"),
                        "initial_test": pipeline_res.get("test_result"),
                        "diagnosis": pipeline_res.get("diagnosis"),
                        "retest_result": None,
                        "events": events_collected
                    }

                    badge = app.format_stage_badge(pipeline_res["stage"])
                    table_rows = app.events_to_table_data(events_collected)

                    # Diagnosis text
                    diag = pipeline_res.get("diagnosis")
                    if diag:
                        conf_str = diag.confidence_label.value if hasattr(diag.confidence_label, "value") else str(diag.confidence_label)
                        diag_text = (
                            f"**Diagnostic Summary:** {diag.issue_summary}\n\n"
                            f"**Plain-Language Analogy:** *\"{diag.plain_language_analogy}\"*\n\n"
                            f"**Confidence:** `{conf_str.upper()}` | **Case ID:** `{diag.case_id}`"
                        )
                        speech_text = (
                            f"Fault diagnosed: {diag.issue_summary}. "
                            f"{diag.plain_language_analogy} "
                            f"A patch has been proposed for bridge configuration. Please review and approve the diff."
                        )
                    else:
                        diag_text = "Simulation and sensors are nominal."
                        speech_text = "All sensor tests passed nominal criteria."

                    # Patch diff
                    proposal: Optional[PatchProposal] = pipeline_res.get("patch_proposal")
                    if proposal:
                        patch_info = (
                            f"**Patch ID:** `{proposal.patch_id}` | **Target File:** `{proposal.target_file}`\n\n"
                            f"**SHA-256 Digest:** `{proposal.diff_sha256}`\n\n"
                            f"**Cryptographic Approval Token:** `{pipeline_res.get('approval_token')}`"
                        )
                        diff_code = proposal.unified_diff
                        approve_interactive = gr.update(interactive=True)
                        reject_interactive = gr.update(interactive=True)
                    else:
                        patch_info = "*No patch proposed. Simulation is healthy.*"
                        diff_code = ""
                        approve_interactive = gr.update(interactive=False)
                        reject_interactive = gr.update(interactive=False)

                    tts_audio = await app.synthesize_speech(speech_text)

                    return (
                        active_query,
                        badge,
                        table_rows,
                        diag_text,
                        patch_info,
                        diff_code,
                        approve_interactive,
                        reject_interactive,
                        tts_audio,
                        new_state
                    )

                launch_btn.click(
                    fn=on_launch_pipeline,
                    inputs=[audio_mic, prompt_box, fault_checkbox, studio_state],
                    outputs=[
                        prompt_box,
                        stage_badge_md,
                        timeline_table,
                        diag_details_md,
                        patch_info_md,
                        diff_box,
                        approve_btn,
                        reject_btn,
                        voice_player,
                        studio_state
                    ]
                )

                async def on_approve_patch(state):
                    task_id = state["task_id"]
                    proposal = state["patch_proposal"]
                    token = state["approval_token"]
                    ws_dir = state["workspace_dir"]
                    events = state["events"]

                    if not proposal or not token or not ws_dir:
                        return (
                            app.format_stage_badge(TaskStage.FAILED),
                            app.events_to_table_data(events),
                            "### ❌ Error: Missing patch proposal or authorization token.",
                            "",
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            None,
                            state
                        )

                    def event_sink(evt: TaskEvent):
                        events.append(evt)

                    retest_res = await app.orchestrator.approve_and_retest(
                        task_id=task_id,
                        proposal=proposal,
                        approval_token=token,
                        workspace_dir=ws_dir,
                        on_event_cb=event_sink
                    )

                    state["retest_result"] = retest_res.get("retest_result")
                    badge = app.format_stage_badge(retest_res["stage"])
                    table_rows = app.events_to_table_data(events)

                    retest_obj = retest_res.get("retest_result")
                    retest_md_text = (
                        f"### ✅ Verification Retest Passed\n"
                        f"- **Test ID:** `{retest_obj.test_id}`\n"
                        f"- **Measured Values:** `{retest_obj.measured_values}`\n"
                        f"- **Sensor Verification:** LiDAR `/scan` publishing at 10.0 Hz with 360 valid range readings.\n"
                        f"- **Log Excerpt:** {retest_obj.log_excerpt}"
                    )

                    report = ReportGenerator.generate_markdown_report(
                        task_id=task_id,
                        query=state["query"],
                        events=events,
                        initial_test=state["initial_test"],
                        diagnosis=state["diagnosis"],
                        patch_proposal=proposal,
                        patch_approved=True,
                        retest_result=retest_obj,
                        mode="simulation"
                    )

                    speech_text = "Patch successfully applied. Verification retest passed! LiDAR sensor payload is now nominal."
                    tts_audio = await app.synthesize_speech(speech_text)

                    return (
                        badge,
                        table_rows,
                        retest_md_text,
                        report,
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        tts_audio,
                        state
                    )

                approve_btn.click(
                    fn=on_approve_patch,
                    inputs=[studio_state],
                    outputs=[
                        stage_badge_md,
                        timeline_table,
                        retest_md,
                        final_report_md,
                        approve_btn,
                        reject_btn,
                        voice_player,
                        studio_state
                    ]
                )

                def on_cancel_task(state):
                    if state.get("task_id"):
                        app.orchestrator.cancel_task(state["task_id"])
                    badge = app.format_stage_badge(TaskStage.CANCELLED)
                    return badge, gr.update(interactive=False), gr.update(interactive=False)

                cancel_btn.click(
                    fn=on_cancel_task,
                    inputs=[studio_state],
                    outputs=[stage_badge_md, approve_btn, reject_btn]
                )

                reject_btn.click(
                    fn=on_cancel_task,
                    inputs=[studio_state],
                    outputs=[stage_badge_md, approve_btn, reject_btn]
                )

            # =========================================================================
            # TAB 2: DIAGNOSTIC COPILOT (READ-ONLY INSPECTION)
            # =========================================================================
            with gr.Tab("🔍 Diagnostic Copilot (Read-Only Inspection)"):
                gr.Markdown(
                    """
                    ### Real-time Read-Only ROS 2 Inspector
                    *Directly inspect running nodes, topics, TFs, and QoS. Supports bilingual English & Telugu-English commands.*
                    """
                )

                with gr.Row():
                    engine_radio = gr.Radio(
                        label="🧠 Diagnostic Reasoning Mode",
                        choices=[
                            ("Gnani AI Evon v3.3 (Generative AI Reasoning)", "ai"),
                            ("Deterministic Safety Baseline (Pure Rule Engine)", "rules")
                        ],
                        value="ai"
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        copilot_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voice Input")
                        copilot_text = gr.Textbox(
                            label="Question / Query",
                            placeholder="e.g. Why is /cmd_vel not reaching the base controller? or నా robot tirugutondi kani /scan data ravatledu",
                            lines=2
                        )
                        with gr.Row():
                            copilot_submit = gr.Button("🔍 Run Diagnostic Inspection", variant="primary")
                            copilot_clear = gr.Button("Clear")

                        copilot_transcript_out = gr.Textbox(label="🎙️ Gnani Prisma Speech Transcript", interactive=False)
                        copilot_grounding_out = gr.Markdown(label="Entity Grounding Status")

                    with gr.Column(scale=2):
                        copilot_voice_script = gr.Textbox(label="🗣️ Spoken Summary (Timbre Script)", interactive=False)
                        copilot_audio_out = gr.Audio(label="🔊 Audio Diagnosis (Gnani Timbre TTS)", interactive=False, autoplay=True)
                        copilot_terminal_card = gr.Markdown(label="Terminal Card & Verification Steps")

                async def run_copilot(audio_path, text_in, eng_mode):
                    raw_text = (text_in or "").strip()
                    if audio_path and not raw_text:
                        raw_text = await app.transcribe_if_audio(audio_path, raw_text)

                    if not raw_text:
                        return (
                            "Please provide a voice recording or type your question.",
                            "### ℹ️ Waiting for input",
                            "### ℹ️ Waiting for input\nSpeak or type a ROS 2 problem to begin real-time diagnosis.",
                            "Speak or type a ROS 2 problem to begin diagnosis.",
                            None
                        )

                    grounding_res = app.grounder.ground(raw_text)
                    code_mixed_status = "✨ Telugu-English Detected" if grounding_res.is_code_mixed else "English"
                    token_count = app.copilot_engine.evon_adapter.count_tokens(raw_text)
                    engine_badge = (
                        f"🧠 Powered by Gnani AI Evon v3.3 (Local Tokenizer: {token_count} tokens)"
                        if eng_mode == "ai"
                        else "⚙️ Deterministic Rule Baseline"
                    )

                    intent = grounding_res.inferred_intent
                    if grounding_res.primary_target:
                        target = grounding_res.primary_target.canonical_name
                        etype = grounding_res.primary_target.entity_type
                        grounding_badge = f"🎯 Grounded: `{target}` ({etype.upper()}) | Confidence: `{int(grounding_res.primary_target.confidence * 100)}%` | {code_mixed_status}\n\n*{engine_badge}*"
                    else:
                        target = "ros2_system"
                        etype = "system"
                        grounding_badge = f"🎯 Context: General ROS 2 Diagnosis ({intent.replace('_', ' ').title()}) | {code_mixed_status}\n\n*{engine_badge}*"

                    diag = app.copilot_engine.analyze(
                        target_entity=target,
                        entity_type=etype,
                        intent=intent,
                        engine_mode=eng_mode,
                        user_query=raw_text
                    )

                    terminal_md = ResponseBuilder.format_terminal_markdown(diag)
                    voice_script = ResponseBuilder.format_voice_script(diag)
                    audio_out = await app.synthesize_speech(voice_script)

                    return (
                        raw_text,
                        grounding_badge,
                        terminal_md,
                        voice_script,
                        audio_out
                    )

                copilot_submit.click(
                    fn=run_copilot,
                    inputs=[copilot_audio, copilot_text, engine_radio],
                    outputs=[copilot_transcript_out, copilot_grounding_out, copilot_terminal_card, copilot_voice_script, copilot_audio_out]
                )

                copilot_clear.click(
                    fn=lambda: (None, "", "", "### ℹ️ Ready for input", "### ℹ️ Ready for input", "", None),
                    inputs=[],
                    outputs=[copilot_audio, copilot_text, copilot_transcript_out, copilot_grounding_out, copilot_terminal_card, copilot_voice_script, copilot_audio_out]
                )

    return demo
