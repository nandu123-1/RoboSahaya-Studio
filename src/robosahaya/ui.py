import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
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
from robosahaya.evidence_models import (
    TaskStage,
    TaskEvent,
    PatchProposal,
    TestResult,
    TaskPlan,
    PermissionTier,
    compute_plan_digest
)
from robosahaya.mcp.client_manager import MCPClientManager
from robosahaya.mcp.models import (
    MCPConnectionConfig,
    MCPConnectionState,
    MCPConnectionStatus,
    MCPToolInfo,
    MCPTransportType
)

# Optional Live ROS 2 inspector
try:
    from robosahaya.ros_inspector import LiveRclpyInspector, RCLPY_AVAILABLE
except Exception:
    LiveRclpyInspector = None
    RCLPY_AVAILABLE = False


def get_gazebo_version() -> str:
    gz_bin = shutil.which("gz")
    if not gz_bin:
        return "Not detected"
    try:
        out = subprocess.check_output([gz_bin, "sim", "--version"], text=True, timeout=2.0)
        return out.strip().split("\n")[0]
    except Exception:
        return "Detected (gz)"


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
        self.mcp_mgr = MCPClientManager()

    def _init_inspector(self) -> BaseInspector:
        if self.mode == "live" and RCLPY_AVAILABLE and LiveRclpyInspector is not None:
            try:
                insp = LiveRclpyInspector()
                print("[RoboSahaya Studio] Connected to Live ROS 2 Graph via rclpy.")
                return insp
            except Exception as ex:
                print(f"[RoboSahaya Studio] Live ROS inspector unavailable: {ex}")
                return None
        if self.mode == "mock":
            return MockInspector(scenario_name=settings.ROBOSAHAYA_MOCK_SCENARIO)
        return None

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
                print(f"[Studio STT] Request failed ({type(ex).__name__}).")
        return (fallback_text or "").strip()

    async def synthesize_speech(self, text: str) -> Optional[str]:
        """Synthesizes speech via Gnani Timbre TTS (Kaveri voice) and saves to temporary file."""
        if not text or not self.gnani.tts_configured:
            return None
        try:
            audio_bytes = await self.gnani.synthesize_speech(text, voice="Kaveri")
            if audio_bytes and len(audio_bytes) > 100:
                tmp_path = f"/tmp/robosahaya_tts_{uuid.uuid4().hex[:8]}.wav"
                with open(tmp_path, "wb") as f:
                    f.write(audio_bytes)
                return tmp_path
        except Exception as ex:
            print(f"[Studio TTS] Request failed ({type(ex).__name__}).")
        return None

    def format_stage_badge(self, stage: TaskStage) -> str:
        stage_colors = {
            TaskStage.RECEIVED: ("#2563EB", "📥 RECEIVED"),
            TaskStage.TRANSCRIBED: ("#0284C7", "🎙️ TRANSCRIBED"),
            TaskStage.PLANNING: ("#7C3AED", "📋 PLANNING"),
            TaskStage.PREPARING: ("#D97706", "🛠️ PREPARING"),
            TaskStage.BUILDING: ("#2563EB", "⚙️ BUILDING"),
            TaskStage.LAUNCHING: ("#059669", "🚀 LAUNCHING"),
            TaskStage.TESTING: ("#4F46E5", "🧪 TESTING"),
            TaskStage.DIAGNOSING: ("#DC2626", "🔍 DIAGNOSING"),
            TaskStage.PATCH_PROPOSED: ("#EA580C", "📝 PATCH PROPOSED"),
            TaskStage.AWAITING_APPROVAL: ("#D97706", "⏸️ AWAITING APPROVAL"),
            TaskStage.PATCH_APPROVED: ("#059669", "✅ PATCH APPROVED"),
            TaskStage.APPLYING_PATCH: ("#D97706", "🔧 APPLYING PATCH"),
            TaskStage.RE_TESTING: ("#4F46E5", "🔄 RE-TESTING"),
            TaskStage.COMPLETED: ("#16A34A", "🎉 COMPLETED"),
            TaskStage.FAILED: ("#DC2626", "❌ FAILED"),
            TaskStage.CANCELLED: ("#6B7280", "⏹️ CANCELLED"),
            TaskStage.BLOCKED: ("#B91C1C", "⛔ LIVE BLOCKED"),
            TaskStage.PARTIAL_SUCCESS: ("#D97706", "⚠️ PARTIAL SUCCESS"),
        }
        color, label = stage_colors.get(stage, ("#4B5563", stage.value.upper() if hasattr(stage, "value") else str(stage)))
        return f"<div style='background-color:{color}; color:white; font-weight:bold; padding:8px 16px; border-radius:6px; display:inline-block; font-size:16px;'>{label}</div>"

    def format_mcp_status_badge(self, status: MCPConnectionStatus) -> str:
        state_colors = {
            MCPConnectionState.NOT_CONFIGURED: ("#6B7280", "⚪ NOT CONFIGURED"),
            MCPConnectionState.DISCONNECTED: ("#F59E0B", "🟠 DISCONNECTED"),
            MCPConnectionState.CONNECTING: ("#2563EB", "🔵 CONNECTING..."),
            MCPConnectionState.CONNECTED: ("#10B981", "🟢 CONNECTED"),
            MCPConnectionState.ERROR: ("#EF4444", "🔴 CONNECTION ERROR"),
            MCPConnectionState.CONNECTION_LOST: ("#F97316", "🟡 CONNECTION LOST"),
        }
        color, label = state_colors.get(status.state, ("#6B7280", status.state.value.upper()))
        return f"<div style='background-color:{color}; color:white; font-weight:bold; padding:6px 14px; border-radius:6px; display:inline-block; font-size:15px;'>{label}</div>"

    def format_mcp_tools_table(self, tools: List[MCPToolInfo]) -> List[List[str]]:
        rows = []
        for t in tools:
            tier_label = f"Tier {t.permission_tier}"
            if t.permission_tier == 0:
                tier_label += " (Read-Only)"
            elif t.permission_tier == 1:
                tier_label += " (Workspace)"
            elif t.permission_tier == 2:
                tier_label += " (Managed Sim/Patch - Approval Required)"
            else:
                tier_label += " (Blocked)"
            avail = "🟢 Ready" if t.is_available else "🔴 Unavailable"
            rows.append([t.name, t.description, tier_label, avail])
        return rows

    def format_plan_card(self, plan: Optional[TaskPlan]) -> str:
        if not plan:
            return "### ℹ️ No Plan Formulated Yet\nEnter your goal in **Phase 1** and click **'📋 Formulate & Review Plan'** to generate the execution plan."

        stale_alert = ""
        if plan.is_stale:
            stale_alert = "> ⚠️ **STALE PLAN WARNING:** Execution-relevant settings or instruction changed since plan review. You must click **'📋 Formulate & Review Plan'** before launching.\n\n"

        if not plan.is_supported:
            return f"""{stale_alert}### ⛔ Plan Status: REJECTED (Unsupported Capability)
**User Goal:** *"{plan.user_goal}"*  
**Requested Capabilities:** `{', '.join(plan.requested_capabilities) or 'None'}`  
**Unsupported Capabilities:** `⚠️ {', '.join(plan.unsupported_capabilities)}`  

> **Rejection Reason:**  
> {plan.rejection_reason}

🛡️ **Policy Gate:** The Studio will not substitute an unrelated mobile robot for a manipulator or aerial vehicle. Please modify your instruction to request a differential-drive robot.
"""

        selection_mode_badge = "⚙️ Manual Override Active" if plan.is_manual_override else f"🤖 Auto-selected by planner ({plan.selection_source.replace('_', ' ').title()})"
        tiers_str = " | ".join(f"`{tool}`: Tier {tier.value if hasattr(tier, 'value') else tier}" for tool, tier in plan.permission_tiers.items())
        stages_str = " ➔ ".join(s.value.upper() for s in plan.ordered_stages)
        tests_str = ", ".join(f"`{t}`" for t in plan.verification_tests)
        sensors_str = ", ".join(f"`{s}`" for s in plan.sensors_selected) if plan.sensors_selected else "`2D Planar LiDAR`, `RGB Monocular Camera`"
        topics_str = ", ".join(f"`{tp}`" for tp in plan.ros_topics) if plan.ros_topics else "`/scan`, `/camera/image_raw`, `/odom`, `/cmd_vel`"

        mcp_section = ""
        if plan.mcp_verified:
            facts = "\n".join(f"  - {f}" for f in plan.mcp_verified_facts)
            mcp_section = f"- **MCP Verified Environment Facts:**\n{facts}"
        else:
            mcp_section = "- **MCP Environment Verification:** *Offline planning mode (MCP server disconnected; verified live checks skipped)*"

        digest_display = f"`{plan.plan_digest[:16]}...`" if plan.plan_digest else "*None*"
        sim_adapter = getattr(plan, "simulator_adapter", "gazebo")
        model_fmt = getattr(plan, "model_format", "URDF/SDF")

        return f"""{stale_alert}### 📋 Plan Status: APPROVED FOR EXECUTION
**Mode:** `{selection_mode_badge}` | **Plan Digest:** {digest_display}

- **User Goal:** *"{plan.user_goal}"*
- **Robot Template:** `{plan.robot_template}` — *{plan.robot_selection_reason}*
- **Simulation World:** `{plan.world_template}` — *{plan.world_selection_reason}*
- **Simulator Adapter:** `{sim_adapter.upper()}` ({model_fmt.upper()})
- **Selected Sensors:** {sensors_str}
- **Target ROS Topics:** {topics_str}
- **Execution Stages:** `{stages_str}`
- **Permission Tiers:** {tiers_str}
- **Planned Verification Tests:** {tests_str}
- **Expected Behavior:** *{plan.expected_behavior or 'Nominal sensor streams and Gazebo simulation.'}*
{mcp_section}
- **Assumptions & Safety Boundaries:**
  - {plan.assumptions[0] if plan.assumptions else 'Standard ROS 2 Jazzy & Gazebo Harmonic runtime'}
  - Workspace directory confined to `/tmp/robosahaya_workspaces`
  - Subprocess group managed with deterministic SIGTERM/SIGKILL lifecycle
"""

    def format_sensor_status(self, test_result: Optional[TestResult], stage: TaskStage = TaskStage.RECEIVED) -> str:
        if not test_result:
            if stage in (TaskStage.TESTING, TaskStage.DIAGNOSING, TaskStage.PATCH_PROPOSED, TaskStage.AWAITING_APPROVAL):
                return "### 📡 Live Sensor Telemetry\n*Sampling in progress...*"
            return "### 📡 Live Sensor Telemetry\n*Sensor metrics will display during the TESTING and RE_TESTING stages.*"
        
        m = test_result.measured_values or {}
        rows = []
        if "joint_states_messages" in m:
            js_msgs = m.get("joint_states_messages", 0)
            js_freq = m.get("joint_states_frequency_hz", 0.0)
            js_valid = m.get("joint_states_valid_samples", 0)
            js_invalid = m.get("joint_states_invalid_samples", 0)
            status_js = "🟢 Active" if js_msgs > 0 and js_invalid == 0 else ("🔴 Missing / Fault" if js_msgs == 0 else "🟡 Degraded")
            rows.append(f"| **Joint States** | `/joint_states` | {status_js} | `{js_msgs}` | `{js_freq:.1f} Hz` | `{js_valid}` | `{js_invalid}` |")
        else:
            scan_msgs = m.get("scan_messages", m.get("messages_received", 0))
            scan_freq = m.get("scan_frequency_hz", m.get("frequency_hz", 0.0))
            scan_valid = m.get("scan_valid_samples", m.get("finite_ranges", 0))
            scan_invalid = m.get("scan_invalid_samples", 0)
            status_scan = "🟢 Active" if scan_msgs > 0 and scan_invalid == 0 else ("🔴 Missing / Fault" if scan_msgs == 0 else "🟡 Degraded")
            rows.append(f"| **2D LiDAR** | `/scan` | {status_scan} | `{scan_msgs}` | `{scan_freq:.1f} Hz` | `{scan_valid}` | `{scan_invalid}` |")
        
        cam_msgs = m.get("camera_messages", 0)
        cam_freq = m.get("camera_frequency_hz", 0.0)
        cam_valid = m.get("camera_valid_samples", 0)
        cam_invalid = m.get("camera_invalid_samples", 0)
        status_cam = "🟢 Active" if cam_msgs > 0 and cam_invalid == 0 else ("🔴 Missing / Fault" if cam_msgs == 0 else "🟡 Degraded")
        rows.append(f"| **RGB Camera** | `/camera/image_raw` | {status_cam} | `{cam_msgs}` | `{cam_freq:.1f} Hz` | `{cam_valid}` | `{cam_invalid}` |")
        
        table_rows = "\n".join(rows)
        table_md = f"""### 📡 Live Sensor Telemetry ({test_result.source_mode.upper()} Mode)
| Sensor Stream | Topic | Status | Messages Received | Frequency | Valid Samples | Invalid/NaN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_rows}

**Log Excerpt:** `{test_result.log_excerpt}`  
**Verification Verdict:** **{'✅ PASSED' if test_result.passed else '❌ FAILED / DISCREPANCY DETECTED'}**
"""
        return table_md

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
    robot_catalog = app.orchestrator.robot_catalog
    robot_template_choices = robot_catalog.choices()
    available_choice_ids = {value for _, value in robot_template_choices}
    # Keep the legacy template ID visible for old plans and generic arm requests.
    if "robotic_arm_6dof" not in available_choice_ids:
        robot_template_choices.insert(1, ("Bundled arm template (legacy ID; Panda-derived)", "robotic_arm_6dof"))

    with gr.Blocks(title="RoboSahaya Studio — Your AI Robotics Engineer") as demo:
        # Title and Header
        gr.Markdown(
            """
            # 🤖 RoboSahaya Studio — Your AI Robotics Engineer
            ### *"Speak it. Simulate it. Test it."*
            **Evidence-led ROS 2 debugging with a supervised simulation repair workflow and Model Context Protocol (MCP) server integration.**
            """
        )

        # System Status Bar
        ros_distro = os.environ.get("ROS_DISTRO", "jazzy").capitalize()
        local_info = app.copilot_engine.evon_adapter.get_local_model_info()
        ros_state = f"🟢 ROS 2 {ros_distro} (Live rclpy connected)" if (RCLPY_AVAILABLE and app.inspector is not None) else "🔴 Live ROS inspection unavailable"
        gz_version = get_gazebo_version()
        gz_state = f"🟢 {gz_version} (Harmonic)" if "version" in gz_version.lower() else "🟡 Gazebo not detected"
        speech_status = "🟢 Gnani Speech (v3/v2.5) configured on api.vachana.ai" if app.gnani.is_configured else "⚪ Gnani speech credentials not configured"
        policy_status = "🛡️ PolicyGate: Tiers 0-2 Enforced | Tier 3 Blocked"
        reasoning_status = "configured endpoint" if app.copilot_engine.evon_adapter.is_live else "deterministic safety baseline (Evon endpoint unconfigured)"

        with gr.Accordion("🖥️ System Pre-requisites & Live Status", open=True):
            gr.Markdown(
                f"**Mode:** `{app.mode.upper()}` | {ros_state} | {gz_state} | {speech_status} | {policy_status}\n\n"
                f"🧠 **Reasoning:** {reasoning_status} | **Local weights detected:** `{local_info['weights_present']}`"
            )
            with gr.Row():
                probe_speech_btn = gr.Button("🩺 Run Live Speech Endpoints Health Probe", size="sm")
                probe_status_md = gr.Markdown("*Click button to verify live roundtrip latency to Gnani Prisma STT and Timbre TTS.*")

            async def on_probe_speech():
                stt_ok, stt_msg = await app.gnani.check_stt_health()
                tts_ok, tts_msg = await app.gnani.check_tts_health()
                stt_icon = "🟢" if stt_ok else "🔴"
                tts_icon = "🟢" if tts_ok else "🔴"
                return f"**Speech Probe Result:** {stt_icon} **STT:** {stt_msg} | {tts_icon} **TTS:** {tts_msg}"

            probe_speech_btn.click(
                fn=on_probe_speech,
                inputs=[],
                outputs=[probe_status_md]
            )

        with gr.Tabs() as tabs:
            # =========================================================================
            # TAB 1: AUTONOMOUS ENGINEERING STUDIO (4-PHASE WORKFLOW)
            # =========================================================================
            with gr.Tab("🚀 Supervised Engineering Studio (Speak it. Simulate it. Test it.)"):
                # State container
                studio_state = gr.State({
                    "task_id": None,
                    "query": "Create a small differential-drive robot with LiDAR in Gazebo, launch it, test sensors, and fix any issue.",
                    "plan": None,
                    "workspace_dir": None,
                    "patch_proposal": None,
                    "initial_test": None,
                    "diagnosis": None,
                    "retest_result": None,
                    "events": [],
                    "source_mode": app.mode,
                    "manual_overrides": {},
                    "auto_robot": None,
                    "auto_world": None
                })

                # -----------------------------------------------------------------
                # PHASE 1: TELL ROBOSAHAYA
                # -----------------------------------------------------------------
                gr.Markdown("## 1️⃣ Phase 1: Tell RoboSahaya (Speak it or Type it)")
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_mic = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="🎙️ Voice Input (Gnani Prisma STT v3)"
                        )
                        prompt_box = gr.Textbox(
                            label="Engineering Instruction / Verified Transcript",
                            placeholder="e.g. Build a 6-DOF robotic arm with a gripper camera to pick and place objects.",
                            lines=3,
                            value=""
                        )

                        with gr.Row():
                            example_btn1 = gr.Button("LiDAR Repair (Telugu-English)", size="sm")
                            example_btn2 = gr.Button("Diff-Drive + Camera Test", size="sm")
                            example_btn3 = gr.Button("6-DOF Arm + Tabletop Pick & Place", size="sm")

                    with gr.Column(scale=1):
                        auto_selection_indicator_md = gr.Markdown(
                            """
                            **Selection Mode:** `🤖 Auto-selected by planner`  
                            *(Type your prompt and click 'Formulate & Review Plan' to automatically detect the robot and simulation world)*
                            """
                        )
                        with gr.Row():
                            template_dropdown = gr.Dropdown(
                                label="Robot Template (Auto-inferred from prompt)",
                                choices=robot_template_choices,
                                value=None,
                                interactive=True,
                                allow_custom_value=True
                            )
                            world_dropdown = gr.Dropdown(
                                label="Simulation World (Auto-inferred from prompt)",
                                choices=["obstacle_world", "tabletop_world", "soccer_pitch"],
                                value=None,
                                interactive=True,
                                allow_custom_value=True
                            )
                        with gr.Row():
                            reset_ai_btn = gr.Button("🔄 Reset to AI-Selected Settings", size="sm", variant="secondary")

                        with gr.Row():
                            gui_checkbox = gr.Checkbox(
                                label="🖥️ Launch 3D Simulation GUI Window",
                                value=True,
                                interactive=True,
                                info="Opens the interactive Gazebo Sim or MuJoCo 3D simulation window on your desktop screen so you can visually watch the robot."
                            )
                            fault_checkbox = gr.Checkbox(
                                label="Simulate Sensor Fault (Autonomous Diagnosis & Patch Repair)",
                                value=False,
                                interactive=True,
                                info="Executes live build, detects missing LiDAR bridge, diagnoses root cause, generates unified diff patch, and prompts for human approval before applying and live retesting."
                            )
                        plan_btn = gr.Button("📋 Formulate & Review Plan", variant="secondary", size="lg")

                # -----------------------------------------------------------------
                # PHASE 2: REVIEW THE PLAN
                # -----------------------------------------------------------------
                gr.Markdown("## 2️⃣ Phase 2: Review the Plan (Supervised Plan Gate)")
                plan_card_md = gr.Markdown(app.format_plan_card(None))

                with gr.Row():
                    launch_btn = gr.Button("🚀 Launch Studio Pipeline", variant="primary", scale=2)
                    cancel_btn = gr.Button("⏹️ Cancel Pipeline", variant="stop", scale=1)
                    open_gui_btn = gr.Button("🖥️ Open Simulator GUI", variant="secondary", scale=1)

                gui_status_md = gr.Markdown("*Gazebo 3D simulation window will open automatically when pipeline launches with GUI enabled.*")

                voice_player = gr.Audio(
                    label="🗣️ Studio Voice Telemetry (Gnani Timbre TTS — Kaveri Voice)",
                    interactive=False,
                    autoplay=True
                )

                # -----------------------------------------------------------------
                # PHASE 3: WATCH IT WORK
                # -----------------------------------------------------------------
                gr.Markdown("## 3️⃣ Phase 3: Watch It Work (Live Execution & Observability)")
                with gr.Row():
                    with gr.Column(scale=1):
                        stage_badge_md = gr.Markdown(app.format_stage_badge(TaskStage.RECEIVED))
                    with gr.Column(scale=2):
                        sensor_status_md = gr.Markdown(app.format_sensor_status(None))

                timeline_table = gr.DataFrame(
                    headers=["Time (UTC)", "Stage", "Severity", "Details", "Tool"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="Real-Time Task Event Timeline"
                )

                # -----------------------------------------------------------------
                # PHASE 4: REVIEW RESULTS & SUPERVISED REPAIR
                # -----------------------------------------------------------------
                gr.Markdown("## 4️⃣ Phase 4: Review Results & Supervised Repair")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Accordion("🔍 Diagnosis, Evidence & Plain-Language Analogy", open=True):
                            diag_details_md = gr.Markdown("### ℹ️ Ready to run\nFormulate a plan and launch the pipeline to begin live inspection.")

                    with gr.Column(scale=1):
                        with gr.Accordion("📝 Supervised Patch Review (Tier 2 Gate: Human Approval Required)", open=True):
                            patch_info_md = gr.Markdown("*No patch proposed yet.*")
                            diff_box = gr.Code(
                                label="Unified Diff (generated workspace file)",
                                language=None,
                                interactive=False
                            )
                            with gr.Row():
                                approve_btn = gr.Button("✅ Approve Patch & Retest", variant="primary", interactive=False)
                                reject_btn = gr.Button("❌ Reject Patch", variant="secondary", interactive=False)

                with gr.Accordion("🧪 Verification Retest & Final Engineering Report", open=True):
                    retest_md = gr.Markdown("*Retest will run automatically once patch is approved.*")
                    final_report_md = gr.Markdown("*Engineering report will be generated upon workflow completion.*")

                # =================================================================
                # EVENT HANDLERS FOR TAB 1
                # =================================================================
                def on_example_selected(text: str, state: dict):
                    infer = app.orchestrator.infer_robot_and_world(text)
                    state["query"] = text
                    state["auto_robot"] = infer["auto_robot"]
                    state["auto_world"] = infer["auto_world"]
                    state["manual_overrides"] = {}
                    if state.get("plan"):
                        state["plan"].is_stale = True

                    indicator = (
                        f"**Selection Mode:** `🤖 Auto-selected by planner ({infer['selection_source'].replace('_', ' ').title()})`  \n"
                        f"- **Robot:** `{infer['auto_robot']}` ({infer['robot_reason']})  \n"
                        f"- **World:** `{infer['auto_world']}` ({infer['world_reason']})"
                    )
                    plan_card = app.format_plan_card(state.get("plan"))
                    return text, infer["auto_robot"], infer["auto_world"], indicator, plan_card, gr.update(interactive=False), state

                example_btn1.click(
                    fn=lambda s: on_example_selected("నా robot launch ayyindi kani laser scan data ravatledu, test chesi fix chey.", s),
                    inputs=[studio_state],
                    outputs=[prompt_box, template_dropdown, world_dropdown, auto_selection_indicator_md, plan_card_md, launch_btn, studio_state]
                )
                example_btn2.click(
                    fn=lambda s: on_example_selected("Create a differential drive robot with LiDAR and camera, verify all sensor streams.", s),
                    inputs=[studio_state],
                    outputs=[prompt_box, template_dropdown, world_dropdown, auto_selection_indicator_md, plan_card_md, launch_btn, studio_state]
                )
                example_btn3.click(
                    fn=lambda s: on_example_selected("Build a 6-DOF robotic arm with a gripper camera to pick and place objects.", s),
                    inputs=[studio_state],
                    outputs=[prompt_box, template_dropdown, world_dropdown, auto_selection_indicator_md, plan_card_md, launch_btn, studio_state]
                )

                def on_manual_template_override(robot_val, world_val, state):
                    auto_robot = state.get("auto_robot")
                    auto_world = state.get("auto_world")

                    # If the values match what the planner auto-selected, or if values are unset, it is NOT a manual override!
                    if (robot_val == auto_robot and world_val == auto_world) or (not robot_val and not world_val):
                        state["manual_overrides"] = {}
                        indicator = (
                            f"**Selection Mode:** `🤖 Auto-selected by planner`  \n"
                            f"- **Robot:** `{robot_val or 'Auto'}`  \n"
                            f"- **World:** `{world_val or 'Auto'}`"
                        )
                        plan_card = app.format_plan_card(state.get("plan"))
                        can_launch = bool(state.get("plan") and state["plan"].is_supported and not state["plan"].is_stale)
                        return indicator, plan_card, gr.update(interactive=can_launch), state

                    state["manual_overrides"]["robot_template"] = robot_val
                    state["manual_overrides"]["world_template"] = world_val
                    if state.get("plan"):
                        state["plan"].is_stale = True

                    indicator = (
                        "**Selection Mode:** `⚠️ Manual Override Active`  \n"
                        f"- **Robot:** `{robot_val}` (User Override)  \n"
                        f"- **World:** `{world_val}` (User Override)  \n"
                        "*Click 'Reset to AI-Selected Settings' to restore planner recommendations.*"
                    )
                    plan_card = app.format_plan_card(state.get("plan"))
                    return indicator, plan_card, gr.update(interactive=False), state

                template_dropdown.change(
                    fn=on_manual_template_override,
                    inputs=[template_dropdown, world_dropdown, studio_state],
                    outputs=[auto_selection_indicator_md, plan_card_md, launch_btn, studio_state]
                )
                world_dropdown.change(
                    fn=on_manual_template_override,
                    inputs=[template_dropdown, world_dropdown, studio_state],
                    outputs=[auto_selection_indicator_md, plan_card_md, launch_btn, studio_state]
                )

                def on_reset_ai(prompt_text, state):
                    infer = app.orchestrator.infer_robot_and_world(prompt_text)
                    state["manual_overrides"] = {}
                    state["auto_robot"] = infer["auto_robot"]
                    state["auto_world"] = infer["auto_world"]
                    if state.get("plan"):
                        state["plan"].is_stale = True

                    indicator = (
                        f"**Selection Mode:** `🤖 Auto-selected by planner ({infer['selection_source'].replace('_', ' ').title()})`  \n"
                        f"- **Robot:** `{infer['auto_robot']}` ({infer['robot_reason']})  \n"
                        f"- **World:** `{infer['auto_world']}` ({infer['world_reason']})"
                    )
                    plan_card = app.format_plan_card(state.get("plan"))
                    return infer["auto_robot"], infer["auto_world"], indicator, plan_card, gr.update(interactive=False), state

                reset_ai_btn.click(
                    fn=on_reset_ai,
                    inputs=[prompt_box, studio_state],
                    outputs=[template_dropdown, world_dropdown, auto_selection_indicator_md, plan_card_md, launch_btn, studio_state]
                )

                async def on_formulate_plan(audio_file, query_text, template_id, world_id, state):
                    active_query = await app.transcribe_if_audio(audio_file, query_text)
                    task_id = f"task_{int(datetime.now(timezone.utc).timestamp())}_{uuid.uuid4().hex[:6]}"
                    plan = app.orchestrator.formulate_plan(
                        task_id=task_id,
                        query=active_query,
                        template_id=template_id,
                        world_id=world_id,
                        manual_overrides=state.get("manual_overrides"),
                        mcp_client=app.mcp_mgr
                    )
                    state["task_id"] = task_id
                    state["query"] = active_query
                    state["plan"] = plan
                    state["auto_robot"] = plan.robot_template
                    state["auto_world"] = plan.world_template

                    plan_card = app.format_plan_card(plan)
                    badge = app.format_stage_badge(TaskStage.PLANNING if plan.is_supported else TaskStage.BLOCKED)
                    launch_update = gr.update(interactive=plan.is_supported and not plan.is_stale)

                    mode_text = "⚠️ Manual Override Active" if plan.is_manual_override else f"🤖 Auto-selected by planner ({plan.selection_source.replace('_', ' ').title()})"
                    indicator = (
                        f"**Selection Mode:** `{mode_text}`  \n"
                        f"- **Robot:** `{plan.robot_template}`  \n"
                        f"- **World:** `{plan.world_template}`"
                    )

                    speech_text = (
                        f"Plan formulated for {plan.robot_template}. Ready to launch."
                        if plan.is_supported
                        else f"Plan rejected: {plan.rejection_reason}"
                    )
                    tts_audio = await app.synthesize_speech(speech_text)
                    return active_query, plan.robot_template, plan.world_template, indicator, plan_card, badge, launch_update, tts_audio, state

                plan_btn.click(
                    fn=on_formulate_plan,
                    inputs=[audio_mic, prompt_box, template_dropdown, world_dropdown, studio_state],
                    outputs=[prompt_box, template_dropdown, world_dropdown, auto_selection_indicator_md, plan_card_md, stage_badge_md, launch_btn, voice_player, studio_state]
                )

                async def on_launch_pipeline(audio_file, query_text, inject_f, template_id, world_id, gui_enabled, state):
                    active_query = await app.transcribe_if_audio(audio_file, query_text)
                    task_id = state.get("task_id") or f"task_{int(datetime.now(timezone.utc).timestamp())}_{uuid.uuid4().hex[:6]}"

                    # Ensure fresh plan exists
                    plan = state.get("plan")
                    if not plan or plan.user_goal != active_query or plan.is_stale:
                        plan = app.orchestrator.formulate_plan(
                            task_id=task_id,
                            query=active_query,
                            template_id=template_id,
                            world_id=world_id,
                            manual_overrides=state.get("manual_overrides"),
                            mcp_client=app.mcp_mgr
                        )
                        state["plan"] = plan

                    plan_card = app.format_plan_card(plan)
                    if not plan.is_supported:
                        badge = app.format_stage_badge(TaskStage.BLOCKED)
                        diag_text = f"### ⛔ Execution Blocked\n{plan.rejection_reason}"
                        tts_audio = await app.synthesize_speech(plan.rejection_reason)
                        yield (
                            active_query, plan_card, badge, [], app.format_sensor_status(None),
                            diag_text, "*No patch proposed.*", "",
                            gr.update(interactive=False), gr.update(interactive=False),
                            tts_audio, "*Report unavailable for blocked task.*", state
                        )
                        return

                    events_collected: List[TaskEvent] = []
                    def event_sink(evt: TaskEvent):
                        events_collected.append(evt)

                    new_state = {
                        "task_id": task_id, "query": active_query, "plan": plan, "events": events_collected,
                        "workspace_dir": None, "patch_proposal": None, "initial_test": None,
                        "diagnosis": None, "retest_result": None, "source_mode": app.mode,
                        "manual_overrides": state.get("manual_overrides", {})
                    }

                    pipeline_task = asyncio.create_task(app.orchestrator.execute_studio_pipeline(
                        task_id=task_id,
                        query=active_query,
                        inject_fault=inject_f,
                        template_id=template_id,
                        world_id=world_id,
                        plan=plan,
                        on_event_cb=event_sink,
                        headless=not bool(gui_enabled)
                    ))

                    seen_events = 0
                    while not pipeline_task.done():
                        await asyncio.sleep(0.3)
                        if len(events_collected) != seen_events:
                            seen_events = len(events_collected)
                            last_event = events_collected[-1]
                            yield (
                                active_query, plan_card, app.format_stage_badge(last_event.stage),
                                app.events_to_table_data(events_collected),
                                app.format_sensor_status(None, last_event.stage),
                                f"**Pipeline running:** {last_event.user_facing_message}",
                                "*Waiting for measured pipeline results.*", "",
                                gr.update(interactive=False), gr.update(interactive=False), None,
                                "*Report will update when the pipeline returns.*", new_state
                            )

                    pipeline_res = await pipeline_task
                    new_state = {
                        "task_id": task_id,
                        "query": active_query,
                        "plan": plan,
                        "workspace_dir": pipeline_res.get("workspace_dir"),
                        "patch_proposal": pipeline_res.get("patch_proposal"),
                        "initial_test": pipeline_res.get("test_result"),
                        "diagnosis": pipeline_res.get("diagnosis"),
                        "retest_result": None,
                        "events": events_collected,
                        "source_mode": pipeline_res.get("source_mode", app.mode),
                        "manual_overrides": state.get("manual_overrides", {})
                    }

                    badge = app.format_stage_badge(pipeline_res["stage"])
                    table_rows = app.events_to_table_data(events_collected)
                    sensor_md = app.format_sensor_status(pipeline_res.get("test_result"), pipeline_res["stage"])

                    # Diagnosis text
                    diag = pipeline_res.get("diagnosis")
                    if diag:
                        conf_str = diag.confidence_label.value if hasattr(diag.confidence_label, "value") else str(diag.confidence_label)
                        diag_text = (
                            f"**Diagnostic Summary:** {diag.issue_summary}\n\n"
                            f"**Plain-Language Analogy:** *\"{diag.plain_language_analogy}\"*\n\n"
                            f"**Confidence:** `{conf_str.upper()}` | **Case ID:** `{diag.case_id}` | **Reasoning:** `{diag.reasoning_source}`"
                        )
                        if diag.evidence_records:
                            diag_text += "\n\n**Evidence Records:**\n" + "\n".join(
                                f"- `{e.evidence_id}` {e.source.value} `{e.target}` — `{e.status.value}`: `{e.result}`"
                                for e in diag.evidence_records
                            )
                        speech_text = diag.voice_script
                    else:
                        diag_text = pipeline_res.get("error", "No diagnosis was generated.")
                        speech_text = diag_text

                    # Patch diff
                    proposal: Optional[PatchProposal] = pipeline_res.get("patch_proposal")
                    if proposal:
                        patch_info = (
                            f"**Patch ID:** `{proposal.patch_id}` | **Target File:** `{proposal.target_file}`\n\n"
                            f"**SHA-256 Digest:** `{proposal.diff_sha256}`\n\n"
                            "Human approval creates an ephemeral, one-time policy token for this exact diff."
                        )
                        diff_code = proposal.unified_diff
                        approve_interactive = gr.update(interactive=True)
                        reject_interactive = gr.update(interactive=True)
                    else:
                        patch_info = f"*No patch proposed. Pipeline status: `{pipeline_res['stage'].value}`.*"
                        diff_code = ""
                        approve_interactive = gr.update(interactive=False)
                        reject_interactive = gr.update(interactive=False)

                    tts_audio = await app.synthesize_speech(speech_text)
                    report = ReportGenerator.generate_markdown_report(
                        task_id=task_id, query=active_query, events=events_collected,
                        initial_test=pipeline_res.get("test_result"), diagnosis=diag,
                        patch_proposal=proposal, patch_approved=False,
                        mode=pipeline_res.get("source_mode", app.mode),
                    )

                    yield (
                        active_query,
                        plan_card,
                        badge,
                        table_rows,
                        sensor_md,
                        diag_text,
                        patch_info,
                        diff_code,
                        approve_interactive,
                        reject_interactive,
                        tts_audio,
                        report,
                        new_state
                    )

                launch_btn.click(
                    fn=on_launch_pipeline,
                    inputs=[audio_mic, prompt_box, fault_checkbox, template_dropdown, world_dropdown, gui_checkbox, studio_state],
                    outputs=[
                        prompt_box,
                        plan_card_md,
                        stage_badge_md,
                        timeline_table,
                        sensor_status_md,
                        diag_details_md,
                        patch_info_md,
                        diff_box,
                        approve_btn,
                        reject_btn,
                        voice_player,
                        final_report_md,
                        studio_state
                    ]
                )

                def on_open_gui(state):
                    plan = (state or {}).get("plan")
                    if plan and getattr(plan, "simulator_adapter", None) == "mujoco":
                        import os
                        from .simulators.registry import get_simulator_registry
                        from .robot_registry import get_robot_registry
                        reg = get_simulator_registry()
                        mujoco_adapter = reg.get("mujoco")
                        robot_reg = get_robot_registry()
                        r_model = robot_reg.get(plan.robot_template)
                        entrypoint = r_model.description.entrypoint if r_model and r_model.description else None
                        if not entrypoint or not os.path.exists(entrypoint):
                            from .model_discovery import ModelDiscovery
                            discovery = ModelDiscovery()
                            pkg = discovery.discover_local_packages().get(plan.robot_template)
                            if pkg and pkg.entrypoint and os.path.exists(pkg.entrypoint):
                                entrypoint = pkg.entrypoint
                        if entrypoint and os.path.exists(entrypoint):
                            res = mujoco_adapter.launch_gui(entrypoint)
                            if res.get("success"):
                                return f"🟢 **MuJoCo GUI Opened**: Viewer launched for `{os.path.basename(entrypoint)}`. Check your desktop for the MuJoCo simulation window."
                            return f"🔴 **Could not open MuJoCo GUI**: {res.get('error', 'Unknown error.')}"
                        return "🔴 **Could not open MuJoCo GUI**: Model entrypoint file not found on disk."

                    res = app.orchestrator.ws_mgr.open_gazebo_gui()
                    if res.get("success"):
                        return "🟢 **Gazebo GUI Client Opened**: Attached to the active simulation server. Check your desktop for the Gazebo 3D simulation window."
                    return f"🔴 **Could not open Gazebo GUI**: {res.get('error', 'Unknown error. Make sure a simulation has been launched.')}"

                open_gui_btn.click(
                    fn=on_open_gui,
                    inputs=[studio_state],
                    outputs=[gui_status_md]
                )

                async def on_approve_patch(state):
                    task_id = state["task_id"]
                    proposal = state["patch_proposal"]
                    ws_dir = state["workspace_dir"]
                    events = state["events"]

                    if not proposal or not ws_dir:
                        return (
                            app.format_stage_badge(TaskStage.FAILED),
                            app.events_to_table_data(events),
                            "",
                            "### ❌ Error: Missing patch proposal or workspace.",
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
                        approval_token=None,
                        workspace_dir=ws_dir,
                        on_event_cb=event_sink
                    )

                    state["retest_result"] = retest_res.get("retest_result")
                    state["source_mode"] = retest_res.get("source_mode", state.get("source_mode", "live"))
                    badge = app.format_stage_badge(retest_res["stage"])
                    table_rows = app.events_to_table_data(events)

                    retest_obj = retest_res.get("retest_result")
                    if retest_obj:
                        retest_md_text = (
                            f"### {'✅' if retest_obj.passed else '❌'} Live verification {'passed' if retest_obj.passed else 'failed'}\n"
                            f"- **Test ID:** `{retest_obj.test_id}`\n"
                            f"- **Measured Values:** `{retest_obj.measured_values}`\n"
                            f"- **Source/status:** `{retest_obj.source_mode}` / `{retest_obj.status}`\n"
                            f"- **Log Excerpt:** {retest_obj.log_excerpt}"
                        )
                    else:
                        retest_md_text = f"### Pipeline status: `{retest_res['stage'].value}`\n{retest_res.get('error', 'No retest result was produced.')}"

                    sensor_md = app.format_sensor_status(retest_obj, retest_res["stage"])
                    report = ReportGenerator.generate_markdown_report(
                        task_id=task_id,
                        query=state["query"],
                        events=events,
                        initial_test=state["initial_test"],
                        diagnosis=state["diagnosis"],
                        patch_proposal=proposal,
                        patch_approved=True,
                        retest_result=retest_obj,
                        mode=state.get("source_mode", "live")
                    )

                    speech_text = "Retest passed! LiDAR scan payload successfully verified. Workflow complete." if (retest_obj and retest_obj.passed) else "Retest finished."
                    tts_audio = await app.synthesize_speech(speech_text)

                    return (
                        badge,
                        table_rows,
                        sensor_md,
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
                        sensor_status_md,
                        retest_md,
                        final_report_md,
                        approve_btn,
                        reject_btn,
                        voice_player,
                        studio_state
                    ]
                )

                def on_cancel_task(state):
                    events = state.get("events", [])
                    if state.get("task_id"):
                        evt = app.orchestrator.cancel_task(state["task_id"])
                        if evt:
                            events.append(evt)
                    badge = app.format_stage_badge(TaskStage.CANCELLED)
                    report = ReportGenerator.generate_markdown_report(
                        task_id=state.get("task_id", "task_cancelled"), query=state.get("query", ""),
                        events=events, initial_test=state.get("initial_test"), diagnosis=state.get("diagnosis"),
                        patch_proposal=state.get("patch_proposal"), patch_approved=False,
                        mode=state.get("source_mode", app.mode),
                    )
                    return badge, app.events_to_table_data(events), gr.update(interactive=False), gr.update(interactive=False), report, state

                def on_reject_patch(state):
                    events = state.get("events", [])
                    if state.get("task_id"):
                        evt = app.orchestrator.cancel_task(state["task_id"], "Patch rejected by user; managed runtime stopped.")
                        if evt:
                            events.append(evt)
                    badge = app.format_stage_badge(TaskStage.CANCELLED)
                    report = ReportGenerator.generate_markdown_report(
                        task_id=state.get("task_id", "task_rejected"), query=state.get("query", ""),
                        events=events, initial_test=state.get("initial_test"), diagnosis=state.get("diagnosis"),
                        patch_proposal=state.get("patch_proposal"), patch_approved=False,
                        mode=state.get("source_mode", app.mode),
                    )
                    return badge, app.events_to_table_data(events), gr.update(interactive=False), gr.update(interactive=False), report, state

                cancel_btn.click(
                    fn=on_cancel_task,
                    inputs=[studio_state],
                    outputs=[stage_badge_md, timeline_table, approve_btn, reject_btn, final_report_md, studio_state]
                )

                reject_btn.click(
                    fn=on_reject_patch,
                    inputs=[studio_state],
                    outputs=[stage_badge_md, timeline_table, approve_btn, reject_btn, final_report_md, studio_state]
                )

            # =========================================================================
            # TAB 2: DIAGNOSTIC COPILOT (READ-ONLY INSPECTION)
            # =========================================================================
            with gr.Tab("📚 Robot Model Library"):
                gr.Markdown(
                    """
                    ## Robot Model Library
                    Browse robot descriptions across arms, grippers, mobile manipulators, wheeled robots, quadrupeds, humanoids, and drones.

                    **Important:** a 3D description is not the same as a working simulation. The current full-pipeline templates bundled here are the differential-drive robot and the Panda-derived arm template. Other entries can be fetched and inspected, but remain blocked for pipeline execution until a Gazebo adapter, controllers, sensors, and tests are verified.
                    """
                )
                if not robot_catalog.descriptions_package_installed:
                    gr.Markdown(
                        "**Full catalogue optional add-on is not installed.** This screen still lists common models. "
                        "To discover/cache the wider `robot_descriptions.py` catalogue, install it in your RoboSahaya environment with `pip install -e '.[robot-catalogue]'` and restart the app. "
                        "Descriptions are fetched only when you explicitly click **Prepare / Cache Description**."
                    )
                else:
                    gr.Markdown("✅ Optional `robot_descriptions` package detected. Model-specific source downloads happen only after an explicit prepare action.")

                from .cloud.client import get_cloud_status
                cloud_info = get_cloud_status()
                gr.Markdown(
                    f"**Cloud Model Backend:** `{cloud_info['status_badge']}` | "
                    f"**Storage Bucket:** `{cloud_info['storage_bucket']}` | "
                    f"*{cloud_info['diagnostics']}*"
                )

                with gr.Row():
                    catalog_search = gr.Textbox(label="Search robots", placeholder="e.g. SO-101, Panda, Go2, UR5e, humanoid", scale=3)
                    family_values = sorted({m.family for m in robot_catalog.list_models(include_discovered=True)})
                    catalog_family = gr.Dropdown(label="Robot family", choices=["All", *family_values], value="All", scale=2)
                    catalog_refresh_btn = gr.Button("🔄 Refresh Catalogue", scale=1)

                catalog_table = gr.Dataframe(
                    headers=["ID", "Robot", "Family", "Format", "Licence", "Execution status", "Availability"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    value=robot_catalog.format_table_rows(),
                    label="Known model descriptions",
                    row_count=(12, "dynamic"),
                    col_count=(7, "fixed"),
                )
                with gr.Row():
                    model_to_prepare = gr.Dropdown(
                        label="Choose a robot description to fetch / inspect",
                        choices=robot_catalog.choices(),
                        value=None,
                        interactive=True,
                        allow_custom_value=False,
                        scale=2,
                    )
                    prepare_model_btn = gr.Button("📥 Prepare / Cache Description", variant="primary", scale=1)
                    verify_integrity_btn = gr.Button("🛡️ Verify Integrity", variant="secondary", scale=1)
                    inspect_readiness_btn = gr.Button("📊 9D Readiness Profile", variant="secondary", scale=1)
                model_prepare_result = gr.Markdown(
                    "*Choose a model and click Prepare / Cache Description. Runtime readiness will not be inferred from download success.*"
                )

                def refresh_catalog(search_text="", family="All", selected=None):
                    models = robot_catalog.list_models(search=search_text or "", family=family or "All")
                    choices = [(f"{m.display_name} — {'bundled pipeline template' if m.runtime_ready else 'description only; not runtime verified'}", m.id) for m in models]
                    values = {v for _, v in choices}
                    chosen = selected if selected in values else None
                    summary = (
                        f"**Showing {len(models)} entries.** "
                        f"`{sum(m.runtime_ready for m in models)}` have a bundled RoboSahaya execution template; "
                        f"`{sum(not m.runtime_ready for m in models)}` are description-only until separately validated."
                    )
                    return robot_catalog.format_table_rows(search=search_text or "", family=family or "All"), gr.update(choices=choices, value=chosen), summary

                def on_prepare_robot_description(model_id):
                    result = robot_catalog.prepare_description(model_id or "")
                    if not result.get("success"):
                        return f"### ⚠️ Description not prepared\n\n{result.get('error', 'No model was selected.')}"
                    if result.get("runtime_ready"):
                        return f"### ✅ {result['display_name']}\n\n{result['message']}\n\n**Runtime template:** `{result['runtime_template_id']}`"
                    stats = result.get("stats") or {}
                    return (
                        f"### ✅ Description cached and inspected: {result['display_name']}\n\n"
                        f"- **Format:** `{result['format']}`\n"
                        f"- **Description path:** `{result['path']}`\n"
                        f"- **Links/bodies:** `{stats.get('links')}` | **Joints:** `{stats.get('joints')}` | **Mesh references:** `{stats.get('mesh_references')}`\n"
                        f"- **SHA-256:** `{result['sha256']}`\n"
                        f"- **Licence:** `{result.get('license', 'Check upstream source')}`\n\n"
                        f"**Not yet Gazebo-ready:** {result['message']} The standard execution pipeline remains blocked for this model until its simulator integration, controllers, sensor topics, and verification tests are implemented and passed."
                    )

                def on_verify_integrity(model_id):
                    if not model_id:
                        return "### ⚠️ No robot model selected."
                    res = robot_catalog.model_cache.validate_cache_integrity(model_id)
                    if res.get("valid"):
                        return f"### 🛡️ Cache Integrity Verified: `{model_id}`\n\n- All `{res.get('file_count', 0)}` assets matched their expected SHA-256 digests.\n- Local cache is cryptographically sound and safe for staging."
                    return f"### ❌ Cache Integrity Check Failed: `{model_id}`\n\n- Reason: `{res.get('reason')}`\n- Recommendation: Click 'Prepare / Cache Description' to re-fetch and restore valid assets."

                def on_inspect_readiness(model_id):
                    if not model_id:
                        return "### ⚠️ No robot model selected."
                    decision = robot_catalog.evaluate_readiness(model_id)
                    d = decision.dimensions
                    blockers_md = "\n".join(f"- 🔴 {b}" for b in decision.blockers) if decision.blockers else "*(None — all required preconditions met)*"
                    return f"""### 📊 Authoritative 9-Dimensional Readiness: `{model_id}` ({decision.model_version})
**Overall Execution Verdict:** `{decision.summary_label}`

| Dimension | Verified Status | Meaning |
|---|---|---|
| **1. Catalogue Identity** | `{d.catalogue_status.value}` | Known canonical model and metadata |
| **2. Cloud Availability** | `{d.cloud_availability.value}` | Upstream / Supabase asset availability |
| **3. Local Cache Status** | `{d.local_cache_status.value}` | Local files downloaded and integrity verified |
| **4. Description Validity** | `{d.description_validity.value}` | XML/MJCF syntax and asset resolution |
| **5. Simulator Compatibility** | `{d.simulator_compatibility.value}` | Compatible runtime adapter registered |
| **6. Simulation Readiness** | `{d.simulation_readiness.value}` | Physics stepping verified |
| **7. Controller Readiness** | `{d.controller_readiness.value}` | Actuator control response verified |
| **8. Sensor Readiness** | `{d.sensor_readiness.value}` | Topic stream telemetry verified |
| **9. Task Verification** | `{d.task_verification.value}` | E2E task success criteria met |

**Identified Blockers / Preconditions:**
{blockers_md}
"""

                catalog_refresh_btn.click(
                    fn=refresh_catalog,
                    inputs=[catalog_search, catalog_family, model_to_prepare],
                    outputs=[catalog_table, model_to_prepare, model_prepare_result],
                )
                catalog_search.submit(
                    fn=refresh_catalog,
                    inputs=[catalog_search, catalog_family, model_to_prepare],
                    outputs=[catalog_table, model_to_prepare, model_prepare_result],
                )
                catalog_family.change(
                    fn=refresh_catalog,
                    inputs=[catalog_search, catalog_family, model_to_prepare],
                    outputs=[catalog_table, model_to_prepare, model_prepare_result],
                )
                prepare_model_btn.click(
                    fn=on_prepare_robot_description,
                    inputs=[model_to_prepare],
                    outputs=[model_prepare_result],
                )
                verify_integrity_btn.click(
                    fn=on_verify_integrity,
                    inputs=[model_to_prepare],
                    outputs=[model_prepare_result],
                )
                inspect_readiness_btn.click(
                    fn=on_inspect_readiness,
                    inputs=[model_to_prepare],
                    outputs=[model_prepare_result],
                )

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
                            ("Configured reasoning endpoint (if available)", "ai"),
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
                    engine_badge = ("🧠 Configured reasoning endpoint with deterministic fallback"
                                    if eng_mode == "ai" else "⚙️ Deterministic Rule Baseline")

                    intent = grounding_res.inferred_intent
                    if grounding_res.primary_target:
                        target = grounding_res.primary_target.canonical_name
                        etype = grounding_res.primary_target.entity_type
                        grounding_badge = f"🎯 Grounded: `{target}` ({etype.upper()}) | Confidence: `{int(grounding_res.primary_target.confidence * 100)}%` | {code_mixed_status}\n\n*{engine_badge}*"
                    else:
                        target = "ros2_system"
                        etype = "system"
                        grounding_badge = f"🎯 Context: General ROS 2 Diagnosis ({intent.replace('_', ' ').title()}) | {code_mixed_status}\n\n*{engine_badge}*"

                    if app.inspector is None:
                        return (raw_text, grounding_badge, "### Live ROS inspection is unavailable in this process.",
                                "No live evidence was collected.", None)

                    diag = await app.copilot_engine.analyze_async(
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

            # =========================================================================
            # TAB 3: MCP SERVER (ROS 2 & GAZEBO INTEGRATION)
            # =========================================================================
            with gr.Tab("🔌 MCP Server (ROS 2 & Gazebo Integration)"):
                gr.Markdown(
                    """
                    ### 🔌 Model Context Protocol (MCP) Server Manager
                    *Connect RoboSahaya Studio to a real MCP server managing your ROS 2 nodes, topics, TF transforms, and Gazebo Harmonic simulation.*
                    
                    `RoboSahaya Web UI ➔ RoboSahaya Backend ➔ MCP Client ➔ MCP Server ➔ Controlled ROS 2/Gazebo Tools`
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        mcp_badge_md = gr.Markdown(app.format_mcp_status_badge(app.mcp_mgr.status))
                    with gr.Column(scale=2):
                        mcp_runtime_status_md = gr.Markdown(
                            f"**Runtime Status:** ROS 2: `{'🟢 Available' if app.mcp_mgr.status.ros2_available else '⚪ Not checked'}` | "
                            f"Gazebo Sim: `{'🟢 Detected' if app.mcp_mgr.status.gazebo_available else '⚪ Not checked'}`\n\n"
                            f"*Notice: MCP connectivity proves protocol handshake; ROS 2 and Gazebo runtime statuses are verified separately.*"
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        mcp_transport_radio = gr.Radio(
                            label="Transport Method",
                            choices=[
                                ("Local stdio (Backend-Managed Process)", "stdio"),
                                ("Streamable HTTP Endpoint", "streamable_http"),
                                ("Server-Sent Events (SSE)", "sse")
                            ],
                            value="stdio"
                        )
                        mcp_cmd_box = gr.Textbox(
                            label="Managed Server Command (stdio)",
                            value=sys.executable,
                            info="Executable path for managed child server process."
                        )
                        mcp_args_box = gr.Textbox(
                            label="Command Arguments (space-separated)",
                            value="-m robosahaya.mcp.server",
                            info="Arguments passed to the child process."
                        )
                        mcp_url_box = gr.Textbox(
                            label="Server URL (HTTP / SSE)",
                            value="http://127.0.0.1:8000/mcp",
                            info="HTTP/SSE destination for remote or external MCP servers."
                        )
                        mcp_token_box = gr.Textbox(
                            label="Authentication Token (Optional)",
                            type="password",
                            placeholder="Bearer token or API key (masked in UI and logs)",
                            info="Never logged or displayed in diagnostic traces."
                        )
                        mcp_timeout_slider = gr.Slider(
                            label="Request Timeout (seconds)",
                            minimum=5.0,
                            maximum=60.0,
                            value=15.0,
                            step=1.0
                        )

                        with gr.Row():
                            mcp_connect_btn = gr.Button("🔌 Connect", variant="primary", scale=1)
                            mcp_test_btn = gr.Button("🩺 Test Connection", variant="secondary", scale=1)
                            mcp_disconnect_btn = gr.Button("⏹️ Disconnect", variant="stop", scale=1)
                            mcp_refresh_btn = gr.Button("🔄 Refresh Tools", variant="secondary", scale=1)

                    with gr.Column(scale=2):
                        gr.Markdown("#### 📋 Discovered MCP Tools")
                        mcp_tools_table = gr.DataFrame(
                            headers=["Tool Name", "Description", "Permission Tier", "Status"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            wrap=True,
                            value=app.format_mcp_tools_table(app.mcp_mgr.status.discovered_tools)
                        )

                        with gr.Accordion("🩺 Connection Diagnostics & Troubleshooting", open=True):
                            mcp_diag_md = gr.Markdown(
                                "**Last Connected:** *None* | **Last Health Check:** *None*\n\n"
                                "**Diagnostics Log:**\n"
                                "*(No connection events recorded yet. Configure settings and click 'Connect'.)*"
                            )

                def format_diagnostics_display(status: MCPConnectionStatus) -> str:
                    last_conn = status.last_connected_at or "*Never*"
                    last_chk = status.last_health_check_at or "*Never*"
                    chk_res = status.health_check_result or "*No check performed*"
                    logs_str = "\n".join(f"- {line}" for line in status.diagnostics_log[-10:]) if status.diagnostics_log else "*(No events)*"
                    return f"**Last Connected:** `{last_conn}` | **Last Health Check:** `{last_chk}`\n\n**Result:** `{chk_res}`\n\n**Recent Events:**\n{logs_str}"

                def format_runtime_display(status: MCPConnectionStatus) -> str:
                    r_icon = "🟢 Available" if status.ros2_available else ("🔴 Unavailable" if status.ros2_available is False else "⚪ Not checked")
                    g_icon = "🟢 Detected" if status.gazebo_available else ("🔴 Not detected" if status.gazebo_available is False else "⚪ Not checked")
                    return (
                        f"**Runtime Status:** ROS 2: `{r_icon}` | Gazebo Sim: `{g_icon}`\n\n"
                        f"*Notice: MCP connectivity proves protocol handshake; ROS 2 and Gazebo runtime statuses are verified separately.*"
                    )

                async def on_mcp_connect(transport_str, cmd, args_str, url, token, timeout):
                    try:
                        args = shlex.split(args_str)
                    except Exception:
                        args = args_str.split()

                    cfg = MCPConnectionConfig(
                        transport=MCPTransportType(transport_str),
                        command=cmd,
                        args=args,
                        server_url=url,
                        auth_token=token if token and token.strip() else None,
                        timeout_sec=float(timeout)
                    )
                    ok, msg = await app.mcp_mgr.connect(cfg)
                    badge = app.format_mcp_status_badge(app.mcp_mgr.status)
                    runtime_md = format_runtime_display(app.mcp_mgr.status)
                    tools_data = app.format_mcp_tools_table(app.mcp_mgr.status.discovered_tools)
                    diag_md = format_diagnostics_display(app.mcp_mgr.status)
                    return badge, runtime_md, tools_data, diag_md

                async def on_mcp_test():
                    ok, msg = await app.mcp_mgr.test_connection()
                    badge = app.format_mcp_status_badge(app.mcp_mgr.status)
                    runtime_md = format_runtime_display(app.mcp_mgr.status)
                    tools_data = app.format_mcp_tools_table(app.mcp_mgr.status.discovered_tools)
                    diag_md = format_diagnostics_display(app.mcp_mgr.status)
                    return badge, runtime_md, tools_data, diag_md

                async def on_mcp_disconnect():
                    ok, msg = await app.mcp_mgr.disconnect()
                    badge = app.format_mcp_status_badge(app.mcp_mgr.status)
                    runtime_md = format_runtime_display(app.mcp_mgr.status)
                    tools_data = app.format_mcp_tools_table([])
                    diag_md = format_diagnostics_display(app.mcp_mgr.status)
                    return badge, runtime_md, tools_data, diag_md

                async def on_mcp_refresh():
                    ok, msg, tools = await app.mcp_mgr.refresh_tools()
                    badge = app.format_mcp_status_badge(app.mcp_mgr.status)
                    runtime_md = format_runtime_display(app.mcp_mgr.status)
                    tools_data = app.format_mcp_tools_table(app.mcp_mgr.status.discovered_tools)
                    diag_md = format_diagnostics_display(app.mcp_mgr.status)
                    return badge, runtime_md, tools_data, diag_md

                mcp_connect_btn.click(
                    fn=on_mcp_connect,
                    inputs=[mcp_transport_radio, mcp_cmd_box, mcp_args_box, mcp_url_box, mcp_token_box, mcp_timeout_slider],
                    outputs=[mcp_badge_md, mcp_runtime_status_md, mcp_tools_table, mcp_diag_md]
                )
                mcp_test_btn.click(
                    fn=on_mcp_test,
                    inputs=[],
                    outputs=[mcp_badge_md, mcp_runtime_status_md, mcp_tools_table, mcp_diag_md]
                )
                mcp_disconnect_btn.click(
                    fn=on_mcp_disconnect,
                    inputs=[],
                    outputs=[mcp_badge_md, mcp_runtime_status_md, mcp_tools_table, mcp_diag_md]
                )
                mcp_refresh_btn.click(
                    fn=on_mcp_refresh,
                    inputs=[],
                    outputs=[mcp_badge_md, mcp_runtime_status_md, mcp_tools_table, mcp_diag_md]
                )

        # Auto-connect MCP server on initial app load if not already connected
        async def on_app_load():
            if not app.mcp_mgr.is_connected:
                await app.mcp_mgr.connect(MCPConnectionConfig())
            return (
                app.format_mcp_status_badge(app.mcp_mgr.status),
                format_runtime_display(app.mcp_mgr.status),
                app.format_mcp_tools_table(app.mcp_mgr.status.discovered_tools),
                format_diagnostics_display(app.mcp_mgr.status)
            )

        demo.load(
            fn=on_app_load,
            inputs=[],
            outputs=[mcp_badge_md, mcp_runtime_status_md, mcp_tools_table, mcp_diag_md]
        )

    return demo
