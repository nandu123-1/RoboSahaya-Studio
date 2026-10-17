import asyncio
from typing import Optional, Tuple
import gradio as gr
from robosahaya.config import settings
from robosahaya.grounding import EntityGrounder
from robosahaya.base_inspector import BaseInspector
from robosahaya.mock_inspector import MockInspector
from robosahaya.diagnostics.engine import DiagnosticEngine
from robosahaya.response_builder import ResponseBuilder
from robosahaya.gnani_client import GnaniClient

# Try importing LiveRclpyInspector
try:
    from robosahaya.ros_inspector import LiveRclpyInspector, RCLPY_AVAILABLE
except Exception:
    LiveRclpyInspector = None
    RCLPY_AVAILABLE = False


class RoboSahayaApp:
    def __init__(self):
        self.mode = settings.ROBOSAHAYA_MODE
        self.scenario = settings.ROBOSAHAYA_MOCK_SCENARIO
        self.gnani = GnaniClient()
        self.inspector = self._init_inspector()
        self.grounder = EntityGrounder()
        self.engine = DiagnosticEngine(self.inspector)

    def _init_inspector(self) -> BaseInspector:
        if self.mode == "live" and RCLPY_AVAILABLE and LiveRclpyInspector is not None:
            try:
                return LiveRclpyInspector()
            except Exception as ex:
                print(f"[Warning] Failed to start LiveRclpyInspector ({ex}). Falling back to MockInspector.")
        return MockInspector(scenario_name=self.scenario)

    def switch_scenario(self, scenario_name: str) -> str:
        self.scenario = scenario_name
        if isinstance(self.inspector, MockInspector):
            self.inspector.load_scenario(scenario_name)
        return f"Active Inspector Mode: Mock Replay | Scenario: `{scenario_name}`"

    async def process_pipeline(
        self,
        audio_path: Optional[str],
        text_input: Optional[str],
        scenario_name: str,
        engine_mode: str = "ai"
    ) -> Tuple[str, str, str, str, Optional[str]]:
        # Update scenario if in mock mode
        if isinstance(self.inspector, MockInspector):
            self.switch_scenario(scenario_name)

        # Step 1: Speech-to-Text or direct text input
        raw_text = (text_input or "").strip()
        if audio_path and not raw_text:
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                raw_text = await self.gnani.transcribe_audio(audio_bytes)
            except Exception as ex:
                raw_text = f"[Audio Transcription Failed: {ex}]"

        if not raw_text:
            return (
                "Please provide a voice recording or type your question.",
                "None",
                "### ℹ️ Waiting for input",
                "Speak or type a ROS 2 problem to begin diagnosis.",
                None
            )

        # Step 2: Entity Grounding
        grounding_res = self.grounder.ground(raw_text)
        code_mixed_status = "✨ Telugu-English Detected" if grounding_res.is_code_mixed else "English"
        engine_badge = "🧠 Powered by Gnani AI Evon v3.3" if engine_mode == "ai" else "⚙️ Deterministic Rule Baseline"
        
        target = "/scan"
        etype = "topic"
        intent = grounding_res.inferred_intent

        if grounding_res.primary_target:
            target = grounding_res.primary_target.canonical_name
            etype = grounding_res.primary_target.entity_type
            grounding_badge = f"🎯 Grounded: `{target}` ({etype.upper()}) | Confidence: `{int(grounding_res.primary_target.confidence * 100)}%` | {code_mixed_status}\n\n*{engine_badge}*"
        else:
            grounding_badge = f"🔍 Analyzing topic `/scan` by default | {code_mixed_status}\n\n*{engine_badge}*"

        # Step 3: Diagnostic Engine execution with selected reasoning mode
        diag = self.engine.analyze(
            target_entity=target,
            entity_type=etype,
            intent=intent,
            engine_mode=engine_mode,
            user_query=raw_text
        )

        # Step 4: Formatting responses
        terminal_md = ResponseBuilder.format_terminal_markdown(diag)
        voice_script = ResponseBuilder.format_voice_script(diag)

        # Step 5: Optional Timbre TTS synthesis
        audio_output = None
        try:
            audio_bytes = await self.gnani.synthesize_speech(voice_script)
            if audio_bytes:
                # Save temp audio file
                temp_audio_path = "/tmp/robosahaya_tts.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_bytes)
                audio_output = temp_audio_path
        except Exception:
            audio_output = None

        return (
            raw_text,
            grounding_badge,
            terminal_md,
            voice_script,
            audio_output
        )


def create_ui() -> gr.Blocks:
    app_instance = RoboSahayaApp()

    with gr.Blocks(title="RoboSahaya — ROS 2 Voice Debugging Copilot") as demo:
        gr.Markdown(
            """
            # 🤖 RoboSahaya — Voice-First ROS 2 Debugging Copilot
            **Speak the problem. Inspect the evidence. Find the next step.**  
            *Powered by Gnani AI (Prisma STT, Timbre TTS & Evon v3.3 Reasoning) | Built for Great Indian AI Internship Challenge 2026*
            """
        )

        with gr.Row():
            scenario_dropdown = gr.Dropdown(
                label="🧪 Evaluator Demo Scenario (Mock Replay Switch)",
                choices=[
                    ("QoS Reliability Mismatch (BEST_EFFORT vs RELIABLE)", "qos_reliability_mismatch"),
                    ("Missing Sensor Publisher (/scan publisher count = 0)", "missing_laser_pub"),
                    ("Broken TF Transform Link (base_link -> camera_link)", "tf_broken_camera_link"),
                    ("Simulation Time Divergence (use_sim_time: false)", "sim_time_divergence"),
                    ("Healthy Operational Robot", "clean_healthy_system")
                ],
                value="qos_reliability_mismatch",
                info="Allows judges to instantly reproduce all 5 failure cases with zero ROS 2 setup required."
            )
            engine_radio = gr.Radio(
                label="🧠 Diagnostic Reasoning Mode",
                choices=[
                    ("Gnani AI Evon v3.3 (Generative AI Reasoning)", "ai"),
                    ("Deterministic Safety Baseline (Pure Rule Engine)", "rules")
                ],
                value="ai",
                info="Toggle between Gnani AI's Evon reasoning model and the deterministic rule safety baseline."
            )

        mode_badge = gr.Markdown(
            f"**System State:** `Mock Replay Active` | **ROS Distro:** `Jazzy` | **Domain:** `{settings.ROS_DOMAIN_ID}`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎙️ 1. Speak or Describe Problem")
                audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voice Input (English or Telugu-English)")
                text_input = gr.Textbox(
                    label="Or Type Question",
                    placeholder="e.g. నా robot tirugutondi kani laser scan data ravatledu, why? or Why is /scan missing?",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("🔍 Diagnose ROS 2 Problem", variant="primary")
                    clear_btn = gr.Button("Clear")

                gr.Markdown("### 🎯 Grounding & ASR")
                transcript_out = gr.Textbox(label="Prisma Speech Transcript", interactive=False)
                grounding_out = gr.Markdown(label="Entity Grounding Status")

            with gr.Column(scale=2):
                gr.Markdown("### 📋 2. Evidence-Based Diagnostic Output")
                voice_script_out = gr.Textbox(label="🗣️ Spoken Summary (Timbre Script)", interactive=False)
                audio_player = gr.Audio(label="🔊 Audio Diagnosis (Gnani Timbre)", interactive=False)
                terminal_card_out = gr.Markdown(label="Terminal Card & Verification Steps")

        # Event Handlers
        async def run_pipeline(audio, text, scen, eng_mode):
            return await app_instance.process_pipeline(audio, text, scen, eng_mode)

        submit_btn.click(
            fn=run_pipeline,
            inputs=[audio_input, text_input, scenario_dropdown, engine_radio],
            outputs=[transcript_out, grounding_out, terminal_card_out, voice_script_out, audio_player]
        )

        scenario_dropdown.change(
            fn=app_instance.switch_scenario,
            inputs=[scenario_dropdown],
            outputs=[mode_badge]
        )

        clear_btn.click(
            fn=lambda: (None, "", "", "", "### ℹ️ Ready for input", "", None),
            inputs=[],
            outputs=[audio_input, text_input, transcript_out, grounding_out, terminal_card_out, voice_script_out, audio_player]
        )

    return demo

