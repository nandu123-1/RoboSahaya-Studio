import asyncio
import os
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
        self.gnani = GnaniClient()
        self.inspector = self._init_inspector()
        self.grounder = EntityGrounder()
        self.engine = DiagnosticEngine(self.inspector)

    def _init_inspector(self) -> BaseInspector:
        if self.mode == "live" and RCLPY_AVAILABLE and LiveRclpyInspector is not None:
            try:
                insp = LiveRclpyInspector()
                print("[RoboSahaya] Connected to Live ROS 2 Graph via rclpy.")
                return insp
            except Exception as ex:
                print(f"[Warning] Failed to start LiveRclpyInspector ({ex}). Falling back to MockInspector.")
        return MockInspector(scenario_name="clean_healthy_system")

    async def process_pipeline(
        self,
        audio_path: Optional[str],
        text_input: Optional[str],
        engine_mode: str = "ai"
    ) -> Tuple[str, str, str, str, Optional[str]]:
        raw_text = (text_input or "").strip()

        # Step 1: Speech-to-Text via live Gnani Prisma STT if audio was provided
        if audio_path and not raw_text:
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                transcript = await self.gnani.transcribe_audio(audio_bytes)
                if transcript:
                    raw_text = transcript
                else:
                    raw_text = "[No clear speech detected in recording]"
            except Exception as ex:
                raw_text = f"[Audio Transcription Failed: {ex}]"

        if not raw_text or raw_text.startswith("[No clear"):
            return (
                raw_text or "Please provide a voice recording or type your question.",
                "### ℹ️ Waiting for input",
                "### ℹ️ Waiting for input\nSpeak or type a ROS 2 problem to begin real-time diagnosis.",
                "Speak or type a ROS 2 problem to begin diagnosis.",
                None
            )

        # Step 2: Dynamic Entity Grounding
        grounding_res = self.grounder.ground(raw_text)
        code_mixed_status = "✨ Telugu-English Detected" if grounding_res.is_code_mixed else "English"
        token_count = self.engine.evon_adapter.count_tokens(raw_text)
        engine_badge = (
            f"🧠 Powered by Gnani AI Evon v3.3 (Local Tokenizer: {token_count} tokens)"
            if engine_mode == "ai"
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

        # Step 3: Diagnostic Engine execution with real-time inspection
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

        # Step 5: Real-time Gnani Timbre TTS speech synthesis
        audio_output = None
        try:
            audio_bytes = await self.gnani.synthesize_speech(voice_script, voice="Kaveri")
            if audio_bytes:
                temp_audio_path = "/tmp/robosahaya_tts.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_bytes)
                audio_output = temp_audio_path
        except Exception as ex:
            print(f"[UI] Timbre TTS Error: {ex}")
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
            **Real-time ROS 2 diagnostic copilot powered by Gnani AI (Prisma STT, Timbre TTS & Evon v3.3 Reasoning).**  
            *Speak or type your robotics problem in English or Telugu-English for instant evidence-based root cause analysis.*
            """
        )

        # System & Model Status Card
        local_info = app_instance.engine.evon_adapter.get_local_model_info()
        inspector_state = "🟢 Live ROS 2 Inspector Active (`rclpy`)" if isinstance(app_instance.inspector, LiveRclpyInspector) else "🟡 Standalone Mode"
        speech_status = "🟢 Gnani Vachana Speech Services Connected (Prisma STT + Timbre TTS)" if app_instance.gnani.is_configured else "🟡 Gnani API Key Not Configured"

        if local_info["weights_present"]:
            model_badge_md = (
                f"🟢 **Local Model Weights Verified:** `{local_info['model_name']}` ({local_info['parameters']}) | "
                f"**Shards:** `{local_info['shard_count']}/{local_info['expected_shards']} Safetensors` | "
                f"**Tokenizer:** `{local_info['vocab_size']:,} Vocab (Telugu+English)` | **Context:** `{local_info['max_context']:,} tokens`"
            )
        else:
            model_badge_md = "🟡 **Local Weights:** Offline / Simulated Fallback Active"

        gr.Markdown(
            f"**System State:** {inspector_state} | **ROS Distro:** `Jazzy` | **Domain:** `{settings.ROS_DOMAIN_ID}` | {speech_status}\n\n{model_badge_md}"
        )

        with gr.Row():
            engine_radio = gr.Radio(
                label="🧠 Diagnostic Reasoning Mode",
                choices=[
                    ("Gnani AI Evon v3.3 (Generative AI Reasoning)", "ai"),
                    ("Deterministic Safety Baseline (Pure Rule Engine)", "rules")
                ],
                value="ai",
                info="Toggle between Gnani AI Evon reasoning model and the deterministic rule safety baseline."
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎙️ 1. Speak or Type Problem")
                audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voice Input (English or Telugu-English)")
                text_input = gr.Textbox(
                    label="Or Type Question",
                    placeholder="e.g. Why is /cmd_vel not reaching the base controller? or నా robot tirugutondi kani /scan data ravatledu",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("🔍 Diagnose ROS 2 Problem", variant="primary")
                    clear_btn = gr.Button("Clear")

                gr.Markdown("### 🎯 Grounding & ASR")
                transcript_out = gr.Textbox(label="🎙️ Gnani Prisma Speech Transcript", interactive=False)
                grounding_out = gr.Markdown(label="Entity Grounding Status")

            with gr.Column(scale=2):
                gr.Markdown("### 📋 2. Evidence-Based Diagnostic Output")
                voice_script_out = gr.Textbox(label="🗣️ Spoken Summary (Timbre Script)", interactive=False)
                audio_player = gr.Audio(label="🔊 Audio Diagnosis (Gnani Timbre TTS)", interactive=False, autoplay=True)
                terminal_card_out = gr.Markdown(label="Terminal Card & Verification Steps")

        # Event Handlers
        async def run_pipeline(audio, text, eng_mode):
            return await app_instance.process_pipeline(audio, text, eng_mode)

        submit_btn.click(
            fn=run_pipeline,
            inputs=[audio_input, text_input, engine_radio],
            outputs=[transcript_out, grounding_out, terminal_card_out, voice_script_out, audio_player]
        )

        clear_btn.click(
            fn=lambda: (None, "", "", "### ℹ️ Ready for input", "### ℹ️ Ready for input", "", None),
            inputs=[],
            outputs=[audio_input, text_input, transcript_out, grounding_out, terminal_card_out, voice_script_out, audio_player]
        )

    return demo
