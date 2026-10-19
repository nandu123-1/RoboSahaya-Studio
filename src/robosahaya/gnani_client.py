import httpx
from typing import Optional
from robosahaya.config import settings

class GnaniClient:
    """Client for Gnani AI Speech Services (Prisma STT and Timbre TTS)."""

    def __init__(self, api_key: Optional[str] = None, stt_url: Optional[str] = None, tts_url: Optional[str] = None):
        self.stt_api_key_id = api_key if api_key is not None else settings.GNANI_STT_API_KEY_ID
        self.tts_api_key = api_key if api_key is not None else settings.GNANI_TTS_API_KEY
        self.stt_url = (stt_url or settings.GNANI_STT_URL).rstrip("/")
        self.tts_url = (tts_url or settings.GNANI_TTS_URL).rstrip("/")

    @property
    def is_configured(self) -> bool:
        return self.stt_configured or self.tts_configured

    @property
    def stt_configured(self) -> bool:
        return self._valid_key(self.stt_api_key_id)

    @property
    def tts_configured(self) -> bool:
        return self._valid_key(self.tts_api_key)

    @staticmethod
    def _valid_key(value: Optional[str]) -> bool:
        return bool(value and len(value) > 5 and value.lower() not in {
            "your_gnani_api_key_here", "change-me", "changeme", "placeholder"
        })

    @staticmethod
    def ensure_wav_bytes(audio_bytes: bytes) -> bytes:
        """Converts incoming audio bytes to 16kHz mono 16-bit WAV PCM if needed."""
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            proc = subprocess.run(
                [ffmpeg_exe, "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                input=audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            if proc.stdout and len(proc.stdout) > 44:
                return proc.stdout
        except Exception:
            pass
        return audio_bytes

    async def transcribe_audio(self, audio_bytes: bytes, lang: str = "en-IN") -> str:
        """
        Transcribe audio using Gnani Prisma STT (v3 on Vachana platform).
        """
        if not self.stt_configured:
            return ""

        wav_bytes = self.ensure_wav_bytes(audio_bytes)
        headers = {
            "X-API-Key-ID": self.stt_api_key_id
        }
        files = {
            "audio_file": ("audio.wav", wav_bytes, "audio/wav")
        }
        data = {
            "language_code": lang,
            "preferred_language": lang,
            "format": "transcribe"
        }

        async with httpx.AsyncClient(timeout=25.0) as client:
            try:
                resp = await client.post(
                    self.stt_url,
                    headers=headers,
                    files=files,
                    data=data
                )
                resp.raise_for_status()
                json_res = resp.json()
                transcript = json_res.get("transcript") or json_res.get("output", {}).get("literal", "")
                return transcript.strip()
            except Exception as ex:
                print(f"[Gnani STT] Request failed ({type(ex).__name__}): {ex}")
                raise RuntimeError(f"Gnani STT request failed ({type(ex).__name__}): {ex}") from ex

    def is_stt_configured(self) -> bool:
        return self.stt_configured

    def is_tts_configured(self) -> bool:
        return self.tts_configured

    async def check_stt_health(self) -> tuple[bool, str]:
        """Perform a live roundtrip probe to verify STT endpoint."""
        if not self.stt_configured:
            return False, "Gnani STT API key not configured."
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00" * 8000)
        try:
            transcript = await self.transcribe_audio(buf.getvalue())
            return True, "Prisma STT (v3/v2.5) connected and verified."
        except Exception as exc:
            return False, f"Prisma STT probe failed: {exc}"

    async def check_tts_health(self) -> tuple[bool, str]:
        """Perform a live synthesis probe to verify TTS endpoint."""
        if not self.tts_configured:
            return False, "Gnani TTS API key not configured."
        try:
            audio = await self.synthesize_speech("RoboSahaya Studio health check.")
            if audio and len(audio) > 100:
                return True, "Timbre TTS (v2.5) connected and verified."
            return False, "Timbre TTS returned empty audio response."
        except Exception as exc:
            return False, f"Timbre TTS probe failed: {exc}"

    async def synthesize_speech(self, text: str, voice: str = "Kaveri", lang: str = "en-IN") -> Optional[bytes]:
        """
        Synthesize text into speech using Gnani Timbre TTS (v2.5 on Vachana platform).
        Returns raw WAV audio bytes or None on failure.
        """
        if not self.tts_configured:
            return None

        # Clean text of markdown characters for natural speech synthesis
        clean_text = text.replace("`", "").replace("*", "").replace("#", "").replace("|", "")
        # Limit length to avoid exceeding TTS buffer
        clean_text = clean_text[:400].strip()
        if not clean_text:
            return None

        headers = {
            "X-API-Key-ID": self.tts_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": clean_text,
            "voice": voice,
            "model": "timbre-v2.5",
            "language": lang,
            "sample_rate": 24000,
        }

        async with httpx.AsyncClient(timeout=25.0) as client:
            try:
                resp = await client.post(self.tts_url, headers=headers, json=payload)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").lower()
                if "json" in content_type:
                    data = resp.json()
                    audio = data.get("audio") or data.get("audio_base64")
                    if audio:
                        import base64
                        return base64.b64decode(audio, validate=True)
                    raise RuntimeError("TTS response JSON did not contain audio bytes.")
                if resp.content and (resp.content.startswith(b"RIFF") or "octet-stream" in content_type or "audio" in content_type):
                    return resp.content
                return resp.content if resp.content else None
            except Exception as ex:
                print(f"[Gnani TTS] Request failed ({type(ex).__name__}): {ex}")
                return None
