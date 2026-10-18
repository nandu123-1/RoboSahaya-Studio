import httpx
from typing import Optional
from robosahaya.config import settings

class GnaniClient:
    """Client for Gnani AI Speech Services (Prisma STT and Timbre TTS)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key if api_key is not None else settings.GNANI_API_KEY
        self.base_url = (base_url or settings.GNANI_BASE_URL).rstrip("/")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 10 and self.api_key != "your_gnani_api_key_here")

    async def transcribe_audio(self, audio_bytes: bytes, lang: str = "en-IN") -> str:
        """
        Transcribe audio using Gnani Prisma STT (v3 on Vachana platform).
        """
        if not self.is_configured:
            return ""

        headers = {
            "X-API-Key-ID": self.api_key
        }
        files = {
            "audio_file": ("audio.wav", audio_bytes, "audio/wav")
        }
        data = {
            "language_code": lang,
            "preferred_language": lang,
            "format": "transcribe"
        }

        async with httpx.AsyncClient(timeout=25.0) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/stt/v3",
                    headers=headers,
                    files=files,
                    data=data
                )
                resp.raise_for_status()
                json_res = resp.json()
                transcript = json_res.get("transcript") or json_res.get("output", {}).get("literal", "")
                return transcript.strip()
            except Exception as ex:
                print(f"[Gnani Prisma STT Error] {ex}")
                raise RuntimeError(f"Gnani Prisma STT failed: {str(ex)}") from ex

    async def synthesize_speech(self, text: str, voice: str = "Kaveri", lang: str = "en-IN") -> Optional[bytes]:
        """
        Synthesize text into speech using Gnani Timbre TTS (v2.5 on Vachana platform).
        Returns raw WAV audio bytes or None on failure.
        """
        if not self.is_configured:
            return None

        # Clean text of markdown characters for natural speech synthesis
        clean_text = text.replace("`", "").replace("*", "").replace("#", "").replace("|", "")
        # Limit length to avoid exceeding TTS buffer
        clean_text = clean_text[:400].strip()
        if not clean_text:
            return None

        headers = {
            "X-API-Key-ID": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": clean_text,
            "voice": voice,
            "model": "timbre-v2.5",
            "language": lang,
            "speed": 1,
            "audio_config": {
                "encoding": "linear_pcm",
                "container": "wav",
                "num_channels": 1,
                "sample_rate": 24000,
                "sample_width": 2
            }
        }

        async with httpx.AsyncClient(timeout=25.0) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/api/v1/tts/inference",
                    headers=headers,
                    json=payload
                )
                resp.raise_for_status()
                return resp.content
            except Exception as ex:
                print(f"[Gnani Timbre TTS Error] {ex}")
                return None
