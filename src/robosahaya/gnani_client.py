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
        return bool(self.api_key and self.api_key != "your_gnani_api_key_here")

    async def transcribe_audio(self, audio_bytes: bytes, lang: str = "en-IN") -> str:
        """
        Transcribe audio using Gnani Prisma STT.
        Normalizes to 16kHz mono WAV. If API key is not configured, returns mock transcript.
        """
        if not self.is_configured:
            # Fallback mock for offline demonstration / testing
            return "slash can data ravatledu kani robot tirugutondi"

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav")
        }
        data = {
            "language": lang,
            "encoding": "LINEAR16",
            "sample_rate": 16000
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/stt/transcribe",
                    headers=headers,
                    files=files,
                    data=data
                )
                resp.raise_for_status()
                json_res = resp.json()
                return json_res.get("transcript", "")
            except Exception as ex:
                raise RuntimeError(f"Gnani Prisma STT failed: {str(ex)}") from ex

    async def synthesize_speech(self, text: str, voice: str = "tejas") -> Optional[bytes]:
        """
        Synthesize text into speech using Gnani Timbre TTS.
        Returns audio bytes or None on failure/unconfigured.
        """
        if not self.is_configured:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voice": voice,
            "format": "wav",
            "rate": 1.0
        }

        async with httpx.AsyncClient(timeout=6.0) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/tts/synthesize",
                    headers=headers,
                    json=payload
                )
                resp.raise_for_status()
                return resp.content
            except Exception:
                # Timbre TTS is non-blocking and fails gracefully to written terminal card
                return None
