import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

class Settings:
    GNANI_API_KEY: str = os.getenv("GNANI_API_KEY", "").strip()
    GNANI_STT_API_KEY_ID: str = (os.getenv("GNANI_STT_API_KEY_ID") or os.getenv("GNANI_API_KEY", "")).strip()
    GNANI_TTS_API_KEY: str = (os.getenv("GNANI_TTS_API_KEY") or os.getenv("GNANI_API_KEY", "")).strip()
    GNANI_STT_URL: str = os.getenv("GNANI_STT_URL", "https://api.vachana.ai/stt/v3").strip().rstrip("/")
    GNANI_TTS_URL: str = os.getenv("GNANI_TTS_URL", "https://api.vachana.ai/api/v1/tts/inference").strip().rstrip("/")
    GNANI_TTS_AUTH_SCHEME: str = os.getenv("GNANI_TTS_AUTH_SCHEME", "X-API-Key-ID").strip()
    ROS_DOMAIN_ID: int = int(os.getenv("ROS_DOMAIN_ID", "0"))
    ROBOSAHAYA_MODE: str = os.getenv("ROBOSAHAYA_MODE", "live").strip().lower()
    ROBOSAHAYA_MOCK_SCENARIO: str = os.getenv("ROBOSAHAYA_MOCK_SCENARIO", "qos_reliability_mismatch").strip()
    HOST: str = os.getenv("HOST", "127.0.0.1").strip()
    PORT: int = int(os.getenv("PORT", "7860"))

    # Paths
    PROJECT_ROOT: Path = ROOT_DIR
    FIXTURES_DIR: Path = ROOT_DIR / "tests" / "fixtures" / "snapshots"
    LOCAL_MODEL_DIR: Path = Path(os.getenv("LOCAL_MODEL_DIR", str(ROOT_DIR / "models")))
    EVON_API_URL: str = os.getenv("EVON_API_URL", "").strip()
    EVON_API_KEY: str = os.getenv("EVON_API_KEY", "").strip()
    EVON_MODEL_ID: str = os.getenv("EVON_MODEL_ID", "evon").strip()

settings = Settings()

