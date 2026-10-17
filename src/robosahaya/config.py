import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

class Settings:
    GNANI_API_KEY: str = os.getenv("GNANI_API_KEY", "").strip()
    GNANI_BASE_URL: str = os.getenv("GNANI_BASE_URL", "https://api.gnani.ai/v1").strip().rstrip("/")
    ROS_DOMAIN_ID: int = int(os.getenv("ROS_DOMAIN_ID", "0"))
    ROBOSAHAYA_MODE: str = os.getenv("ROBOSAHAYA_MODE", "mock").strip().lower()
    ROBOSAHAYA_MOCK_SCENARIO: str = os.getenv("ROBOSAHAYA_MOCK_SCENARIO", "qos_reliability_mismatch").strip()
    HOST: str = os.getenv("HOST", "127.0.0.1").strip()
    PORT: int = int(os.getenv("PORT", "7860"))

    # Paths
    PROJECT_ROOT: Path = ROOT_DIR
    FIXTURES_DIR: Path = ROOT_DIR / "tests" / "fixtures" / "snapshots"

settings = Settings()
