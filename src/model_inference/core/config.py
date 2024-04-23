import secrets
from pathlib import Path

from pydantic import validator
from pydantic_settings import BaseSettings


# Get the absolute path of the current script
current_script_path = Path(__file__).resolve()

# Define the project root by going up one level (adjust as needed)
project_root = current_script_path.parent.parent.parent

# Example: Define other paths relative to the project root
data_path = project_root / "data"
code_path = project_root / "src/dataset_cleaning"


class Settings:
    SECRET_KEY: str = secrets.token_urlsafe(32)
    PINECONE_API_KEY: str = "63185059-eafa-4759-bf16-121673c4ee94"
    # find ENV (cloud region) next to API key in console
    PINECONE_ENV: str = "us-west1-gcp-free"
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8

    TEMPERATURE: float = 0.0

    PROJECT_PATH = project_root
    DATA_PATH = data_path
    CODE_PATH = code_path

    class Config:
        case_sensitive = True


settings = Settings()
