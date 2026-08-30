"""Project config file"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import computed_field

class Settings(BaseSettings):
    # --- 1. DIRECT FIELDS ----------------------------------------------------
    RANDOM_SEED:int = 488
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

    DATASET_ID: str = "splcher/animefacedataset"
    DATASET_DIR:str = "images"
    DATASET_FILE:str = "0_2000.jpg" # sample image path
    DATASET_FILE_CLEANED:str = ""

    #--------------------------------------------------------------------------
    #

    # --- 2. COMPUTED FIELDS --------------------------------------------------
    @computed_field
    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @property
    def OUT_DIR(self) -> Path:
        return self.PROJECT_ROOT / "out"

    @computed_field
    @property
    def IMG_DIR(self) -> Path:
        return self.PROJECT_ROOT / "report/latex/img"

    @computed_field
    @property
    def DATASET_PATH(self) -> Path:
        if self.DATASET_DIR:
            return  self.DATA_DIR/self.DATASET_DIR/self.DATASET_FILE
        else:
            return  self.DATA_DIR/self.DATASET_FILE

    # --- 3. CONFIGURATION ----------------------------------------------------
    # Configuration for the settings class
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- 4. METHODS ----------------------------------------------------------
    def create_directories(self):
        for path in [self.DATA_DIR, self.OUT_DIR, self.IMG_DIR]:
            path.mkdir(parents=True, exist_ok=True)

#------------------------------------------------------------------------------
# Create a single instance to be used everywhere
settings = Settings()
settings.create_directories()

