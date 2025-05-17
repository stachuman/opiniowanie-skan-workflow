"""
Konfiguracja modułu OCR.
"""
import logging
import os
from pathlib import Path

# Stałe dla modelu OCR
DEFAULT_OCR_INSTRUCTION = "Read all text in the image. Extract all visible text including headers, footers, paragraphs, lists, and tables. Preserve original formatting as much as possible. Output to be a plain text. Ignore watermarks. Text is in Polish."
OCR_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS = 6096

# Konfiguracja logowania
LOG_DIR = os.getenv("OCR_LOG_DIR", "/var/log")
LOG_FILE = os.getenv("OCR_LOG_FILE", "ocr_runner.log")

# Upewnij się, że katalog logów istnieje
try:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    LOG_PATH = Path(LOG_DIR) / LOG_FILE
except PermissionError:
    # Jeśli nie mamy uprawnień, używamy katalogu tymczasowego
    LOG_DIR = "/tmp"
    LOG_PATH = Path(LOG_DIR) / LOG_FILE

# Konfiguracja loggera
def setup_logger():
    """Konfiguruje i zwraca logger dla modułu OCR."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("ocr")
    logger.info(f"========== INICJALIZACJA MODUŁU OCR ==========")
    logger.info(f"Logi zapisywane do: {LOG_PATH}")
    return logger

logger = setup_logger()

# Ustawienia dla timeout'ów
OCR_TIMEOUT_SECONDS = 600  # 10 minut na stronę
WATCHDOG_TIMEOUT_SECONDS = 1800  # 30 minut na cały dokument

# Ustawienia dla preprocessingu
DPI = 300  # Rozdzielczość przy konwersji PDF -> obraz
# 'single'  → cały model na widoczną kartę (CUDA_VISIBLE_DEVICES)
# 'auto'    → HuggingFace rozdziela warstwy na wszystkie karty
DEVICE_STRATEGY = os.getenv("OCR_DEVICE_STRATEGY", "single").lower()

# Ile pamięci zostawiamy na GPU (GiB) – aby uniknąć OOM przy single
GPU_MEM_LIMIT_GB = int(os.getenv("OCR_GPU_MEM_LIMIT_GB", "22"))

GPU_SELECT_MODE  = os.getenv("OCR_GPU_SELECT", "auto").lower()
