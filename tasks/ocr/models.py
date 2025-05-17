from __future__ import annotations

import gc
import os
import signal
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple
import pynvml

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from .config import (
    DEFAULT_OCR_INSTRUCTION,
    DEVICE_STRATEGY as CFG_STRATEGY,
    GPU_MEM_LIMIT_GB,
    GPU_SELECT_MODE,
    OCR_MODEL_PATH,
    OCR_TIMEOUT_SECONDS,
    MAX_NEW_TOKENS,
    logger,
)

# Wyłączamy globalnie Flash‑Attention 2 – znany source segfaultów na Ampere
os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"


class TimeoutError(Exception):
    """Sygnalizuje przekroczenie limitu czasu generacji jednej strony."""


# ---------------------------------------------------------------------------
#  Wybór najlepszej karty GPU (tryb single)
# ---------------------------------------------------------------------------


def get_gpu_performance(handle):
    pynvml.nvmlInit()
    cuda_cores = 10496  # RTX 3090 has 10496 CUDA cores; adjust if your model varies significantly
    clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    logger.info(f"MHz {clock_mhz} ")
    return cuda_cores * clock_mhz

def _pick_best_gpu(threshold_gb: int) -> int | None:
    best_gpu = None
    best_free = 0.0
    best_perf = 0

    for i in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        free, _ = torch.cuda.mem_get_info(i)
        free_gb = free / (1024 ** 3)

        if free_gb >= threshold_gb:
            performance = get_gpu_performance(handle)

            if (free_gb > best_free) or (free_gb == best_free and performance > best_perf):
                best_gpu = i
                best_free = free_gb
                best_perf = performance

    pynvml.nvmlShutdown()
    return best_gpu


# ---------------------------------------------------------------------------
#  Model + processor – singleton w pamięci procesu
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_once() -> Tuple[AutoModelForVision2Seq, AutoProcessor]:
    """Ładuje model i processor dokładnie raz."""

    strategy = CFG_STRATEGY  # lokalna kopia

    if strategy == "single" and GPU_SELECT_MODE == "auto":
        gpu = _pick_best_gpu(GPU_MEM_LIMIT_GB)
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            free = torch.cuda.mem_get_info(0)[0] / 1e9
            logger.info(f"Wybrano GPU {gpu} (wolne ≈ {free:.1f} GB)")
        else:
            logger.warning("Brak karty z ≥ % s GB – przełączam na device_map='auto'", GPU_MEM_LIMIT_GB)
            strategy = "auto"

    params: Dict[str, Any] = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if strategy == "single":
        params.update({"device_map": None, "max_memory": {0: f"{GPU_MEM_LIMIT_GB}GiB"}})
    else:
        params["device_map"] = "auto"

    try:
        model = AutoModelForVision2Seq.from_pretrained(OCR_MODEL_PATH, **params).eval()
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM – ponawiam z device_map='auto'")
        params.pop("max_memory", None)
        params["device_map"] = "auto"
        model = AutoModelForVision2Seq.from_pretrained(OCR_MODEL_PATH, **params).eval()

    processor = AutoProcessor.from_pretrained(OCR_MODEL_PATH)

    if strategy == "single":
        model.to(torch.device("cuda"))
        if torch.cuda.get_device_capability()[0] == 8:
            model.config.attn_implementation = "eager"
        logger.info("Model na GPU %s – gotowy", torch.cuda.current_device())

    return model, processor


def get_ocr_model() -> Tuple[AutoModelForVision2Seq, AutoProcessor]:
    return _load_once()


# ---------------------------------------------------------------------------
#  OCR jednej strony
# ---------------------------------------------------------------------------

def _timeout_handler(_signum, _frame):
    raise TimeoutError("Timeout podczas generacji tekstu")


def process_image_to_text(
    image_path: str | Path,
    instruction: str = DEFAULT_OCR_INSTRUCTION,
    model=None,
    processor=None,
):
    """Rozpoznaje tekst z obrazu i zwraca go jako string."""

    from qwen_vl_utils import process_vision_info
    
    # Jeśli nie podano modelu lub procesora, załaduj je
    if model is None or processor is None:
        model, processor = get_ocr_model()

    if isinstance(image_path, Path):
        image_path = str(image_path)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are OCR system for text recognition."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    logger.debug("model=%s pixels=%s", model.device, inputs["pixel_values"].device)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(OCR_TIMEOUT_SECONDS)
    try:
        logger.info("Instrukcja: %s", instruction)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS
                #eos_token_id=processor.tokenizer.eos_token_id,
                #pad_token_id=processor.tokenizer.pad_token_id,
                )
    except TimeoutError:
        logger.error("Timeout > %s s – pominięto stronę", OCR_TIMEOUT_SECONDS)
        return ""
    finally:
        signal.alarm(0)

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    # cleanup RAM
    del inputs, gen_ids, image_inputs, video_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text.strip()
# ---------------------------------------------------------------------------
#  Zwalnianie zasobów (legacy helper)
# ---------------------------------------------------------------------------

def clean_resources(*resources: Any) -> None:
    for r in resources:
        if r is not None:
            del r
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
