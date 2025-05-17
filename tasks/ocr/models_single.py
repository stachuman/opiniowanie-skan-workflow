import os, torch
from functools import lru_cache
from transformers import AutoModelForVision2Seq, AutoProcessor
from .config import OCR_MODEL_PATH, logger
import pynvml
from .config import (
    GPU_MEM_LIMIT_GB
)


# wyłącz FA2 (stabilność)
os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"
pynvml.nvmlInit()

def get_gpu_performance(handle):
    cuda_cores = 10496  # RTX 3090 has 10496 CUDA cores; adjust if your model varies significantly
    clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
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


@lru_cache(maxsize=1)
def get_model():
    gpu = _pick_best_gpu(GPU_MEM_LIMIT_GB)
    device = "cuda:" + str(gpu)

    
    logger.info(f"OLD get_model!")
        
    model = AutoModelForVision2Seq.from_pretrained(
        OCR_MODEL_PATH,
        torch_dtype=torch.float16
    ).to(device).eval()
    proc = AutoProcessor.from_pretrained(OCR_MODEL_PATH)
    logger.info(f"OCR model loaded once on {device}")
    return model, proc

