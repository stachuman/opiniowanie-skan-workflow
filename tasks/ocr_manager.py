import asyncio, traceback, time, gc
from pathlib import Path
from sqlmodel import Session
from app.db import engine
from app.models import Document
from tasks.ocr.models_single import get_model
from tasks.ocr.pipeline import process_document

queue: asyncio.Queue[int] = asyncio.Queue()   # doc_id jako int

async def ocr_manager():
    pass

async def enqueue(doc_id: int):
    """
    Dodaje dokument do kolejki OCR.
    Ta funkcja jest zmodyfikowana, aby używać nowego systemu zadań w tle.
    """
    from app.background_tasks import enqueue_ocr_task
    await enqueue_ocr_task(doc_id)

