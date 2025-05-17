"""
Moduł obsługi zadań w tle dla aplikacji.
Zarządza kolejkami zadań i ich asynchronicznym wykonywaniem.
"""
import asyncio
import logging
from typing import Dict, List, Set

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("background_tasks")

# Globalne kolejki zadań
task_queues: Dict[str, asyncio.Queue] = {
    "ocr": asyncio.Queue(),
    "notifications": asyncio.Queue(),
}

# Aktualnie przetwarzane zadania (dla deduplicji)
active_tasks: Dict[str, Set[int]] = {
    "ocr": set(),
}

# Funkcja do wstawiania zadania OCR
async def enqueue_ocr_task(doc_id: int):
    """Dodaje zadanie OCR do kolejki."""
    # Sprawdź czy dokument nie jest już przetwarzany
    if doc_id in active_tasks["ocr"]:
        logger.info(f"Dokument {doc_id} jest już w kolejce OCR - pomijam")
        return
    
    # Dodaj do aktywnych zadań
    active_tasks["ocr"].add(doc_id)
    
    # Dodaj do kolejki
    await task_queues["ocr"].put(doc_id)
    logger.info(f"Dodano dokument {doc_id} do kolejki OCR")
    
    # Natychmiast oddaj kontrolę do pętli zdarzeń
    await asyncio.sleep(0)


# Funkcja do usuwania zadania z aktywnych
def remove_active_task(queue_name: str, task_id: int):
    """Usuwa zadanie z listy aktywnych po zakończeniu."""
    if task_id in active_tasks.get(queue_name, set()):
        active_tasks[queue_name].remove(task_id)
        logger.info(f"Usunięto zadanie {task_id} z aktywnych zadań {queue_name}")

# Asynchroniczny worker OCR
async def ocr_worker():
    """Worker przetwarzający zadania OCR z kolejki."""
    logger.info("Uruchomiono worker OCR")
    
    while True:
        try:
            # Pobierz dokument z kolejki (z krótkim timeoutem)
            try:
                doc_id = await asyncio.wait_for(task_queues["ocr"].get(), timeout=0.1)
            except asyncio.TimeoutError:
                # Brak zadań w kolejce - oddaj kontrolę do pętli zdarzeń
                await asyncio.sleep(0.1)
                continue
                
            logger.info(f"Przetwarzanie dokumentu {doc_id} z kolejki OCR")
            
            # Uruchom przetwarzanie OCR w oddzielnym zadaniu i NIE CZEKAJ na jego zakończenie
            asyncio.create_task(_process_ocr_document(doc_id))
            
            # Oznacz zadanie jako pobrane z kolejki
            task_queues["ocr"].task_done()
            
            # Oddaj kontrolę do pętli zdarzeń
            await asyncio.sleep(0)
            
        except Exception as e:
            logger.error(f"Błąd w workerze OCR: {str(e)}")
            # Poczekaj przed ponowną próbą i oddaj kontrolę
            await asyncio.sleep(1)

# Pomocnicza funkcja do przetwarzania dokumentu OCR
async def _process_ocr_document(doc_id: int):
    """Przetwarza pojedynczy dokument OCR w tle."""
    from tasks.ocr_manager import process_document
    
    try:
        # Wywołaj funkcję OCR
        await process_document(doc_id)
        logger.info(f"Zakończono OCR dla dokumentu {doc_id}")
    except Exception as e:
        logger.error(f"Błąd podczas przetwarzania OCR dla dokumentu {doc_id}: {str(e)}")
        # Aktualizuj status na błąd
        with Session(engine) as session:
            doc = session.get(Document, doc_id)
            if doc:
                doc.ocr_status = "fail"
                doc.ocr_progress_info = f"Błąd: {str(e)}"
                session.add(doc)
                session.commit()
    finally:
        # Usuń z aktywnych zadań
        remove_active_task("ocr", doc_id)

# Funkcja startująca workery
async def start_background_workers():
    """Uruchamia wszystkie workery zadań w tle."""
    # Uruchom worker OCR
    asyncio.create_task(ocr_worker())
    logger.info("Uruchomiono workery zadań w tle")
