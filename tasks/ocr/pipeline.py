"""
Główny pipeline przetwarzania OCR.
"""
import asyncio
import signal
import time
import uuid
import os
import gc
import torch
import tempfile
from datetime import datetime
from pathlib import Path
from sqlmodel import Session

from app.db import engine, FILES_DIR
from app.models import Document

# Importujemy funkcje z innych modułów OCR
from .config import logger, WATCHDOG_TIMEOUT_SECONDS, DPI
from .models import process_image_to_text, clean_resources
from .preprocessors import preprocess_image
from .postprocessors import clean_ocr_text, estimate_ocr_confidence
from .utils import clean_temp_files, get_available_gpu_memory

class WatchdogTimeoutError(Exception):
    """Wyjątek dla przekroczenia limitu czasu całego zadania."""
    pass

def watchdog_timeout_handler(signum, frame):
    """Handler dla przekroczenia limitu czasu całego zadania."""
    raise WatchdogTimeoutError("Watchdog timeout - przekroczenie maksymalnego czasu przetwarzania")

def aggressive_memory_cleanup():
    """
    Bardziej agresywne czyszczenie pamięci.
    """
    with open("/tmp/ocr_debug.log", "a") as f:
        f.write(f"MEMORY_CLEANUP: Rozpoczynam agresywne czyszczenie pamięci\n")
    
    # Standardowe czyszczenie CUDA
    if torch.cuda.is_available():
        # Wyświetl informacje o pamięci przed czyszczeniem
        with open("/tmp/ocr_debug.log", "a") as f:
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            f.write(f"MEMORY_CLEANUP: Przed czyszczeniem - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB\n")
        
        # Próba zwolnienia pamięci CUDA
        torch.cuda.empty_cache()
        
        # Dodatkowe czyszczenie
        collected = gc.collect()
        
        # Wyświetl informacje po czyszczeniu
        with open("/tmp/ocr_debug.log", "a") as f:
            allocated_after = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved_after = torch.cuda.memory_reserved() / (1024 * 1024)
            f.write(f"MEMORY_CLEANUP: Po czyszczeniu - Allocated: {allocated_after:.2f}MB, Reserved: {reserved_after:.2f}MB, GC objects: {collected}\n")
        
        # Próba wymuszenia czyszczenia niewykorzystanej pamięci
        try:
            torch.cuda.synchronize()
        except Exception as e:
            with open("/tmp/ocr_debug.log", "a") as f:
                f.write(f"MEMORY_CLEANUP: Błąd synchronizacji CUDA: {str(e)}\n")


async def process_document(doc_id, model=None, proc=None):
    """Asynchroniczna funkcja do przetwarzania dokumentu OCR."""
    with Session(engine) as session:
        # Pobierz dane dokumentu
        doc = session.get(Document, doc_id)
        if not doc:
            logger.error(f"Nie znaleziono dokumentu o ID={doc_id}")
            return
        
        # Oznacz dokument jako przetwarzany
        doc.ocr_status = "running"
        doc.ocr_progress = 0.0  # Inicjalizacja postępu
        doc.ocr_progress_info = "Inicjalizacja procesu OCR"
        session.add(doc)
        session.commit()
        
        # Pobierz ścieżkę do pliku
        file_path = FILES_DIR / doc.stored_filename
        if not file_path.exists():
            error_msg = f"Plik źródłowy nie istnieje: {file_path}"
            logger.error(error_msg)
            doc.ocr_status = "fail"
            doc.comments = error_msg
            session.add(doc)
            session.commit()
            return
            
        # Sprawdź czy to obraz czy PDF
        is_image = doc.content_type == 'image' or (doc.mime_type and doc.mime_type.startswith('image/'))
        
        try:
            if is_image:
                # Przetwarzanie obrazu
                logger.info(f"Przetwarzam obraz: {doc.original_filename}")
                
                # Aktualizuj postęp - jeden obraz to 100%
                doc.ocr_total_pages = 1
                doc.ocr_current_page = 1
                doc.ocr_progress = 0.3  # Początkowy postęp
                doc.ocr_progress_info = "Przygotowanie obrazu do OCR"
                session.add(doc)
                session.commit()
                
                # Przygotuj instrukcję
                if doc.sygnatura:
                    instruction += f" Document reference number: {doc.sygnatura}."
                
                # Aktualizuj postęp
                doc.ocr_progress = 0.5
                doc.ocr_progress_info = "Wykonywanie OCR"
                session.add(doc)
                session.commit()
                
                # Rozpoznaj tekst
                page_text = process_image_to_text(
                    str(file_path), 
                    model=model,
                    processor=proc
                )
                
                # Aktualizuj postęp
                doc.ocr_progress = 0.9
                doc.ocr_progress_info = "Przetwarzanie końcowe"
                session.add(doc)
                session.commit()
                
                # Oczyść tekst
                clean_text = clean_ocr_text(page_text)
                confidence_score = estimate_ocr_confidence(clean_text)
                
                text_all = clean_text
            else:
                # Przetwarzanie PDF
                logger.info(f"Przetwarzam PDF: {doc.original_filename}")
                
                # Importuj w tym miejscu, aby uniknąć długiego ładowania przy starcie
                from pdf2image import convert_from_path
                
                # Aktualizuj postęp
                doc.ocr_progress = 0.1
                doc.ocr_progress_info = "Analizowanie dokumentu PDF"
                session.add(doc)
                session.commit()
                
                # Konwertuj PDF na obrazy
                pages = convert_from_path(str(file_path), dpi=200)
                total_pages = len(pages)
                
                # Aktualizuj informacje o postępie
                doc.ocr_total_pages = total_pages
                doc.ocr_current_page = 0
                doc.ocr_progress = 0.2
                doc.ocr_progress_info = f"Wyodrębniono {total_pages} stron z PDF"
                session.add(doc)
                session.commit()
                
                logger.info(f"Wyodrębniono {total_pages} stron z PDF")
                
                # Przetwarzaj każdą stronę
                page_texts = []
                confidence_scores = []
                
                for page_number, img in enumerate(pages, 1):
                    # Aktualizuj postęp dla aktualnej strony
                    progress = 0.2 + (0.7 * page_number / total_pages)  # Postęp od 0.2 do 0.9
                    doc.ocr_progress = progress
                    doc.ocr_current_page = page_number
                    doc.ocr_progress_info = f"Przetwarzanie strony {page_number}/{total_pages}"
                    session.add(doc)
                    session.commit()
                    
                    # Zapisz obraz do pliku tymczasowego
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                        img_path = tmp_img.name
                    
                    try:
                        # Zapisz obraz strony
                        img.save(img_path, "PNG")
                        
                        # Przygotuj instrukcję
                        if doc.sygnatura:
                            instruction += f" Document reference number: {doc.sygnatura}. Page {page_number} of {total_pages}."
                        
                        logger.info(f"Wykonuję OCR dla strony {page_number}/{total_pages}")
                        
                        # Rozpoznaj tekst
                        page_text = process_image_to_text(
                            img_path, 
                            model=model,
                            processor=proc
                        )
                        
                        # Oczyść tekst
                        clean_text = clean_ocr_text(page_text)
                        confidence = estimate_ocr_confidence(clean_text)
                        
                        page_texts.append(clean_text)
                        confidence_scores.append(confidence)
                        
                        logger.info(f"Strona {page_number}: tekst rozpoznany, {len(clean_text)} znaków, pewność: {confidence:.2f}")
                    
                    except Exception as e:
                        logger.error(f"Błąd OCR dla strony {page_number}: {str(e)}")
                        page_texts.append(f"[Błąd OCR dla strony {page_number}]")
                        confidence_scores.append(0.0)
                    
                    finally:
                        # Usuń plik tymczasowy
                        import os
                        if os.path.exists(img_path):
                            os.remove(img_path)
                
                # Aktualizuj postęp końcowy
                doc.ocr_progress = 0.9
                doc.ocr_progress_info = "Finalizacja wyników OCR"
                session.add(doc)
                session.commit()
                
                # Łącz teksty stron
                text_all = ""
                for i, page_text in enumerate(page_texts, 1):
                    text_all += f"\n\n=== Strona {i} ===\n\n"
                    text_all += page_text
                
                # Oblicz średnią pewność
                confidence_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

                if not is_image and doc.mime_type == 'application/pdf':
                    logger.info(f"Osadzanie tekstu w PDF dla dokumentu {doc_id}")
                    doc.ocr_progress = 0.95
                    doc.ocr_progress_info = "Osadzanie tekstu w pliku PDF"
                    session.add(doc)
                    session.commit()
		    
                    embed_result = embed_text_in_pdf(file_path)
                    if embed_result:
                        logger.info(f"Pomyślnie osadzono tekst w PDF dla dokumentu {doc_id}")
                    else:
                        logger.warning(f"Nie udało się osadzić tekstu w PDF dla dokumentu {doc_id}")           

 
            # Zapisz tekst do pliku
            txt_stored = f"{uuid.uuid4().hex}.txt"
            txt_path = FILES_DIR / txt_stored
            txt_path.write_text(text_all, encoding="utf-8")
            
            # Utwórz dokument tekstowy OCR
            txt_doc = Document(
                sygnatura=doc.sygnatura,
                doc_type="OCR TXT",
                original_filename=f"{Path(doc.original_filename).stem}.txt",
                stored_filename=txt_stored,
                step=doc.step,
                ocr_status="done",
                ocr_parent_id=doc_id,
                ocr_confidence=confidence_score,
                mime_type="text/plain",
                content_type="document",
                upload_time=datetime.utcnow()
            )
            session.add(txt_doc)
            
            # Zaktualizuj status dokumentu źródłowego
            doc.ocr_status = "done"
            doc.ocr_confidence = confidence_score
            doc.ocr_progress = 1.0  # Zakończono - 100%
            doc.ocr_progress_info = "Zakończono"
            session.add(doc)
            session.commit()
            
            logger.info(f"OCR zakończony dla dokumentu {doc_id}, utworzono dokument TXT (ID: {txt_doc.id})")
            
            return txt_doc.id
            
        except Exception as e:
            logger.error(f"Błąd OCR: {str(e)}", exc_info=True)
            doc.ocr_status = "fail"
            doc.comments = f"Błąd: {str(e)}"
            doc.ocr_progress_info = f"Błąd: {str(e)}"
            session.add(doc)
            session.commit()
            raise


def embed_text_in_pdf(pdf_path):
    """
    Osadza rozpoznany tekst w pliku PDF.
    
    Args:
        pdf_path: Ścieżka do pliku PDF
        
    Returns:
        bool: True jeśli operacja się powiodła
    """
    try:
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"EMBED_TEXT: Rozpoczynam osadzanie tekstu w PDF: {pdf_path}\n")
            
        import subprocess
        import tempfile
        import shutil
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_out:
            tmp_path = tmp_out.name
        
        logger.info(f"Uruchamiam ocrmypdf dla {pdf_path}")
        result = subprocess.run(
            ["ocrmypdf", "--skip-text", "--sidecar", "/dev/null", str(pdf_path), tmp_path],
            check=True, capture_output=True, text=True
        )
        
        logger.info(f"ocrmypdf zakończony: {result.stdout}")
        if result.stderr:
            logger.warning(f"ocrmypdf ostrzeżenia: {result.stderr}")
            
        shutil.move(tmp_path, str(pdf_path))
        
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"EMBED_TEXT: Tekst osadzony pomyślnie w PDF\n")
            
        return True
    except Exception as e:
        logger.error(f"Błąd podczas osadzania tekstu w PDF: {str(e)}")
        
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"EMBED_TEXT: BŁĄD podczas osadzania tekstu w PDF: {str(e)}\n")
            
        return False

def save_ocr_results(doc_id, text_all, confidence_score):
    """
    Zapisuje wyniki OCR do bazy danych i na dysk.
    
    Args:
        doc_id: ID dokumentu w bazie danych
        text_all: Rozpoznany tekst
        confidence_score: Poziom pewności OCR
    
    Returns:
        int: ID nowego dokumentu tekstowego
    """
    with open("/tmp/ocr_debug.log", "a") as f:
        f.write(f"SAVE_RESULTS: Rozpoczynam zapisywanie wyników OCR dla ID={doc_id}\n")
    
    with Session(engine) as session:
        # Pobierz dane dokumentu źródłowego
        doc = session.get(Document, doc_id)
        if not doc:
            logger.error(f"Nie znaleziono dokumentu o ID={doc_id}")
            with open("/tmp/ocr_debug.log", "a") as f:
                f.write(f"SAVE_RESULTS: BŁĄD - Nie znaleziono dokumentu o ID={doc_id}\n")
            return None
        
        # Zapisz tekst do pliku
        txt_stored = f"{uuid.uuid4().hex}.txt"
        txt_path = FILES_DIR / txt_stored
        
        try:
            with open("/tmp/ocr_debug.log", "a") as f:
                f.write(f"SAVE_RESULTS: Zapisuję tekst do pliku {txt_stored} ({len(text_all)} znaków)\n")
                
            txt_path.write_text(text_all, encoding="utf-8")
            logger.info(f"Zapisano tekst do pliku {txt_stored} ({len(text_all)} znaków)")
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania tekstu do pliku: {str(e)}")
            with open("/tmp/ocr_debug.log", "a") as f:
                f.write(f"SAVE_RESULTS: BŁĄD podczas zapisywania tekstu do pliku: {str(e)}\n")
            return None
        
        # Utwórz nowy dokument tekstowy
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"SAVE_RESULTS: Tworzę wpis w bazie danych dla dokumentu tekstowego\n")
            
        txt_doc = Document(
            sygnatura=doc.sygnatura,
            doc_type="OCR TXT",
            original_filename=Path(doc.original_filename).stem + ".txt",
            stored_filename=txt_stored,
            step=doc.step,
            ocr_status="done",
            ocr_parent_id=doc_id,
            ocr_confidence=confidence_score,
            mime_type="text/plain",
            content_type="document",
            upload_time=datetime.utcnow()
        )
        session.add(txt_doc)
        
        # Zaktualizuj status dokumentu źródłowego
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"SAVE_RESULTS: Aktualizuję status dokumentu źródłowego\n")
            
        doc.ocr_status = "done"
        doc.ocr_confidence = confidence_score
        session.add(doc)
        session.commit()
        
        logger.info(f"Utworzono dokument TXT (ID: {txt_doc.id}), pewność OCR: {confidence_score:.2f}")
        
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"SAVE_RESULTS: Wyniki OCR zapisane pomyślnie, ID dokumentu TXT: {txt_doc.id}\n")
            
        return txt_doc.id

def run_ocr_pipeline(doc_id: int):
    """
    Główna funkcja przetwarzania OCR.
    
    Args:
        doc_id: ID dokumentu do przetworzenia
    """
    # Ustaw watchdog na cały proces
    signal.signal(signal.SIGALRM, watchdog_timeout_handler)
    signal.alarm(WATCHDOG_TIMEOUT_SECONDS)
    
    start_time = time.time()
    
    with open("/tmp/ocr_debug.log", "a") as f:
        f.write(f"PIPELINE: Rozpoczęcie dla ID={doc_id}\n")
    
    logger.info(f"Rozpoczynam OCR dla dokumentu ID={doc_id}")
    
    # Wyświetl informacje o dostępnej pamięci GPU
    gpu_info = get_available_gpu_memory()
    if gpu_info.get("available", False):
        logger.info(f"Dostępna pamięć GPU: {gpu_info['free_gb']:.2f} GB / {gpu_info['total_gb']:.2f} GB")
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"PIPELINE: Dostępna pamięć GPU: {gpu_info['free_gb']:.2f} GB / {gpu_info['total_gb']:.2f} GB\n")
    
    try:
        # 1) Pobranie metadanych + oznaczenie 'running'
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"PIPELINE: Pobieranie metadanych dokumentu\n")
        
        filename = None  # Zdefiniuj zmienne przed użyciem
        is_image = False
        sygnatura = None
        
        with Session(engine) as session:
            doc = session.get(Document, doc_id)
            if doc is None:
                logger.error(f"Nie znaleziono dokumentu o ID={doc_id}")
                with open("/tmp/ocr_debug.log", "a") as f:
                    f.write(f"PIPELINE: BŁĄD - Nie znaleziono dokumentu o ID={doc_id}\n")
                return
            
            # Oznacz dokument jako przetwarzany
            doc.ocr_status = "running"
            session.add(doc)
            session.commit()
            
            # Pobierz potrzebne dane
            filename = doc.stored_filename
            sygnatura = doc.sygnatura
            step = doc.step
            
            # Sprawdź typ dokumentu
            content_type = getattr(doc, 'content_type', '')
            mime_type = getattr(doc, 'mime_type', '')
            
            if content_type == 'image' or (mime_type and mime_type.startswith('image/')):
                is_image = True
                
            with open("/tmp/ocr_debug.log", "a") as f:
                f.write(f"PIPELINE: Metadane pobrane, filename={filename}, is_image={is_image}\n")
        
        logger.info(f"Dokument: {doc.original_filename}, Sygnatura: {sygnatura or 'brak'}, Status: {step}")
        
        # 2) Przetwarzanie dokumentu
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"PIPELINE: Rozpoczynam przetwarzanie dokumentu\n")
        
        file_path = FILES_DIR / filename
        
        # Sprawdź czy plik istnieje
        if not file_path.exists():
            error_msg = f"Plik źródłowy nie istnieje: {file_path}"
            logger.error(error_msg)
            with open("/tmp/ocr_debug.log", "a") as f:
                f.write(f"PIPELINE: BŁĄD - {error_msg}\n")
            raise FileNotFoundError(error_msg)
        
        # Wykonaj OCR
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"PIPELINE: Wywołuję process_document\n")
            
        text_all, confidence_score = process_document(doc_id, file_path, is_image, sygnatura)
        
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"PIPELINE: process_document zakończony, długość tekstu: {len(text_all)}, pewność: {confidence_score:.2f}\n")
        
        # 3) Jeśli to PDF, osadź tekst w pliku (opcjonalne)
        if not is_image:
            with open("/tmp/ocr_debug.log", "a") as f:
                f.write(f"PIPELINE: Osadzam tekst w PDF\n")
                
            embed_text_in_pdf(file_path)
        
        # 4) Zapisanie wyników
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"PIPELINE: Zapisuję wyniki OCR\n")
            
        save_ocr_results(doc_id, text_all, confidence_score)
        
        # 5) Podsumowanie
        total_time = time.time() - start_time
        logger.info(f"OCR zakończony pomyślnie dla dokumentu ID={doc_id}, czas: {total_time:.2f}s")
        
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"PIPELINE: OCR zakończony pomyślnie, czas: {total_time:.2f}s\n")
        
    except WatchdogTimeoutError:
        # Obsługa przekroczenia czasu watchdoga
        error_msg = f"Watchdog timeout dla dokumentu ID={doc_id} - przekroczenie maksymalnego czasu"
        logger.error(error_msg)
        
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"PIPELINE: BŁĄD - {error_msg}\n")
            
        with Session(engine) as session:
            doc = session.get(Document, doc_id)
            if doc:
                doc.ocr_status = "fail"
                doc.comments = "Timeout - przekroczenie maksymalnego czasu przetwarzania"
                session.add(doc)
                session.commit()
    
    except Exception as e:
        # Obsługa innych błędów
        logger.error(f"OCR nie powiódł się: {str(e)}", exc_info=True)
        
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"PIPELINE: BŁĄD - {str(e)}\n")
            
        with Session(engine) as session:
            doc = session.get(Document, doc_id)
            if doc:
                doc.ocr_status = "fail"
                doc.comments = f"Błąd: {str(e)}"
                session.add(doc)
                session.commit()
    
    finally:
        # Zawsze wyłącz alarm i zwolnij zasoby
        signal.alarm(0)
        clean_resources()
        aggressive_memory_cleanup()
        
        total_time = time.time() - start_time
        logger.info(f"Zakończono przetwarzanie dokumentu ID={doc_id}, całkowity czas: {total_time:.2f}s")
        
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"PIPELINE: Finalizacja dla ID={doc_id}, całkowity czas: {total_time:.2f}s\n")
