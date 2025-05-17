from __future__ import annotations

import asyncio
import mimetypes
import shutil
import time
import PyPDF2
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Field, Session, SQLModel, select
from datetime import datetime
import uuid, shutil, os, pathlib, mimetypes
from PIL import Image
from pathlib import Path
import io

from app.db import engine, FILES_DIR, BASE_DIR, init_db
from app.models import Document
from tasks.ocr_manager import enqueue, ocr_manager

from tasks.ocr.config import (
    logger,
)

# Próba importu biblioteki python-magic dla dokładnego wykrywania MIME type
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

# Dodanie obsługiwanych typów plików
ALLOWED_EXTENSIONS = {
    # PDFy
    '.pdf': 'application/pdf',
    # Obrazy
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.tif': 'image/tiff',
    '.tiff': 'image/tiff',
    '.bmp': 'image/bmp',
    '.webp': 'image/webp',
    # Dokumenty tekstowe
    '.txt': 'text/plain',
    # Dokumenty (opcjonalnie)
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
}

# Redis configuration
#redis_conn = Redis(host="localhost", port=6379, db=0)
#ocr_q = Queue("ocr", connection=redis_conn)


app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/files", StaticFiles(directory=str(FILES_DIR)), name="files")
app.mount("/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

@app.on_event("startup")
async def _startup():
    # Ensure database tables are created
    init_db()
    # asyncio.create_task(ocr_manager())
    # Uruchomienie nowego systemu workerów zadań w tle
    from app.background_tasks import start_background_workers
    asyncio.create_task(start_background_workers())


STEP_ICON = {
    "k1": "bi bi-exclamation-triangle-fill text-danger",
    "k2": "bi bi-exclamation-circle-fill text-warning",
    "k3": "bi bi-check-circle-fill text-success",
    "k4": "bi bi-archive-fill text-secondary",
}

def detect_mime_type(file_path):
    """
    Wykrywa faktyczny MIME type pliku na podstawie jego zawartości.
    Używa biblioteki python-magic jeśli jest dostępna, w przeciwnym razie
    bazuje na rozszerzeniu pliku.
    
    Args:
        file_path: Ścieżka do pliku
    
    Returns:
        str: MIME type pliku
    """
    # Jeśli mamy bibliotekę python-magic, używamy jej
    if HAS_MAGIC:
        try:
            mime = magic.Magic(mime=True)
            detected_mime = mime.from_file(str(file_path))
            return detected_mime
        except Exception as e:
            print(f"Błąd wykrywania MIME type: {e}")
            # W przypadku błędu, używamy fallbacku
            pass
    
    # Fallback - używamy rozszerzenia pliku
    suffix = file_path.suffix.lower()
    if suffix in ALLOWED_EXTENSIONS:
        return ALLOWED_EXTENSIONS[suffix]
    
    # Ostatecznie próbujemy użyć mimetypes
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type:
        return mime_type
    
    # Domyślnie zwróć application/octet-stream
    return 'application/octet-stream'

# Dodaj ten nowy endpoint do main.py
@app.get("/document/{doc_id}/pdf-viewer")
def document_pdf_viewer(request: Request, doc_id: int):
    """Zaawansowany podgląd PDF z funkcją zaznaczania i OCR."""
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie znaleziono dokumentu")
        
        # Sprawdź czy dokument to PDF
        if not doc.mime_type or doc.mime_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="Ten widok jest dostępny tylko dla plików PDF")
        
        # Sprawdź czy dokument ma wyniki OCR
        ocr_txt_query = select(Document).where(
            Document.ocr_parent_id == doc_id,
            Document.doc_type == "OCR TXT"
        )
        ocr_txt = session.exec(ocr_txt_query).first()
        
    
    return templates.TemplateResponse(
        "pdf_view_with_selection.html", 
        {
            "request": request, 
            "doc": doc,
            "title": f"Podgląd PDF z zaznaczaniem - {doc.original_filename}"
        }
    )



@app.get("/create_empty_opinion")
def create_empty_opinion_form(request: Request):
    """Formularz tworzenia nowej pustej opinii."""
    return templates.TemplateResponse(
        "create_empty_opinion.html", 
        {"request": request, "title": "Utwórz nową pustą opinię"}
    )

@app.post("/create_empty_opinion")
def create_empty_opinion(
    request: Request,
    sygnatura: str | None = Form(None),
    doc_type: str = Form(...),
    step: str = Form("k1")
):
    """Utworzenie nowej pustej opinii bez dokumentu."""
    # Generowanie unikalnej nazwy dla "pustego" dokumentu
    unique_name = f"{uuid.uuid4().hex}.empty"
    
    with Session(engine) as session:
        # Utworzenie nowej opinii w bazie danych
        opinion = Document(
            original_filename="Nowa opinia",
            stored_filename=unique_name,
            step=step,
            ocr_status="none",
            is_main=True,
            content_type="opinion",
            doc_type=doc_type,
            sygnatura=sygnatura,
            creator=None  # Tu można dodać current_user gdy będzie system użytkowników
        )
        session.add(opinion)
        session.commit()
        opinion_id = opinion.id
    
    # Przekieruj do widoku szczegółów opinii
    return RedirectResponse(request.url_for("opinion_detail", doc_id=opinion_id), status_code=303)

@app.get("/quick_ocr")
def quick_ocr_form(request: Request):
    """Formularz do szybkiego OCR dokumentów bez przypisywania do opinii."""
    allowed_types = ", ".join([k for k in ALLOWED_EXTENSIONS.keys() 
                              if k not in ['.doc', '.docx']])  # Bez plików Word
    return templates.TemplateResponse(
        "quick_ocr.html", 
        {"request": request, "title": "Szybki OCR", "allowed_types": allowed_types}
    )

@app.post("/quick_ocr")
async def quick_ocr(request: Request, files: list[UploadFile] = File(...)):
    """Szybki OCR - dodawanie dokumentów bez wiązania z opinią."""
    uploaded_docs = []
    
    # Utwórz lub pobierz specjalną "opinię" dla dokumentów niezwiązanych
    with Session(engine) as session:
        # Sprawdź czy istnieje specjalna opinia dla dokumentów niezwiązanych
        special_opinion_query = select(Document).where(
            Document.is_main == True,
            Document.doc_type == "Dokumenty niezwiązane z opiniami"
        )
        special_opinion = session.exec(special_opinion_query).first()
        
        # Jeśli nie istnieje, utwórz ją
        if not special_opinion:
            special_opinion = Document(
                original_filename="Dokumenty niezwiązane z opiniami",
                stored_filename=f"{uuid.uuid4().hex}.empty",
                step="k1",
                ocr_status="none",
                is_main=True,
                content_type="container",  # Specjalny typ dla kontenera dokumentów
                doc_type="Dokumenty niezwiązane z opiniami",
                sygnatura="UNASSIGNED",
                creator=None
            )
            session.add(special_opinion)
            session.commit()
            special_opinion_id = special_opinion.id
        else:
            special_opinion_id = special_opinion.id
    
    # Przetwarzanie wgranych plików
    for file in files:
        # Sprawdzenie rozszerzenia pliku
        suffix = check_file_extension(file.filename)
        
        # Ignorujemy pliki Word w szybkim OCR
        if suffix.lower() in ['.doc', '.docx']:
            continue
        
        # Generowanie unikalnej nazwy pliku
        unique_name = f"{uuid.uuid4().hex}{suffix}"
        dest = FILES_DIR / unique_name
        
        # Zapisanie pliku
        content = await file.read()
        with open(dest, "wb") as buffer:
            buffer.write(content)
        
        # Wykrywanie właściwego MIME typu pliku
        actual_mime_type = detect_mime_type(dest)
        
        # Określanie content_type na podstawie MIME type
        content_type = "document"
        if actual_mime_type.startswith('image/'):
            content_type = "image"
        
        # Ustal właściwy status OCR
        ocr_status = "pending"
        
        # Zapisanie do bazy danych
        with Session(engine) as session:
            new_doc = Document(
                doc_type="Dokument OCR",
                original_filename=file.filename,
                stored_filename=unique_name,
                step="k1",
                ocr_status=ocr_status,
                parent_id=special_opinion_id,  # Przypisz do specjalnej "opinii"
                is_main=False,
                content_type=content_type,
                mime_type=actual_mime_type,
                creator=None,
                upload_time=datetime.utcnow()
            )
            session.add(new_doc)
            session.commit()
            uploaded_docs.append(new_doc.id)
    
    # Uruchom OCR dla wgranych dokumentów
    asyncio.create_task(_enqueue_ocr_documents_nonblocking(uploaded_docs))
    
    # Przekieruj do listy dokumentów
    return RedirectResponse(request.url_for("list_documents"), status_code=303)

# Zmodyfikowana funkcja document_ocr_selection w main.py
@app.post("/api/document/{doc_id}/ocr-selection")
async def document_ocr_selection(request: Request, doc_id: int):
    """Zwraca OCR dla zaznaczonego fragmentu dokumentu."""
    try:
        # Pobierz dane z POST
        data = await request.json()
        page = data.get('page', 1)     # Numer strony (1-based)
        x1 = data.get('x1', 0)         # Współrzędne zaznaczenia (0-1)
        y1 = data.get('y1', 0)
        x2 = data.get('x2', 1)
        y2 = data.get('y2', 1)
        skip_pdf_embed = data.get('skip_pdf_embed', False)  # Czy pomijać osadzanie tekstu w PDF
        
        # Zapisz informacje diagnostyczne
        logger.info(f"Współrzędne zaznaczenia: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        logger.info(f"Parametry: strona={page}, skip_pdf_embed={skip_pdf_embed}")
        
        # Sprawdź czy dokument istnieje
        with Session(engine) as session:
            doc = session.get(Document, doc_id)
            if not doc:
                return {"error": "Nie znaleziono dokumentu"}
            
            # Sprawdź, czy to jest PDF
            if not doc.mime_type or doc.mime_type != 'application/pdf':
                return {"error": "Ta funkcja obsługuje tylko pliki PDF"}
            
            # Ścieżka do pliku PDF
            pdf_path = FILES_DIR / doc.stored_filename
            if not pdf_path.exists():
                return {"error": "Nie znaleziono pliku PDF"}
            
            # Pobierz najpierw liczbę stron z PDF - niezależnie od istnienia OCR
            import PyPDF2
            try:
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    total_pages = len(pdf_reader.pages)
                    logger.info(f"Dokument ma {total_pages} stron")
                    
                    # Sprawdź, czy żądana strona istnieje
                    if page <= 0 or page > total_pages:
                        return {"error": f"Strona {page} nie istnieje. Dokument ma {total_pages} stron."}
            except Exception as e:
                logger.error(f"Błąd odczytu dokumentu PDF: {str(e)}", exc_info=True)
                return {"error": f"Nie można odczytać dokumentu PDF: {str(e)}"}
            
            # Pobierz wynik OCR, jeśli istnieje
            ocr_txt_query = select(Document).where(
                Document.ocr_parent_id == doc_id,
                Document.doc_type == "OCR TXT"
            )
            ocr_txt = session.exec(ocr_txt_query).first()
            
            # Zmienne do przechowywania tekstu strony i wszystkich stron
            page_text = None
            pages = []
            
            # Jeśli istnieje OCR, próbujemy odczytać tekst
            if ocr_txt:
                ocr_file_path = FILES_DIR / ocr_txt.stored_filename
                try:
                    encodings = ['utf-8', 'latin-1', 'cp1250']
                    full_text = None
                    
                    for encoding in encodings:
                        try:
                            full_text = ocr_file_path.read_text(encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if full_text:
                        # Podziel tekst na strony (jeśli zawiera znaczniki stron)
                        pages = full_text.split("=== Strona")
                        
                        # Jeśli nie ma znaczników stron, traktuj cały tekst jako jedną stronę
                        if len(pages) <= 1:
                            pages = [full_text]
                        else:
                            # Pierwszy element to tekst przed pierwszym znacznikiem "=== Strona"
                            # Usuwamy go, jeśli jest pusty, lub dodajemy jako stronę 0
                            if pages[0].strip():
                                pages = pages
                            else:
                                pages = pages[1:]  # Usuwamy pierwszy element (pusty)
                                # Dodajemy prefix "=== Strona" z powrotem do każdego elementu
                                pages = ["=== Strona" + page for page in pages]
                        
                        # Pobierz tekst dla wybranej strony (indeks page-1, bo numeracja stron zaczyna się od 1)
                        if 0 <= page-1 < len(pages):
                            page_text = pages[page-1]
                        else:
                            # Jeśli nie mamy tekstu dla tej strony, tworzymy pusty
                            page_text = f"=== Strona {page} ===\n\n"
                except Exception as e:
                    logger.warning(f"Nie udało się odczytać tekstu OCR: {str(e)}")
                    # Kontynuujemy bez tekstu OCR
                    page_text = None
                    pages = []
            
            # Sprawdź, czy to jest zaznaczenie całej strony i czy mamy już tekst OCR
            is_full_page = (abs(x1) < 0.01 and abs(y1) < 0.01 and abs(x2 - 1.0) < 0.01 and abs(y2 - 1.0) < 0.01)
            
            if is_full_page and page_text:
                # Jeśli to zaznaczenie całej strony i mamy już OCR, po prostu zwróć istniejący tekst
                return {
                    "success": True,
                    "text": page_text.strip(),
                    "page": page,
                    "total_pages": total_pages,
                    "is_full_page": True
                }
            
            # Przygotuj do OCR zaznaczonego fragmentu
            logger.info(f"Przygotowanie do OCR fragmentu strony {page}")
            
            # Konwertuj stronę PDF na obraz
            import tempfile
            from pdf2image import convert_from_path
            
            try:
                # Konwertuj tylko wybraną stronę (numeracja w pdf2image zaczyna się od 0)
                images = convert_from_path(str(pdf_path), first_page=page, last_page=page, dpi=300)
                
                if not images:
                    return {"error": "Nie można skonwertować strony PDF na obraz"}
                
                # Weź pierwszy (i jedyny) obraz strony
                image = images[0]
                
                # Oblicz współrzędne zaznaczenia w pikselach
                width, height = image.size
                crop_x1 = int(x1 * width)
                crop_y1 = int(y1 * height)
                crop_x2 = int(x2 * width)
                crop_y2 = int(y2 * height)
                
                # Dodaj margines do zaznaczenia, aby uchwycić cały tekst
                margin = 5  # Pikselowy margines
                crop_x1 = max(0, crop_x1 - margin)
                crop_y1 = max(0, crop_y1 - margin)
                crop_x2 = min(width, crop_x2 + margin)
                crop_y2 = min(height, crop_y2 + margin)
                
                # Zapisz informacje o wymiarach obrazu i zaznaczenia
                logger.info(f"Wymiary obrazu: {width}x{height}, Zaznaczenie (px): ({crop_x1},{crop_y1})-({crop_x2},{crop_y2})")
                
                # Wytnij zaznaczony fragment
                crop_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                # Zapisz wycięty fragment do pliku tymczasowego
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                
                # Opcjonalne powiększenie małych fragmentów
                crop_width, crop_height = crop_image.size
                scale_factor = 1
                
                # Jeśli wycinek jest bardzo mały, powiększamy go dla lepszego OCR
                min_dimension = 300  # Minimalna szerokość/wysokość dla dobrego OCR
                if crop_width < min_dimension or crop_height < min_dimension:
                    if crop_width < crop_height:
                        scale_factor = min_dimension / crop_width
                    else:
                        scale_factor = min_dimension / crop_height
                
                if scale_factor > 1:
                    new_width = int(crop_width * scale_factor)
                    new_height = int(crop_height * scale_factor)
                    crop_image = crop_image.resize((new_width, new_height), Image.LANCZOS)
                    logger.info(f"Przeskalowano fragment do {new_width}x{new_height} (współczynnik: {scale_factor:.2f})")
                
                # Zapisz obraz fragmentu
                crop_image.save(tmp_path, format="PNG", quality=95)
                
                try:
                    # Uruchom OCR na wyciętym fragmencie
                    from tasks.ocr.models import process_image_to_text
                    
                    # Dostosuj instrukcję do fragmentu
                    instruction = "Extract all the text visible in this image fragment. Keep all formatting."
                    
                    # Rozpoznaj tekst z fragmentu
                    fragment_text = process_image_to_text(tmp_path, instruction=instruction)
                    
                    # Usuń plik tymczasowy
                    import os
                    os.unlink(tmp_path)
                    
                    # Jeśli tekst fragmentu jest pusty, ale zaznaczenie nie jest małe, 
                    # to prawdopodobnie to część obrazu bez tekstu
                    if not fragment_text.strip() and (crop_width * crop_height) > (width * height * 0.01):
                        fragment_text = "W zaznaczonym fragmencie nie wykryto tekstu."
                    
                    # Zwróć wynik
                    result = {
                        "success": True,
                        "text": fragment_text.strip(),
                        "page": page,
                        "total_pages": total_pages
                    }
                    
                    # Dodaj tekst pełnej strony, jeśli jest dostępny
                    if page_text:
                        result["full_page_text"] = page_text.strip()
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Błąd OCR fragmentu: {str(e)}", exc_info=True)
                    # W przypadku błędu OCR dla fragmentu, zwróć tekst całej strony jeśli jest dostępny
                    error_result = {
                        "success": True,
                        "text": "Nie udało się rozpoznać tekstu z fragmentu. Spróbuj zaznaczyć większy obszar.",
                        "page": page,
                        "total_pages": total_pages,
                        "error_fragment_ocr": str(e)
                    }
                    
                    if page_text:
                        error_result["full_page_text"] = page_text.strip()
                    
                    return error_result
                    
            except Exception as e:
                logger.error(f"Błąd konwersji PDF na obraz: {str(e)}", exc_info=True)
                return {"error": f"Błąd podczas przetwarzania zaznaczenia: {str(e)}"}
    
    except Exception as e:
        logger.error(f"Globalny błąd OCR zaznaczenia: {str(e)}", exc_info=True)
        return {"error": f"Błąd: {str(e)}"}

# Dodaj nowy endpoint w app/main.py
@app.get("/api/document/{doc_id}/ocr-progress")
def document_ocr_progress(doc_id: int):
    """Zwraca informacje o postępie OCR w formacie JSON."""
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie znaleziono dokumentu")
        
        # Przygotuj dane o postępie
        progress_data = {
            "status": doc.ocr_status,
            "progress": doc.ocr_progress or 0.0,
            "info": doc.ocr_progress_info or "",
            "current_page": doc.ocr_current_page or 0,
            "total_pages": doc.ocr_total_pages or 0,
            "confidence": doc.ocr_confidence
        }
        
        return progress_data

# Zaktualizuj tę funkcję w app/main.py
async def _enqueue_ocr(doc_id: int):
    """
    Wrzuć dokument do kolejki OCR.
    Ta funkcja jest wywoływana asynchronicznie i nie blokuje interfejsu.
    """
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            logger.error(f"Nie znaleziono dokumentu o ID={doc_id}")
            return
        
        # Zaktualizuj status dokumentu
        if doc.ocr_status != "running":  # Unikaj ponownego uruchamiania już działającego OCR
            doc.ocr_status = "pending"
            session.add(doc)
            session.commit()
    
    # Przekaż do kolejki OCR
    await enqueue(doc_id)

def check_file_extension(filename: str):
    """Sprawdź czy rozszerzenie pliku jest dozwolone."""
    suffix = pathlib.Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Niedozwolony typ pliku. Dozwolone typy: {allowed}"
        )
    return suffix

def get_text_preview(doc_id, max_length=None):
    """
    Pobiera tekst OCR dla podglądu.
    
    Args:
        doc_id: ID dokumentu
        max_length: Opcjonalne ograniczenie długości (None = bez ograniczeń)
    
    Returns:
        str: Tekst dokumentu
    """
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            return "Nie znaleziono dokumentu"
        
        try:
            text_path = FILES_DIR / doc.stored_filename
            if not text_path.exists():
                return "Plik tekstowy nie istnieje"
            
            # Próba odczytu z różnymi kodowaniami
            encodings = ['utf-8', 'latin-1', 'cp1250']
            for encoding in encodings:
                try:
                    # Odczytaj pełny tekst dokumentu
                    text = text_path.read_text(encoding=encoding)
                    if max_length and len(text) > max_length:
                        return text[:max_length] + "...\n[Skrócone - pobierz pełny tekst, aby zobaczyć więcej]"
                    return text
                except UnicodeDecodeError:
                    continue
            
            # Jeśli żadne kodowanie nie zadziałało
            return "Nie można odczytać tekstu - nieobsługiwane kodowanie znaków"
        except Exception as e:
            return f"Błąd podczas odczytu tekstu: {str(e)}"

# Middleware do wykrywania blokujących operacji
@app.middleware("http")
async def detect_blocking_operations(request: Request, call_next):
    # Rozpocznij liczenie czasu
    start_time = time.time()
    
    # Zidentyfikuj żądanie
    request_id = str(uuid.uuid4())[:8]
    path = request.url.path
    print(f"[{request_id}] Rozpoczęto żądanie: {path}")
    
    # Wykonaj żądanie
    response = await call_next(request)
    
    # Sprawdź czas wykonania
    elapsed = time.time() - start_time
    print(f"[{request_id}] Zakończono żądanie: {path} w {elapsed:.4f}s")
    
    # Loguj długie żądania
    if elapsed > 1.0:
        print(f"[{request_id}] UWAGA: Długie żądanie ({elapsed:.4f}s): {path}")
    
    return response

@app.post("/document/{doc_id}/delete")
async def document_delete(request: Request, doc_id: int):
    """Usuwa dokument. Jeśli to opinia, usuwa także wszystkie powiązane dokumenty."""
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie ma takiego dokumentu")
        
        # Sprawdź czy to opinia (dokument główny)
        if doc.is_main:
            # Pobierz wszystkie powiązane dokumenty
            related_docs = session.exec(
                select(Document)
                .where(Document.parent_id == doc_id)
            ).all()
            
            # Usuń pliki powiązanych dokumentów
            for related_doc in related_docs:
                file_path = FILES_DIR / related_doc.stored_filename
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    print(f"Błąd podczas usuwania pliku {related_doc.stored_filename}: {e}")
            
            # Usuń powiązane dokumenty z bazy danych
            for related_doc in related_docs:
                session.delete(related_doc)
            
            # Komunikat o usuniętych powiązanych dokumentach
            delete_message = f"Usunięto opinię i {len(related_docs)} powiązanych dokumentów."
        else:
            delete_message = "Dokument został usunięty."
        
        # Usuń plik głównego dokumentu
        file_path = FILES_DIR / doc.stored_filename
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Błąd podczas usuwania pliku {doc.stored_filename}: {e}")
        
        # Zapisz informacje o usuwanym dokumencie przed jego usunięciem
        was_opinion = doc.is_main
        parent_id = doc.parent_id
        
        # Usuń dokument z bazy danych
        session.delete(doc)
        session.commit()
    
    # Przekieruj w zależności od typu usuniętego dokumentu
    if was_opinion:
        base_url = str(request.url_for("list_opinions"))
        return RedirectResponse(f"{base_url}?delete_message={delete_message}", status_code=303)
    elif parent_id:
        base_url = str(request.url_for("opinion_detail", doc_id=parent_id))
        return RedirectResponse(f"{base_url}?delete_message={delete_message}", status_code=303)
    else:
        base_url = str(request.url_for("list_documents"))
        return RedirectResponse(f"{base_url}?delete_message={delete_message}", status_code=303)

@app.get("/document/{doc_id}/preview-content")
def document_preview_content(request: Request, doc_id: int):
    """Zwraca HTML z zawartością podglądu dokumentu do wyświetlenia w modalu."""
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie ma takiego dokumentu")
        
        # Sprawdź, czy istnieje wynik OCR (dokument TXT)
        ocr_txt = None
        if doc.ocr_status == "done":
            ocr_txt_query = select(Document).where(
                Document.ocr_parent_id == doc_id,
                Document.doc_type == "OCR TXT"
            )
            ocr_txt = session.exec(ocr_txt_query).first()
    
    # Przygotuj ścieżkę do pliku
    file_path = FILES_DIR / doc.stored_filename
    
    # Jeśli plik nie istnieje, zwróć błąd
    if not file_path.exists():
        return """
        <div class="alert alert-danger">
            <i class="bi bi-exclamation-triangle-fill me-2"></i>
            Plik nie istnieje.
        </div>
        """
    
    # Sprawdź typ pliku i przygotuj odpowiedni podgląd
    mime_type = doc.mime_type
    if not mime_type:
        mime_type = detect_mime_type(file_path)
    
    # Generuj poprawny URL do podglądu PDF
    pdf_url = str(request.url_for("document_preview", doc_id=doc.id))
    download_url = str(request.url_for("document_download", doc_id=doc.id))
  

    # Przygotuj przycisk ponownego uruchomienia OCR (jeśli dokument może mieć OCR)
    rerun_ocr_button = ""
    if doc.content_type in ['document', 'image'] and doc.mime_type != 'text/plain':
        # Status OCR i odpowiedni przycisk
        if doc.ocr_status == 'fail':
            rerun_ocr_button = f"""
            <form action="/document/{doc_id}/run_ocr" method="post" class="mt-3 text-center">
                <div class="alert alert-danger">
                    <i class="bi bi-x-circle-fill me-2"></i> OCR zakończony błędem
                </div>
                <button type="submit" class="btn btn-danger">
                    <i class="bi bi-arrow-clockwise"></i> Spróbuj ponownie OCR
                </button>
            </form>
            """
        elif doc.ocr_status == 'none':
            rerun_ocr_button = f"""
            <form action="/document/{doc_id}/run_ocr" method="post" class="mt-3 text-center">
                <button type="submit" class="btn btn-primary">
                    <i class="bi bi-play"></i> Uruchom OCR
                </button>
            </form>
            """
        elif doc.ocr_status == 'done':
            rerun_ocr_button = f"""
            <form action="/document/{doc_id}/run_ocr" method="post" class="mt-3 text-center">
                <button type="submit" class="btn btn-outline-secondary">
                    <i class="bi bi-arrow-clockwise"></i> Wykonaj OCR ponownie
                </button>
            </form>
            """
  
    if mime_type == 'application/pdf':
        # Użyj prostego widoku bez osadzania dużych danych
      return f"""
       <div style="text-align: center; padding: 30px;">
        <div style="margin-bottom: 20px;">
            <i class="bi bi-file-earmark-pdf text-danger" style="font-size: 64px;"></i>
            <h4 style="margin-top: 15px;">{doc.original_filename}</h4>
            <p class="text-muted">
                Dokument PDF • Dodano: {doc.upload_time.strftime('%Y-%m-%d %H:%M')}
            </p>
        </div>
        
        <div class="alert alert-info">
            <i class="bi bi-info-circle-fill me-2"></i>
            Podgląd PDF jest dostępny po otwarciu w nowej karcie.
        </div>
        
        <div style="margin-top: 20px;">
            <a href="{pdf_url}" class="btn btn-primary" target="_blank">
                <i class="bi bi-eye"></i> Otwórz PDF
            </a>
            <a href="{download_url}" class="btn btn-outline-secondary ms-3">
                <i class="bi bi-download"></i> Pobierz PDF
            </a>
        </div>
        
        {rerun_ocr_button}
      </div>
      """        


    # Dla obrazów
    elif mime_type and mime_type.startswith('image/'):
        img_url = request.url_for("document_preview", doc_id=doc.id)
        return f"""
        <div class="document-preview image-preview">
            <div class="document-info mb-3">
                <h6 class="document-title">{doc.original_filename}</h6>
                <div class="document-metadata text-muted small">
                    <span>Obraz {mime_type.split('/')[1].upper()}</span>
                    <span class="mx-2">•</span>
                    <span>Dodano: {doc.upload_time.strftime('%Y-%m-%d %H:%M')}</span>
                </div>
            </div>
            
            <div class="text-center">
                <img src="{img_url}" class="img-fluid border rounded" alt="Podgląd obrazu" 
                     style="max-height: 70vh; max-width: 100%;">
            </div>
            
            <div class="document-actions mt-3 text-center">
                <a href="{img_url}" class="btn btn-sm btn-outline-primary" target="_blank">
                    <i class="bi bi-fullscreen"></i> Otwórz w nowej karcie
                </a>
                <a href="{request.url_for('document_download', doc_id=doc.id)}" class="btn btn-sm btn-outline-secondary ms-2">
                    <i class="bi bi-download"></i> Pobierz obraz
                </a>
            </div>
        </div>
        """
    
    # Dla plików tekstowych (zarówno pliki TXT jak i wyniki OCR)
    elif mime_type == 'text/plain' or ocr_txt or (doc.original_filename and doc.original_filename.lower().endswith('.txt')):
        text_content = ""
        preview_source = ""
        
        if mime_type == 'text/plain' or (doc.original_filename and doc.original_filename.lower().endswith('.txt')):
            # Próba odczytu pliku tekstowego
            try:
                encodings = ['utf-8', 'latin-1', 'cp1250']
                for encoding in encodings:
                    try:
                        text_content = file_path.read_text(encoding=encoding)
                        preview_source = "Plik tekstowy"
                        break
                    except UnicodeDecodeError:
                        continue
                
                if not text_content:
                    return """
                    <div class="alert alert-warning">
                        <i class="bi bi-exclamation-triangle-fill me-2"></i>
                        Nie można odczytać zawartości pliku tekstowego.
                    </div>
                    """
            except Exception as e:
                return f"""
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    Błąd podczas odczytu pliku: {str(e)}
                </div>
                """
        
        # Jeśli nie ma zawartości z pliku tekstowego, spróbuj z OCR
        if not text_content and ocr_txt:
            ocr_file_path = FILES_DIR / ocr_txt.stored_filename
            try:
                encodings = ['utf-8', 'latin-1', 'cp1250']
                for encoding in encodings:
                    try:
                        text_content = ocr_file_path.read_text(encoding=encoding)
                        preview_source = "Wynik OCR"
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception:
                pass
        
        # Skróć tekst do podglądu jeśli jest zbyt długi
        max_length = 5000
        truncated = False
        if len(text_content) > max_length:
            text_content = text_content[:max_length]
            truncated = True
        
        # Utwórz element HTML do kopiowania tekstu
        copy_button = """
        <button id="copyTextBtn" class="btn btn-sm btn-outline-secondary float-end" onclick="copyTextToClipboard()">
            <i class="bi bi-clipboard"></i> Kopiuj do schowka
        </button>
        <script>
        function copyTextToClipboard() {
            const textElement = document.querySelector('.text-preview');
            if (textElement) {
                const textToCopy = textElement.textContent;
                navigator.clipboard.writeText(textToCopy).then(function() {
                    // Zmień wygląd przycisku na chwilę
                    const btn = document.getElementById('copyTextBtn');
                    const originalHtml = btn.innerHTML;
                    btn.innerHTML = '<i class="bi bi-check2"></i> Skopiowano!';
                    btn.classList.add('btn-success');
                    btn.classList.remove('btn-outline-secondary');
                    
                    setTimeout(function() {
                        btn.innerHTML = originalHtml;
                        btn.classList.remove('btn-success');
                        btn.classList.add('btn-outline-secondary');
                    }, 2000);
                });
            }
        }
        </script>
        """
        
        # Zwróć sformatowany podgląd tekstu
        truncation_notice = f"""
        <div class="alert alert-info mt-2">
            <i class="bi bi-info-circle-fill me-2"></i>
            Wyświetlono skróconą wersję. Pełny tekst dostępny po otwarciu pełnego podglądu.
        </div>
        """ if truncated else ""
        
        return f"""
        <div class="document-preview text-preview">
            <div class="document-info mb-3">
                <h6 class="document-title">{doc.original_filename}</h6>
                <div class="document-metadata text-muted small">
                    <span>{preview_source or "Dokument tekstowy"}</span>
                    <span class="mx-2">•</span>
                    <span>Dodano: {doc.upload_time.strftime('%Y-%m-%d %H:%M')}</span>
                </div>
            </div>
            
            <div class="card bg-light">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <span>Zawartość dokumentu</span>
                    {copy_button}
                </div>
                <div class="card-body">
                    <pre class="text-preview" style="max-height: 60vh; overflow-y: auto; white-space: pre-wrap;">{text_content}</pre>
                </div>
            </div>
            {truncation_notice}
            
            <div class="document-actions mt-3 text-center">
                <a href="{request.url_for('document_text_preview', doc_id=doc.id)}" class="btn btn-sm btn-outline-primary" target="_blank">
                    <i class="bi bi-fullscreen"></i> Otwórz pełny podgląd
                </a>
                <a href="{request.url_for('document_download', doc_id=doc.id)}" class="btn btn-sm btn-outline-secondary ms-2">
                    <i class="bi bi-download"></i> Pobierz dokument
                </a>
            </div>
        </div>
        """
    
    # Dla dokumentów Word i innych typów
    else:
        # Określ odpowiednią ikonę dla typu pliku
        icon_class = "bi-file-earmark"
        file_type_name = "Dokument"
        
        if mime_type:
            if "pdf" in mime_type:
                icon_class = "bi-file-earmark-pdf"
                file_type_name = "Dokument PDF"
            elif "word" in mime_type:
                icon_class = "bi-file-earmark-word"
                file_type_name = "Dokument Word"
            elif "image" in mime_type:
                icon_class = "bi-file-earmark-image"
                file_type_name = "Obraz"
            elif "text" in mime_type:
                icon_class = "bi-file-earmark-text"
                file_type_name = "Dokument tekstowy"
        
        # Przygotuj podgląd OCR, jeśli dostępny
        ocr_preview = ""
        if ocr_txt:
            try:
                ocr_file_path = FILES_DIR / ocr_txt.stored_filename
                ocr_text = ""
                encodings = ['utf-8', 'latin-1', 'cp1250']
                for encoding in encodings:
                    try:
                        ocr_text = ocr_file_path.read_text(encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                # Skróć tekst OCR do podglądu
                max_length = 1000
                truncated = False
                if len(ocr_text) > max_length:
                    ocr_text = ocr_text[:max_length]
                    truncated = True
                
                truncation_notice = f"""
                <div class="text-center mt-2">
                    <a href="{request.url_for('document_text_preview', doc_id=ocr_txt.id)}" class="btn btn-sm btn-outline-info">
                        <i class="bi bi-eye"></i> Zobacz pełny tekst OCR
                    </a>
                </div>
                """ if truncated else ""
                
                ocr_preview = f"""
                <div class="mt-4">
                    <h6 class="mb-3">Rozpoznany tekst (OCR):</h6>
                    <div class="card bg-light">
                        <div class="card-body">
                            <pre class="text-preview" style="max-height: 30vh; overflow-y: auto; white-space: pre-wrap;">{ocr_text}</pre>
                        </div>
                    </div>
                    {truncation_notice}
                </div>
                """
            except Exception as e:
                ocr_preview = f"""
                <div class="alert alert-warning mt-3">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    Nie można załadować podglądu tekstu OCR: {str(e)}
                </div>
                """
        
        # Zwróć informacje o pliku i podgląd OCR jeśli dostępny
        return f"""
        <div class="document-preview generic-preview">
            <div class="text-center py-4">
                <i class="bi {icon_class}" style="font-size: 5rem; color: #6c757d;"></i>
                <h5 class="mt-3">{doc.original_filename}</h5>
                <p class="text-muted">
                    {file_type_name}<br>
                    <small>Typ pliku: {mime_type or "Nieznany"}</small><br>
                    <small>Dodano: {doc.upload_time.strftime('%Y-%m-%d %H:%M')}</small>
                </p>
                <div class="mt-3">
                    <a href="{request.url_for('document_download', doc_id=doc.id)}" class="btn btn-primary">
                        <i class="bi bi-download"></i> Pobierz dokument
                    </a>
                    <a href="{request.url_for('document_detail', doc_id=doc.id)}" class="btn btn-outline-secondary ms-2">
                        <i class="bi bi-eye"></i> Szczegóły dokumentu
                    </a>
                </div>
            </div>
            
            {ocr_preview}
        </div>
        """

@app.get("/document/{doc_id}/update")
def document_update_form(request: Request, doc_id: int):
    """Formularz aktualizacji istniejącego dokumentu lub wgrania dokumentu Word dla pustej opinii."""
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie znaleziono dokumentu")

        #sprawdź, czy to pusta opinia (bez pliku) lub dokument w formacie Word
        is_empty_opinion = doc.is_main and doc.stored_filename.endswith('.empty')
        is_word_doc = doc.mime_type and 'word' in doc.mime_type
    
        if not (is_empty_opinion or is_word_doc):
            raise HTTPException(
                status_code=400, 
                detail="Tylko dokumenty Word lub puste opinie mogą być aktualizowane tą metodą"
            )

    return templates.TemplateResponse(
        "document_update.html", 
        {
            "request": request, 
            "doc": doc, 
            "is_empty_opinion": is_empty_opinion,
            "title": f"{'Wgraj' if is_empty_opinion else 'Aktualizuj'} dokument: {doc.original_filename}"
        }
    )


@app.get("/document/{doc_id}/history")
def document_history(request: Request, doc_id: int):
    """Historia wersji dokumentu."""
    with Session(engine) as session:
        # Pobierz aktualny dokument
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie znaleziono dokumentu")
        
        # Pobierz historyczne wersje (dokumenty z parent_id równym ID aktualnego dokumentu)
        history_docs = session.exec(
            select(Document)
            .where(Document.parent_id == doc_id, Document.doc_type == "Archiwalna wersja")
            .order_by(Document.upload_time.desc())
        ).all()
    
    return templates.TemplateResponse(
        "document_history.html", 
        {
            "request": request, 
            "doc": doc, 
            "history_docs": history_docs,
            "title": f"Historia dokumentu: {doc.original_filename}"
        }
    )

@app.post("/document/{doc_id}/update")
async def document_update(request: Request, doc_id: int, updated_file: UploadFile = File(...), 
    keep_history: bool = Form(True),comments: str | None = Form(None)):
   
    """Aktualizacja istniejącego dokumentu lub wgranie dokumentu Word dla pustej opinii."""
    with Session(engine) as session:
        # Pobierz istniejący dokument
        doc = session.get(Document, doc_id)
        if not doc:
          raise HTTPException(status_code=404, detail="Nie znaleziono dokumentu")
    
    # Sprawdź, czy to pusta opinia
    is_empty_opinion = doc.is_main and doc.stored_filename.endswith('.empty')
    
    # Sprawdź rozszerzenie pliku - dla pustej opinii musi być .doc lub .docx
    # Dla istniejącego dokumentu musi być zgodne z oryginalnym
    new_ext = Path(updated_file.filename).suffix.lower()
    
    if is_empty_opinion:
        # Dla pustej opinii akceptujemy tylko pliki Word
        if new_ext.lower() not in ['.doc', '.docx']:
            raise HTTPException(
                status_code=400,
                detail="Dla pustej opinii można wgrać tylko pliki Word (.doc, .docx)"
            )
    else:
        # Dla istniejącego dokumentu sprawdzamy zgodność rozszerzenia
        original_ext = Path(doc.original_filename).suffix.lower()
        if original_ext != new_ext:
            raise HTTPException(
                status_code=400,
                detail=f"Rozszerzenie pliku musi być zgodne z oryginałem ({original_ext})"
            )
    
    # Jeśli zachowujemy historię i nie jest to pusta opinia, oznacz stary dokument jako wersję historyczną
    if keep_history and not is_empty_opinion:
        # Utwórz kopię oryginalnego dokumentu jako "wersję historyczną"
        history_doc = Document(
            sygnatura=doc.sygnatura,
            doc_type="Archiwalna wersja",
            original_filename=f"Archiwalna_{doc.original_filename}",
            stored_filename=doc.stored_filename,  # Zachowujemy oryginalny plik
            step=doc.step,
            ocr_status=doc.ocr_status,
            parent_id=doc.id,  # Powiązanie z aktualnym dokumentem
            is_main=False,  # To nie jest główny dokument
            content_type=doc.content_type,
            mime_type=doc.mime_type,
            creator=doc.creator,
            upload_time=doc.upload_time,
            last_modified=doc.last_modified,
            comments=f"Wersja archiwalna z {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}. {doc.comments or ''}"
        )
        session.add(history_doc)
        
    # Zapisz nowy plik
    unique_name = f"{uuid.uuid4().hex}{new_ext}"
    file_path = FILES_DIR / unique_name
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(updated_file.file, buffer)
    
    # Zaktualizuj informacje o dokumencie
    doc.stored_filename = unique_name
    doc.original_filename = updated_file.filename
    doc.mime_type = detect_mime_type(file_path)
    doc.last_modified = datetime.utcnow()
    # doc.last_modified_by = current_user  # Gdy będzie system użytkowników
    
    if comments:
        if doc.comments:
            doc.comments = f"{comments}\n\n---\n\n{doc.comments}"
        else:
            doc.comments = comments
    
    session.add(doc)
    session.commit()
    
    # Przekieruj do widoku dokumentu
    return RedirectResponse(
        request.url_for("document_detail", doc_id=doc.id),
        status_code=303
    )

@app.get("/")
def root_redirect():
    """Przekierowanie ze strony głównej do listy opinii."""
    return RedirectResponse(url="/opinions", status_code=303)

@app.get("/documents", name="list_documents")
def list_documents(request: Request):
    """Lista wszystkich dokumentów."""
    with Session(engine) as session:
        docs = session.exec(select(Document).order_by(Document.upload_time.desc())).all()
        return templates.TemplateResponse(
         "index.html", 
         {"request": request, "docs": docs, "icons": STEP_ICON, "title": "Dokumenty"})

@app.get("/opinions")
def list_opinions(request: Request):
    """Lista głównych dokumentów (opinii)."""
    with Session(engine) as session:
        # Pobierz tylko dokumenty główne (is_main=True)
        opinions = session.exec(
            select(Document)
            .where(Document.is_main == True)
            .order_by(Document.upload_time.desc())
        ).all()
        return templates.TemplateResponse(
            "opinions.html", 
            {"request": request, "opinions": opinions, "icons": STEP_ICON, "title": "Opinie sądowe"}
        )

@app.get("/opinion/{doc_id}")
def opinion_detail(request: Request, doc_id: int):
    """Szczegóły opinii wraz z dokumentami powiązanymi."""
    with Session(engine) as session:
        # Pobierz główny dokument
        opinion = session.get(Document, doc_id)
        if not opinion or not opinion.is_main:
            raise HTTPException(status_code=404, detail="Nie znaleziono opinii")
        
        # Pobierz dokumenty powiązane
        related_docs = session.exec(
            select(Document)
            .where(Document.parent_id == doc_id)
            .order_by(Document.upload_time.desc())
        ).all()
        
        # Grupuj dokumenty powiązane według doc_type
        grouped_docs = {}
        
        # Przygotuj statystyki OCR
        total_docs = 0
        pending_docs = 0
        running_docs = 0
        done_docs = 0
        failed_docs = 0
        
        for doc in related_docs:
            # Zliczanie dokumentów według statusu OCR
            total_docs += 1
            if doc.ocr_status == 'pending':
                pending_docs += 1
            elif doc.ocr_status == 'running':
                running_docs += 1
            elif doc.ocr_status == 'done':
                done_docs += 1
            elif doc.ocr_status == 'fail':
                failed_docs += 1
            
            # Grupowanie według typu dokumentu
            doc_type = doc.doc_type or "Inne"
            if doc_type not in grouped_docs:
                grouped_docs[doc_type] = []
            grouped_docs[doc_type].append(doc)
        
        steps = [("k1", "k1 – Wywiad"),
                 ("k2", "k2 – Wyciąg z akt"),
                 ("k3", "k3 – Opinia"),
                 ("k4", "k4 – Archiwum")]
        
        return templates.TemplateResponse(
            "opinion_detail.html",
            {
                "request": request, 
                "opinion": opinion, 
                "grouped_docs": grouped_docs,
                "steps": steps, 
                "title": f"Opinia #{opinion.id}: {opinion.sygnatura or opinion.original_filename}",
                # Dodaj statystyki OCR do kontekstu
                "total_docs": total_docs,
                "pending_docs": pending_docs,
                "running_docs": running_docs,
                "done_docs": done_docs,
                "failed_docs": failed_docs,
                "has_active_ocr": pending_docs > 0 or running_docs > 0
            }
        )

@app.post("/opinion/{doc_id}/update")
def opinion_update(request: Request, doc_id: int,
                   step: str = Form(...),
                   sygnatura: str | None = Form(None),
                   comments: str | None = Form(None)):
    """Aktualizacja statusu opinii."""
    with Session(engine) as session:
        opinion = session.get(Document, doc_id)
        if not opinion or not opinion.is_main:
            raise HTTPException(status_code=404, detail="Nie znaleziono opinii")
        
        # Aktualizuj dane opinii
        opinion.step = step
        opinion.sygnatura = sygnatura
        opinion.comments = comments
        opinion.last_modified = datetime.utcnow()
        # Tu można dodać last_modified_by = current_user
        
        session.add(opinion)
        session.commit()
    
    return RedirectResponse(request.url_for("opinion_detail", doc_id=doc_id), status_code=303)

@app.get("/opinion/{doc_id}/upload")
def opinion_upload_form(request: Request, doc_id: int):
    """Formularz dodawania dokumentów do opinii."""
    with Session(engine) as session:
        opinion = session.get(Document, doc_id)
        if not opinion or not opinion.is_main:
            raise HTTPException(status_code=404, detail="Nie znaleziono opinii")
    
    allowed_types = ", ".join(ALLOWED_EXTENSIONS.keys())
    return templates.TemplateResponse(
        "upload_to_opinion.html", 
        {
            "request": request, 
            "opinion": opinion,
            "allowed_types": allowed_types,
            "title": f"Dodaj dokumenty do opinii: {opinion.sygnatura or opinion.original_filename}"
        }
    )

async def _enqueue_ocr_documents_nonblocking(doc_ids: list[int]):
    """
    Asynchronicznie wstawia dokumenty do kolejki OCR bez blokowania.
    Ta funkcja jest całkowicie odłączona od głównego wątku.
    """
    from app.background_tasks import enqueue_ocr_task
    
    for doc_id in doc_ids:
        try:
            # Ważne - tworzymy nową sesję wewnątrz tej funkcji
            with Session(engine) as session:
                doc = session.get(Document, doc_id)
                if doc and doc.ocr_status == "pending":
                    # Dodajemy dokument do kolejki OCR
                    await enqueue_ocr_task(doc_id)
                    
                    # Natychmiast oddajemy kontrolę do pętli zdarzeń
                    await asyncio.sleep(0)
        except Exception as e:
            print(f"Błąd podczas dodawania dokumentu {doc_id} do kolejki OCR: {str(e)}")
            # Bez błędu kontynuujemy z kolejnymi dokumentami
            continue

# Dodaj tę funkcję w app/main.py
async def _enqueue_ocr_documents(doc_ids: list[int]):
    """
    Asynchronicznie wstawia dokumenty do kolejki OCR.
    Ta funkcja jest uruchamiana w tle i nie blokuje odpowiedzi HTTP.
    """
    for doc_id in doc_ids:
        with Session(engine) as session:
            doc = session.get(Document, doc_id)
            if doc and doc.ocr_status == "pending":
                # Tylko dokumenty oczekujące na OCR
                await enqueue(doc_id)

# Zmodyfikuj istniejącą funkcję upload
@app.get("/upload")
def upload_form(request: Request):
    allowed_types = ", ".join(ALLOWED_EXTENSIONS.keys())
    return templates.TemplateResponse(
        "upload.html", 
        {"request": request, "title": "Załaduj nową opinię", "allowed_types": allowed_types}
    )

# Zaktualizuj tę funkcję w app/main.py
@app.post("/document/{doc_id}/run_ocr")
async def document_run_ocr(request: Request, doc_id: int):
    """Endpoint do ręcznego uruchomienia OCR ponownie."""
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie ma takiego dokumentu")
        doc.ocr_status = "pending"
        doc.ocr_progress = 0.0
        doc.ocr_progress_info = "Oczekuje w kolejce"
        session.add(doc)
        session.commit()
    
    # Dodaj do kolejki OCR
    from app.background_tasks import enqueue_ocr_task
    asyncio.create_task(enqueue_ocr_task(doc_id))
    
    # Dodaj parametr do URL przekierowania, aby pokazać powiadomienie
    redirect_url = request.url_for("document_detail", doc_id=doc_id)
    return RedirectResponse(f"{redirect_url}?ocr_restarted=true", status_code=303)

# Modyfikacja funkcji upload w app/main.py
@app.post("/upload")
async def upload(request: Request, files: list[UploadFile] = File(...)):
    """Dodawanie nowych głównych dokumentów (opinii) bez blokowania interfejsu."""
    uploaded_docs = []
    
    with Session(engine) as session:
        for file in files:
            # Sprawdzenie rozszerzenia pliku
            suffix = check_file_extension(file.filename)
            
            # Dla opinii akceptujemy tylko pliki Word
            if suffix.lower() not in ['.doc', '.docx']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Opinie muszą być w formacie Word (.doc, .docx). Przesłano: {suffix}"
                )
            
            # Generowanie unikalnej nazwy pliku
            unique_name = f"{uuid.uuid4().hex}{suffix}"
            dest = FILES_DIR / unique_name
            
            # Zapisanie pliku
            with dest.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Wykrywanie właściwego MIME typu pliku
            actual_mime_type = detect_mime_type(dest)
            
            # Zapisanie do bazy danych jako dokument główny
            doc = Document(
                original_filename=file.filename,
                stored_filename=unique_name,
                step="k1",  # Nowe opinie zaczynają od k1
                ocr_status="none",  # Word nie wymaga OCR
                is_main=True,  # Oznacz jako dokument główny
                content_type="opinion",
                mime_type=actual_mime_type,
                doc_type="Opinia",
                creator=None  # Tu można dodać current_user
            )
            session.add(doc)
            session.commit()
            uploaded_docs.append(doc.id)

    # Przekieruj do listy opinii lub do szczegółów pierwszej dodanej opinii
    if len(uploaded_docs) == 1:
        return RedirectResponse(request.url_for("opinion_detail", doc_id=uploaded_docs[0]), status_code=303)
    else:
        return RedirectResponse(request.url_for("list_opinions"), status_code=303)

# Modyfikacja funkcji opinion_upload w app/main.py
@app.post("/opinion/{doc_id}/upload")
async def opinion_upload(request: Request, doc_id: int, 
                         doc_type: str = Form(...),
                         files: list[UploadFile] = File(...),
                         run_ocr: bool = Form(False)):
    """Dodawanie dokumentów do opinii bez blokowania interfejsu."""
    # Sprawdź czy opinia istnieje
    with Session(engine) as session:
        opinion = session.get(Document, doc_id)
        if not opinion or not opinion.is_main:
            raise HTTPException(status_code=404, detail="Nie znaleziono opinii")
    
    uploaded_docs = []
    has_ocr_docs = False  # Flaga wskazująca, czy jakiekolwiek dokumenty wymagają OCR

    # Najpierw zapisujemy metadane dokumentów w bazie danych
    for file in files:
      # Sprawdzenie rozszerzenia pliku
      suffix = check_file_extension(file.filename)
    
      # Generowanie unikalnej nazwy pliku
      unique_name = f"{uuid.uuid4().hex}{suffix}"
      dest = FILES_DIR / unique_name
    
      # Asynchroniczne zapisanie pliku
      content = await file.read()
    
      # Synchroniczne zapisanie zawartości (to jest operacja procesora, nie I/O)
      with open(dest, "wb") as buffer:
        buffer.write(content)
    
      # Wykrywanie właściwego MIME typu pliku
      actual_mime_type = detect_mime_type(dest)
    
      # Określanie content_type na podstawie MIME type
      content_type = "document"
      if actual_mime_type.startswith('image/'):
          content_type = "image"
      elif suffix.lower() in ['.doc', '.docx']:
          content_type = "opinion"  # Dokumenty Word mogą być opiniami
    
      # Jeśli to nowy dokument główny, nie powiązuj go z obecną opinią
      is_main = content_type == "opinion" and doc_type == "Opinia"
      parent_id = None if is_main else doc_id
    
      # Ustal właściwy status OCR
      # Pliki PDF i obrazy będą miały status "pending"
      ocr_status = "none"
      if run_ocr and content_type != "opinion":
        ocr_status = "pending"
        has_ocr_docs = True  # Zaznacz, że co najmniej jeden dokument wymaga OCR
    
      # Zapisanie do bazy danych
      with Session(engine) as session:
        new_doc = Document(
            sygnatura=opinion.sygnatura,
            doc_type=doc_type,
            original_filename=file.filename,
            stored_filename=unique_name,
            step="k1" if is_main else opinion.step,  # Nowe opinie zaczynają od k1
            ocr_status=ocr_status,
            parent_id=parent_id,
            is_main=is_main,
            content_type=content_type,
            mime_type=actual_mime_type,
            creator=None,  # Tu można dodać current_user
            upload_time=datetime.utcnow()
        )
        session.add(new_doc)
        session.commit()
        
        uploaded_docs.append(new_doc.id)

    # Uruchom OCR dla wgranych dokumentów w tle
    asyncio.create_task(_enqueue_ocr_documents_nonblocking(uploaded_docs))

    # Przekieruj do widoku opinii z odpowiednim komunikatem
    redirect_url = request.url_for("opinion_detail", doc_id=doc_id)

    # Jeśli są dokumenty wymagające OCR, dodaj parametr informujący o tym
    if has_ocr_docs:
      asyncio.create_task(_enqueue_ocr_documents_nonblocking(uploaded_docs))
      return RedirectResponse(
        f"{redirect_url}?ocr_started=true&count={len(uploaded_docs)}", 
        status_code=303
      )
    else:
      return RedirectResponse(redirect_url, status_code=303)

@app.get("/document/{doc_id}", name="document_detail")
def document_detail(request: Request, doc_id: int):
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie ma takiego dokumentu")
        
        # Sprawdź, czy istnieje wynik OCR (dokument TXT)
        ocr_txt = None
        if doc.ocr_status == "done":
            ocr_txt_query = select(Document).where(
                Document.ocr_parent_id == doc_id,
                Document.doc_type == "OCR TXT"
            )
            ocr_txt = session.exec(ocr_txt_query).first()
            
    steps = [("k1", "k1 – Wywiad"),
             ("k2", "k2 – Wyciąg z akt"),
             ("k3", "k3 – Opinia"),
             ("k4", "k4 – Archiwum")]
             
    # Przygotuj kontekst odpowiedzi
    context = {
        "request": request, 
        "doc": doc, 
        "ocr_txt": ocr_txt, 
        "steps": steps, 
        "title": f"Dokument #{doc.id}"
    }
    
    # Jeśli dokument to plik tekstowy, dodaj pełny tekst
    if doc.mime_type == "text/plain":
        context["doc_text_preview"] = get_text_preview(doc.id, max_length=None)  # Bez ograniczenia długości
    
    # Jeśli istnieje dokument TXT z OCR, dodaj pełny tekst OCR
    if ocr_txt:
        context["ocr_text_preview"] = get_text_preview(ocr_txt.id, max_length=None)  # Bez ograniczenia długości
             
    return templates.TemplateResponse("document.html", context)

@app.post("/document/{doc_id}")
def document_update(request: Request, doc_id: int,
                    step: str = Form(...),
                    sygnatura: str | None = Form(None),
                    doc_type: str | None = Form(None)):
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie ma takiego dokumentu")
        doc.step = step
        doc.sygnatura = sygnatura or None
        doc.doc_type = doc_type or None
        session.add(doc)
        session.commit()
    return RedirectResponse(request.url_for("document_detail", doc_id=doc_id), status_code=303)

@app.get("/document/{doc_id}/download")
def document_download(doc_id: int):
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie ma takiego dokumentu")
    file_path = FILES_DIR / doc.stored_filename
    
    # Określamy MIME type na podstawie zapisanego typu lub wykrywamy na nowo
    mime_type = doc.mime_type
    if not mime_type:
        mime_type = detect_mime_type(file_path)
    
    return FileResponse(
        file_path,
        filename=doc.original_filename,
        media_type=mime_type
    )

# Zastąp istniejącą funkcję document_preview poniższą wersją
@app.get("/document/{doc_id}/preview")
def document_preview(request: Request, doc_id: int):
    """Podgląd dokumentu bezpośrednio w przeglądarce."""
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie ma takiego dokumentu")
    
    file_path = FILES_DIR / doc.stored_filename
    
    # Sprawdź, czy plik istnieje
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Plik nie istnieje")
    
    # Dla obrazów, PDF i plików tekstowych wyświetlamy bezpośrednio
    mime_type = doc.mime_type
    if not mime_type:
        mime_type = detect_mime_type(file_path)
    
    # Obrazy i PDF obsługujemy bezpośrednio
    if mime_type and mime_type.startswith('image/'): 
        return FileResponse(
            file_path,
            filename=doc.original_filename,
            media_type=mime_type
        )
    if mime_type and (mime_type == 'application/pdf'):
        return FileResponse(file_path, media_type="application/pdf")

    # Dla plików tekstowych otwieramy specjalny podgląd
    if mime_type == 'text/plain' or doc.original_filename.lower().endswith('.txt'):
        return RedirectResponse(
            request.url_for("document_text_preview", doc_id=doc.id)
        )
    
    # Dla innych typów, przekierowujemy do pobierania
    return RedirectResponse(
        request.url_for("document_download", doc_id=doc.id)
    )

# Dodaj tę nową funkcję do main.py
@app.get("/document/{doc_id}/text-preview")
def document_text_preview(request: Request, doc_id: int):
    """Podgląd pliku tekstowego w formacie HTML."""
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Nie ma takiego dokumentu")
    
    file_path = FILES_DIR / doc.stored_filename
    
    # Sprawdź, czy plik istnieje
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Plik nie istnieje")
    
    # Odczytaj zawartość pliku
    content = None
    encodings = ['utf-8', 'latin-1', 'cp1250']
    error_message = None
    
    for encoding in encodings:
        try:
            content = file_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        error_message = "Nie można odczytać pliku - nieobsługiwane kodowanie znaków"
    
    # Renderuj stronę HTML z zawartością tekstu
    return templates.TemplateResponse(
        "text_preview.html",
        {
            "request": request,
            "doc": doc,
            "content": content,
            "error_message": error_message,
            "title": f"Podgląd tekstowy: {doc.original_filename}"
        }
    )
