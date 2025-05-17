from datetime import datetime
from sqlmodel import SQLModel, Field

class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    sygnatura: str | None = None
    doc_type: str | None = None
    original_filename: str
    stored_filename: str
    step: str
    ocr_status: str = "none"            # none/pending/running/done/fail
    ocr_parent_id: int | None = None    # Relacja do dokumentu źródłowego OCR
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    
    # Nowe pola
    content_type: str = "document"      # document/image/opinion - typ treści
    mime_type: str | None = None        # MIME type pliku
    ocr_confidence: float | None = None # Średni poziom pewności OCR
    
    # Relacje hierarchiczne
    is_main: bool = False               # Czy jest to dokument główny (opinia)
    parent_id: int | None = Field(default=None, foreign_key="document.id")  # Relacja do dokumentu nadrzędnego
    
    # Informacje dodatkowe
    creator: str | None = None          # Osoba, która utworzyła dokument
    last_modified_by: str | None = None # Osoba, która ostatnio modyfikowała dokument
    last_modified: datetime | None = None  # Data ostatniej modyfikacji
    comments: str | None = None         # Komentarze/uwagi do dokumentu

    # Nowe pola do śledzenia postępu OCR
    ocr_progress: float | None = None  # Postęp od 0.0 do 1.0
    ocr_progress_info: str | None = None  # Dodatkowe informacje o postępie
    ocr_total_pages: int | None = None  # Całkowita liczba stron
    ocr_current_page: int | None = None  # Aktualna przetwarzana strona
