from pathlib import Path
from sqlmodel import create_engine, SQLModel
import os, pathlib, logging

# Konfiguracja ścieżek
DB_URL = os.getenv("DB_URL", "sqlite:///data.db")
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
FILES_DIR = BASE_DIR / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db")

# Konfiguracja połączenia do bazy danych
engine = create_engine(
    DB_URL, 
    connect_args={"check_same_thread": False},
    echo=False  # Zmień na True, aby włączyć logowanie SQL
)

def init_db():
    """Inicjalizacja bazy danych i aktualizacja schematu."""
    logger.info("Inicjalizacja bazy danych...")
    
    # Importujemy tutaj, aby uniknąć cyklicznych importów
    from app.models import Document
    
    # Tworzenie tabel, jeśli nie istnieją
    SQLModel.metadata.create_all(engine)
    
    # Sprawdzenie, czy potrzebna jest migracja dla istniejących dokumentów
    try:
        from sqlalchemy import inspect
        inspector = inspect(engine)
        
        # Sprawdź, czy kolumny content_type, mime_type istnieją
        existing_columns = {col['name'] for col in inspector.get_columns('document')}
        needed_columns = {'content_type', 'mime_type', 'ocr_confidence'}
        
        # Jeśli brakuje którejś kolumny, dodaj ją
        missing_columns = needed_columns - existing_columns
        if missing_columns:
            logger.info(f"Brakujące kolumny: {missing_columns}")
            
            # Wykonaj migrację - dodaj brakujące kolumny
            from sqlalchemy import Column, String, Float
            from sqlalchemy.sql import text
            
            with engine.connect() as connection:
                if 'content_type' in missing_columns:
                    logger.info("Dodawanie kolumny 'content_type'...")
                    connection.execute(text("ALTER TABLE document ADD COLUMN content_type VARCHAR DEFAULT 'document'"))
                
                if 'mime_type' in missing_columns:
                    logger.info("Dodawanie kolumny 'mime_type'...")
                    connection.execute(text("ALTER TABLE document ADD COLUMN mime_type VARCHAR"))
                
                if 'ocr_confidence' in missing_columns:
                    logger.info("Dodawanie kolumny 'ocr_confidence'...")
                    connection.execute(text("ALTER TABLE document ADD COLUMN ocr_confidence FLOAT"))
                
                connection.commit()
            
            logger.info("Migracja zakończona pomyślnie")
    except Exception as e:
        logger.error(f"Błąd podczas migracji: {str(e)}")
        # Kontynuuj mimo błędu - najgorsze co się stanie to brak nowych kolumn
    
    logger.info("Inicjalizacja bazy danych zakończona")
