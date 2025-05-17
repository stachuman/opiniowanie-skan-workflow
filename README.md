# Court Workflow

System do zarządzania opiniami sądowymi i dokumentami z zaawansowaną funkcją OCR (Optical Character Recognition).

## Główne funkcje

- **Hierarchiczna struktura dokumentów**:
  - Opinie główne (dokumenty Word)
  - Dokumenty powiązane (PDF, obrazy, pliki tekstowe) przypisane do opinii
  - Specjalny kontener dla dokumentów niezwiązanych z opiniami

- **Obieg pracy (workflow)**:
  - **k1 (Wywiad)** - początkowy etap tworzenia opinii
  - **k2 (Wyciąg z akt)** - dodawanie dokumentów źródłowych
  - **k3 (Opinia)** - finalizacja opinii
  - **k4 (Archiwum)** - archiwizacja zakończonej sprawy

- **Zaawansowane OCR**:
  - Automatyczne rozpoznawanie tekstu w plikach PDF i obrazach
  - Możliwość zaznaczania fragmentów do selektywnego OCR
  - Szybkie OCR niezależne od opinii
  - Możliwość włączenia/wyłączenia OCR przy wgrywaniu dokumentów

- **Zarządzanie wersjami dokumentów**:
  - Historia zmian dokumentów
  - Możliwość aktualizacji dokumentów Word i zachowania poprzednich wersji

- **Przetwarzanie asynchroniczne**:
  - Przetwarzanie OCR w tle bez blokowania interfejsu
  - Śledzenie postępu OCR w czasie rzeczywistym

## Technologie

- **Backend**: FastAPI, Python 3.8+
- **Baza danych**: SQLite z SQLModel
- **Frontend**: Bootstrap 5, JavaScript
- **OCR**: Tesseract, model AI Qwen2.5-VL
- **Przetwarzanie plików**: PyPDF2, pdf2image, PIL

## 🚀 Instalacja

### Wymagania wstępne

- Python 3.8 lub nowszy
- Tesseract OCR
- Poppler (dla pdf2image)

### Kroki instalacji

1. **Klonowanie repozytorium**
   ```bash
   git clone https://github.com/stachuman/opiniowanie-skan-workflow.git
   cd opiniowanie-skan-workflow
   ```

2. **Utworzenie i aktywacja wirtualnego środowiska**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # lub
   venv\Scripts\activate     # Windows
   ```

3. **Instalacja zależności**
   ```bash
   pip install -r requirements.txt
   ```

4. **Przygotowanie katalogów**
   ```bash
   mkdir -p files
   ```

5. **Uruchomienie aplikacji**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Dostęp do aplikacji**
   Otwórz przeglądarkę i przejdź do: `http://localhost:8000`

## 📂 Struktura projektu

```
opiniowanie-skan-workflow/
├── app/                  # Główny katalog aplikacji
│   ├── main.py           # Główny plik aplikacji FastAPI
│   ├── db.py             # Konfiguracja bazy danych
│   ├── models.py         # Modele danych (SQLModel)
│   └── background_tasks.py # System zadań w tle
├── tasks/                # Moduł zarządzania zadaniami OCR
│   ├── ocr_manager.py    # Zarządzanie kolejką OCR
│   └── ocr/              # Implementacja OCR
│       ├── models.py     # Modele OCR
│       ├── pipeline.py   # Pipeline przetwarzania OCR
│       └── config.py     # Konfiguracja OCR
├── templates/            # Szablony HTML
│   ├── base.html         # Szablon bazowy
│   ├── document.html     # Widok dokumentu
│   ├── opinion_detail.html # Widok szczegółów opinii
│   └── ...
├── static/               # Pliki statyczne (CSS, JS, ikony)
├── files/                # Katalog na przechowywane pliki (gitignore)
└── requirements.txt      # Zależności Pythona
```

## Przepływ pracy

### Tworzenie nowej opinii

1. Utwórz nową opinię na trzy sposoby:
   - Wgraj dokument Word z opinią
   - Utwórz pustą opinię (bez dokumentu)
   - Wykonaj szybkie OCR niezależne od opinii

2. Dodaj dokumenty do opinii:
   - PDF, obrazy, dokumenty Word, pliki tekstowe
   - Opcjonalnie włącz/wyłącz OCR dla wgrywanych plików

3. Zarządzaj obiegiem pracy:
   - Przesuwaj opinie przez kolejne etapy (k1 → k2 → k3 → k4)
   - Dodawaj komentarze i metadane

### Praca z OCR

1. OCR jest wykonywane asynchronicznie w tle
2. Dostępne są informacje o postępie przetwarzania OCR
3. Możliwość ponownego uruchomienia OCR jeśli potrzeba
4. Dla plików PDF - zaawansowany podgląd z możliwością zaznaczania fragmentów do selektywnego OCR

## Rozwój projektu

### Znane problemy

- Obcinanie tekstu OCR do około 4800 znaków (prawdopodobna przyczyna: ograniczenie parametru max_new_tokens w funkcji process_image_to_text)

### Planowane ulepszenia

- Zwiększenie limitu tokenów dla OCR
- Implementacja przetwarzania OCR w mniejszych fragmentach
- Optymalizacja zarządzania pamięcią GPU podczas przetwarzania OCR


