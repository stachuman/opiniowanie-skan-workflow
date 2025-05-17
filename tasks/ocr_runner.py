"""
Moduł wejściowy dla zadań OCR uruchamianych przez Redis Queue.
"""
import os
import time
import signal
import traceback
import logging
from pathlib import Path

# Konfiguracja podstawowego loggera dla tego skryptu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ocr_runner")

def run_ocr_pipeline(doc_id: int):
    """
    Główna funkcja zadania uruchamianego przez RQ.
    """
    try:
        # Importuj potrzebne moduły
        import psutil
        import threading
    except ImportError:
        # Jeśli psutil nie jest zainstalowany, zainstaluj go
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"BRAK MODUŁU PSUTIL - PRÓBA INSTALACJI\n")
        import subprocess
        subprocess.run(["pip", "install", "psutil"])
        import psutil
        import threading
    
    start_time = time.time()
    pid = os.getpid()
    
    # Zapisz podstawowe informacje diagnostyczne
    with open("/tmp/ocr_debug.log", "a") as f:
        f.write(f"====================\n")
        f.write(f"START OCR dla ID={doc_id}, PID={pid}, TIME={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        process = psutil.Process(pid)
        mem = process.memory_info()
        f.write(f"POCZĄTKOWE ZUŻYCIE PAMIĘCI: RSS={mem.rss/1024/1024:.2f}MB, VMS={mem.vms/1024/1024:.2f}MB\n")
    
    logger.info(f"Uruchamiam zadanie OCR dla dokumentu ID={doc_id}, PID={pid}")
    
    # Funkcja do logowania stanu
    def log_status():
        try:
            if not psutil.pid_exists(pid):
                return False
            
            process = psutil.Process(pid)
            mem = process.memory_info()
            cpu = process.cpu_percent(interval=0.1)
            
            try:
                io = process.io_counters()
                io_read = io.read_bytes/1024/1024
                io_write = io.write_bytes/1024/1024
            except:
                io_read = 0
                io_write = 0
            
            with open("/tmp/ocr_debug.log", "a") as f:
                runtime = time.time() - start_time
                f.write(f"STATUS [{time.strftime('%H:%M:%S')}]: Runtime={runtime:.1f}s, MEM_RSS={mem.rss/1024/1024:.2f}MB, CPU={cpu}%, IO_READ={io_read:.2f}MB, IO_WRITE={io_write:.2f}MB\n")
            return True
        except Exception as e:
            with open("/tmp/ocr_debug.log", "a") as f:
                f.write(f"BŁĄD LOGOWANIA: {str(e)}\n")
            return False
    
    # Ustaw handler dla sygnału SIGTERM, aby wyłapać próby zabicia procesu
    def sigterm_handler(signum, frame):
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"OTRZYMANO SYGNAŁ SIGTERM! Runtime={time.time()-start_time:.1f}s\n")
            f.write(f"STACK TRACE:\n{traceback.format_stack()}\n")
        raise SystemExit("Proces przerwany przez sygnał SIGTERM")
    
    old_handler = signal.signal(signal.SIGTERM, sigterm_handler)
    
    # Uruchom wątek logujący
    stop_logging = threading.Event()
    
    def logging_thread():
        while not stop_logging.is_set():
            log_status()
            time.sleep(10)  # Log co 10 sekund
    
    logger_thread = threading.Thread(target=logging_thread)
    logger_thread.daemon = True
    logger_thread.start()
    
    try:
        # Loguj etapy wykonania
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"IMPORTUJĘ MODUŁ PIPELINE\n")
        
        # Importuj funkcję z podmodułu ocr
        from tasks.ocr.pipeline import run_ocr_pipeline as _run_ocr_pipeline
        
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"ROZPOCZYNAM WŁAŚCIWĄ FUNKCJĘ OCR\n")
        
        # Uruchom właściwą implementację
        _run_ocr_pipeline(doc_id)
        
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"FUNKCJA OCR ZAKOŃCZONA POMYŚLNIE\n")
        
    except Exception as e:
        with open("/tmp/ocr_debug.log", "a") as f:
            f.write(f"WYJĄTEK: {type(e).__name__}: {str(e)}\n")
            f.write(f"STACK TRACE:\n{traceback.format_exc()}\n")
        logger.error(f"Błąd podczas uruchamiania OCR: {str(e)}", exc_info=True)
        raise
    
    finally:
        # Zatrzymaj wątek logujący
        stop_logging.set()
        # Przywróć poprzedni handler
        signal.signal(signal.SIGTERM, old_handler)
        
        # Loguj zakończenie
        runtime = time.time() - start_time
        
        try:
            process = psutil.Process(pid)
            mem = process.memory_info()
            with open("/tmp/ocr_debug.log", "a") as f:
                f.write(f"KONIEC OCR dla ID={doc_id}, Runtime={runtime:.1f}s, EXIT_MEM={mem.rss/1024/1024:.2f}MB\n")
                f.write(f"====================\n")
        except:
            with open("/tmp/ocr_debug.log", "a") as f:
                f.write(f"KONIEC OCR dla ID={doc_id}, Runtime={runtime:.1f}s (nie można odczytać pamięci)\n")
                f.write(f"====================\n")
