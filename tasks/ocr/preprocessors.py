"""
Przetwarzanie wstępne dokumentów przed OCR.
"""
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .config import logger, DPI

def preprocess_image(image_path):
    """
    Przetwarza obraz, aby poprawić wyniki OCR.
    
    Args:
        image_path: Ścieżka do pliku obrazu
        
    Returns:
        str: Ścieżka do przetworzonego obrazu
    """
    try:
        # Wczytaj obraz
        image = Image.open(image_path)
        
        # Konwersja do RGB jeśli obraz ma kanał alpha
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Skalowanie obrazu jeśli jest zbyt mały
        width, height = image.size
        min_dimension = 1000  # Minimalna szerokość lub wysokość
        if width < min_dimension or height < min_dimension:
            scale_factor = min_dimension / min(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Przeskalowano obraz z {width}x{height} do {new_width}x{new_height}")
        
        # Zapisz przetworzony obraz
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            tmp_path = tmp_img.name
        
        image.save(tmp_path, "PNG")
        return tmp_path
        
    except Exception as e:
        logger.error(f"Błąd podczas przetwarzania obrazu: {str(e)}")
        return str(image_path)  # Zwróć oryginalną ścieżkę w przypadku błędu

def extract_pages_from_pdf(pdf_path, max_batch_size=5):
    """
    Konwertuje PDF na listy obrazów stron podzielone na partie.
    
    Args:
        pdf_path: Ścieżka do pliku PDF
        max_batch_size: Maksymalna liczba stron w partii
        
    Returns:
        list: Lista list ścieżek do obrazów stron (pogrupowane w partie)
    """
    try:
        from pdf2image import convert_from_path
        
        # Konwertuj PDF na obrazy
        pages = convert_from_path(pdf_path, dpi=DPI)
        total_pages = len(pages)
        logger.info(f"Wyodrębniono {total_pages} stron z PDF")
        
        # Podziel strony na partie
        batches = []
        for i in range(0, total_pages, max_batch_size):
            batch_pages = pages[i:i+max_batch_size]
            
            # Zapisz strony z tej partii jako obrazy
            batch_paths = []
            for j, img in enumerate(batch_pages):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                    tmp_path = tmp_img.name
                
                img.save(tmp_path, "PNG")
                batch_paths.append(tmp_path)
            
            batches.append(batch_paths)
            
        return batches
        
    except Exception as e:
        logger.error(f"Błąd podczas konwersji PDF na obrazy: {str(e)}")
        return []
