"""
Przetwarzanie końcowe wyników OCR.
"""
import re
from .config import logger

def clean_ocr_text(text):
    """
    Czyszczenie i formatowanie tekstu OCR, specjalnie 
    dostosowane dla dokumentów sądowych.
    
    Args:
        text: Tekst rozpoznany przez OCR
        
    Returns:
        str: Oczyszczony i sformatowany tekst
    """
    if not text:
        return ""
    
    try:
        # Definicje dla tekstu prawniczego
        legal_terms = ["sąd", "wyrok", "postanowienie", "uzasadnienie", 
                      "art.", "ust.", "pkt.", "zł", "sygn.", "k.p.c.", "k.c."]
        
        # Ustawienia dla dokumentów sądowych
        is_heading = lambda line: (line.isupper() and len(line) > 5) or line.startswith('§') or any(marker in line for marker in ['WYROK', 'POSTANOWIENIE', 'UZASADNIENIE', 'SYGNATURA', 'AKT'])
        is_paragraph_break = lambda line: len(line.strip()) <= 3 or line.strip() in ['-', '–', '—', '*', '•']
        
        # Wstępne czyszczenie
        text = text.replace('\r', '\n')
        
        # Usunięcie nadmiarowych znaków nowej linii i białych znaków
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        
        # Zachowaj numerację punktów/artykułów
        for i, line in enumerate(lines):
            # Jeśli linia wygląda jak numer artykułu lub punktu
            if re.match(r'^[0-9]+[.)]', line) or re.match(r'^[a-z][.)]', line):
                # Dodaj spację po kropce/nawiasie, jeśli jej nie ma
                if re.match(r'^[0-9]+[.)]$', line) and i+1 < len(lines):
                    lines[i] = line + ' ' + lines[i+1]
                    lines[i+1] = ''
        
        # Łączenie linii w paragrafy
        paragraphs = []
        current_lines = []
        
        for line in lines:
            # Pomiń puste linie
            if not line:
                if current_lines:
                    paragraphs.append(' '.join(current_lines))
                    current_lines = []
                continue
                
            # Sprawdź, czy to nagłówek lub specjalny element
            if is_heading(line):
                # Zapisz poprzedni paragraf, jeśli istnieje
                if current_lines:
                    paragraphs.append(' '.join(current_lines))
                    current_lines = []
                # Nagłówki zachowujemy jako oddzielne paragrafy
                paragraphs.append(line)
            elif is_paragraph_break(line):
                # Zapisz poprzedni paragraf i dodaj separator
                if current_lines:
                    paragraphs.append(' '.join(current_lines))
                    current_lines = []
                paragraphs.append(line)  # Zachowaj separator
            elif not current_lines:
                # Start nowego paragrafu
                current_lines.append(line)
            elif line[0].islower() or line[0] in ',;:)]}' or not line[0].isalpha():
                # Ta linia prawdopodobnie jest kontynuacją poprzedniej
                current_lines.append(line)
            elif line.endswith((':', '.', ',', ';', '!', '?', '…', '"')) and len(line) < 50:
                # Krótkie fragmenty zakończone znakami przestankowymi mogą być częścią listy
                if current_lines:
                    paragraphs.append(' '.join(current_lines))
                paragraphs.append(line)
                current_lines = []
            else:
                # W innych przypadkach zaczynamy nowy paragraf
                if current_lines:
                    paragraphs.append(' '.join(current_lines))
                current_lines = [line]
        
        # Dodajemy ostatni paragraf, jeśli istnieje
        if current_lines:
            paragraphs.append(' '.join(current_lines))
        
        # Usuwamy puste paragrafy
        paragraphs = [p for p in paragraphs if p.strip()]
        
        # Poprawiamy artykuły i paragrafy
        for i, paragraph in enumerate(paragraphs):
            # Poprawka dla art. / § 
            paragraphs[i] = re.sub(r'art\s+(\d+)', r'art. \1', paragraph)
            paragraphs[i] = re.sub(r'Art\s+(\d+)', r'Art. \1', paragraphs[i])
            paragraphs[i] = re.sub(r'§\s+(\d+)', r'§ \1', paragraphs[i])
        
        # Łączymy paragrafy z pustą linią między nimi
        result = '\n\n'.join(paragraphs)
        logger.info(f"Oczyszczono tekst: {len(text)} -> {len(result)} znaków")
        return result
        
    except Exception as e:
        logger.error(f"Błąd podczas czyszczenia tekstu: {str(e)}")
        return text  # Zwróć oryginalny tekst w przypadku błędu

def estimate_ocr_confidence(text):
    """
    Szacuje poziom pewności OCR na podstawie heurystyk.
    Funkcja pomocnicza, gdy model nie zwraca pewności.
    
    Args:
        text: Tekst rozpoznany przez OCR
        
    Returns:
        float: Szacowana pewność (0.0-1.0)
    """
    if not text or len(text) < 10:
        return 0.0
    
    # Stosujemy proste heurystyki do oceny jakości OCR
    score = 0.9  # Domyślnie zakładamy wysoką jakość dla modelu Qwen
    
    # Zmniejszamy ocenę, jeśli tekst zawiera znaki zapytania lub nieznane znaki
    unknown_chars = text.count('?') + text.count('�')
    if unknown_chars > 0:
        score -= min(0.3, unknown_chars / len(text) * 5)
    
    # Zmniejszamy ocenę, jeśli są anomalie w długości słów
    words = [w for w in text.split() if w]
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length < 2 or avg_word_length > 15:
            score -= 0.1
    
    # Zmniejszamy ocenę, jeśli tekst jest bardzo krótki w stosunku do długości oczekiwanej
    expected_min_length = 100  # Oczekujemy co najmniej 100 znaków na stronę
    if len(text) < expected_min_length:
        score -= 0.2
    
    return max(0.0, min(1.0, score))  # Normalizacja do przedziału [0.0, 1.0]
