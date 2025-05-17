// Inicjalizacja PDF.js
const pdfjsLib = window['pdfjs-dist/build/pdf'];
pdfjsLib.GlobalWorkerOptions.workerSrc = '/static/js/pdf.worker.js';

// Zmienne globalne
let pdfDoc = null;
let pageNum = 1;
let pageRendering = false;
let pageNumPending = null;
let scale = 1.0;
let canvas = document.getElementById('pdfCanvas');
let ctx = canvas.getContext('2d');

/**
 * Renderuje stronę PDF.
 * @param {Number} num Numer strony.
 */
function renderPage(num) {
  pageRendering = true;
  
  // Aktualizuj UI
  document.getElementById('currentPage').textContent = num;
  
  // Pobierz stronę
  pdfDoc.getPage(num).then(function(page) {
    // Dostosuj skalę aby strona mieściła się w kontenerze
    const viewport = page.getViewport({scale: scale});
    
    // Przygotuj canvas
    canvas.height = viewport.height;
    canvas.width = viewport.width;
    
    // Renderuj PDF na canvas
    const renderContext = {
      canvasContext: ctx,
      viewport: viewport
    };
    
    const renderTask = page.render(renderContext);
    
    // Poczekaj na zakończenie renderowania
    renderTask.promise.then(function() {
      pageRendering = false;
      
      // Dostosuj rozmiar kontenera canvasa
      const container = document.getElementById('pdfCanvasContainer');
      container.style.width = `${viewport.width}px`;
      container.style.height = `${viewport.height}px`;
      
      // Jeśli jest oczekująca strona, renderuj ją
      if (pageNumPending !== null) {
        renderPage(pageNumPending);
        pageNumPending = null;
      }
    });
  });
}

/**
 * Zmienia stronę jeśli nie trwa rendering.
 * W przeciwnym przypadku zaplanuj zmianę na później.
 */
function queueRenderPage(num) {
  if (pageRendering) {
    pageNumPending = num;
  } else {
    renderPage(num);
  }
}

/**
 * Wyświetla poprzednią stronę.
 */
function onPrevPage() {
  if (pageNum <= 1) {
    return;
  }
  pageNum--;
  queueRenderPage(pageNum);
  updateButtonsState();
}

/**
 * Wyświetla następną stronę.
 */
function onNextPage() {
  if (pageNum >= pdfDoc.numPages) {
    return;
  }
  pageNum++;
  queueRenderPage(pageNum);
  updateButtonsState();
}

/**
 * Aktualizuje stan przycisków (włączone/wyłączone)
 */
function updateButtonsState() {
  document.getElementById('prevPage').disabled = pageNum <= 1;
  document.getElementById('nextPage').disabled = pageNum >= pdfDoc.numPages;
}

/**
 * Zmienia poziom powiększenia.
 */
function onZoomChange(zoomIn) {
  if (zoomIn) {
    scale = Math.min(scale * 1.25, 3.0); // Maksymalna skala 300%
  } else {
    scale = Math.max(scale / 1.25, 0.5); // Minimalna skala 50%
  }
  
  // Aktualizuj wyświetlany poziom zoomu
  document.getElementById('zoomLevel').textContent = `${Math.round(scale * 100)}%`;
  
  // Ponownie renderuj aktualną stronę z nową skalą
  queueRenderPage(pageNum);
}

/**
 * Inicjalizuj PDF Viewer.
 */
function initPdfViewer(url) {
  // Pokaż spinner ładowania
  document.getElementById('pdfContent').innerHTML = `
    <div class="d-flex justify-content-center align-items-center h-100">
      <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">Ładowanie PDF...</span>
      </div>
    </div>
  `;
  
  // Załaduj dokument PDF
  pdfjsLib.getDocument(url).promise.then(function(pdf) {
    pdfDoc = pdf;
    document.getElementById('totalPages').textContent = pdf.numPages;
    
    // Przywróć oryginalny kontener PDF
    document.getElementById('pdfContent').innerHTML = `
      <div id="pdfCanvasContainer" class="pdf-canvas-container">
        <canvas id="pdfCanvas"></canvas>
      </div>
    `;
    
    // Odśwież referencje
    canvas = document.getElementById('pdfCanvas');
    ctx = canvas.getContext('2d');
    
    // Renderuj pierwszą stronę
    renderPage(pageNum);
    updateButtonsState();
  }).catch(function(error) {
    // Obsługa błędów
    document.getElementById('pdfContent').innerHTML = `
      <div class="alert alert-danger m-4">
        <i class="bi bi-exclamation-triangle-fill me-2"></i>
        Nie udało się załadować dokumentu PDF: ${error.message}
      </div>
    `;
    console.error('Błąd podczas ładowania PDF:', error);
  });
}

// Inicjalizacja po załadowaniu strony
document.addEventListener('DOMContentLoaded', function() {
  // Pobierz url PDF, który został ustawiony w szablonie
  initPdfViewer(pdfUrl);
  
  // Dodaj obsługę przycisków
  document.getElementById('prevPage').addEventListener('click', onPrevPage);
  document.getElementById('nextPage').addEventListener('click', onNextPage);
  document.getElementById('zoomIn').addEventListener('click', function() { onZoomChange(true); });
  document.getElementById('zoomOut').addEventListener('click', function() { onZoomChange(false); });
});
