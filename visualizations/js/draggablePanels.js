class DraggablePanels {
    constructor() {
        this.draggingElement = null;
        this.offset = { x: 0, y: 0 };
        this.initializePanels();
    }

    initializePanels() {
        const panels = document.querySelectorAll('.draggable-panel');
        
        panels.forEach(panel => {
            const header = panel.querySelector('.panel-header');
            const minimizeButton = panel.querySelector('.minimize-button');

            // Gestion du drag
            header.addEventListener('mousedown', (e) => this.startDragging(e, panel));
            
            // Gestion du bouton minimize
            minimizeButton.addEventListener('click', () => {
                panel.classList.toggle('panel-minimized');
                minimizeButton.textContent = panel.classList.contains('panel-minimized') ? '+' : '-';
            });
        });

        // Événements globaux
        document.addEventListener('mousemove', (e) => this.onDrag(e));
        document.addEventListener('mouseup', () => this.stopDragging());
    }

    startDragging(e, panel) {
        if (e.button !== 0) return; // Seulement clic gauche
        
        this.draggingElement = panel;
        const rect = panel.getBoundingClientRect();
        
        this.offset = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };

        panel.style.cursor = 'grabbing';
        e.preventDefault(); // Empêcher la sélection de texte
    }

    onDrag(e) {
        if (!this.draggingElement) return;

        const x = e.clientX - this.offset.x;
        const y = e.clientY - this.offset.y;

        // Limiter aux bords de la fenêtre
        const rect = this.draggingElement.getBoundingClientRect();
        const maxX = window.innerWidth - rect.width;
        const maxY = window.innerHeight - rect.height;

        this.draggingElement.style.left = `${Math.min(Math.max(0, x), maxX)}px`;
        this.draggingElement.style.top = `${Math.min(Math.max(0, y), maxY)}px`;
    }

    stopDragging() {
        if (this.draggingElement) {
            this.draggingElement.style.cursor = 'move';
            this.draggingElement = null;
        }
    }
}

// Initialisation au chargement de la page
window.addEventListener('load', () => {
    new DraggablePanels();
});