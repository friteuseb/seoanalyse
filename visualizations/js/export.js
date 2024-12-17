class ExportManager {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.initializeButtons();
    }

    initializeButtons() {
        this.exportSvgButton = document.getElementById('exportSVG');
        this.exportPngButton = document.getElementById('exportPNG');
        this.shareButton = document.getElementById('shareView');
        this.downloadButton = document.getElementById('downloadData');

        if (this.exportSvgButton) {
            this.exportSvgButton.addEventListener('click', () => this.exportSVG());
        }
        if (this.exportPngButton) {
            this.exportPngButton.addEventListener('click', () => this.exportPNG());
        }
        if (this.shareButton) {
            this.shareButton.addEventListener('click', () => this.shareView());
        }
        if (this.downloadButton) {
            this.downloadButton.addEventListener('click', () => this.downloadData());
        }
    }

    prepareSVGForExport(withBackground = true) {
        const originalSvg = this.graphRenderer.container.querySelector('svg');
        const svgClone = originalSvg.cloneNode(true);

        // Ajouter un fond noir si demandé
        if (withBackground) {
            const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            rect.setAttribute("width", "100%");
            rect.setAttribute("height", "100%");
            rect.setAttribute("fill", "#1a1a1a");
            svgClone.insertBefore(rect, svgClone.firstChild);
        } else {
            // Adapter les couleurs pour fond blanc
            svgClone.querySelectorAll('.node-text').forEach(text => {
                text.style.fill = '#000000';
                text.style.textShadow = 'none';
            });
            svgClone.querySelectorAll('.link-line').forEach(link => {
                if (link.getAttribute('stroke') === '#999') {
                    link.setAttribute('stroke', '#666');
                }
            });
            svgClone.querySelectorAll('marker path').forEach(marker => {
                marker.setAttribute('fill', '#666');
            });
        }

        return svgClone;
    }

    exportSVG(withBackground = false) {
        try {
            const svgClone = this.prepareSVGForExport(withBackground);
            const svgData = new XMLSerializer().serializeToString(svgClone);
            const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
            const url = URL.createObjectURL(svgBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = withBackground ? 'graph-dark.svg' : 'graph.svg';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Erreur lors de l\'export SVG:', error);
        }
    }

    exportPNG(withBackground = false) {
        try {
            const svgClone = this.prepareSVGForExport(withBackground);
            const svgData = new XMLSerializer().serializeToString(svgClone);
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            // Créer une image à partir du SVG
            const img = new Image();
            const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
            const url = URL.createObjectURL(svgBlob);

            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                
                // Appliquer le fond si nécessaire
                if (withBackground) {
                    context.fillStyle = '#1a1a1a';
                    context.fillRect(0, 0, canvas.width, canvas.height);
                }
                
                context.drawImage(img, 0, 0);
                
                canvas.toBlob((blob) => {
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = withBackground ? 'graph-dark.png' : 'graph.png';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(url);
                });
            };

            img.src = url;
        } catch (error) {
            console.error('Erreur lors de l\'export PNG:', error);
        }
    }
}