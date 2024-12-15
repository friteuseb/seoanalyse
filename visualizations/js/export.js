class ExportManager {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.setupButtons();
    }

    exportSVG() {
        const svg = this.graphRenderer.container.querySelector('svg');
        const svgData = new XMLSerializer().serializeToString(svg);
        this.downloadFile(svgData, 'graph.svg', 'image/svg+xml');
    }

    exportPNG() {
        const svg = this.graphRenderer.container.querySelector('svg');
        const canvas = document.createElement('canvas');
        // Conversion SVG vers PNG
        // ...
    }
}