class MinimapManager {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.minimap = document.getElementById('minimap');
        this.scale = 0.15; // échelle de la minimap
    }

    createMinimap() {
        const minimapSvg = d3.select(this.minimap)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%');
            
        // Créer version réduite du graphe
        // ...
    }

    updateViewport() {
        // Mettre à jour le rectangle de visualisation
        // ...
    }
}