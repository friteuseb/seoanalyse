class MinimapManager {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.scale = 0.15;
        this.minimapSize = 200;
        this.minimapContainer = null;
        this.createMinimapContainer();
    }

    clearMinimap() {
        if (this.minimapContainer) {
            this.minimapContainer.innerHTML = '';
        }
    }

    createMinimapContainer() {
        const existingMinimap = document.querySelector('.minimap-container');
        if (existingMinimap) {
            existingMinimap.remove();
        }

        this.minimapContainer = document.createElement('div');
        this.minimapContainer.className = 'minimap-container';
        
        this.minimapContainer.style.position = 'fixed';
        this.minimapContainer.style.bottom = '20px';
        this.minimapContainer.style.right = '20px';
        this.minimapContainer.style.width = '200px';
        this.minimapContainer.style.height = '200px';
        this.minimapContainer.style.background = 'rgba(0, 0, 0, 0.8)';
        this.minimapContainer.style.border = '1px solid rgba(255, 255, 255, 0.2)';
        this.minimapContainer.style.borderRadius = '8px';
        this.minimapContainer.style.zIndex = '1000';
        
        document.body.appendChild(this.minimapContainer);
    }

    createMinimap() {
        if (!this.graphRenderer.data || !this.minimapContainer) return;
        
        this.clearMinimap();

        if (!this.graphRenderer.data.nodes.some(node => isNaN(node.x) || isNaN(node.y))) {
            const minimapSvg = d3.select(this.minimapContainer)
                .append('svg')
                .attr('width', '100%')
                .attr('height', '100%');

            // Calcul des limites du graphe
            const xExtent = d3.extent(this.graphRenderer.data.nodes, d => d.x);
            const yExtent = d3.extent(this.graphRenderer.data.nodes, d => d.y);
            
            // Calcul des échelles
            const xScale = d3.scaleLinear()
                .domain([xExtent[0], xExtent[1]])
                .range([10, this.minimapSize - 10]);

            const yScale = d3.scaleLinear()
                .domain([yExtent[0], yExtent[1]])
                .range([10, this.minimapSize - 10]);

            // Création d'un groupe pour le contenu
            const minimapG = minimapSvg.append('g');

            // Dessiner les liens
            minimapG.selectAll('line')
                .data(this.graphRenderer.data.links)
                .join('line')
                .attr('stroke', '#666')
                .attr('stroke-width', 0.5)
                .attr('x1', d => xScale(d.source.x))
                .attr('y1', d => yScale(d.source.y))
                .attr('x2', d => xScale(d.target.x))
                .attr('y2', d => yScale(d.target.y));

            // Dessiner les nœuds
            minimapG.selectAll('circle')
                .data(this.graphRenderer.data.nodes)
                .join('circle')
                .attr('r', 2)
                .attr('fill', d => getColor(d.group))
                .attr('cx', d => xScale(d.x))
                .attr('cy', d => yScale(d.y));

            // Viewport
            const viewport = minimapSvg.append('rect')
                .attr('class', 'minimap-viewport')
                .attr('stroke', '#fff')
                .attr('stroke-width', 1)
                .attr('fill', 'none')
                .attr('pointer-events', 'none');

            this.updateViewport();
        }
    }

    updateViewport() {
        if (!this.minimapContainer) return;

        const transform = d3.zoomTransform(this.graphRenderer.container.querySelector('svg'));
        const viewport = d3.select(this.minimapContainer).select('.minimap-viewport');
        
        // Calcul des limites du graphe
        const xExtent = d3.extent(this.graphRenderer.data.nodes, d => d.x);
        const yExtent = d3.extent(this.graphRenderer.data.nodes, d => d.y);
        
        // Échelles pour la transformation
        const xScale = d3.scaleLinear()
            .domain([xExtent[0], xExtent[1]])
            .range([10, this.minimapSize - 10]);

        const yScale = d3.scaleLinear()
            .domain([yExtent[0], yExtent[1]])
            .range([10, this.minimapSize - 10]);
        
        if (!viewport.empty()) {
            const width = Math.abs(xScale(this.graphRenderer.width / transform.k) - xScale(0));
            const height = Math.abs(yScale(this.graphRenderer.height / transform.k) - yScale(0));
            
            viewport
                .attr('x', xScale(-transform.x / transform.k))
                .attr('y', yScale(-transform.y / transform.k))
                .attr('width', width)
                .attr('height', height);
        }
    }
}