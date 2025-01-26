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
        if (!this.graphRenderer || !this.graphRenderer.data || !this.minimapContainer) return;
    
        // Nettoyer la minimap existante
        this.clearMinimap();
    
        // Calculer les limites du graphe
        const xExtent = d3.extent(this.graphRenderer.data.nodes, d => d.x);
        const yExtent = d3.extent(this.graphRenderer.data.nodes, d => d.y);
    
        // Définir les échelles
        const xScale = d3.scaleLinear()
            .domain([xExtent[0], xExtent[1]])
            .range([10, this.minimapSize - 10]);
    
        const yScale = d3.scaleLinear()
            .domain([yExtent[0], yExtent[1]])
            .range([10, this.minimapSize - 10]);
    
        // Créer le SVG de la minimap
        const minimapSvg = d3.select(this.minimapContainer)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%');
    
        // Dessiner les liens
        minimapSvg.selectAll('line')
            .data(this.graphRenderer.data.links)
            .join('line')
            .attr('stroke', '#666')
            .attr('stroke-width', 0.5)
            .attr('x1', d => xScale(d.source.x))
            .attr('y1', d => yScale(d.source.y))
            .attr('x2', d => xScale(d.target.x))
            .attr('y2', d => yScale(d.target.y));
    
        // Dessiner les nœuds
        minimapSvg.selectAll('circle')
            .data(this.graphRenderer.data.nodes)
            .join('circle')
            .attr('r', 2)
            .attr('fill', d => getColor(d.group))
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y));
    
        // Ajouter le viewport
        minimapSvg.append('rect')
            .attr('class', 'minimap-viewport')
            .attr('stroke', '#fff')
            .attr('stroke-width', 1)
            .attr('fill', 'none')
            .attr('pointer-events', 'none');
    
        // Mettre à jour le viewport après un court délai
        setTimeout(() => {
            this.updateViewport();
        }, 10000); // Délai de 100 ms
    }

    updateViewport() {
        if (!this.minimapContainer || !this.graphRenderer || !this.graphRenderer.data) {
            console.error("Données ou conteneur manquants pour la mise à jour du viewport");
            return;
        }
    
        const transform = d3.zoomTransform(this.graphRenderer.container.querySelector('svg'));
        const viewport = d3.select(this.minimapContainer).select('.minimap-viewport');
    
        if (viewport.empty()) {
            console.error("Viewport non trouvé dans la minimap");
            return;
        }
    
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
    
        const width = Math.abs(xScale(this.graphRenderer.width / transform.k) - xScale(0));
        const height = Math.abs(yScale(this.graphRenderer.height / transform.k) - yScale(0));
    
        viewport
            .attr('x', xScale(-transform.x / transform.k))
            .attr('y', yScale(-transform.y / transform.k))
            .attr('width', width)
            .attr('height', height);
    }
}