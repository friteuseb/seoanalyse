class SearchManager {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.searchInput = document.getElementById('nodeSearch');
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        if (this.searchInput) {
            this.searchInput.addEventListener('input', () => this.handleSearch());
        }
    }

    reset() {
        if (this.searchInput) {
            this.searchInput.value = '';
        }
        if (this.graphRenderer && this.graphRenderer.node) {
            this.graphRenderer.node.style('opacity', 1);
            this.graphRenderer.link.style('opacity', 0.6);
        }
    }

    handleSearch() {
        const searchTerm = this.searchInput.value.toLowerCase();
        if (!this.graphRenderer || !this.graphRenderer.node) return;
    
        // Définir le filtre de halo s'il n'existe pas déjà
        const svg = d3.select('#graph svg');
        if (!svg.select('#glow').size()) {
            const filter = svg.append('defs')
                .append('filter')
                .attr('id', 'searchGlow')
                .attr('x', '-50%')
                .attr('y', '-50%')
                .attr('width', '200%')
                .attr('height', '200%');
    
            filter.append('feGaussianBlur')
                .attr('stdDeviation', '3')
                .attr('result', 'coloredBlur');
    
            const feMerge = filter.append('feMerge');
            feMerge.append('feMergeNode')
                .attr('in', 'coloredBlur');
            feMerge.append('feMergeNode')
                .attr('in', 'SourceGraphic');
        }
    
        this.graphRenderer.node.each(function(d) {
            const node = d3.select(this);
            const label = d.label ? d.label.toLowerCase() : '';
            const matches = label.includes(searchTerm);
            
            node.select('circle')
                .style('filter', matches && searchTerm ? 'url(#searchGlow)' : null)
                .style('stroke', matches && searchTerm ? '#2ecc71' : null)
                .style('stroke-width', matches && searchTerm ? '3px' : '1px');
        });
    }
}