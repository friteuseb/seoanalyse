class SearchManager {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.searchInput = document.getElementById('nodeSearch');
        this.filterType = document.getElementById('filterType');
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.searchInput.addEventListener('input', () => this.handleSearch());
        this.filterType.addEventListener('change', () => this.applyFilters());
    }

    handleSearch() {
        const searchTerm = this.searchInput.value.toLowerCase();
        const nodes = this.graphRenderer.node;
        nodes.style('opacity', d => 
            d.label.toLowerCase().includes(searchTerm) ? 1 : 0.1
        );
    }
}