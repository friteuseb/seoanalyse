class SearchManager {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.searchInput = document.getElementById('nodeSearch');
        if (!this.searchInput) {
            this.createSearchInterface();
        }
        this.initializeEventListeners();
    }

    createSearchInterface() {
        const controls = document.getElementById('controls');
        const searchContainer = document.createElement('div');
        searchContainer.className = 'search-container';
        searchContainer.innerHTML = `
            <input type="text" id="nodeSearch" placeholder="Rechercher une page...">
            <div class="search-results"></div>
        `;
        controls.appendChild(searchContainer);
        this.searchInput = document.getElementById('nodeSearch');
    }

    initializeEventListeners() {
        if (this.searchInput) {
            this.searchInput.addEventListener('input', () => this.handleSearch());
        }
    }

    handleSearch() {
        const searchTerm = this.searchInput.value.toLowerCase();
        if (!this.graphRenderer || !this.graphRenderer.node) return;

        this.graphRenderer.node.style('opacity', d => 
            d.label.toLowerCase().includes(searchTerm) ? 1 : 0.1
        );

        this.graphRenderer.link.style('opacity', d => {
            const sourceMatch = d.source.label.toLowerCase().includes(searchTerm);
            const targetMatch = d.target.label.toLowerCase().includes(searchTerm);
            return (sourceMatch || targetMatch) ? 0.6 : 0.1;
        });
    }

    reset() {
        if (this.searchInput) {
            this.searchInput.value = '';
        }
        if (this.graphRenderer) {
            this.graphRenderer.node?.style('opacity', 1);
            this.graphRenderer.link?.style('opacity', 0.6);
        }
    }
}