class SearchManager {
    constructor(graphRenderer) {
        console.log("SearchManager: Initialisation...");
        this.graphRenderer = graphRenderer;
        this.searchInput = document.getElementById('nodeSearch');
        console.log("SearchManager: Input trouvé:", !!this.searchInput);
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        console.log("SearchManager: Configuration des événements...");
        if (this.searchInput) {
            this.searchInput.addEventListener('input', () => {
                console.log("SearchManager: Événement de recherche déclenché");
                this.handleSearch();
            });
        } else {
            console.warn("SearchManager: Élément de recherche non trouvé");
        }
    }

    reset() {
        if (this.searchInput) {
            this.searchInput.value = '';
        }
        if (this.graphRenderer && this.graphRenderer.node) {
            this.graphRenderer.node.each(function() {
                const node = d3.select(this);
                const circle = node.select('circle');
                // Restaurer les couleurs et styles originaux
                circle
                    .attr("fill", "#1E90FF")
                    .attr("stroke", "#104E8B")
                    .attr("stroke-width", "2")
                    .style("filter", "url(#glow)");
                node.style('opacity', 1);
            });
            
            if (this.graphRenderer.link) {
                this.graphRenderer.link.style('opacity', 0.6);
            }
        }
    }

    
    handleSearch() {
        const searchTerm = this.searchInput.value.toLowerCase();
        if (!this.graphRenderer || !this.graphRenderer.node) return;
    
        this.graphRenderer.node.each(function(d) {
            const node = d3.select(this);
            const circle = node.select('circle');
            const label = d.label ? d.label.toLowerCase() : '';
            const matches = label.includes(searchTerm);
            
            if (matches && searchTerm) {
                // Style pour les résultats de recherche
                circle
                    .attr("fill", "#ffeb3b")
                    .attr("stroke", "#ffd700")
                    .attr("stroke-width", "3")
                    .style("filter", "url(#searchGlow)");
            } else {
                // Restaurer les couleurs originales
                circle
                    .attr("fill", "#1E90FF")
                    .attr("stroke", "#104E8B")
                    .attr("stroke-width", "2")
                    .style("filter", "url(#glow)");
            }
            
            node.style('opacity', matches || !searchTerm ? 1 : 0.2);
        });
    
        if (this.graphRenderer.link) {
            this.graphRenderer.link.style('opacity', l => {
                const sourceMatch = l.source.label.toLowerCase().includes(searchTerm);
                const targetMatch = l.target.label.toLowerCase().includes(searchTerm);
                return (sourceMatch || targetMatch || !searchTerm) ? 0.6 : 0.1;
            });
        }
    }
}