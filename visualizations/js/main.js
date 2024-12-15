class AppManager {
    constructor() {
        // Initialisation des composants de base
        this.graphRenderer = new GraphRenderer('graph');
        this.graphSelect = document.getElementById('graphSelect');
        this.loadGraphButton = document.getElementById('loadGraph');
        this.resetViewButton = document.getElementById('resetView');
        
        // Initialisation des managers de fonctionnalités
        this.metricsManager = new MetricsManager(this.graphRenderer);
        this.searchManager = new SearchManager(this.graphRenderer);
        this.layoutManager = new LayoutManager(this.graphRenderer);
        this.exportManager = new ExportManager(this.graphRenderer);
        this.minimapManager = new MinimapManager(this.graphRenderer);
        
        this.setupEventListeners();
        this.loadAvailableGraphs();
    }

    setupEventListeners() {
        // Évènements de base
        this.graphSelect.addEventListener('change', () => {
            this.loadGraphButton.disabled = !this.graphSelect.value;
        });

        this.loadGraphButton.addEventListener('click', () => {
            this.loadGraph(this.graphSelect.value);
        });

        this.resetViewButton.addEventListener('click', () => {
            this.graphRenderer.resetView();
        });

        window.addEventListener('resize', () => {
            this.graphRenderer.resize();
        });
    }

    loadAvailableGraphs() {
        fetch('get_available_graphs.php')
            .then(response => response.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                
                const graphKeys = data.filter(key => 
                    key.includes('_simple_graph') || key.includes('_clustered_graph')
                );

                this.graphSelect.innerHTML = '<option value="">Sélectionnez un graphe</option>';
                
                graphKeys.forEach(graph => {
                    const option = document.createElement('option');
                    option.value = graph;
                    option.textContent = graph;
                    this.graphSelect.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Erreur de chargement:', error);
                showError(`Erreur de chargement des graphes: ${error.message}`);
                this.graphSelect.innerHTML = '<option value="">Erreur de chargement</option>';
            });
    }

    loadGraph(graphId) {
        if (!graphId) return;

        fetch(`get_graph_data.php?graph=${graphId}`)
            .then(response => response.json())
            .then(data => {
                console.log("Received data from server:", data);
                console.log("Number of nodes:", data.nodes.length);
                console.log("Number of links:", data.links.length);
                
                // Rendu du graphe
                this.graphRenderer.render(data);
                
                // Mise à jour des managers
                this.metricsManager.updateMetricsDisplay();
                this.minimapManager.createMinimap();
                this.searchManager.reset();
                this.layoutManager.setDefaultLayout();
            })
            .catch(error => {
                console.error('Error loading graph:', error);
                showError(`Erreur de chargement du graphe: ${error.message}`);
            });
    }
}

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AppManager();
});