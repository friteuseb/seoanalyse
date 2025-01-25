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
        // Passer le graphRenderer à la minimap
        this.minimapManager = new MinimapManager(this.graphRenderer);
        // Ajouter la référence dans le graphRenderer
        this.graphRenderer.minimapManager = this.minimapManager;
        
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

        window.addEventListener('resize', () => {
            this.graphRenderer.resize();
        });
    }

    loadAvailableGraphs() {
        fetch('get_available_graphs.php')
        .then(response => response.json())
        .then(data => {
            if (data.success && Array.isArray(data.graphs)) {
                // Utiliser data.graphs au lieu de data directement
                const graphSelect = document.getElementById('graphSelect');
                graphSelect.innerHTML = '';
                
                // Ajouter une option par défaut
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = 'Sélectionnez un graphe...';
                graphSelect.appendChild(defaultOption);

                data.graphs.forEach(graph => {
                    const option = document.createElement('option');
                    option.value = graph.id + '_' + graph.type + '_graph';
                    
                    // Construire un libellé informatif
                    let label = `${graph.id.split('___')[0]} `;
                    label += `(${graph.type})`;
                    if (graph.created) {
                        label += ` - ${new Date(graph.created).toLocaleDateString()}`;
                    }
                    if (graph.nodes && graph.links) {
                        label += ` - ${graph.nodes} nœuds, ${graph.links} liens`;
                    }
                    
                    option.textContent = label;
                    graphSelect.appendChild(option);
                });
            } else {
                throw new Error('Invalid graph data format');
            }
        })
        .catch(error => {
            console.error('Erreur de chargement des graphes:', error);
            showError('Erreur lors du chargement des graphes disponibles');
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
                this.minimapManager.createMinimap(); 

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