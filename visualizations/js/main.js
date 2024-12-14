function loadAvailableGraphs(graphSelect) {
    fetch('get_available_graphs.php')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            const graphKeys = data.filter(key => 
                key.includes('_simple_graph') || key.includes('_clustered_graph')
            );

            graphSelect.innerHTML = '<option value="">Sélectionnez un graphe</option>';
            
            graphKeys.forEach(graph => {
                const option = document.createElement('option');
                option.value = graph;
                option.textContent = graph;
                graphSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Erreur de chargement:', error);
            showError(`Erreur de chargement des graphes: ${error.message}`);
            graphSelect.innerHTML = '<option value="">Erreur de chargement</option>';
        });
}

function loadGraph(graphId, renderer) {
    if (!graphId) return;

    fetch(`get_graph_data.php?graph=${graphId}`)
        .then(response => response.json())
        .then(data => {
            console.log("Received data from server:", data);
            console.log("Number of nodes:", data.nodes.length);
            console.log("Number of links:", data.links.length);
            
            renderer.render(data);
        })
        .catch(error => {
            console.error('Error loading graph:', error);
            showError(`Erreur de chargement du graphe: ${error.message}`);
        });
}

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', () => {
    const graphRenderer = new GraphRenderer('graph');
    const graphSelect = document.getElementById('graphSelect');
    const loadGraphButton = document.getElementById('loadGraph');

    loadAvailableGraphs(graphSelect);

    graphSelect.addEventListener('change', () => {
        loadGraphButton.disabled = !graphSelect.value;
    });

    loadGraphButton.addEventListener('click', () => {
        loadGraph(graphSelect.value, graphRenderer);
    });

    window.addEventListener('resize', () => {
        graphRenderer.resize();
    });
});