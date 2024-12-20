class MetricsManager {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.metricsPanel = document.querySelector('.metrics-panel');
    }

    updateMetricsDisplay() {
        if (!this.metricsPanel) return;
        
        const metrics = this.calculateMetrics();
        
        // Mise à jour des valeurs
        this.metricsPanel.querySelector('.nodes-count').textContent = metrics.nodesCount || '-';
        this.metricsPanel.querySelector('.links-count').textContent = metrics.linksCount || '-';
        this.metricsPanel.querySelector('.density').textContent = 
            metrics.density ? `${(metrics.density * 100).toFixed(1)}%` : '-';
        this.metricsPanel.querySelector('.avg-depth').textContent = 
            metrics.avgDepth ? metrics.avgDepth.toFixed(1) : '-';
        this.metricsPanel.querySelector('.orphan-count').textContent = 
            `${metrics.orphanCount || 0} (${metrics.orphanPercentage?.toFixed(1)}%)`;
        this.metricsPanel.querySelector('.components').textContent = metrics.components || '-';
        this.metricsPanel.querySelector('.bidirectional').textContent = 
            `${metrics.bidirectionalCount || 0} (${metrics.bidirectionalPercentage?.toFixed(1)}%)`;
        this.metricsPanel.querySelector('.clustering').textContent = 
            metrics.clustering ? metrics.clustering.toFixed(3) : '-';
    }


    updateMetricsDisplay() {
        const metrics = this.calculateMetrics();
        
        if (this.metricsPanel) {
            // Mise à jour des valeurs
            this.metricsPanel.querySelector('.nodes-count').textContent = metrics.nodesCount || '-';
            this.metricsPanel.querySelector('.links-count').textContent = metrics.linksCount || '-';
            this.metricsPanel.querySelector('.density').textContent = 
                metrics.density ? `${(metrics.density * 100).toFixed(1)}%` : '-';
            this.metricsPanel.querySelector('.avg-depth').textContent = 
                metrics.avgDepth ? metrics.avgDepth.toFixed(1) : '-';
            this.metricsPanel.querySelector('.orphan-count').textContent = 
                `${metrics.orphanCount || 0} (${metrics.orphanPercentage?.toFixed(1)}%)`;
            this.metricsPanel.querySelector('.components').textContent = metrics.components || '-';
            this.metricsPanel.querySelector('.bidirectional').textContent = 
                `${metrics.bidirectionalCount || 0} (${metrics.bidirectionalPercentage?.toFixed(1)}%)`;
            this.metricsPanel.querySelector('.clustering').textContent = 
                metrics.clustering ? metrics.clustering.toFixed(3) : '-';
        }
    }

    calculateMetrics() {
        const data = this.graphRenderer.data;
        if (!data || !data.nodes || !data.links) return {};

        const nodesCount = data.nodes.length;
        const linksCount = data.links.length;
        
        // Calcul des liens bidirectionnels
        const bidirectionalLinks = this.countBidirectionalLinks(data);
        
        // Calcul des pages isolées
        const orphanNodes = data.nodes.filter(node => 
            !data.links.some(link => 
                link.source.id === node.id || link.target.id === node.id
            )
        );

        return {
            nodesCount,
            linksCount,
            density: this.calculateDensity(data),
            avgDepth: this.calculateAverageDepth(data),
            orphanCount: orphanNodes.length,
            orphanPercentage: (orphanNodes.length / nodesCount) * 100,
            components: this.countConnectedComponents(data),
            bidirectionalCount: bidirectionalLinks,
            bidirectionalPercentage: (bidirectionalLinks / linksCount) * 100,
            clustering: this.calculateClustering(data)
        };
    }

    countBidirectionalLinks(data) {
        let count = 0;
        const linkMap = new Map();
        
        data.links.forEach(link => {
            const key1 = `${link.source.id}-${link.target.id}`;
            const key2 = `${link.target.id}-${link.source.id}`;
            
            if (!linkMap.has(key1)) {
                linkMap.set(key1, true);
            }
            if (linkMap.has(key2)) {
                count++;
            }
        });
        
        return count;
    }


    calculateDensity(data) {
        const n = data.nodes.length;
        const m = data.links.length;
        // Densité = nombre de liens / nombre de liens possibles
        return n > 1 ? (2 * m) / (n * (n - 1)) : 0;
    }

    calculateAverageDepth(data) {
        // Créer un graphe pour le calcul des chemins
        const graph = new Map();
        data.nodes.forEach(node => {
            graph.set(node.id, new Set());
        });
        data.links.forEach(link => {
            graph.get(link.source.id).add(link.target.id);
            graph.get(link.target.id).add(link.source.id);
        });

        // Calculer la profondeur moyenne
        let totalDepth = 0;
        let totalPaths = 0;

        data.nodes.forEach(startNode => {
            const visited = new Set();
            const queue = [[startNode.id, 0]];
            visited.add(startNode.id);

            while (queue.length > 0) {
                const [currentId, depth] = queue.shift();
                totalDepth += depth;
                totalPaths++;

                graph.get(currentId).forEach(neighborId => {
                    if (!visited.has(neighborId)) {
                        visited.add(neighborId);
                        queue.push([neighborId, depth + 1]);
                    }
                });
            }
        });

        return totalPaths > 0 ? totalDepth / totalPaths : 0;
    }

    calculateClustering(data) {
        let totalCoefficient = 0;
        const nodeNeighbors = new Map();

        // Construire la liste des voisins pour chaque nœud
        data.nodes.forEach(node => {
            nodeNeighbors.set(node.id, new Set());
        });

        data.links.forEach(link => {
            nodeNeighbors.get(link.source.id).add(link.target.id);
            nodeNeighbors.get(link.target.id).add(link.source.id);
        });

        // Calculer le coefficient de clustering pour chaque nœud
        data.nodes.forEach(node => {
            const neighbors = nodeNeighbors.get(node.id);
            if (neighbors.size < 2) {
                totalCoefficient += 0;
                return;
            }

            let triangles = 0;
            const neighborArray = Array.from(neighbors);
            
            for (let i = 0; i < neighborArray.length; i++) {
                for (let j = i + 1; j < neighborArray.length; j++) {
                    if (nodeNeighbors.get(neighborArray[i]).has(neighborArray[j])) {
                        triangles++;
                    }
                }
            }

            const possibleTriangles = (neighbors.size * (neighbors.size - 1)) / 2;
            totalCoefficient += possibleTriangles > 0 ? triangles / possibleTriangles : 0;
        });

        return data.nodes.length > 0 ? totalCoefficient / data.nodes.length : 0;
    }

    countConnectedComponents(data) {
        const visited = new Set();
        let components = 0;

        const dfs = (nodeId) => {
            visited.add(nodeId);
            const neighbors = data.links
                .filter(link => link.source.id === nodeId || link.target.id === nodeId)
                .map(link => link.source.id === nodeId ? link.target.id : link.source.id);

            neighbors.forEach(neighbor => {
                if (!visited.has(neighbor)) {
                    dfs(neighbor);
                }
            });
        };

        data.nodes.forEach(node => {
            if (!visited.has(node.id)) {
                components++;
                dfs(node.id);
            }
        });

        return components;
    }


}