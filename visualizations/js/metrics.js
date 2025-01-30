class MetricsManager {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.metricsPanel = document.querySelector('#draggable-metrics-panel');
    }

    updateMetricsDisplay() {
        if (!this.metricsPanel) return;
        
        const metrics = this.calculateMetrics();
        
        try {
            // Mise à jour des valeurs
            this.updateMetricValue('.nodes-count', metrics.nodesCount || '-');
            this.updateMetricValue('.links-count', metrics.linksCount || '-');
            this.updateMetricValue('.density', 
                metrics.density ? `${(metrics.density * 100).toFixed(1)}%` : '-');
            this.updateMetricValue('.avg-depth', 
                metrics.avgDepth ? metrics.avgDepth.toFixed(1) : '-');
            this.updateMetricValue('.orphan-count', 
                `${metrics.orphanCount || 0} (${metrics.orphanPercentage?.toFixed(1)}%)`);
            this.updateMetricValue('.components', metrics.components || '-');
            this.updateMetricValue('.bidirectional', 
                `${metrics.bidirectionalCount || 0} (${metrics.bidirectionalPercentage?.toFixed(1)}%)`);
            this.updateMetricValue('.clustering', 
                metrics.clustering ? metrics.clustering.toFixed(3) : '-');
                
            // Mise à jour du tableau des nœuds les plus connectés
            this.updateTopNodesTable();
            
        } catch (error) {
            console.error('Erreur lors de la mise à jour des métriques:', error);
        }
    }
    
    // Ajoutez cette méthode auxiliaire si elle n'existe pas déjà
    updateMetricValue(selector, value) {
        const element = this.metricsPanel.querySelector(selector);
        if (element) {
            element.textContent = value;
        }
    }

    calculateMetrics() {
        const data = this.graphRenderer.data;
        if (!data || !data.nodes || !data.links) return {};
    
        const nodesCount = data.nodes.length;
        const linksCount = data.links.length;
        
        // Créer un Map pour une recherche plus rapide
        const nodeConnections = new Map();
        
        // Initialiser la Map avec tous les nœuds comme non connectés
        data.nodes.forEach(node => {
            nodeConnections.set(node.id, false);
        });
        
        // Marquer les nœuds connectés
        data.links.forEach(link => {
            const sourceId = link.source.id || link.source;
            const targetId = link.target.id || link.target;
            
            if (nodeConnections.has(sourceId)) {
                nodeConnections.set(sourceId, true);
            }
            if (nodeConnections.has(targetId)) {
                nodeConnections.set(targetId, true);
            }
        });
        
        // Compter les nœuds orphelins
        let orphanCount = 0;
        nodeConnections.forEach(isConnected => {
            if (!isConnected) orphanCount++;
        });
        
        const orphanPercentage = (orphanCount / nodesCount) * 100;
    
        console.log(`Métriques calculées : 
            Nodes total: ${nodesCount}
            Links total: ${linksCount}
            Orphan count: ${orphanCount}
            Orphan %: ${orphanPercentage.toFixed(2)}%`);
    
        return {
            nodesCount,
            linksCount,
            density: this.calculateDensity(data),
            avgDepth: this.calculateAverageDepth(data),
            orphanCount,
            orphanPercentage,
            components: this.countConnectedComponents(data),
            bidirectionalCount: this.countBidirectionalLinks(data),
            bidirectionalPercentage: (linksCount > 0 ? (this.countBidirectionalLinks(data) / linksCount) * 100 : 0),
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



    updateTopNodesTable() {
        // Supprimer la table existante si elle existe
        d3.select("#topNodesTable").remove();

        // Calculer les liens entrants et sortants pour chaque nœud
        const nodeStats = this.calculateNodeStats();
        
        // Créer la table
        const table = d3.select("#controls")
            .append("table")
            .attr("id", "topNodesTable")
            .style("width", "100%")
            .style("margin-top", "20px");

        // En-tête
        table.append("thead")
            .append("tr")
            .selectAll("th")
            .data(["Top liens entrants", "Count", "Top liens sortants", "Count"])
            .enter()
            .append("th")
            .text(d => d);

        // Corps du tableau
        const tbody = table.append("tbody");
        const rows = tbody.selectAll("tr")
            .data(d3.range(5))
            .enter()
            .append("tr");

        // Données des liens entrants
        rows.append("td")
            .text(i => nodeStats.topIncoming[i] ? nodeStats.topIncoming[i].label : "-")
            .style("cursor", "pointer")
            .on("contextmenu", (event, i) => {
                event.preventDefault();
                if (nodeStats.topIncoming[i]) {
                    if (confirm(`Supprimer le nœud "${nodeStats.topIncoming[i].label}" ?`)) {
                        this.graphRenderer.removeNode(nodeStats.topIncoming[i]);
                    }
                }
            });

        rows.append("td")
            .text(i => nodeStats.topIncoming[i] ? nodeStats.topIncoming[i].count : "-");

        // Données des liens sortants
        rows.append("td")
            .text(i => nodeStats.topOutgoing[i] ? nodeStats.topOutgoing[i].label : "-")
            .style("cursor", "pointer")
            .on("contextmenu", (event, i) => {
                event.preventDefault();
                if (nodeStats.topOutgoing[i]) {
                    if (confirm(`Supprimer le nœud "${nodeStats.topOutgoing[i].label}" ?`)) {
                        this.graphRenderer.removeNode(nodeStats.topOutgoing[i]);
                    }
                }
            });

        rows.append("td")
            .text(i => nodeStats.topOutgoing[i] ? nodeStats.topOutgoing[i].count : "-");
    }

    calculateNodeStats() {
        const incomingLinks = new Map();
        const outgoingLinks = new Map();
        
        // Compter les liens entrants et sortants
        this.graphRenderer.data.links.forEach(link => {
            const sourceId = link.source.id;
            const targetId = link.target.id;
            
            outgoingLinks.set(sourceId, (outgoingLinks.get(sourceId) || 0) + 1);
            incomingLinks.set(targetId, (incomingLinks.get(targetId) || 0) + 1);
        });

        // Convertir en tableaux et trier
        const topIncoming = Array.from(incomingLinks.entries())
            .map(([id, count]) => ({
                id,
                label: this.graphRenderer.data.nodes.find(n => n.id === id)?.label || id,
                count
            }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 5);

        const topOutgoing = Array.from(outgoingLinks.entries())
            .map(([id, count]) => ({
                id,
                label: this.graphRenderer.data.nodes.find(n => n.id === id)?.label || id,
                count
            }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 5);

        return { topIncoming, topOutgoing };
    }


}