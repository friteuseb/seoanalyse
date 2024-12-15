class LayoutManager {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.layoutSelect = document.getElementById('layoutType');
        this.currentLayout = 'force'; // layout par défaut
        this.initializeEventListeners(); 
    }

    initializeEventListeners() {
        if (this.layoutSelect) {
            this.layoutSelect.addEventListener('change', (e) => {
                this.switchLayout(e.target.value);
            });
        } else {
            console.warn("Element layoutType non trouvé dans le DOM");
        }
    }

    switchLayout(type) {
        if (!this.graphRenderer) {
            console.warn("GraphRenderer non initialisé");
            return;
        }

        this.currentLayout = type;
        switch(type) {
            case 'force':
                this.applyForceLayout();
                break;
            case 'radial':
                this.applyRadialLayout();
                break;
            case 'tree':
                this.applyTreeLayout();
                break;
            default:
                console.warn(`Type de layout inconnu: ${type}`);
                break;
        }
    }

    applyForceLayout() {
        if (!this.graphRenderer || !this.graphRenderer.simulation) return;

        this.graphRenderer.simulation
            .force("link", d3.forceLink()
                .id(d => d.id)
                .distance(100))
            .force("charge", d3.forceManyBody().strength(-800))
            .force("center", d3.forceCenter(0, 0))
            .alpha(1)
            .restart();
    }

    applyRadialLayout() {
        if (!this.graphRenderer || !this.graphRenderer.simulation) return;

        const radius = Math.min(this.graphRenderer.width, this.graphRenderer.height) / 3;
        
        this.graphRenderer.simulation
            .force("r", d3.forceRadial(radius))
            .force("charge", d3.forceManyBody().strength(-1000))
            .force("link", d3.forceLink().id(d => d.id).distance(50))
            .alpha(1)
            .restart();
    }

    applyTreeLayout() {
        if (!this.graphRenderer || !this.graphRenderer.simulation) return;

        const width = this.graphRenderer.width;
        const height = this.graphRenderer.height;

        // Arrêter la simulation existante
        this.graphRenderer.simulation.stop();

        // Créer une hiérarchie basique si pas de données parent
        const nodes = this.graphRenderer.data.nodes.map(node => ({
            ...node,
            parent: node.parent || null
        }));

        try {
            const hierarchy = d3.stratify()
                .id(d => d.id)
                .parentId(d => d.parent)(nodes);

            const treeLayout = d3.tree()
                .size([width, height]);

            const root = treeLayout(hierarchy);

            // Mettre à jour les positions
            this.graphRenderer.node
                .transition()
                .duration(750)
                .attr("transform", d => {
                    const nodeData = root.find(n => n.id === d.id);
                    return nodeData ? `translate(${nodeData.x},${nodeData.y})` : '';
                });

            this.graphRenderer.link
                .transition()
                .duration(750)
                .attr("x1", d => {
                    const sourceData = root.find(n => n.id === d.source.id);
                    return sourceData ? sourceData.x : 0;
                })
                .attr("y1", d => {
                    const sourceData = root.find(n => n.id === d.source.id);
                    return sourceData ? sourceData.y : 0;
                })
                .attr("x2", d => {
                    const targetData = root.find(n => n.id === d.target.id);
                    return targetData ? targetData.x : 0;
                })
                .attr("y2", d => {
                    const targetData = root.find(n => n.id === d.target.id);
                    return targetData ? targetData.y : 0;
                });
        } catch (error) {
            console.error("Erreur lors de l'application du layout en arbre:", error);
            this.applyForceLayout(); // Fallback sur le layout force
        }
    }

    setDefaultLayout() {
        this.switchLayout(this.currentLayout);
    }
}