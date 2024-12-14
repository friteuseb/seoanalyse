class GraphRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        this.simulation = null;
        this.tooltipManager = new TooltipManager();
        this.filterManager = null;
        this.data = null;
    }


    render(data) {
        if (!data || !data.nodes || !data.links) {
            console.error("Invalid data provided to render:", data);
            return;
        }
        
        this.data = data;
        console.log("Starting render with data:", data);
    
        // Nettoyage et création du SVG
        const svg = this.initializeSVG();
        const g = svg.append("g");
        
        // Ajout des marqueurs de flèches
        this.addArrowMarkers(svg);
    
        // Initialiser la simulation en PREMIER
        this.setupSimulation(data);
        
        // Création des éléments
        const elements = this.createElements(g, data);
        
        // Configuration des forces et démarrage de la simulation
        if (this.simulation) {
            this.setupSimulationForces(elements.node, elements.link);
            this.simulation.alpha(1).restart();
        } else {
            console.error("Simulation not properly initialized");
        }
        
        // Initialisation des filtres
        this.filterManager = new FilterManager(elements.node, elements.link, data);
    }
    
    setupSimulation(data) {
        console.log("Setting up simulation with", data.nodes.length, "nodes and", data.links.length, "links");
        
        // Arrêter la simulation existante si elle existe
        if (this.simulation) {
            this.simulation.stop();
        }
    
        try {
            this.simulation = d3.forceSimulation()
                .nodes(data.nodes)
                .force("link", d3.forceLink()
                    .id(d => d.id)
                    .links(data.links)
                    .distance(100)
                    .strength(1))
                .force("charge", d3.forceManyBody()
                    .strength(-2000))
                .force("x", d3.forceX(this.width / 2).strength(0.1))
                .force("y", d3.forceY(this.height / 2).strength(0.1))
                .force("collide", d3.forceCollide().radius(30))
                .velocityDecay(0.6)
                .alpha(1)
                .alphaTarget(0);
    
            console.log("Simulation initialized:", this.simulation);
            return this.simulation;
        } catch (error) {
            console.error("Error initializing simulation:", error);
            return null;
        }
    }
    
    setupSimulationForces(node, link) {
        if (!this.simulation) {
            console.error("Simulation not initialized in setupSimulationForces");
            return;
        }
    
        try {
            this.simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
    
                node.attr("transform", d => {
                    const x = Math.max(20, Math.min(this.width - 20, d.x || 0));
                    const y = Math.max(20, Math.min(this.height - 20, d.y || 0));
                    return `translate(${x},${y})`;
                });
            });
        } catch (error) {
            console.error("Error in setupSimulationForces:", error);
        }
    }

    initializeSVG() {
        d3.select("#" + this.container.id).selectAll("*").remove();
        const svg = d3.select("#" + this.container.id)
            .append("svg")
            .attr("width", this.width)
            .attr("height", this.height);

        svg.call(d3.zoom()
            .extent([[0, 0], [this.width, this.height]])
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => this.handleZoom(event)));

        return svg;
    }

    handleZoom(event) {
        d3.select("#" + this.container.id)
            .select("g")
            .attr("transform", event.transform);
    }



    createElements(g, data) {
        // Calcul du nombre de liens entrants
        const incomingLinksCount = this.calculateIncomingLinks(data.links);

        // Création de l'échelle pour la taille des nœuds
        const nodeScale = d3.scaleLinear()
            .domain([0, d3.max(Object.values(incomingLinksCount)) || 1])
            .range([CONFIG.nodeMinSize, CONFIG.nodeMaxSize]);

        // Création des liens
        const link = this.createLinks(g, data.links);

        // Création des nœuds
        const node = this.createNodes(g, data.nodes, nodeScale, incomingLinksCount);

        // Configuration de la simulation
        this.setupSimulationForces(node, link);

        return { node, link };
    }

    calculateIncomingLinks(links) {
        const incomingLinksCount = {};
        links.forEach(link => {
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            incomingLinksCount[targetId] = (incomingLinksCount[targetId] || 0) + 1;
        });
        return incomingLinksCount;
    }

    createLinks(g, links) {
        return g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", d => Math.sqrt(d.value))
            .attr("marker-end", "url(#arrowhead)");
    }

    createNodes(g, nodes, nodeScale, incomingLinksCount) {
        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            .data(nodes)
            .join("g")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("contextmenu", (event, d) => {
                event.preventDefault();
                if (confirm(`Supprimer le nœud "${d.label}" ?`)) {
                    this.removeNode(d);
                }
            })
            .on("mouseover", (event, d) => this.tooltipManager.show(event, d, incomingLinksCount))
            .on("mouseout", () => this.tooltipManager.hide());

        // Ajout des cercles
        node.append("circle")
            .attr("r", d => nodeScale(incomingLinksCount[d.id] || 0))
            .attr("fill", d => getColor(d.group))
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5);

        // Ajout des labels
        node.append("text")
            .attr("dx", d => nodeScale(incomingLinksCount[d.id] || 0) + 5)
            .attr("dy", ".35em")
            .text(d => d.label)
            .attr("class", "node-text")
            .style("font-size", "10px")
            .style("fill", "#fff");

        return node;
    }


    removeNode(nodeToRemove) {
        if (!this.data) return;

        // Mettre à jour les données
        this.data.nodes = this.data.nodes.filter(n => n.id !== nodeToRemove.id);
        this.data.links = this.data.links.filter(l => 
            l.source.id !== nodeToRemove.id && l.target.id !== nodeToRemove.id
        );

        // Redessiner le graphe
        this.render(this.data);
    }

    addArrowMarkers(svg) {
        svg.append("defs").selectAll("marker")
            .data(["end"])
            .enter().append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 15)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .append("svg:path")
            .attr("d", "M 0,-5 L 10 ,0 L 0,5")
            .attr("fill", "#999");
    }

    resize() {
        if (this.data) {
            this.width = this.container.clientWidth;
            this.height = this.container.clientHeight;
            this.render(this.data);
        }
    }
}