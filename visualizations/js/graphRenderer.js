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
        this.createLegend(svg, data);

        
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
        if (!this.simulation) return;
    
        try {
            // Ajuster les forces de la simulation
            this.simulation
                .force("center", d3.forceCenter(this.width / 2, this.height / 2))
                .force("charge", d3.forceManyBody().strength(-1000)) // Force de répulsion plus forte
                .force("collision", d3.forceCollide().radius(40)) // Éviter le chevauchement
                .force("x", d3.forceX(this.width / 2).strength(0.05)) // Force plus faible vers le centre X
                .force("y", d3.forceY(this.height / 2).strength(0.05)); // Force plus faible vers le centre Y
    
            this.simulation.on("tick", () => {
                link.each(function(d) {
                    const sourceRadius = d3.select(d.source.node).select('circle').attr('r');
                    const targetRadius = d3.select(d.target.node).select('circle').attr('r');
    
                    const deltaX = d.target.x - d.source.x;
                    const deltaY = d.target.y - d.source.y;
                    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    
                    if (distance === 0) return;
    
                    const sourceX = d.source.x + (deltaX * sourceRadius) / distance;
                    const sourceY = d.source.y + (deltaY * sourceRadius) / distance;
                    const targetX = d.target.x - (deltaX * targetRadius) / distance;
                    const targetY = d.target.y - (deltaY * targetRadius) / distance;
    
                    d3.select(this)
                        .attr("x1", sourceX)
                        .attr("y1", sourceY)
                        .attr("x2", targetX)
                        .attr("y2", targetY);
                });
    
                // Permettre aux nœuds de sortir légèrement des limites
                node.attr("transform", d => {
                    const padding = 100; // Zone de dépassement autorisée
                    const x = Math.max(-padding, Math.min(this.width + padding, d.x || 0));
                    const y = Math.max(-padding, Math.min(this.height + padding, d.y || 0));
                    return `translate(${x},${y})`;
                });
            });
        } catch (error) {
            console.error("Error in setupSimulationForces:", error);
        }
    }
    
    createLinks(g, links) {
        return g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("stroke", "#2f4f4f")  // Couleur plus profonde pour les liens
            .attr("stroke-opacity", 0.3) // Plus transparent
            .attr("stroke-width", d => Math.sqrt(d.value))
            .attr("class", "link-line")  // Pour l'animation
            .attr("marker-end", "url(#arrowhead)");
    }
    addArrowMarkers(svg) {
        svg.selectAll("defs").remove();
        
        svg.append("defs")
            .append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-10 -5 20 10")
            .attr("refX", 15)  // Ajusté pour correspondre au rayon du nœud
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M -10,-5 L 0,0 L -10,5")
            .attr("fill", "#999");
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
    
    addArrowMarkers(svg) {
        svg.selectAll("defs").remove();
        
        svg.append("defs")
            .append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-10 -5 20 10")
            .attr("refX", 15)  // Ajusté pour correspondre au rayon du nœud
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M -10,-5 L 0,0 L -10,5")
            .attr("fill", "#999");
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
            .attr("marker-end", "url(#arrowhead)")
            // Ajustement des points de départ/arrivée pour tenir compte de la taille des nœuds
            .attr("x1", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                return d.source.x + (dx * 20) / dr;
            })
            .attr("y1", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                return d.source.y + (dy * 20) / dr;
            })
            .attr("x2", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                return d.target.x - (dx * 20) / dr;
            })
            .attr("y2", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);
                return d.target.y - (dy * 20) / dr;
            });
    }

        dragstarted = (event) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }

        dragged = (event) => {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        dragended = (event) => {
            if (!event.active) this.simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        createNodes(g, nodes, nodeScale, incomingLinksCount) {
            const node = g.append("g")
                .attr("class", "nodes")
                .selectAll("g")
                .data(nodes)
                .join("g")
                .each(function(d) {
                    // Stocker une référence au nœud DOM
                    d.node = this;
                })
                .call(d3.drag()
                    .on("start", this.dragstarted)
                    .on("drag", this.dragged)
                    .on("end", this.dragended))
                .on("contextmenu", (event, d) => {
                    event.preventDefault();
                    if (confirm(`Supprimer le nœud "${d.label}" ?`)) {
                        this.removeNode(d);
                    }
                })
                .on("mouseover", (event, d) => this.tooltipManager.show(event, d, incomingLinksCount))
                .on("mouseout", () => this.tooltipManager.hide());
        
                // Ajout des cercles avec effet de lueur
            node.append("circle")
                .attr("r", d => nodeScale(incomingLinksCount[d.id] || 0))
                .attr("fill", d => getColor(d.group))
                .attr("stroke", d => d3.color(getColor(d.group)).brighter(0.5))
                .attr("stroke-width", 2)
                .style("filter", "url(#glow)"); // Effet de lueur
        
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

        createLegend(svg, data) {
            const legendWidth = 200;
            const legendPadding = 10;
            
            // Créer le tooltip
            const tooltip = d3.select("body").append("div")
                .attr("class", "legend-tooltip")
                .style("position", "absolute")
                .style("visibility", "hidden")
                .style("background-color", "rgba(0, 0, 0, 0.9)")
                .style("padding", "8px")
                .style("border-radius", "4px")
                .style("color", "white")
                .style("max-width", "300px")
                .style("z-index", "1000");
        
            const legend = svg.append("g")
                .attr("class", "legend")
                .attr("transform", `translate(10, 20)`); // Positionner à gauche plutôt qu'à droite
        
            const uniqueClusters = [...new Set(data.nodes.map(n => n.group))].sort((a, b) => a - b);
        
            // Fond semi-transparent
            const background = legend.append("rect")
                .attr("fill", "rgba(0, 0, 0, 0.7)")
                .attr("rx", 5)
                .attr("ry", 5);
        
            const legendItems = legend.selectAll(".legend-item")
                .data(uniqueClusters)
                .enter()
                .append("g")
                .attr("class", "legend-item")
                .attr("transform", (d, i) => `translate(${legendPadding}, ${20 + i * 25})`)
                .style("cursor", "pointer")
                .on("mouseover", function(event, d) {
                    const description = d === -1 ? "Pages non classées" : 
                                      data.nodes.find(n => n.group === d)?.title || `Cluster ${d}`;
                    tooltip.style("visibility", "visible")
                        .html(description)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                })
                .on("mouseout", function() {
                    tooltip.style("visibility", "hidden");
                });
        
            // Carré de couleur
            legendItems.append("rect")
                .attr("width", 15)
                .attr("height", 15)
                .attr("fill", d => getColor(d));
        
            // Texte court pour la légende
            legendItems.append("text")
                .attr("x", 25)
                .attr("y", 12)
                .attr("fill", "white")
                .text(d => {
                    const count = data.nodes.filter(n => n.group === d).length;
                    return `Cluster ${d} (${count})`;
                })
                .each(function() {
                    // Tronquer le texte si nécessaire
                    const textWidth = legendWidth - 40;
                    const text = d3.select(this);
                    if (this.getComputedTextLength() > textWidth) {
                        let textContent = text.text();
                        while (this.getComputedTextLength() > textWidth) {
                            textContent = textContent.slice(0, -1);
                            text.text(textContent + "...");
                        }
                    }
                });
        
            // Ajuster la taille du fond
            const bbox = legend.node().getBBox();
            background
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", legendWidth)
                .attr("height", bbox.height + legendPadding * 2);
        }

        removeNode(nodeToRemove) {
            if (!this.data) return;
        
            try {
                this.data.nodes = this.data.nodes.filter(n => n.id !== nodeToRemove.id);
                this.data.links = this.data.links.filter(l => 
                    l.source.id !== nodeToRemove.id && l.target.id !== nodeToRemove.id
                );
        
                // Mise à jour de la simulation
                this.simulation.nodes(this.data.nodes);
                this.simulation.force("link").links(this.data.links);
                
                // Redémarrage plus progressif de la simulation
                this.simulation
                    .alpha(0.5) // Augmentation de l'alpha pour une meilleure réorganisation
                    .alphaTarget(0)
                    .alphaDecay(0.02) // Ralentissement de la décroissance
                    .restart();
        
                // Mise à jour du DOM
                const nodes = d3.select("#" + this.container.id)
                    .selectAll(".nodes g")
                    .data(this.data.nodes, d => d.id);
                    
                nodes.exit().remove();
        
                const links = d3.select("#" + this.container.id)
                    .selectAll(".links line")
                    .data(this.data.links, d => `${d.source.id}-${d.target.id}`);
                    
                links.exit().remove();
        
            } catch (error) {
                console.error("Error removing node:", error);
            }
        }

        addArrowMarkers(svg) {
            // Ajouter un filtre pour l'effet de lueur
            const defs = svg.append("defs");
            
            const filter = defs.append("filter")
                .attr("id", "glow")
                .attr("height", "300%")
                .attr("width", "300%")
                .attr("x", "-100%")
                .attr("y", "-100%");
                
            filter.append("feGaussianBlur")
                .attr("stdDeviation", "2")
                .attr("result", "coloredBlur");
                
            const feMerge = filter.append("feMerge");
            feMerge.append("feMergeNode")
                .attr("in", "coloredBlur");
            feMerge.append("feMergeNode")
                .attr("in", "SourceGraphic");
        
            // Marqueur de flèche amélioré
            defs.append("marker")
                .attr("id", "arrowhead")
                .attr("viewBox", "-10 -5 20 10")
                .attr("refX", 15)
                .attr("refY", 0)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M -10,-5 L 0,0 L -10,5")
                .attr("fill", "#2f4f4f")
                .style("opacity", 0.6);
        }

    resize() {
        if (this.data) {
            this.width = this.container.clientWidth;
            this.height = this.container.clientHeight;
            this.render(this.data);
        }
    }
}