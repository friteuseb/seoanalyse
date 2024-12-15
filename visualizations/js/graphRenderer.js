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
        this.data = data;
        const svg = this.initializeSVG();
        const g = svg.select("g");  // Sélectionne le groupe créé dans initializeSVG
        
        this.addArrowMarkers(svg);
        
        // Configuration de la simulation avant création des éléments
        this.setupSimulation(data);
        
        // Création des éléments
        const elements = this.createElements(g, data);
        
        // Configuration des forces et démarrage
        this.setupSimulationForces(elements.node, elements.link);
        this.simulation.alpha(1).restart();
        
        // Création de la légende
        this.createLegend(svg, data);
        
        // Configuration des filtres
        this.filterManager = new FilterManager(elements.node, elements.link, data);
    }


    
    setupSimulation(data) {
        if (this.simulation) {
            this.simulation.stop();
        }

        const centerX = 0;
        const centerY = 0;
        
        this.simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink()
                .id(d => d.id)
                .links(data.links)
                .distance(100))
            .force("charge", d3.forceManyBody()
                .strength(-800))
            .force("center", d3.forceCenter(centerX, centerY))
            .force("collide", d3.forceCollide().radius(30))
            .velocityDecay(0.6);

        return this.simulation;
    }

    setupSimulationForces(node, link) {
        if (!this.simulation) return;

        this.simulation.on("tick", () => {
            // Mise à jour des positions des liens
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            // Mise à jour des positions des nœuds
            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });
    }


    setupSimulationForces(node, link) {
        if (!this.simulation) return;
    
        try {
            // Ajuster les forces pour utiliser tout l'espace disponible
            this.simulation
                .force("center", d3.forceCenter(0, 0))
                .force("charge", d3.forceManyBody()
                    .strength(d => -2000 * Math.sqrt(this.data.nodes.length / 100)))
                .force("collision", d3.forceCollide().radius(40))
                // Retirer les forces de contrainte x et y
                .force("x", null)
                .force("y", null);
    
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
    
                // Ne plus contraindre les nœuds aux limites
                node.attr("transform", d => `translate(${d.x},${d.y})`);
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
      

    initializeSVG() {
        d3.select("#" + this.container.id).selectAll("*").remove();
        
        // Création du conteneur de légende d'abord
        const container = d3.select("#" + this.container.id);
        container.append("div")
            .attr("class", "legend-container")
            .style("position", "absolute")
            .style("top", "20px")
            .style("right", "20px")
            .style("background", "rgba(0, 0, 0, 0.8)")
            .style("border-radius", "10px")
            .style("padding", "15px")
            .style("z-index", "1000");
        
        // Création du SVG principal
        const svg = container.append("svg")
            .attr("width", this.width)
            .attr("height", this.height);
        
        // Groupe principal pour le graphe
        const g = svg.append("g");
        
        // Configuration du zoom
        const zoom = d3.zoom()
            .scaleExtent([0.2, 4])  // Limites du zoom
            .on("zoom", (event) => {
                g.attr("transform", event.transform);
            });
        
        // Application du zoom avec transformation initiale
        svg.call(zoom);
        
        // Centrage initial - ajusté pour un meilleur centrage
        const initialScale = 0.8;
        svg.call(zoom.transform, d3.zoomIdentity
            .translate(this.width / 2, this.height / 2)
            .scale(initialScale));

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
            // Sélectionner le conteneur de légende
            const legendContainer = d3.select(".legend-container");
            legendContainer.selectAll("*").remove();
        
            // Ajouter le titre
            legendContainer.append("h3")
                .style("color", "white")
                .style("margin", "0 0 15px 0")
                .style("font-size", "16px")
                .text("Groupes thématiques");
        
            // Créer la liste des clusters
            const uniqueClusters = [...new Set(data.nodes.map(n => n.group))]
                .sort((a, b) => a - b);
        
            const legendItems = legendContainer.selectAll(".legend-item")
                .data(uniqueClusters)
                .enter()
                .append("div")
                .attr("class", "legend-item")
                .style("display", "flex")
                .style("align-items", "center")
                .style("margin-bottom", "10px")
                .style("cursor", "pointer")
                .on("mouseover", (event, d) => {
                    // Mise en surbrillance des nœuds du cluster
                    svg.selectAll(".nodes circle")
                        .style("opacity", node => node.group === d ? 1 : 0.3);
                    
                    // Afficher le tooltip
                    const description = d === -1 ? "Pages non classées" : 
                                      data.nodes.find(n => n.group === d)?.title || `Cluster ${d}`;
                    legendContainer.append("div")
                        .attr("class", "legend-tooltip")
                        .style("position", "absolute")
                        .style("left", (event.offsetX + 20) + "px")
                        .style("top", event.offsetY + "px")
                        .style("background", "rgba(0, 0, 0, 0.9)")
                        .style("padding", "8px")
                        .style("border-radius", "4px")
                        .style("font-size", "14px")
                        .text(description);
                })
                .on("mouseout", () => {
                    // Restaurer l'opacité normale
                    svg.selectAll(".nodes circle")
                        .style("opacity", 1);
                    
                    // Supprimer le tooltip
                    legendContainer.selectAll(".legend-tooltip").remove();
                });
        
            // Ajouter les indicateurs de couleur
            legendItems.append("div")
                .style("width", "20px")
                .style("height", "20px")
                .style("border-radius", "4px")
                .style("margin-right", "10px")
                .style("background", d => getColor(d))
                .style("border", d => `2px solid ${d3.color(getColor(d)).brighter(0.5)}`);
        
            // Ajouter les labels
            legendItems.append("div")
                .style("color", "white")
                .style("font-size", "14px")
                .text(d => {
                    const count = data.nodes.filter(n => n.group === d).length;
                    return `Cluster ${d} (${count} pages)`;
                });
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
            this.baseWidth = this.container.clientWidth;
            this.baseHeight = this.container.clientHeight;
            this.width = this.baseWidth * 3;
            this.height = this.baseHeight * 3;
            this.render(this.data);
        }
    }


    resetView() {
        const svg = d3.select("#" + this.container.id).select("svg");
        const g = svg.select("g");
        const zoom = d3.zoom().transform(svg);
        
        svg.transition()
           .duration(750)
           .call(zoom.transform, d3.zoomIdentity
                .translate(this.baseWidth/2, this.baseHeight/2)
                .scale(0.5));
    }

}