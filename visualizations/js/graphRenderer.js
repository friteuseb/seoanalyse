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
        // Calcul préalable du nombre de liens entrants/sortants pour chaque nœud
        const linkCounts = new Map();
        links.forEach(link => {
            if (!linkCounts.has(link.source.id)) {
                linkCounts.set(link.source.id, { outgoing: 0, incoming: 0 });
            }
            if (!linkCounts.has(link.target.id)) {
                linkCounts.set(link.target.id, { outgoing: 0, incoming: 0 });
            }
            linkCounts.get(link.source.id).outgoing++;
            linkCounts.get(link.target.id).incoming++;
        });
    
        return g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("class", "link-line")
            .attr("stroke", d => {
                // Un lien est considéré comme "fort" s'il fait partie d'une relation bidirectionnelle
                const isReciprocal = links.some(l => 
                    (l.source.id === d.target.id && l.target.id === d.source.id)
                );
                if (isReciprocal) {
                    return "#4a90e2"; // Bleu pour les liens bidirectionnels
                }
                // Sinon on regarde le ratio entrant/sortant
                const sourceCounts = linkCounts.get(d.source.id);
                if (sourceCounts.outgoing > sourceCounts.incoming) {
                    return "#2ecc71"; // Vert pour les liens sortants dominants
                } else {
                    return "#e74c3c"; // Rouge pour les liens entrants dominants
                }
            })
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
        
        const container = d3.select("#" + this.container.id);
        
        // Légende
        container.append("div")
            .attr("class", "legend-container")
            .style("position", "absolute")
            .style("top", "20px")
            .style("right", "20px")
            .style("background", "rgba(0, 0, 0, 0.8)")
            .style("border-radius", "10px")
            .style("padding", "15px")
            .style("z-index", "1000");
        
        const svg = container.append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("viewBox", [0, 0, this.width, this.height]);
        
        const g = svg.append("g");
        
        const zoom = d3.zoom()
            .scaleExtent([0.2, 4])
            .on("zoom", (event) => g.attr("transform", event.transform));
        
        svg.call(zoom);
        
        svg.call(zoom.transform, d3.zoomIdentity
            .translate(this.width/2, this.height/2)
            .scale(0.8));
        
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
        
            // Titre principal
            legendContainer.append("h3")
                .style("color", "white")
                .style("margin", "0 0 15px 0")
                .style("font-size", "16px")
                .text("Légende");
        
            // Section des clusters
            const clusterSection = legendContainer.append("div")
                .attr("class", "legend-section");
        
            clusterSection.append("h4")
                .style("color", "white")
                .style("margin", "0 0 10px 0")
                .style("font-size", "14px")
                .text("Groupes thématiques");
        
            // Légende des clusters (code existant)
            const uniqueClusters = [...new Set(data.nodes.map(n => n.group))]
                .sort((a, b) => a - b);
        
            const clusterItems = clusterSection.selectAll(".cluster-item")
                .data(uniqueClusters)
                .enter()
                .append("div")
                .attr("class", "legend-item")
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
        
             // Indicateurs de couleur des clusters
                        clusterItems.append("div")
                        .style("width", "20px")
                        .style("height", "20px")
                        .style("border-radius", "4px")
                        .style("margin-right", "10px")
                        .style("background", d => getColor(d))
                        .style("border", d => `2px solid ${d3.color(getColor(d)).brighter(0.5)}`);

                    // Labels des clusters
                    clusterItems.append("div")
                        .style("color", "white")
                        .style("font-size", "12px")
                        .text(d => {
                            const count = data.nodes.filter(n => n.group === d).length;
                            return `Cluster ${d} (${count} pages)`;
                        });

                    // Section des liens
                    const linkSection = legendContainer.append("div")
                        .attr("class", "legend-section")
                        .style("margin-top", "20px");

                    linkSection.append("h4")
                        .style("color", "white")
                        .style("margin", "0 0 10px 0")
                        .style("font-size", "14px")
                        .text("Types de liens");

                    // Légende pour les liens sortants
                    const linkTypes = [
                        {color: "#2ecc71", label: "Liens majoritairement sortants"},
                        {color: "#e74c3c", label: "Liens majoritairement entrants"},
                        {color: "#4a90e2", label: "Liens bidirectionnels"}
                    ];
                    linkSection.selectAll(".link-item")
                        .data(linkTypes)
                        .enter()
                        .append("div")
                        .attr("class", "legend-item")
                        .style("display", "flex")
                        .style("align-items", "center")
                        .style("margin-bottom", "5px")
                        .each(function(d) {
                            // Ligne de démonstration
                            d3.select(this).append("div")
                                .style("width", "30px")
                                .style("height", "2px")
                                .style("background", d.color)
                                .style("margin-right", "10px");

                            // Label
                            d3.select(this).append("div")
                                .style("color", "white")
                                .style("font-size", "12px")
                                .text(d.label);
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
                .style("opacity", 0.6)
                .attr("class", "arrowhead-path") // Ajoutez une classe
                .style("fill", "currentColor"); // Utilisez la couleur courante au lieu d'une couleur fixe
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