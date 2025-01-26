class GraphRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = Math.max(this.container.clientWidth * 2, 3000);
        this.height = Math.max(this.container.clientHeight * 2, 3000);
        this.simulation = null;
        this.tooltipManager = new TooltipManager();
        this.filterManager = null;
        this.data = null;
        this.nodeScale = null; 
        this.metricsManager = null;

    }

    setMetricsManager(metricsManager) {
        this.metricsManager = metricsManager;
    }

    render(data) {
        console.log("Début du rendu du graphe...");
    
        // Vérifier que les données sont valides
        if (!data || !data.nodes || !data.links) {
            console.error("Données invalides pour le rendu du graphe");
            return;
        }
    
        console.log("Données reçues :", data);
        console.log("Nombre de nœuds :", data.nodes.length);
        console.log("Nombre de liens :", data.links.length);
    
        this.data = data;
    
        // Initialiser le SVG et le groupe principal
        console.log("Initialisation du SVG...");
        const svg = this.initializeSVG();
        const g = svg.select("g");
    
        // Calculer le nombre de liens entrants pour chaque nœud
        console.log("Calcul des liens entrants...");
        const incomingLinksCount = this.calculateIncomingLinks(data.links);
    
        // Initialiser l'échelle pour la taille des nœuds
        console.log("Initialisation de l'échelle pour la taille des nœuds...");
        this.nodeScale = d3.scaleLinear()
            .domain([0, d3.max(Object.values(incomingLinksCount)) || 1])
            .range([CONFIG.nodeMinSize, CONFIG.nodeMaxSize]);
    
        // Ajouter les marqueurs de flèche
        console.log("Ajout des marqueurs de flèche...");
        this.addArrowMarkers(svg);
    
        // Créer les éléments du graphe (nœuds et liens)
        console.log("Création des éléments du graphe...");
        const elements = this.createElements(g, data);
    
        // Configurer la simulation
        console.log("Configuration de la simulation...");
        this.setupSimulation(data);
    
        // Configurer les forces de la simulation
        console.log("Configuration des forces de la simulation...");
        this.setupSimulationForces(elements.node, elements.link);
    
        // Démarrer la simulation
        console.log("Démarrage de la simulation...");
        this.simulation.alphaTarget(0.3).restart();
    
        // Réduire l'alphaTarget après un court délai pour stabiliser la simulation
        console.log("Réduction de l'alphaTarget après 1 seconde...");
        setTimeout(() => {
            this.simulation.alphaTarget(0);
            console.log("Simulation stabilisée.");
        }, 1000);
    
        // Créer la légende
        console.log("Création de la légende...");
        this.createLegend(svg, data);
    
        // Configurer les filtres
        console.log("Configuration des filtres...");
        this.filterManager = new FilterManager(elements.node, elements.link, data);
    
        // Forcer un rendu avec requestAnimationFrame
        console.log("Forcer un rendu avec requestAnimationFrame...");
        requestAnimationFrame(() => {
            this.simulation.alphaTarget(0);
            console.log("Rendu forcé terminé.");
        });
    
        console.log("Rendu du graphe terminé.");
    }


    setupSimulation(data) {
        if (this.simulation) {
            this.simulation.stop();
        }

        const centerX = 0;
        const centerY = 0;

        // Calculer incomingLinksCount
        const incomingLinksCount = this.calculateIncomingLinks(data.links);

        this.simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink()
                .id(d => d.id)
                .links(data.links)
                .distance(100))
            .force("charge", d3.forceManyBody()
                .strength(-800))
            .force("center", d3.forceCenter(centerX, centerY))
            .force("collide", d3.forceCollide().radius(d => this.nodeScale(incomingLinksCount[d.id] || 0) + 10)) // Ajustez le rayon de collision
            .velocityDecay(0.6);

        this.simulation.on("end", () => {
            console.log("Simulation stabilisée");
        });

        return this.simulation;
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
                // Mise à jour des positions des liens
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

                // Mise à jour des positions des nœuds
                node.attr("transform", d => {
                    if (isNaN(d.x) || isNaN(d.y)) {
                        console.error("Invalid node position:", d);
                        return `translate(0, 0)`; // Valeur par défaut
                    }
                    return `translate(${d.x},${d.y})`;
                });
            });
        } catch (error) {
            console.error("Error in setupSimulationForces:", error);
        }
    }
    
    createLinks(g, links) {
        // Calculer d'abord les liens par nœud
        const linkCounts = new Map();
        links.forEach(link => {
            // Assurer que source et target sont des objets
            const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            
            if (!linkCounts.has(sourceId)) {
                linkCounts.set(sourceId, { outgoing: 0, incoming: 0 });
            }
            if (!linkCounts.has(targetId)) {
                linkCounts.set(targetId, { outgoing: 0, incoming: 0 });
            }
            linkCounts.get(sourceId).outgoing++;
            linkCounts.get(targetId).incoming++;
        });
    
        // Ajouter les marqueurs de flèche
        const svg = g.ownerSVGElement;
        if (svg) {
            this.addArrowMarkers(svg);
        }
    
        return g.append("g")
            .attr("class", "links")
            .lower() // Mettre les liens en arrière-plan
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("class", "link-line")
            .attr("stroke", d => {
                // Obtenir les IDs de source et target de manière sûre
                const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                const targetId = typeof d.target === 'object' ? d.target.id : d.target;
    
                // Vérifier s'il y a un lien réciproque
                const isReciprocal = links.some(l => {
                    const lSourceId = typeof l.source === 'object' ? l.source.id : l.source;
                    const lTargetId = typeof l.target === 'object' ? l.target.id : l.target;
                    return lSourceId === targetId && lTargetId === sourceId;
                });
    
                if (isReciprocal) {
                    return "#4a90e2"; // bleu pour les liens bidirectionnels
                }
    
                // Obtenir les statistiques des liens
                const sourceCounts = linkCounts.get(sourceId);
                if (sourceCounts && sourceCounts.outgoing > sourceCounts.incoming) {
                    return "#2ecc71"; // vert pour les liens sortants dominants
                } else {
                    return "#e74c3c"; // rouge pour les liens entrants dominants
                }
            })
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", d => Math.sqrt(d.value || 1))
            .attr("marker-end", d => {
                // Déterminer le type de flèche en fonction de la couleur du lien
                const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                const targetId = typeof d.target === 'object' ? d.target.id : d.target;
                
                const isReciprocal = links.some(l => {
                    const lSourceId = typeof l.source === 'object' ? l.source.id : l.source;
                    const lTargetId = typeof l.target === 'object' ? l.target.id : l.target;
                    return lSourceId === targetId && lTargetId === sourceId;
                });
    
                if (isReciprocal) {
                    return "url(#arrowhead-blue)";
                }
                
                const sourceCounts = linkCounts.get(sourceId);
                if (sourceCounts && sourceCounts.outgoing > sourceCounts.incoming) {
                    return "url(#arrowhead-green)";
                }
                return "url(#arrowhead-red)";
            });
    }

    
    addArrowMarkers(svg) {
        if (!svg) {
            console.error("SVG element is required for adding markers");
            return;
        }
    
        // Supprimer les anciens marqueurs s'ils existent
        const oldDefs = svg.querySelector("defs");
        if (oldDefs) {
            oldDefs.remove();
        }
    
        // Créer un nouvel élément defs
        const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
        svg.appendChild(defs);
        
        // Créer trois marqueurs différents pour chaque couleur de lien
        const colors = {
            'blue': '#4a90e2',
            'green': '#2ecc71',
            'red': '#e74c3c'
        };
    
        Object.entries(colors).forEach(([key, color]) => {
            const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
            marker.setAttribute("id", `arrowhead-${key}`);
            marker.setAttribute("viewBox", "-10 -5 20 10");
            marker.setAttribute("refX", "25"); // Augmenter cette valeur pour éloigner la flèche
            marker.setAttribute("refY", "0");
            marker.setAttribute("markerWidth", "6");
            marker.setAttribute("markerHeight", "6");
            marker.setAttribute("orient", "auto");
    
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", "M -10,-5 L 0,0 L -10,5");
            path.setAttribute("fill", color);
    
            marker.appendChild(path);
            defs.appendChild(marker);
        });
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
        .on("zoom", (event) => {
            g.attr("transform", event.transform);
        });
        
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
                    d.node = this;
                })
                .call(d3.drag()
                    .on("start", this.dragstarted)
                    .on("drag", this.dragged)
                    .on("end", this.dragended))
                .on("click", (event, d) => {
                    if (event.ctrlKey) {
                        window.open(d.url, '_blank');
                    }
                })
                .on("contextmenu", (event, d) => {
                    event.preventDefault();
                    this.showContextMenu(event, d);
                })
                .on("mouseover", (event, d) => this.tooltipManager.show(event, d, incomingLinksCount))
                .on("mouseout", () => this.tooltipManager.hide());
        
            // Création des cercles avec une taille basée sur incomingLinksCount
            node.append("circle")
                .attr("r", d => {
                    const count = incomingLinksCount[d.id] || 0;
                    console.log(`Node ${d.id} has ${count} incoming links`); // Debug
                    return nodeScale(count);
                })
                .attr("fill", "#1E90FF") // Bleu électrique
                .attr("stroke", "#104E8B") // Contour plus foncé
                .attr("stroke-width", 2) // Épaisseur du contour
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



        showContextMenu(event, node) {
            // Supprimer tout menu contextuel existant
            d3.selectAll('.context-menu').remove();
        
            // Créer le menu contextuel
            const menu = d3.select('body')
                .append('div')
                .attr('class', 'context-menu')
                .style('position', 'absolute')
                .style('left', `${event.pageX}px`)
                .style('top', `${event.pageY}px`)
                .style('background-color', 'rgba(0, 0, 0, 0.9)')
                .style('border', '1px solid #444')
                .style('border-radius', '4px')
                .style('padding', '5px')
                .style('z-index', '1000');
        
            // Ajouter les options du menu
            this.addMenuItem(menu, "Supprimer ce nœud", () => this.removeNode(node));
            this.addMenuItem(menu, "Ne garder que ce nœud et ses connexions", () => this.keepNodeAndConnections(node));
        
            // Fermer le menu au clic ailleurs
            d3.select('body').on('click.context-menu', () => {
                d3.selectAll('.context-menu').remove();
            });
        }

        keepNodeAndConnections(nodeToKeep) {
            if (!this.data) return;
        
            try {
                // Trouver tous les nœuds connectés directement à ce nœud
                const connectedNodeIds = new Set();
                connectedNodeIds.add(nodeToKeep.id);
        
                // Ajouter les nœuds liés par des liens entrants ou sortants
                this.data.links.forEach(link => {
                    if (link.source.id === nodeToKeep.id) {
                        connectedNodeIds.add(link.target.id);
                    }
                    if (link.target.id === nodeToKeep.id) {
                        connectedNodeIds.add(link.source.id);
                    }
                });
        
                // Filtrer les nœuds et les liens
                this.data.nodes = this.data.nodes.filter(n => connectedNodeIds.has(n.id));
                this.data.links = this.data.links.filter(l => 
                    connectedNodeIds.has(l.source.id) && connectedNodeIds.has(l.target.id)
                );
        
                // Mise à jour de la simulation
                this.simulation.nodes(this.data.nodes);
                this.simulation.force("link").links(this.data.links);
                
                // Redémarrage plus progressif de la simulation
                this.simulation
                    .alpha(0.5)
                    .alphaTarget(0)
                    .alphaDecay(0.02)
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
                console.error("Error keeping node and connections:", error);
            }
        }
        
        addMenuItem(menu, text, onClick) {
            menu.append('div')
                .attr('class', 'context-menu-item')
                .style('padding', '8px 12px')
                .style('cursor', 'pointer')
                .style('color', 'white')
                .style('hover', 'background-color: #444')
                .text(text)
                .on('click', (event) => {
                    event.stopPropagation();
                    onClick();
                    d3.selectAll('.context-menu').remove();
                })
                .on('mouseover', function() {
                    d3.select(this).style('background-color', '#444');
                })
                .on('mouseout', function() {
                    d3.select(this).style('background-color', 'transparent');
                });
        }
        
        
        createLegend(svg, data) {
            // Sélectionner uniquement le contenu du panneau déplaçable
            const legendContainer = d3.select("#draggable-legend-panel .panel-content");
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
                            // Suppression du nœud et des liens associés
                            this.data.nodes = this.data.nodes.filter(n => n.id !== nodeToRemove.id);
                            this.data.links = this.data.links.filter(l => 
                                l.source.id !== nodeToRemove.id && l.target.id !== nodeToRemove.id
                            );
                    
                            // Mise à jour de la simulation
                            this.simulation.nodes(this.data.nodes);
                            this.simulation.force("link").links(this.data.links);
                            
                            // Mise à jour du DOM
                            const nodes = d3.select("#" + this.container.id)
                                .selectAll(".nodes g")
                                .data(this.data.nodes, d => d.id);
                                
                            nodes.exit().remove();
                    
                            const links = d3.select("#" + this.container.id)
                                .selectAll(".links line")
                                .data(this.data.links, d => `${d.source.id}-${d.target.id}`);
                                
                            links.exit().remove();
                    
                            // Mise à jour des métriques
                            if (this.metricsManager) {
                                this.metricsManager.updateMetricsDisplay();
                                this.metricsManager.updateTopNodesTable();
                            }
                    
                            // Redémarrage plus progressif de la simulation
                            this.simulation
                                .alpha(0.5)
                                .alphaTarget(0)
                                .alphaDecay(0.02)
                                .restart();
                    
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