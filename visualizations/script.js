document.addEventListener('DOMContentLoaded', () => {
    const graphSelect = document.getElementById('graphSelect');
    const loadGraphButton = document.getElementById('loadGraph');

    let graphData = null;
    let isClustered = false;
    let currentSortColumn = null;
    let currentSortOrder = 'asc';
    let svg; 


    const colorList = [
        "#2D7DD2",  // Bleu plus intense
        "#EEB902",  // Orange/Jaune
        "#97CC04",  // Vert vif
        "#F45D01",  // Orange brûlé
        "#474647"   // Gris pour les non classés
    ];
    
    function getColor(group) {
        if (group === undefined || group === -1) {
            return colorList[4];  // Gris pour les non classés
        }
        return colorList[group % colorList.length];
    }

    fetch('get_available_graphs.php')
        .then(response => response.json())
        .then(graphs => {
            graphs.forEach(graph => {
                const option = document.createElement('option');
                option.value = graph;
                option.textContent = graph;
                graphSelect.appendChild(option);
            });
        });

    loadGraphButton.addEventListener('click', loadGraph);

    // MODIFIED: loadGraph function
    function loadGraph() {
        const selectedGraph = graphSelect.value;
        if (!selectedGraph) return;
    
        fetch(`get_graph_data.php?graph=${selectedGraph}`)
            .then(response => response.json())
            .then(data => {
                // Vérifier si les données sont dans le bon format
                if (!Array.isArray(data.nodes) || !Array.isArray(data.links)) {
                    console.error('Invalid data format:', data);
                    return;
                }
    
                // Créer un map des nœuds pour une recherche plus rapide
                const nodeMap = new Map(data.nodes.map(node => [node.id, node]));
    
                // Convert links to the correct format
                data.links = data.links.map(link => {
                    let source = typeof link.source === 'object' ? link.source : nodeMap.get(link.source);
                    let target = typeof link.target === 'object' ? link.target : nodeMap.get(link.target);
                
                    if (!source || !target) {
                        console.error('Invalid link:', link);
                        return null;
                    }
                
                    return {
                        source: source,
                        target: target,
                        value: link.value
                    };
                }).filter(link => link !== null);
    
                graphData = data;
                isClustered = selectedGraph.includes('clustered');
                createGraph(data, isClustered);
                populateTable(data.nodes, data.links);
            })
            .catch(error => {
                console.error('Error loading graph:', error);
            });
    }

    function createGraph(data, isClustered) {
        d3.select("#graph").selectAll("*").remove();
    
        const width = document.getElementById('graph').clientWidth;
        const height = document.getElementById('graph').clientHeight;
    
        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
    
        const g = svg.append("g");
    
        svg.call(d3.zoom().on("zoom", (event) => {
            g.attr("transform", event.transform);
        }));
    
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));
    
        if (isClustered) {
            simulation.force("x", d3.forceX().strength(0.1).x(d => {
                return width / 2 + 100 * Math.cos(2 * Math.PI * d.group / 10);
            }))
            .force("y", d3.forceY().strength(0.1).y(d => {
                return height / 2 + 100 * Math.sin(2 * Math.PI * d.group / 10);
            }));
        }
    
        // Ajout des flèches pour les liens
        svg.append("defs").selectAll("marker")
            .data(["end"])
            .enter().append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 13)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("xoverflow", "visible")
            .append("svg:path")
            .attr("d", "M 0,-5 L 10 ,0 L 0,5")
            .attr("fill", "#999")
            .style("stroke", "none");
    
        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", d => Math.sqrt(d.value))
            .attr("marker-end", "url(#arrowhead)");
    
        const incomingLinksCount = {};
        data.links.forEach(link => {
            incomingLinksCount[link.target.id] = (incomingLinksCount[link.target.id] || 0) + 1;
        });
    
        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            .data(data.nodes)
            .enter().append("g")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("contextmenu", (event, d) => {
                event.preventDefault();
                removeNode(d);
            });
    
        node.append("circle")
            .attr("r", d => 5 + Math.sqrt(incomingLinksCount[d.id] || 1) * 3)
            .attr("fill", d => getColor(d.group));
    
        node.append("title")
            .text(d => d.title);
    
        node.append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .attr("class", "node-text")
            .text(d => d.label);
    
        node.append("title")
            .text(d => {
                let tooltip = `URL: ${d.label}\n`;
                tooltip += `Cluster: ${d.group}\n`;
                tooltip += `Description: ${d.title}\n`;
                tooltip += `Liens: ${d.internal_links_count}`;
                return tooltip;
            });
    
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => {
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const radius = 5 + Math.sqrt(incomingLinksCount[d.target.id] || 1) * 3;
                    return d.target.x - (dx * radius) / distance;
                })
                .attr("y2", d => {
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const radius = 5 + Math.sqrt(incomingLinksCount[d.target.id] || 1) * 3;
                    return d.target.y - (dy * radius) / distance;
                });
    
            node
                .attr("transform", d => `translate(${d.x},${d.y})`);
        });
    
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
    
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
    
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
    
        const legend = svg.append("g")
            .attr("class", "legend")
            .attr("transform", `translate(${width - 250}, 20)`);
    
        const clusters = {};
        data.nodes.forEach(node => {
            if (!clusters[node.group]) {
                clusters[node.group] = {
                    color: getColor(node.group),
                    title: node.title,
                    count: 1
                };
            } else {
                clusters[node.group].count++;
            }
        });
    
        Object.entries(clusters).forEach(([group, info], i) => {
            const legendRow = legend.append("g")
                .attr("transform", `translate(0, ${i * 50})`);
    
            legendRow.append("rect")
                .attr("width", 20)
                .attr("height", 20)
                .attr("fill", info.color);
    
            legendRow.append("text")
                .attr("x", 30)
                .attr("y", 10)
                .attr("dy", "0.32em")
                .attr("class", "legend-text")
                .text(`Cluster ${group} (${info.count} pages)`);
    
            legendRow.append("text")
                .attr("x", 30)
                .attr("y", 30)
                .attr("dy", "0.32em")
                .attr("class", "legend-description")
                .style("font-size", "0.8em")
                .text(info.title.length > 60 ? info.title.substring(0, 57) + "..." : info.title);
        });
    }
    
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
    
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
    
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
    
        const legend = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - 250}, 20)`);
    
        // Grouper les nœuds par cluster
        const clusters = {};
        data.nodes.forEach(node => {
            if (!clusters[node.group]) {
                clusters[node.group] = {
                    color: getColor(node.group),
                    title: node.title,
                    count: 1
                };
            } else {
                clusters[node.group].count++;
            }
        });
    
        // Créer la légende
        Object.entries(clusters).forEach(([group, info], i) => {
            const legendRow = legend.append("g")
                .attr("transform", `translate(0, ${i * 50})`);  // Plus d'espace entre les entrées

            // Carré de couleur
            legendRow.append("rect")
                .attr("width", 20)
                .attr("height", 20)
                .attr("fill", info.color);

            // Description du cluster
            legendRow.append("text")
                .attr("x", 30)
                .attr("y", 10)
                .attr("dy", "0.32em")
                .attr("class", "legend-text")
                .text(`Cluster ${group} (${info.count} pages)`);

            // Description en dessous
            legendRow.append("text")
                .attr("x", 30)
                .attr("y", 30)
                .attr("dy", "0.32em")
                .attr("class", "legend-description")
                .style("font-size", "0.8em")
                .text(info.title.length > 60 ? info.title.substring(0, 57) + "..." : info.title);
        });
            
    
    
    function populateTable(nodes, links) {
            const tableBody = document.querySelector('#nodesTable tbody');
            tableBody.innerHTML = '';
    
            nodes.forEach(node => {
                const row = document.createElement('tr');
                row.setAttribute('data-node-id', node.id || 'undefined');
    
                const colorCell = document.createElement('td');
                colorCell.style.backgroundColor = getColor(node.group || 0);
                colorCell.style.width = '5px';
                row.appendChild(colorCell);
    
                const nodeNameCell = document.createElement('td');
                nodeNameCell.textContent = node.label || node.id || 'Unnamed';
                row.appendChild(nodeNameCell);
    
                const incomingLinksCount = links.filter(link => 
                    (typeof link.target === 'object' ? link.target.id : link.target) === node.id
                ).length;
                const outgoingLinksCount = links.filter(link => 
                    (typeof link.source === 'object' ? link.source.id : link.source) === node.id
                ).length;
    
                const incomingLinksCell = document.createElement('td');
                incomingLinksCell.textContent = incomingLinksCount;
                row.appendChild(incomingLinksCell);
    
                const outgoingLinksCell = document.createElement('td');
                outgoingLinksCell.textContent = outgoingLinksCount;
                row.appendChild(outgoingLinksCell);
    
                const actionsCell = document.createElement('td');
                const deleteButton = document.createElement('button');
                deleteButton.textContent = 'Supprimer';
                deleteButton.addEventListener('click', () => {
                    removeNode(node);
                });
                actionsCell.appendChild(deleteButton);
                row.appendChild(actionsCell);
    
                tableBody.appendChild(row);
            });
        }
    
   
    function removeNode(d) {
        const index = graphData.nodes.findIndex(node => node.id === d.id);
        if (index > -1) {
            graphData.nodes.splice(index, 1);
            graphData.links = graphData.links.filter(l => l.source.id !== d.id && l.target.id !== d.id);
            createGraph(graphData, isClustered);
            populateTable(graphData.nodes, graphData.links);
            
            // Réappliquer le tri actuel après la suppression
            if (currentSortColumn !== null) {
                sortTable(currentSortColumn, currentSortOrder);
            }
        }
    }



    // fonction pour gérer le tri
    function sortTable(columnIndex, order) {
        const tableBody = document.querySelector('#nodesTable tbody');
        const rows = Array.from(tableBody.querySelectorAll('tr'));
    
        rows.sort((a, b) => {
            let aValue = a.children[columnIndex].textContent.trim();
            let bValue = b.children[columnIndex].textContent.trim();
    
            // Vérifier si les valeurs sont des nombres et les convertir
            if (!isNaN(aValue) && !isNaN(bValue)) {
                aValue = parseFloat(aValue);
                bValue = parseFloat(bValue);
            }
    
            if (aValue < bValue) return order === 'asc' ? -1 : 1;
            if (aValue > bValue) return order === 'asc' ? 1 : -1;
            return 0;
        });
    
        rows.forEach(row => tableBody.appendChild(row));
    }
    

    function addTableSorting() {
        const table = document.querySelector('#nodesTable');
        if (!table) {
            console.error('Table #nodesTable not found!');
            return;
        }

        const headers = table.querySelectorAll('th');
        if (headers.length === 0) {
            console.error('No headers <th> found in #nodesTable!');
            return;
        }

        headers.forEach((header, index) => {
            header.addEventListener('click', () => {
                currentSortOrder = header.classList.contains('asc') ? 'desc' : 'asc';
                currentSortColumn = index;

                headers.forEach(h => {
                    h.classList.remove('asc', 'desc');
                });

                header.classList.add(currentSortOrder);

                sortTable(currentSortColumn, currentSortOrder);
            });
        });
    }


    addTableSorting(); // Appel de la fonction pour ajouter le tri aux colonnes
}); 
