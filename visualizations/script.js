document.addEventListener('DOMContentLoaded', () => {
    const graphSelect = document.getElementById('graphSelect');
    const loadGraphButton = document.getElementById('loadGraph');

    let graphData = null;
    let isClustered = false;
    let currentSortColumn = null;
    let currentSortOrder = 'asc';

    const colorList = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ];

    function getColor(group) {
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

    // MODIFIED: createGraph function
    function createGraph(data, isClustered) {
        d3.select("#graph").selectAll("*").remove();

        const width = document.getElementById('graph').clientWidth;
        const height = 600;

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

        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("stroke", "#999")
            .attr("stroke-opacity", 0.6)
            .attr("stroke-width", d => Math.sqrt(d.value));

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
            .attr("r", d => 5 + Math.sqrt(d.internal_links_count || 1) * 2)
            .attr("fill", d => getColor(d.group));

        node.append("title")
            .text(d => d.title);

        node.append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .attr("class", "node-text")
            .text(d => d.label);

        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

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
            .attr("transform", `translate(${width - 200}, 20)`);

        const labels = {};
        data.nodes.forEach(node => {
            if (!labels[node.group]) {
                labels[node.group] = node.title;
            }
        });

        Object.entries(labels).forEach(([group, label], i) => {
            const legendRow = legend.append("g")
                .attr("transform", `translate(0, ${i * 20})`);

            legendRow.append("rect")
                .attr("width", 10)
                .attr("height", 10)
                .attr("fill", getColor(group));

            legendRow.append("text")
                .attr("x", 20)
                .attr("y", 10)
                .attr("text-anchor", "start")
                .text(label)
                .attr("fill", getColor(group));
        });
    }

    function populateTable(nodes, links) {
        console.log("Nodes:", nodes);
        console.log("Links:", links);
    
        const tableBody = document.querySelector('#nodesTable tbody');
        tableBody.innerHTML = '';
    
        nodes.forEach(node => {
            console.log("Processing node:", node);
    
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
    
            console.log(`Node ${node.id}: Incoming links: ${incomingLinksCount}, Outgoing links: ${outgoingLinksCount}`);
    
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
    // Modifiez la fonction removeNode comme suit
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



    // Ajoutez cette nouvelle fonction pour gérer le tri
    function sortTable(columnIndex, order) {
        const tableBody = document.querySelector('#nodesTable tbody');
        const rows = Array.from(tableBody.querySelectorAll('tr'));

        rows.sort((a, b) => {
            let aValue = a.children[columnIndex].textContent;
            let bValue = b.children[columnIndex].textContent;

            if (columnIndex === 0) {  // Tri par couleur
                aValue = a.children[columnIndex].style.backgroundColor;
                bValue = b.children[columnIndex].style.backgroundColor;
            } else if (!isNaN(aValue) && !isNaN(bValue)) {  // Tri numérique
                aValue = Number(aValue);
                bValue = Number(bValue);
            }

            if (aValue < bValue) return order === 'asc' ? -1 : 1;
            if (aValue > bValue) return order === 'asc' ? 1 : -1;
            return 0;
        });

        rows.forEach(row => tableBody.appendChild(row));
    }

    function addTableSorting() {
        const headers = document.querySelectorAll('#nodesTable th');
        headers.forEach((header, index) => {
            header.addEventListener('click', () => {
                const table = header.closest('table');
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

    addTableSorting();
});