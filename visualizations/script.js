document.addEventListener('DOMContentLoaded', () => {
    const graphSelect = document.getElementById('graphSelect');
    const loadGraphButton = document.getElementById('loadGraph');

    let graphData = null;
    let isClustered = false;

    // Liste de couleurs définie en dur
    const colorList = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ];

    // Fonction pour récupérer la couleur en fonction du groupe
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

    function loadGraph() {
        const selectedGraph = graphSelect.value;
        if (!selectedGraph) return;

        fetch(`get_graph_data.php?graph=${selectedGraph}`)
            .then(response => response.json())
            .then(data => {
                graphData = data;
                isClustered = selectedGraph.includes('clustered');
                createGraph(data, isClustered);
                populateTable(data.nodes, data.links);
            });
    }

    function createGraph(data, isClustered) {
        d3.select("#graph").selectAll("*").remove();

        const width = document.getElementById('graph').clientWidth;
        const height = 600;

        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        // Création d'un groupe pour le zoom
        const g = svg.append("g");

        // Ajout du zoom
        svg.call(d3.zoom().on("zoom", (event) => {
            g.attr("transform", event.transform);
        }));

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id))
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
            .attr("r", d => Math.sqrt(d.internal_links_count || 1) * 5)
            .attr("fill", d => getColor(d.group));

        node.append("title")
            .text(d => d.title);

        node.append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .attr("class", "node-text")
            .text(d => d.id.split('/').pop());

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

        // Ajouter une légende pour les couleurs utilisant le champ label
        const legend = svg.append("g")
            .attr("class", "legend")
            .attr("transform", `translate(${width - 200}, 20)`); // Ajuster la position

        // Récupérer les labels uniques pour chaque groupe
        const labels = {};
        data.nodes.forEach(node => {
            if (!labels[node.group]) {
                labels[node.group] = node.label;
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
                .attr("fill", getColor(group)); // Assurer que la couleur du texte est la même que celle du nœud
        });
    }

    function populateTable(nodes, links) {
        const tableBody = document.querySelector('#nodesTable tbody');
        tableBody.innerHTML = '';

        nodes.forEach(node => {
            const row = document.createElement('tr');
            row.setAttribute('data-node-id', node.id);

            const colorCell = document.createElement('td');
            colorCell.style.backgroundColor = getColor(node.group);
            colorCell.style.width = '5px';  // Cellule fine pour la thématique de couleur
            row.appendChild(colorCell);

            const nodeNameCell = document.createElement('td');
            nodeNameCell.textContent = node.id.split('/').pop();
            row.appendChild(nodeNameCell);

            // Calculer le nombre de liens entrants
            const incomingLinksCount = links.filter(link => link.target.id === node.id).length;
            // Calculer le nombre de liens sortants
            const outgoingLinksCount = links.filter(link => link.source.id === node.id).length;

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

    // Fonction pour supprimer un nœud et mettre à jour le tableau
    function removeNode(d) {
        const index = graphData.nodes.indexOf(d);
        if (index > -1) {
            graphData.nodes.splice(index, 1);
            graphData.links = graphData.links.filter(l => l.source !== d && l.target !== d);
            createGraph(graphData, isClustered); // Recreate the graph with the node removed
            populateTable(graphData.nodes, graphData.links); // Update the table
        }
    }

    // Fonction pour ajouter des fonctionnalités de tri au tableau
    function addTableSorting() {
        const headers = document.querySelectorAll('#nodesTable th');
        headers.forEach(header => {
            header.addEventListener('click', () => {
                const table = header.parentElement.parentElement.parentElement;
                const rows = Array.from(table.querySelectorAll('tbody tr'));
                const index = Array.from(header.parentElement.children).indexOf(header);
                const ascending = header.classList.contains('asc');

                rows.sort((a, b) => {
                    const cellA = a.children[index].textContent;
                    const cellB = b.children[index].textContent;

                    if (!isNaN(cellA) && !isNaN(cellB)) {
                        return ascending ? cellA - cellB : cellB - cellA;
                    } else {
                        return ascending ? cellA.localeCompare(cellB) : cellB.localeCompare(cellA);
                    }
                });

                rows.forEach(row => table.querySelector('tbody').appendChild(row));
                header.classList.toggle('asc', !ascending);
                header.classList.toggle('desc', ascending);
            });
        });
    }

    // Ajouter la fonctionnalité de tri après le chargement initial du DOM
    addTableSorting();
});