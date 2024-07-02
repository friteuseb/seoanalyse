document.addEventListener('DOMContentLoaded', () => {
    const graphSelect = document.getElementById('graphSelect');
    const loadGraphButton = document.getElementById('loadGraph');

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
                createGraph(data, selectedGraph.includes('clustered'));
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

        const color = d3.scaleOrdinal(d3.schemeCategory10);

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
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.value));

        const node = g.append("g")
            .selectAll("g")
            .data(data.nodes)
            .enter().append("g")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", 5)
            .attr("fill", d => color(d.group));

        node.append("title")
            .text(d => d.title);

        node.append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .text(d => d.id.split('/').pop())
            .style("font-size", "8px");

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
    }
});