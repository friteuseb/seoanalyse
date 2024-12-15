class FilterManager {
    constructor(node, link, data) {
        this.node = node;
        this.link = link;
        this.data = data;
        
        // Nettoyer les contrôles existants
        d3.select("#controls")
            .selectAll(".filter-controls")
            .remove();
            
        this.setupControls();
    }

    setupControls() {
        const controls = d3.select("#controls")
            .append("div")
            .attr("class", "filter-controls")
            .style("margin-top", "10px");

        this.setupClusterFilter(controls);
        this.setupLinksFilter(controls);
        this.setupResetButton(controls);
    }
    
    setupClusterFilter(controls) {
        const clusters = [...new Set(this.data.nodes.map(n => n.group))].sort();
        controls.append("select")
            .attr("id", "clusterFilter")
            .style("margin-right", "10px")
            .on("change", e => this.filterByCluster(e.target.value))
            .selectAll("option")
            .data([{value: "all", text: "Tous les clusters"}, 
                  ...clusters.map(c => ({value: c, text: `Cluster ${c}`}))])
            .enter()
            .append("option")
            .attr("value", d => d.value)
            .text(d => d.text);
    }

    setupLinksFilter(controls) {
        controls.append("input")
            .attr("type", "number")
            .attr("id", "minLinksFilter")
            .attr("placeholder", "Liens minimum")
            .attr("min", "0")
            .style("width", "100px")
            .style("margin-right", "10px")
            .on("input", e => this.filterByLinks(e.target.value));
    }

    setupResetButton(controls) {
        controls.append("button")
            .text("Réinitialiser les filtres")
            .on("click", () => this.resetFilters());
    }

    filterByCluster(selectedCluster) {
        this.node.style("opacity", d => 
            selectedCluster === "all" || d.group === parseInt(selectedCluster) ? 1 : 0.1
        );

        this.link.style("opacity", d => {
            if (selectedCluster === "all") return 0.6;
            const sourceVisible = d.source.group === parseInt(selectedCluster);
            const targetVisible = d.target.group === parseInt(selectedCluster);
            return sourceVisible || targetVisible ? 0.6 : 0.1;
        });
    }

    filterByLinks(minLinks) {
        const incomingLinksCount = {};
        this.data.links.forEach(link => {
            const targetId = typeof link.target === 'object' ? link.target.id : link.target;
            incomingLinksCount[targetId] = (incomingLinksCount[targetId] || 0) + 1;
        });

        const threshold = parseInt(minLinks) || 0;
        
        this.node.style("opacity", d => 
            (incomingLinksCount[d.id] || 0) >= threshold ? 1 : 0.1
        );

        this.link.style("opacity", d => {
            const sourceVisible = (incomingLinksCount[d.source.id] || 0) >= threshold;
            const targetVisible = (incomingLinksCount[d.target.id] || 0) >= threshold;
            return sourceVisible && targetVisible ? 0.6 : 0.1;
        });
    }

    resetFilters() {
        d3.select("#clusterFilter").property("value", "all");
        d3.select("#minLinksFilter").property("value", "");
        this.node.style("opacity", 1);
        this.link.style("opacity", 0.6);
    }
}