class TooltipManager {
    constructor() {
        this.tooltip = d3.select("body")
            .append("div")
            .attr("class", "tooltip")
            .style("position", "absolute")
            .style("visibility", "hidden")
            .style("background-color", "rgba(0, 0, 0, 0.8)")
            .style("color", "white")
            .style("padding", "10px")
            .style("border-radius", "5px")
            .style("font-size", "12px");
    }

    show(event, d, incomingLinksCount) {
        const linksIn = incomingLinksCount[d.id] || 0;
        const linksOut = d.internal_links_count || 0;
        
        this.tooltip.html(`
            <strong>${d.label}</strong><br>
            Liens entrants: ${linksIn}<br>
            Liens sortants: ${linksOut}<br>
            Cluster: ${d.group}<br>
            ${d.title ? `Description: ${d.title}` : ''}
        `)
        .style("visibility", "visible")
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 10) + "px");
    }

    hide() {
        this.tooltip.style("visibility", "hidden");
    }
}