const CONFIG = {
    colors: [
        "#2D7DD2", // Bleu intense
        "#EEB902", // Orange/Jaune
        "#97CC04", // Vert vif
        "#F45D01", // Orange brûlé
        "#474647"  // Gris pour non classés
    ],
    nodeMinSize: 5,
    nodeMaxSize: 20,
    linkStrength: -300,
    linkDistance: 100
};

function getColor(group) {
    if (group === undefined || group === -1) {
        return CONFIG.colors[4];
    }
    return CONFIG.colors[group % CONFIG.colors.length];
}