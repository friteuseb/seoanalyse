const CONFIG = {
    colors: [
        "#4FC3F7", // Bleu électrique clair
        "#2196F3", // Bleu material
        "#1976D2", // Bleu profond
        "#0D47A1", // Bleu intense
        "#666666"  // Gris pour non classés
    ],
    nodeMinSize: 5,
    nodeMaxSize: 25,
    linkStrength: -300,
    linkDistance: 100
};

function getColor(group) {
    if (group === undefined || group === -1) {
        return CONFIG.colors[4];
    }
    return CONFIG.colors[group % CONFIG.colors.length];
}


