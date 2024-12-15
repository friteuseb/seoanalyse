const CONFIG = {
    colors: [
        "#4FC3F7", // Bleu clair lumineux
        "#00B0FF", // Bleu électrique
        "#2979FF", // Bleu intense
        "#304FFE", // Bleu indigo
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