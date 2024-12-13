<?php
function get_redis_connection() {
    try {
        // Exécuter la commande shell pour récupérer le port Redis
        $portOutput = shell_exec("ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'");
        
        // Log du port récupéré
        error_log("Port Redis récupéré : $portOutput");

        // Vérifier si la commande a réussi
        if ($portOutput === null || $portOutput === false) {
            throw new Exception("Échec de l'exécution de la commande shell pour récupérer le port Redis");
        }

        // Nettoyer et valider la sortie de la commande
        $port = trim($portOutput);
        if (!is_numeric($port)) {
            throw new Exception("Le port Redis récupéré est invalide : '$port'");
        }

        // Connexion à Redis
        $redis = new Redis();
        $redis->connect('localhost', intval($port));

        // Log de la connexion réussie
        error_log("Connexion réussie à Redis sur le port : $port");

        return $redis;
    } catch (Exception $e) {
        // Retourner une réponse JSON avec l'erreur
        error_log("Erreur lors de la connexion Redis : " . $e->getMessage());
        header('Content-Type: application/json');
        echo json_encode(['error' => 'Redis connection failed: ' . $e->getMessage()]);
        exit;
    }
}
