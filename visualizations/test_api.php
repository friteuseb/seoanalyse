<?php
// get_available_graphs.php
require_once 'redis_config.php';

// Désactiver l'affichage des erreurs PHP dans la sortie
ini_set('display_errors', 0);
error_reporting(0);

// S'assurer que la réponse est toujours en JSON
header('Content-Type: application/json');

try {
    $redis = get_redis_connection();
    
    // Debug: Vérifier la connexion
    if (!$redis->ping()) {
        throw new Exception("Redis connection failed");
    }
    
    // Récupérer les graphes
    $keys = $redis->keys('*_graph');
    
    // Debug: Vérifier le contenu
    error_log("Found graphs: " . print_r($keys, true));
    
    echo json_encode([
        'status' => 'success',
        'graphs' => $keys
    ]);
    
} catch (Exception $e) {
    // Log l'erreur pour debug
    error_log("Redis error: " . $e->getMessage());
    
    // Renvoyer une erreur formatée en JSON
    echo json_encode([
        'status' => 'error',
        'message' => $e->getMessage()
    ]);
}