<?php
require_once __DIR__ . '/redis_config.php';
header('Content-Type: application/json');

try {
    // Établir la connexion Redis en utilisant notre configuration centralisée
    $redis = get_redis_connection();
    if (!$redis) {
        throw new Exception('Impossible d\'établir la connexion Redis');
    }

    // Test de la connexion
    $redis->ping();

    // Récupération de tous les graphes disponibles
    $allKeys = $redis->keys('*');
    $graphs = [];

    foreach ($allKeys as $key) {
        // Ne récupérer que les clés qui sont des graphes (suffixe _simple_graph ou _clustered_graph)
        if (str_ends_with($key, '_simple_graph') || str_ends_with($key, '_clustered_graph')) {
            $graphId = preg_replace('/_(?:simple|clustered)_graph$/', '', $key);
            
            // Récupérer les méta-données si disponibles
            $metadata = [];
            try {
                $metadata = [
                    'id' => $graphId,
                    'type' => str_ends_with($key, '_simple_graph') ? 'simple' : 'clustered',
                    'created' => $redis->hGet($key . '_metadata', 'created_at') ?? null,
                    'url' => $redis->hGet($key . '_metadata', 'base_url') ?? null,
                    'nodes' => $redis->hGet($key . '_metadata', 'node_count') ?? 0,
                    'links' => $redis->hGet($key . '_metadata', 'link_count') ?? 0
                ];
            } catch (Exception $e) {
                error_log("Erreur lors de la récupération des métadonnées pour $key: " . $e->getMessage());
            }
            
            $graphs[] = $metadata ?: $key;
        }
    }

    // Trier les graphes par date de création (si disponible)
    usort($graphs, function($a, $b) {
        if (is_array($a) && is_array($b) && isset($a['created']) && isset($b['created'])) {
            return strtotime($b['created']) - strtotime($a['created']);
        }
        return 0;
    });

    error_log("Graphes trouvés : " . count($graphs));
    echo json_encode([
        'success' => true,
        'graphs' => $graphs,
        'count' => count($graphs)
    ], JSON_PRETTY_PRINT);

} catch (Exception $e) {
    error_log("Erreur dans get_available_graphs.php: " . $e->getMessage());
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Redis connection failed: ' . $e->getMessage(),
        'details' => 'Unable to fetch available graphs'
    ], JSON_PRETTY_PRINT);
    exit;
}