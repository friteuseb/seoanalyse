<?php
require_once 'redis_config.php';
header('Content-Type: application/json');

try {
    $graph = $_GET['graph'] ?? '';
    error_log("Requested graph: " . $graph);
    
    if (empty($graph)) {
        throw new Exception('No graph specified');
    }

    $redis = get_redis_connection();
    if (!$redis) {
        throw new Exception('Redis connection failed');
    }
    
    $nodes = [];
    $links = [];
    $urlToId = [];
    $nodeId = 0;

    // Récupérer tous les documents du crawl
    $pattern = str_replace(['_simple_graph', '_clustered_graph'], ':doc:*', $graph);
    error_log("Searching pattern: " . $pattern);
    
    // Première passe : créer tous les nœuds
    $keys = $redis->keys($pattern);
    error_log("Found " . count($keys) . " documents");
    
    foreach ($keys as $key) {
        $doc = $redis->hGetAll($key);
        if (!$doc) {
            error_log("Skipping empty doc for key: " . $key);
            continue;
        }

        $url = $doc['url'];
        $id = 'node_' . $nodeId++;
        $urlToId[$url] = $id;
        
        // Simplifier le label
        $label = basename(rtrim($url, '/'));
        
        // Créer le nœud
        $nodes[] = [
            'id' => $id,
            'url' => $url,
            'label' => $label,
            'internal_links_count' => (int)($doc['links_count'] ?? 0),
            'group' => (int)($doc['cluster'] ?? -1),
            'title' => $doc['cluster_description'] ?? "Cluster " . ($doc['cluster'] ?? -1)
        ];
        
        error_log("Created node for URL: " . $url);
    }

    // Deuxième passe : créer tous les liens
    error_log("Starting link processing...");
    foreach ($keys as $key) {
        $doc = $redis->hGetAll($key);
        if (!$doc || !isset($doc['url']) || !isset($doc['internal_links_out'])) {
            continue;
        }

        $sourceUrl = $doc['url'];
        $internal_links = json_decode($doc['internal_links_out'], true);
        
        if ($internal_links) {
            error_log("Processing links for: " . $sourceUrl . " - Found " . count($internal_links) . " links");
            
            foreach ($internal_links as $targetUrl) {
                if (isset($urlToId[$sourceUrl]) && isset($urlToId[$targetUrl])) {
                    $processed_links[] = [
                        'source' => $urlToId[$sourceUrl],
                        'target' => $urlToId[$targetUrl],
                        'value' => 1
                    ];
                }
            }
        }
    }

    // Log final pour débogage
    error_log("Final counts - Nodes: " . count($nodes) . ", Links: " . count($processed_links));
    if (!empty($processed_links)) {
        error_log("Sample links: " . json_encode(array_slice($processed_links, 0, 5)));
    }

    $result = [
        'nodes' => array_values($nodes),
        'links' => array_values($processed_links)
    ];
    
    echo json_encode($result);

} catch (Exception $e) {
    error_log("Error in get_graph_data.php: " . $e->getMessage());
    http_response_code(500);
    echo json_encode([
        'error' => $e->getMessage(),
        'details' => 'Error occurred while fetching graph data'
    ]);
}
?>