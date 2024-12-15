<?php
require_once __DIR__ . '/redis_config.php';
error_reporting(E_ALL);
ini_set('display_errors', 0);
header('Content-Type: application/json');

function normalizeUrl($url) {
    // Retire le protocole (http:// ou https://)
    $url = preg_replace('#^https?://#', '', $url);
    
    // Retire les slashes de fin
    $url = rtrim($url, '/');
    
    // Retire les paramètres d'URL (tout ce qui suit ?)
    $url = preg_replace('/\?.*$/', '', $url);
    
    // Retire les ancres (tout ce qui suit #)
    $url = preg_replace('/#.*$/', '', $url);
    
    // Convertit en minuscules pour une comparaison insensible à la casse
    $url = strtolower($url);
    
    // Remplace les doubles slashes par des simples
    $url = preg_replace('#/+#', '/', $url);
    
    // Retire index.php, index.html, etc.
    $url = preg_replace('/\/index\.(php|html|htm)$/', '', $url);
    
    return $url;
}

function getDisplayLabel($url) {
    // Extrait le dernier segment de l'URL pour l'affichage
    $path = parse_url($url, PHP_URL_PATH);
    if (!$path) return $url;
    
    $segments = explode('/', trim($path, '/'));
    $lastSegment = end($segments);
    
    // Retire les extensions courantes
    $lastSegment = preg_replace('/\.(html|htm|php)$/', '', $lastSegment);
    
    // Remplace les tirets et underscores par des espaces
    $lastSegment = str_replace(['-', '_'], ' ', $lastSegment);
    
    // Limite la longueur pour l'affichage
    if (strlen($lastSegment) > 50) {
        $lastSegment = substr($lastSegment, 0, 47) . '...';
    }
    
    return $lastSegment;
}

try {
    $graph = $_GET['graph'] ?? '';
    error_log("Processing request for graph: " . $graph);
    
    if (empty($graph)) {
        throw new Exception('No graph specified');
    }

    $redis = get_redis_connection();
    if (!$redis) {
        throw new Exception('Redis connection failed');
    }

    $pattern = str_replace(['_simple_graph', '_clustered_graph'], ':doc:*', $graph);
    error_log("Using pattern: " . $pattern);
    
    $keys = $redis->keys($pattern);
    error_log("Found " . count($keys) . " keys");
    
    if (empty($keys)) {
        throw new Exception('No data found for pattern: ' . $pattern);
    }

    $nodes = [];
    $links = [];
    $urlToId = [];
    $nodeId = 0;
    $urlMapping = []; // Pour stocker la correspondance entre URLs normalisées et originales

    // Premier passage : création des nœuds
    foreach ($keys as $key) {
        try {
            $doc = $redis->hGetAll($key);
            if (empty($doc) || empty($doc['url'])) {
                continue;
            }

            $originalUrl = $doc['url'];
            $normalizedUrl = normalizeUrl($originalUrl);
            
            // Évite les doublons d'URLs normalisées
            if (isset($urlMapping[$normalizedUrl])) {
                continue;
            }
            
            $id = 'node_' . $nodeId++;
            $urlToId[$normalizedUrl] = $id;
            $urlMapping[$normalizedUrl] = $originalUrl;

            $nodes[] = [
                'id' => $id,
                'url' => $originalUrl,
                'label' => getDisplayLabel($originalUrl),
                'internal_links_count' => (int)($doc['links_count'] ?? 0),
                'group' => (int)($doc['cluster'] ?? -1),
                'title' => $doc['cluster_description'] ?? "Cluster " . ($doc['cluster'] ?? -1)
            ];
            
        } catch (Exception $e) {
            error_log("Error processing node: " . $e->getMessage());
        }
    }

    // Second passage : création des liens
    foreach ($keys as $key) {
        try {
            $doc = $redis->hGetAll($key);
            if (!isset($doc['url'], $doc['internal_links_out'])) continue;

            $sourceUrl = normalizeUrl($doc['url']);
            $internalLinks = json_decode($doc['internal_links_out'], true);

            if ($internalLinks) {
                foreach ($internalLinks as $targetUrl) {
                    $normalizedTargetUrl = normalizeUrl($targetUrl);
                    
                    if (isset($urlToId[$sourceUrl], $urlToId[$normalizedTargetUrl])) {
                        // Vérifie si le lien n'existe pas déjà
                        $linkExists = false;
                        foreach ($links as $existingLink) {
                            if ($existingLink['source'] === $urlToId[$sourceUrl] && 
                                $existingLink['target'] === $urlToId[$normalizedTargetUrl]) {
                                $linkExists = true;
                                break;
                            }
                        }
                        
                        if (!$linkExists) {
                            $links[] = [
                                'source' => $urlToId[$sourceUrl],
                                'target' => $urlToId[$normalizedTargetUrl],
                                'value' => 1
                            ];
                        }
                    }
                }
            }
        } catch (Exception $e) {
            error_log("Error processing links: " . $e->getMessage());
        }
    }

    error_log("Final count - Nodes: " . count($nodes) . ", Links: " . count($links));

    $result = [
        'nodes' => array_values($nodes),
        'links' => array_values($links)
    ];

    if (empty($nodes)) {
        throw new Exception('No valid nodes found');
    }

    echo json_encode($result, JSON_PRETTY_PRINT);
    exit;

} catch (Exception $e) {
    $error = [
        'error' => $e->getMessage(),
        'details' => 'Failed to process graph data'
    ];
    error_log("Error in get_graph_data.php: " . $e->getMessage());
    http_response_code(500);
    echo json_encode($error);
    exit;
}
?>