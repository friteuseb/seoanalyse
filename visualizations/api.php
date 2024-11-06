<?php
header('Content-Type: application/json');

$redis = new Redis();
$redis->connect('redis', 6379);

function getCrawls($redis) {
    $crawls = [];
    foreach ($redis->keys("*:doc_count") as $key) {
        $crawl_id = explode(':', $key)[0];
        $count = $redis->get($key);
        
        // Récupérer les graphes
        $simple_graph = $redis->get($crawl_id . "_simple_graph");
        $clustered_graph = $redis->get($crawl_id . "_clustered_graph");
        
        $documents = getCrawlDocuments($redis, $crawl_id);
        
        $crawls[] = [
            'id' => $crawl_id,
            'count' => $count,
            'documents' => $documents,
            'simple_graph' => json_decode($simple_graph, true),
            'clustered_graph' => json_decode($clustered_graph, true)
        ];
    }
    return $crawls;
}

function getCrawlDocuments($redis, $crawl_id) {
    $documents = [];
    foreach ($redis->keys("$crawl_id:doc:*") as $key) {
        $doc_data = $redis->hGetAll($key);
        if (!$doc_data) continue;
        
        $url = $doc_data['url'] ?? '';
        $content = $doc_data['content'] ?? '';
        $label = $doc_data['label'] ?? '';
        $cluster = isset($doc_data['cluster']) ? (int)$doc_data['cluster'] : 0;
        $internal_links_out = isset($doc_data['internal_links_out']) ? json_decode($doc_data['internal_links_out'], true) : [];
        
        $documents[] = [
            'doc_id' => $key,
            'url' => $url,
            'content' => $content,
            'label' => $label,
            'cluster' => $cluster,
            'internal_links_out' => $internal_links_out
        ];
    }
    return $documents;
}

// Nouveau point d'entrée pour récupérer les graphes
function getGraphs($redis, $crawl_id) {
    $simple_graph = $redis->get($crawl_id . "_simple_graph");
    $clustered_graph = $redis->get($crawl_id . "_clustered_graph");
    
    return [
        'simple_graph' => json_decode($simple_graph, true),
        'clustered_graph' => json_decode($clustered_graph, true)
    ];
}

if (isset($_GET['action'])) {
    switch ($_GET['action']) {
        case 'getCrawls':
            echo json_encode(getCrawls($redis));
            break;
        case 'getCrawlDetails':
            if (isset($_GET['id'])) {
                echo json_encode(getCrawlDocuments($redis, $_GET['id']));
            } else {
                echo json_encode(['error' => 'Missing parameter: id']);
            }
            break;
        case 'getGraphs':
            if (isset($_GET['id'])) {
                echo json_encode(getGraphs($redis, $_GET['id']));
            } else {
                echo json_encode(['error' => 'Missing parameter: id']);
            }
            break;
        default:
            echo json_encode(['error' => 'Unknown action']);
            break;
    }
} else {
    echo json_encode(['error' => 'No action specified']);
}
?>