<?php
require_once 'redis_config.php';
header('Content-Type: application/json');

$redis = get_redis_connection();

function getCrawls($redis) {
    $crawls = [];
    foreach ($redis->keys("*:doc:*") as $key) {
        $doc_data = $redis->hgetall($key);
        if (!$doc_data) continue;

        $url = $doc_data[b'url'].decode('utf-8');
        $cluster = isset($doc_data[b'cluster']) ? (int)$doc_data[b'cluster'].decode('utf-8') : 0;
        $label = $doc_data[b'label'].decode('utf-8') ?? '';
        $internal_links_out = json_decode($doc_data[b'internal_links_out'].decode('utf-8'), true);
        
        $crawls[] = [
            'id' => $key.decode('utf-8'),
            'url' => $url,
            'cluster' => $cluster,
            'label' => $label,
            'internal_links' => $internal_links_out
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