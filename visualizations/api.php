<?php
header('Content-Type: application/json');

$redis = new Redis();
$redis->connect('redis', 6379);

function getCrawls($redis) {
    $crawls = [];
    foreach ($redis->keys("*:doc_count") as $key) {
        $crawl_id = explode(':', $key)[0];
        $count = $redis->get($key);
        $documents = getCrawlDocuments($redis, $crawl_id); // Ajout des documents
        $crawls[] = [
            'id' => $crawl_id,
            'count' => $count,
            'documents' => $documents // Inclure les documents
        ];
    }
    return $crawls;
}

function getCrawlDocuments($redis, $crawl_id) {
    $documents = [];
    foreach ($redis->keys("$crawl_id:doc:*") as $key) {
        $doc_data = $redis->hGetAll($key);
        $doc_id = $key;
        $url = $doc_data['url'];
        $content = $doc_data['content'];
        $label = isset($doc_data['label']) ? $doc_data['label'] : '';
        $cluster = isset($doc_data['cluster']) ? (int)$doc_data['cluster'] : 0;
        $internal_links_out = isset($doc_data['internal_links_out']) ? explode(',', $doc_data['internal_links_out']) : [];
        $documents[] = [
            'doc_id' => $doc_id,
            'url' => $url,
            'content' => $content,
            'label' => $label,
            'cluster' => $cluster,
            'internal_links_out' => $internal_links_out
        ];
    }
    return $documents;
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
        default:
            echo json_encode(['error' => 'Unknown action']);
            break;
    }
} else {
    echo json_encode(['error' => 'No action specified']);
}
?>
