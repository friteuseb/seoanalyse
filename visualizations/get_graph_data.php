<?php
require_once 'redis_config.php';
header('Content-Type: application/json');

$graph = $_GET['graph'] ?? '';

if (empty($graph)) {
    echo json_encode(['error' => 'No graph specified']);
    exit;
}

$redis = get_redis_connection();
$data = $redis->get($graph);

if ($data === false) {
    echo json_encode(['error' => 'Graph not found']);
    exit;
}

echo $data;