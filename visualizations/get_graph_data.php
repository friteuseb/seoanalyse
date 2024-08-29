<?php
header('Content-Type: application/json');

$redis = new Redis();
$redis->connect('redis', 6379);

$graph = $_GET['graph'] ?? '';

if (empty($graph)) {
    echo json_encode(['error' => 'No graph specified']);
    exit;
}

$data = $redis->get($graph);

if ($data === false) {
    echo json_encode(['error' => 'Graph not found']);
    exit;
}

echo $data;