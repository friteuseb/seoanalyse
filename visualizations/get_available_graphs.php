<?php
header('Content-Type: application/json');

$redis = new Redis();
$redis->connect('redis', 6379);

$keys = $redis->keys('*_graph');

echo json_encode($keys);