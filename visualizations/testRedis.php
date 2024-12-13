<?php
require 'redis_config.php';

$redis = get_redis_connection();
$redis->set('test_key', 'test_value');
echo json_encode(['success' => 'Redis connecté avec succès', 'value' => $redis->get('test_key')]);
?>
