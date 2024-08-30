<?php
// testRedis.php

$redis = new Redis();
try {
    $redis->connect('127.0.0.1', 6379);
    echo "Connection to server sucessfully";
    
    // Set the data in redis string
    $redis->set("key", "value");
    // Get the stored data and print it
    $value = $redis->get("key");
    echo "La valeur de 'key' est : " . $value;
} catch (Exception $e) {
    echo "Failed to connect to Redis: " . $e->getMessage();
}
?>

