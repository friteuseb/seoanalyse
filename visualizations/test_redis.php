<?php
// Connexion à Redis
$redis = new Redis();
$redis->connect('localhost', 6379);

// Récupération des clés pour les graphes
$keys = $redis->keys('www_xalis-finances_com__*_graph');

$graphs = [];

foreach ($keys as $key) {
    $graphData = $redis->get($key);
    if ($graphData) {
        $graphs[$key] = json_decode($graphData, true);
    }
}

// Affichage des données (pour le débogage)
echo "<pre>";
print_r($graphs);
echo "</pre>";
?>