<?php
header('Content-Type: application/json');

try {
    // Récupération des paramètres Redis
    $host = getenv('REDIS_HOST') ?: 'localhost';
    $port = getenv('REDIS_PORT') ?: 6379;

    // Si exécuté à l'extérieur du conteneur, utiliser les variables EXTERNAL_*
    if (getenv('DDEV_HOSTNAME') === false) {
        $host = getenv('EXTERNAL_REDIS_HOST') ?: 'localhost';
        $port = getenv('EXTERNAL_REDIS_PORT') ?: 6379;
    }

    // Connexion à Redis
    $redis = new Redis();
    $redis->connect($host, intval($port));

    // Récupération des données
    $graphs = $redis->keys('*');
    echo json_encode($graphs);

} catch (Exception $e) {
    echo json_encode(['error' => 'Redis connection failed: ' . $e->getMessage()]);
    exit;
}
?>
