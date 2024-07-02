<?php
header('Content-Type: application/json');

require 'vendor/autoload.php';

$redis = new Predis\Client();

$crawl_id = isset($_GET['crawl_id']) ? $_GET['crawl_id'] : 'default_crawl';
$suggestions_key = "{$crawl_id}:link_suggestions";

$suggestions = $redis->get($suggestions_key);

if ($suggestions) {
    echo $suggestions;
} else {
    echo json_encode(['error' => 'No suggestions found']);
}
?>
