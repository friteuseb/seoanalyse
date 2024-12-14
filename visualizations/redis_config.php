<?php
function get_redis_connection() {
    try {
        // Configuration Redis pour DDEV
        $redis = new Redis();
        $redis->connect('redis', 6379); // Utiliser le nom du service DDEV
        
        if (!$redis->ping()) {
            throw new Exception("Impossible de pinguer Redis");
        }
        
        error_log("Connexion Redis établie avec succès");
        return $redis;
        
    } catch (Exception $e) {
        error_log("Erreur Redis: " . $e->getMessage());
        return false;
    }
}