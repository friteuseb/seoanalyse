<?php
function get_redis_connection() {
    try {
        // Si on est dans DDEV (environnement conteneurisé)
        if (getenv('IS_DDEV_PROJECT') === 'true') {
            $redis = new Redis();
            
            $host = getenv('REDIS_HOST') ?: 'redis';
            $port = getenv('REDIS_PORT') ?: 6379;
            
            $redis->connect($host, $port);
            error_log("Connexion Redis interne établie sur $host:$port");
            return $redis;
        }
        
        // En dehors de DDEV - récupérer le port exposé
        $redis = new Redis();
        
        // Essayer d'abord de récupérer le port depuis le descriptor DDEV
        $port = null;
        $json = shell_exec('ddev describe -j 2>/dev/null');
        if ($json) {
            $config = json_decode($json, true);
            if (isset($config['raw']['services']['redis-1']['host_ports'])) {
                $port = explode(',', $config['raw']['services']['redis-1']['host_ports'])[0];
            }
        }
        
        // Si on n'a pas pu récupérer le port, utiliser la variable d'environnement ou la valeur par défaut
        if (!$port) {
            $port = getenv('REDIS_EXTERNAL_PORT') ?: 6379;
        }
        
        $redis->connect('127.0.0.1', $port);
        error_log("Connexion Redis externe établie sur le port $port");
        
        // Vérifier que la connexion est bien établie
        if ($redis->ping()) {
            return $redis;
        } else {
            throw new Exception("Échec du ping Redis");
        }

    } catch (Exception $e) {
        error_log("Erreur Redis : " . $e->getMessage());
        
        // Dernière tentative sur le port par défaut
        try {
            $redis = new Redis();
            $redis->connect('127.0.0.1', 6379);
            if ($redis->ping()) {
                error_log("Connexion Redis de secours établie sur port 6379");
                return $redis;
            }
        } catch (Exception $e) {
            error_log("Échec de la connexion de secours : " . $e->getMessage());
        }
        
        return false;
    }
}