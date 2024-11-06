import redis
import json
import subprocess

def export_redis_data(crawl_id):
    # Connexion à Redis
    redis_port = subprocess.check_output("ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", shell=True)
    r = redis.Redis(host='localhost', port=int(redis_port), db=0)
    
    # Récupération de toutes les clés pour ce crawl
    all_data = {}
    
    # Pattern pour les documents
    pattern = f"{crawl_id}:doc:*"
    
    # Pour chaque clé correspondant au pattern
    for key in r.scan_iter(pattern):
        key_str = key.decode('utf-8')
        doc_data = r.hgetall(key)
        
        # Conversion des données binaires en strings, en excluant le contenu complet
        decoded_data = {}
        for field, value in doc_data.items():
            field_str = field.decode('utf-8')
            # On saute le contenu complet
            if field_str == 'content':
                decoded_data[field_str] = "content truncated..."
                continue
                
            value_str = value.decode('utf-8')
            # Tenter de décoder le JSON pour internal_links_out
            if field_str == 'internal_links_out':
                try:
                    value_str = json.loads(value_str)
                except:
                    pass
            decoded_data[field_str] = value_str
            
        all_data[key_str] = decoded_data
    
    # Export en JSON
    with open(f'{crawl_id}_metadata.json', 'w') as f:
        json.dump(all_data, f, indent=2)
        
    print(f"Données exportées dans {crawl_id}_metadata.json")

# Usage
crawl_id = "chadyagamma_fr_guide-sonotherapie__9369b67c-6244-4113-8549-0245325ccbbe"
export_redis_data(crawl_id)
