import redis
import json
import subprocess
from datetime import datetime
from tabulate import tabulate
import re

class RedisExportCLI:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.crawls = []
        
    def list_available_crawls(self):
        """Liste tous les crawls disponibles dans Redis"""
        crawls = []
        pattern = "*:doc:*"
        
        # Ensemble pour stocker les IDs de crawl uniques
        crawl_ids = set()
        
        # Parcours des clés pour extraire les IDs de crawl uniques
        for key in self.redis.scan_iter(pattern):
            try:
                key_str = key.decode('utf-8')
                # Extraction de l'ID du crawl (tout ce qui est avant ':doc:')
                crawl_id = key_str.split(':doc:')[0]
                if crawl_id:  # Vérifier que l'ID n'est pas vide
                    crawl_ids.add(crawl_id)
            except Exception as e:
                print(f"Avertissement : Impossible de traiter la clé {key}: {str(e)}")
        
        # Pour chaque ID de crawl, récupérer les informations
        for index, crawl_id in enumerate(sorted(crawl_ids), 1):
            try:
                # Recherche d'un document valide pour ce crawl
                sample_key = None
                for key in self.redis.scan_iter(f"{crawl_id}:doc:*"):
                    sample_key = key
                    break
                
                if not sample_key:
                    continue
                
                doc_data = self.redis.hgetall(sample_key)
                
                # Extraction de la date du crawl avec gestion d'erreur
                crawl_date = doc_data.get(b'crawl_date', b'Unknown')
                try:
                    date_str = crawl_date.decode('utf-8', errors='ignore')
                    date_obj = datetime.fromisoformat(date_str)
                    formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
                except:
                    formatted_date = 'Date inconnue'
                
                # Comptage sécurisé des pages
                try:
                    page_count = len(list(self.redis.scan_iter(f"{crawl_id}:doc:*")))
                except:
                    page_count = 'N/A'
                
                crawls.append({
                    'index': index,
                    'id': crawl_id,
                    'date': formatted_date,
                    'pages': page_count
                })
                
            except Exception as e:
                print(f"Avertissement : Erreur lors du traitement du crawl {crawl_id}: {str(e)}")
                # Ajouter quand même le crawl avec des informations minimales
                crawls.append({
                    'index': index,
                    'id': crawl_id,
                    'date': 'Erreur de date',
                    'pages': 'Erreur de comptage'
                })
        
        self.crawls = crawls
        return crawls
    
    def display_crawls(self):
        """Affiche les crawls disponibles dans un tableau formaté"""
        headers = ['#', 'ID du Crawl', 'Date', 'Nombre de Pages']
        rows = []
        for c in self.crawls:
            # Formatage plus robuste des données
            row = [
                str(c['index']),
                str(c['id']),
                str(c['date']),
                str(c['pages'])
            ]
            rows.append(row)
        
        print("\n=== CRAWLS DISPONIBLES ===")
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print("\nEntrez le numéro du crawl à exporter (ex: 2)")
    
    def get_crawl_selection(self):
        """Gère la sélection du crawl par l'utilisateur"""
        while True:
            try:
                selection = input("Votre choix : ").strip()
                idx = int(selection)
                
                if 1 <= idx <= len(self.crawls):
                    return next(c['id'] for c in self.crawls if c['index'] == idx)
                else:
                    print(f"Erreur: Veuillez entrer un numéro entre 1 et {len(self.crawls)}")
            except ValueError:
                print("Erreur: Veuillez entrer un nombre valide")
            except Exception as e:
                print(f"Erreur inattendue: {str(e)}")
    
    def export_redis_data(self, crawl_id):
        """Exporte les données Redis pour un crawl donné"""
        try:
            all_data = {}
            pattern = f"{crawl_id}:doc:*"
            
            for key in self.redis.scan_iter(pattern):
                try:
                    key_str = key.decode('utf-8')
                    doc_data = self.redis.hgetall(key)
                    
                    decoded_data = {}
                    for field, value in doc_data.items():
                        field_str = field.decode('utf-8')
                        if field_str == 'content':
                            decoded_data[field_str] = "content truncated..."
                            continue
                            
                        value_str = value.decode('utf-8')
                        if field_str == 'internal_links_out':
                            try:
                                value_str = json.loads(value_str)
                            except:
                                pass
                        decoded_data[field_str] = value_str
                        
                    all_data[key_str] = decoded_data
                except Exception as e:
                    print(f"Avertissement: Impossible de traiter la clé {key}: {str(e)}")
                    continue
            
            # Générer un nom de fichier plus court et sécurisé
            safe_filename = re.sub(r'[^\w\-_]', '_', crawl_id)[:30]
            filename = f"{safe_filename}_metadata.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
                
            print(f"\nDonnées exportées dans {filename}")
            print(f"Nombre d'entrées exportées : {len(all_data)}")
            
        except Exception as e:
            print(f"Erreur lors de l'export: {str(e)}")

def main():
    try:
        # Récupération du port Redis avec gestion d'erreur
        try:
            redis_port = int(subprocess.check_output(
                "ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", 
                shell=True,
                stderr=subprocess.PIPE
            ))
        except subprocess.CalledProcessError as e:
            print("Erreur lors de la récupération du port Redis via ddev.")
            print("Utilisation du port par défaut 6379")
            redis_port = 6379
        except ValueError as e:
            print("Erreur lors de la conversion du port Redis.")
            print("Utilisation du port par défaut 6379")
            redis_port = 6379
        
        # Connexion à Redis avec timeout
        redis_client = redis.Redis(
            host='localhost', 
            port=redis_port, 
            db=0,
            socket_timeout=5,
            decode_responses=False  # Important pour la gestion des bytes
        )
        
        # Test de la connexion
        try:
            redis_client.ping()
        except redis.ConnectionError:
            print("Impossible de se connecter à Redis. Vérifiez que le service est démarré.")
            sys.exit(1)
        
        # Création et exécution du CLI
        cli = RedisExportCLI(redis_client)
        
        # Lister et afficher les crawls
        cli.list_available_crawls()
        cli.display_crawls()
        
        # Obtenir la sélection de l'utilisateur et exporter
        selected_crawl = cli.get_crawl_selection()
        cli.export_redis_data(selected_crawl)
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {str(e)}")

if __name__ == "__main__":
    main()