import redis
import json
from datetime import datetime
from typing import List, Tuple, Dict
import sys
import subprocess
from enhanced_analyzer import EnhancedCoconAnalyzer
from tabulate import tabulate
import re

class CrawlComparisonCLI:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.crawls = []
        
    def list_available_crawls(self) -> List[Dict]:
        """Liste tous les crawls disponibles dans Redis"""
        crawls = []
        # Pattern pour identifier les clés de crawl
        pattern = "*:doc:*"
        
        # Ensemble pour stocker les IDs de crawl uniques
        crawl_ids = set()
        
        # Parcours des clés pour extraire les IDs de crawl uniques
        for key in self.redis.scan_iter(pattern):
            try:
                # Décodage plus robuste
                key_str = key.decode('utf-8', errors='ignore')
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
    
    def display_crawls(self, crawls: List[Dict]):
        """Affiche les crawls disponibles dans un tableau formaté"""
        headers = ['#', 'ID du Crawl', 'Date', 'Nombre de Pages']
        rows = []
        for c in crawls:
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
        print("\nEntrez les numéros des deux crawls à comparer (ex: 1,3)")
    
    def get_crawl_selection(self) -> Tuple[str, str]:
        """Gère la sélection des crawls par l'utilisateur"""
        while True:
            try:
                selection = input("Votre choix (2 numéros séparés par une virgule) : ").strip()
                # Gestion plus flexible de la saisie
                numbers = re.findall(r'\d+', selection)
                if len(numbers) != 2:
                    raise ValueError("Veuillez entrer exactement deux numéros")
                    
                idx1, idx2 = map(int, numbers)
                
                # Vérifier que les indices sont valides
                if 1 <= idx1 <= len(self.crawls) and 1 <= idx2 <= len(self.crawls):
                    crawl1 = next(c['id'] for c in self.crawls if c['index'] == idx1)
                    crawl2 = next(c['id'] for c in self.crawls if c['index'] == idx2)
                    return crawl1, crawl2
                else:
                    print(f"Erreur: Veuillez entrer des numéros entre 1 et {len(self.crawls)}")
            except ValueError as ve:
                print(f"Erreur: {str(ve)}")
            except Exception as e:
                print(f"Erreur inattendue: {str(e)}")
    
    def run_comparison(self):
        """Exécute la comparaison complète"""
        try:
            # Lister et afficher les crawls disponibles
            crawls = self.list_available_crawls()
            if not crawls:
                print("Aucun crawl trouvé dans Redis")
                return
                
            self.display_crawls(crawls)
            
            # Obtenir la sélection de l'utilisateur
            crawl_id1, crawl_id2 = self.get_crawl_selection()
            
            print(f"\nAnalyse comparative en cours...")
            print(f"Crawl 1 (AVANT): {crawl_id1}")
            print(f"Crawl 2 (APRÈS): {crawl_id2}\n")
            
            try:
                analyzer1 = EnhancedCoconAnalyzer(self.redis, crawl_id1)
                metrics1 = analyzer1.calculate_scientific_metrics()
            except Exception as e:
                print(f"Erreur lors de l'analyse du premier crawl: {str(e)}")
                return
                
            try:
                analyzer2 = EnhancedCoconAnalyzer(self.redis, crawl_id2)
                metrics2 = analyzer2.calculate_scientific_metrics()
            except Exception as e:
                print(f"Erreur lors de l'analyse du second crawl: {str(e)}")
                return
            
            # Génération du rapport dans le bon ordre
            try:
                report = analyzer1.generate_scientific_report(metrics1, metrics2)  # Ordre normal
                
                # Afficher le rapport
                print("\n" + "="*50)
                print("RAPPORT D'ANALYSE COMPARATIVE")
                print("="*50)
                print(report)
                
                # Proposer la sauvegarde du rapport
                save = input("\nSouhaitez-vous sauvegarder ce rapport ? (o/n) : ").strip().lower()
                if save == 'o':
                    filename = f"comparison_report_{crawl_id1}_vs_{crawl_id2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"Rapport sauvegardé dans : {filename}")
                    
            except Exception as e:
                print(f"Erreur lors de la génération du rapport: {str(e)}")
                
        except Exception as e:
            print(f"Une erreur est survenue: {str(e)}")
            raise

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
        cli = CrawlComparisonCLI(redis_client)
        cli.run_comparison()
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()