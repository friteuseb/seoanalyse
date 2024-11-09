import redis
import json
from datetime import datetime
from typing import List, Tuple, Dict
import sys
from enhanced_analyzer import EnhancedCoconAnalyzer
from tabulate import tabulate
import subprocess

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
            crawl_id = key.decode('utf-8').split(':')[0]
            crawl_ids.add(crawl_id)
        
        # Pour chaque ID de crawl, récupérer les informations
        for index, crawl_id in enumerate(sorted(crawl_ids), 1):
            # Récupérer un document pour obtenir la date de crawl
            sample_key = next(self.redis.scan_iter(f"{crawl_id}:doc:*"))
            doc_data = self.redis.hgetall(sample_key)
            
            # Extraire la date du crawl
            crawl_date = doc_data.get(b'crawl_date', b'Unknown').decode('utf-8')
            try:
                date_obj = datetime.fromisoformat(crawl_date)
                formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_date = 'Date inconnue'
            
            # Compter le nombre de pages
            page_count = len(list(self.redis.scan_iter(f"{crawl_id}:doc:*")))
            
            crawls.append({
                'index': index,
                'id': crawl_id,
                'date': formatted_date,
                'pages': page_count
            })
        
        self.crawls = crawls
        return crawls
    
    def display_crawls(self, crawls: List[Dict]):
        """Affiche les crawls disponibles dans un tableau formaté"""
        headers = ['#', 'ID du Crawl', 'Date', 'Nombre de Pages']
        rows = [[c['index'], c['id'], c['date'], c['pages']] for c in crawls]
        
        print("\n=== CRAWLS DISPONIBLES ===")
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        print("\nEntrez les numéros des deux crawls à comparer (ex: 1,3)")
    
    def get_crawl_selection(self) -> Tuple[str, str]:
        """Gère la sélection des crawls par l'utilisateur"""
        while True:
            try:
                selection = input("Votre choix (2 numéros séparés par une virgule) : ")
                idx1, idx2 = map(int, selection.split(','))
                
                # Vérifier que les indices sont valides
                if 1 <= idx1 <= len(self.crawls) and 1 <= idx2 <= len(self.crawls):
                    crawl1 = next(c['id'] for c in self.crawls if c['index'] == idx1)
                    crawl2 = next(c['id'] for c in self.crawls if c['index'] == idx2)
                    return crawl1, crawl2
                else:
                    print(f"Erreur: Veuillez entrer des numéros entre 1 et {len(self.crawls)}")
            except ValueError:
                print("Erreur: Format invalide. Utilisez deux numéros séparés par une virgule (ex: 1,3)")
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
            print(f"Crawl 1: {crawl_id1}")
            print(f"Crawl 2: {crawl_id2}")
            
            # Initialiser les analyseurs
            analyzer1 = EnhancedCoconAnalyzer(self.redis, crawl_id1)
            analyzer2 = EnhancedCoconAnalyzer(self.redis, crawl_id2)
            
            # Calculer les métriques
            metrics1 = analyzer1.calculate_scientific_metrics()
            metrics2 = analyzer2.calculate_scientific_metrics()
            
            # Générer le rapport
            report = analyzer1.generate_scientific_report(metrics1, metrics2)
            
            # Afficher le rapport
            print("\n" + "="*50)
            print("RAPPORT D'ANALYSE COMPARATIVE")
            print("="*50)
            print(report)
            
            # Proposer la sauvegarde du rapport
            save = input("\nSouhaitez-vous sauvegarder ce rapport ? (o/n) : ")
            if save.lower() == 'o':
                filename = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"Rapport sauvegardé dans : {filename}")
                
        except Exception as e:
            print(f"Une erreur est survenue: {str(e)}")

def main():
    try:
        # Récupération du port Redis comme dans votre code original
        redis_port = int(subprocess.check_output(
            "ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", 
            shell=True
        ))
        redis_client = redis.Redis(host='localhost', port=redis_port, db=0)
        
        # Création et exécution du CLI
        cli = CrawlComparisonCLI(redis_client)
        cli.run_comparison()
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
