# analyze_semantic_juice.py

import redis
import sys
import subprocess
import json
import logging
from termcolor import colored
from semantic_juice_analyzer import SemanticJuiceAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_redis_port():
    try:
        port = subprocess.check_output(
            "ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", 
            shell=True
        )
        return int(port.strip())
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du port Redis : {e}")
        sys.exit(1)

def print_metrics_table(metrics):
    """Affiche les métriques dans un tableau coloré"""
    print("\n" + "="*60)
    print(colored("MÉTRIQUES DE TRANSFERT DE JUS SÉMANTIQUE", "cyan", attrs=["bold"]))
    print("="*60)
    
    metrics_info = {
        'semantic_coherence': ('Cohérence sémantique', 'Mesure la cohérence thématique globale'),
        'juice_efficiency': ('Efficacité du transfert', 'Évalue l\'efficacité de la distribution du jus'),
        'theme_preservation': ('Préservation thématique', 'Mesure la conservation des thématiques'),
        'link_relevance': ('Pertinence des liens', 'Évalue la pertinence sémantique des liens')
    }
    
    for key, value in metrics.items():
        name, description = metrics_info[key]
        score = value
        color = 'green' if score > 0.7 else 'yellow' if score > 0.4 else 'red'
        
        print(f"\n{colored(name, 'white', attrs=['bold'])}")
        print(f"Score: {colored(f'{score:.2f}', color)}")
        print(f"Description: {description}")

def generate_visualizations(analyzer, results):
    """Génère les visualisations des résultats"""
    
    # 1. Heatmap de la matrice de transition sémantique
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        analyzer.semantic_transition_matrix,
        cmap='YlOrRd',
        xticklabels=False,
        yticklabels=False
    )
    plt.title('Matrice de Transition Sémantique')
    plt.savefig('semantic_transition_matrix.png')
    plt.close()
    
    # 2. Distribution du PageRank sémantique
    plt.figure(figsize=(12, 6))
    pd.Series(analyzer.semantic_pagerank).hist(bins=30)
    plt.title('Distribution du PageRank Sémantique')
    plt.xlabel('Score PageRank')
    plt.ylabel('Nombre de Pages')
    plt.savefig('semantic_pagerank_distribution.png')
    plt.close()
    
    # 3. Comparaison des thématiques avant/après
    themes_before = results['themes_before']
    themes_after = results['themes_after']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Thématiques avant
    theme_strengths_before = [np.mean(theme) for theme in themes_before.values()]
    ax1.bar(range(len(theme_strengths_before)), theme_strengths_before)
    ax1.set_title('Forces des Thématiques Avant Maillage')
    
    # Thématiques après
    theme_strengths_after = [np.mean(theme) for theme in themes_after.values()]
    ax2.bar(range(len(theme_strengths_after)), theme_strengths_after)
    ax2.set_title('Forces des Thématiques Après Maillage')
    
    plt.tight_layout()
    plt.savefig('thematic_comparison.png')
    plt.close()

def save_results(results, crawl_id):
    """Sauvegarde les résultats dans un fichier JSON"""
    filename = f"semantic_analysis_{crawl_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"Résultats sauvegardés dans {filename}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_semantic_juice.py <crawl_id>")
        sys.exit(1)

    crawl_id = sys.argv[1]
    
    print(colored("\n=== ANALYSE DU JUS SÉMANTIQUE ===", "cyan", attrs=["bold"]))
    print(f"Crawl ID: {crawl_id}")
    
    try:
        # Initialisation
        redis_client = redis.Redis(host='localhost', port=get_redis_port(), db=0)
        analyzer = SemanticJuiceAnalyzer(redis_client, crawl_id)
        
        # Analyse
        logging.info("Démarrage de l'analyse sémantique...")
        results = analyzer.analyze_semantic_juice()
        
        if not results:
            logging.error("L'analyse n'a pas pu être effectuée.")
            return
        
        # Affichage des résultats
        print_metrics_table(results['metrics'])
        
        # Génération des visualisations
        logging.info("Génération des visualisations...")
        generate_visualizations(analyzer, results)
        
        # Sauvegarde des résultats
        save_results(results, crawl_id)
        
        print(colored("\n✅ Analyse terminée avec succès!", "green"))
        print("""
        📊 Visualisations générées:
        - semantic_transition_matrix.png: Matrice de transition sémantique
        - semantic_pagerank_distribution.png: Distribution du PageRank
        - thematic_comparison.png: Comparaison des thématiques
        """)
        
    except Exception as e:
        logging.error(f"Une erreur est survenue: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()