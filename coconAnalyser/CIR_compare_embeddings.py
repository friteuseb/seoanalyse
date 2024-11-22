import redis
import sys
import subprocess
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from enhanced_analyzer import EnhancedCoconAnalyzer

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingsComparator:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def compare_multiple_crawls(self, base_crawl_id, model_configs):
        """Compare plusieurs crawls d'embeddings avec un crawl de base"""
        try:
            print(f"Analyse du crawl de base: {base_crawl_id}")
            base_analyzer = EnhancedCoconAnalyzer(self.redis, base_crawl_id)
            base_metrics = base_analyzer.calculate_scientific_metrics()
            print("Métriques de base calculées avec succès")
            
            # Analyse de chaque variation
            embedding_metrics = []
            model_names = []
            
            print("\nAnalyse des différentes variations:")
            for crawl_id, model, threshold in model_configs:
                try:
                    print(f"Traitement de {model} (threshold={threshold})")
                    print(f"Crawl ID: {crawl_id}")
                    analyzer = EnhancedCoconAnalyzer(self.redis, crawl_id)
                    metrics = analyzer.calculate_scientific_metrics()
                    embedding_metrics.append(metrics)
                    model_names.append(f"{model}-t{threshold}")
                    print(f"✓ {model}-t{threshold} analysé avec succès\n")
                except Exception as e:
                    print(f"❌ Erreur lors de l'analyse de {model}-t{threshold}: {str(e)}\n")
                    continue
            
            if not embedding_metrics:
                print("Aucune métrique n'a pu être calculée")
                return None
            
            # Génération du tableau comparatif
            print("Génération du tableau comparatif...")
            comparison_table = self.generate_embeddings_comparison(
                base_metrics, 
                embedding_metrics,
                model_names
            )
            
            # Export en différents formats et génération des visualisations
            if comparison_table is not None:
                self._export_results(comparison_table)
                self._generate_comparison_plots(comparison_table)
            
            return comparison_table
            
        except Exception as e:
            logging.error(f"Erreur lors de la comparaison des crawls: {str(e)}")
            traceback.print_exc()
            return None

    def generate_embeddings_comparison(self, base_metrics, embedding_metrics_list, model_names):
        """Génère un tableau comparatif des différents modèles d'embeddings"""
        try:
            # Définition des métriques clés à comparer
            key_metrics = {
                'Densité & Connectivité': [
                    ('structural_metrics', 'density', '0.1-0.3 est optimal pour la lisibilité'),
                    ('structural_metrics', 'average_clustering', 'Mesure la formation de communautés'),
                    ('structural_metrics', 'reciprocity', 'Équilibre des liens bidirectionnels'),
                ],
                'Accessibilité': [
                    ('accessibility_metrics', 'pages_within_3_clicks', 'Idéalement >80%'),
                    ('accessibility_metrics', 'mean_depth', 'Idéalement entre 1.5 et 2.5'),
                    ('accessibility_metrics', 'pagerank_entropy', 'Plus élevé = meilleure répartition'),
                ],
                'Structure des Clusters': [
                    ('cluster_metrics', 'cluster_density', 'Force des liens thématiques'),
                    ('structural_metrics', 'thematic_ratio', 'Balance thématique/transversal')
                ]
            }

            data = []
            for i, metrics in enumerate(embedding_metrics_list):
                model_data = {
                    'Modèle': model_names[i],
                    'Threshold': int(model_names[i].split('-t')[1]),
                    'Liens totaux': metrics['structural_metrics'].get('number_of_edges', 0),
                }
                
                # Calcul des métriques et variations
                for category, metric_list in key_metrics.items():
                    for metric_path, metric_name, label in metric_list:
                        try:
                            value = metrics.get(metric_path, {}).get(metric_name, 0)
                            base_value = base_metrics.get(metric_path, {}).get(metric_name, 0)
                            
                            if base_value > 0:
                                variation = ((value - base_value) / base_value) * 100
                            else:
                                variation = float('inf') if value > 0 else 0
                            
                            model_data[label] = f"{value:.2f} ({variation:+.1f}%)"
                        except Exception as e:
                            logging.error(f"Erreur pour la métrique {metric_name}: {str(e)}")
                            model_data[label] = "0.00 (+0.0%)"
                
                # Calcul du score global objectif
                model_data['Score Global'] = self._calculate_objective_score(metrics)
                data.append(model_data)
                
            # Réorganisation des colonnes dans un ordre logique
            column_order = [
                'Modèle',
                'Threshold',
                'Liens totaux',
                '0.1-0.3 est optimal pour la lisibilité',
                'Mesure la formation de communautés',
                'Équilibre des liens bidirectionnels',
                'Idéalement >80%',
                'Idéalement entre 1.5 et 2.5',
                'Plus élevé = meilleure répartition',
                'Force des liens thématiques',
                'Balance thématique/transversal',
                'Score Global'
            ]

            # Création et tri du DataFrame
            df = pd.DataFrame(data)
            df = df.sort_values('Score Global', ascending=False)
            df = df[column_order]  # Réorganisation des colonnes
            
            # Réinitialisation de l'index pour avoir une numérotation propre
            df = df.reset_index(drop=True)
            
            return df

        except Exception as e:
            logging.error(f"Erreur lors de la génération du tableau comparatif: {str(e)}")
            traceback.print_exc()
            return None

    def _calculate_objective_score(self, metrics):
        """Calcule un score objectif basé sur des critères scientifiques"""
        try:
            score = 0
            
            # Densité (0-20 points)
            density = metrics['structural_metrics'].get('density', 0)
            if 0.1 <= density <= 0.3:
                score += 20 * (1 - abs(0.2 - density) / 0.1)
            
            # Accessibilité (0-30 points)
            accessibility = metrics['accessibility_metrics'].get('pages_within_3_clicks', 0)
            score += min(30, accessibility * 30)
            
            # Profondeur (0-15 points)
            depth = metrics['accessibility_metrics'].get('mean_depth', 0)
            if 1.5 <= depth <= 2.5:
                score += 15 * (1 - abs(2 - depth) / 0.5)
            
            # Distribution PageRank (0-15 points)
            entropy = metrics['accessibility_metrics'].get('pagerank_entropy', 0)
            score += min(15, entropy * 2.5)  # Normalisé pour un maximum autour de 6
            
            # Cohérence thématique (0-20 points)
            cluster_density = metrics['cluster_metrics'].get('cluster_density', 0)
            score += min(20, cluster_density * 40)
            
            return round(score, 1)
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul du score: {str(e)}")
            return 0

    def _export_results(self, comparison_table):
        """Export les résultats dans différents formats"""
        try:
            comparison_table.to_csv('embedding_comparison.csv')
            with open('embedding_comparison.md', 'w') as f:
                f.write(comparison_table.to_markdown())
            print("Exports générés avec succès")
        except Exception as e:
            logging.error(f"Erreur lors de l'export des résultats: {str(e)}")

    def _generate_comparison_plots(self, comparison_table):
        """Génère des visualisations des comparaisons"""
        try:
            # Score global par modèle
            plt.figure(figsize=(15, 8))
            ax = plt.gca()
            models = comparison_table['Modèle']
            scores = comparison_table['Score Global']
            
            bars = ax.bar(range(len(models)), scores)
            ax.set_title('Comparaison des Scores Globaux par Modèle')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('scores_comparison.png')
            plt.close()

            # Métriques clés
            metrics_to_plot = ['0.1-0.3 est optimal pour la lisibilité', 
                            'Idéalement >80%', 
                            'Force des liens thématiques']
            
            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 6))
            
            for i, metric in enumerate(metrics_to_plot):
                ax = axes[i]
                data = comparison_table[['Modèle', metric]].copy()
                data[metric] = data[metric].apply(lambda x: float(x.split()[0]))
                
                bars = ax.bar(range(len(models)), data[metric])
                ax.set_title(metric)
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig('metrics_comparison.png')
            plt.close()
                
        except Exception as e:
            logging.error(f"Erreur lors de la génération des graphiques: {str(e)}")

def main():
    try:
        # Configuration Redis
        redis_port = subprocess.check_output(
            "ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", 
            shell=True
        ).decode('utf-8').strip()
        
        redis_client = redis.Redis(host='localhost', port=int(redis_port), db=0)
        
        # Configuration des modèles à comparer
        base_crawl = "semantic-suggestion_ecotechlab_fr__17247f41-c326-42a6-9919-4668ddb49453"
        
        model_configs = [
        # Ada3
        ("0_0_0_0_8000___00f54723-34a9-4164-b9da-640c47eb694b", "ada3", 60),
        ("0_0_0_0_8000___3571e24e-9e70-4dba-afa5-b3f1ece7d360", "ada3", 80),
        ("0_0_0_0_8000___9d207f21-4588-405a-be9c-b5513c4abf51", "ada3", 75),

         # CamemBERT
        ("0_0_0_0_8000___6f3f5c37-59ce-4cdc-990d-fb5a01bd8ee9", "camembert", 60),
        ("0_0_0_0_8000___62ca7e18-5ab6-4bb2-82f4-0f3fb229ccf8", "camembert", 80),
        ("0_0_0_0_8000___14d39e14-38ee-4f69-94f0-56d8b34dd8ff", "camembert", 75),

        # Ada2
        ("0_0_0_0_8000___1192b5da-775f-431d-86cf-db1a3f890d81", "ada2", 60),
        ("0_0_0_0_8000___99db2848-fa45-4dc6-9c77-6893944425df", "ada2", 80),
        ("0_0_0_0_8000___bb99a2fc-a475-4340-bfda-b00eac39d79d", "ada2", 75),

        # MiniLM
        ("0_0_0_0_8000___2c19db21-5c22-431d-adbe-c27c741571c7", "minilm", 60),
        ("0_0_0_0_8000___0bb7a293-c77f-41f9-b643-8989a15da831", "minilm", 40),
        ("0_0_0_0_8000___c859dffc-ad44-4e62-8875-9e3ab9f83c35", "minilm", 20),
    ]
        
        # Exécution de la comparaison
        comparator = EmbeddingsComparator(redis_client)
        comparison_table = comparator.compare_multiple_crawls(base_crawl, model_configs)
        
        if comparison_table is not None:
            print("\nTableau comparatif des modèles d'embeddings:")
            print(comparison_table.to_markdown())
        else:
            print("\nErreur lors de la génération du tableau comparatif")

    except Exception as e:
        logging.error(f"Erreur dans le programme principal: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()