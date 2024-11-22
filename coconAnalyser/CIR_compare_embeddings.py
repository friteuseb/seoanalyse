import redis
import sys
import subprocess
from enhanced_analyzer import EnhancedCoconAnalyzer
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingsComparator:
    def __init__(self, redis_client):
        self.redis = redis_client
    

    def generate_embeddings_comparison(self, base_metrics, embedding_metrics_list, model_names):
        """Génère un tableau comparatif des différents modèles d'embeddings"""
        try:
            # Définition des métriques clés à comparer
            key_metrics = {
                'Qualité Structurelle': [
                    ('density', 'Densité du réseau', '0.1-0.3 est optimal pour la lisibilité'),
                    ('average_clustering', 'Cohésion locale', 'Mesure la formation de communautés'),
                    ('reciprocity', 'Réciprocité', 'Équilibre des liens bidirectionnels')
                ],
                'Accessibilité': [
                    ('pages_within_3_clicks', 'Accessibilité à 3 clics', 'Idéalement >80%'),
                    ('mean_depth', 'Profondeur moyenne', 'Idéalement entre 1.5 et 2.5'),
                    ('pagerank_entropy', 'Distribution de l\'autorité', 'Plus élevé = meilleure répartition')
                ],
                'Cohérence Thématique': [
                    ('cluster_density', 'Cohésion intra-cluster', 'Force des liens thématiques'),
                    ('thematic_ratio', 'Ratio thématique/total', 'Balance thématique/transversal')
                ]
            }

            data = []
            for i, metrics in enumerate(embedding_metrics_list):
                model_data = {
                    'Modèle': model_names[i],
                    'Threshold': int(model_names[i].split('-t')[1]),
                    'Liens totaux': metrics['structural_metrics'].get('number_of_edges', 0),
                }
                
                # Calcul des variations par rapport au crawl de base
                for category, metric_list in key_metrics.items():
                    for metric_path, metric_name, metric_label in metric_list:
                        try:
                            base_value = base_metrics.get(metric_path, {}).get(metric_name, 0)
                            current_value = metrics.get(metric_path, {}).get(metric_name, 0)
                            if base_value > 0:
                                variation = ((current_value - base_value) / base_value) * 100
                            else:
                                variation = float('inf') if current_value > 0 else 0
                            
                            model_data[metric_label] = f"{current_value:.2f} ({variation:+.1f}%)"
                        except Exception as e:
                            logging.error(f"Erreur pour la métrique {metric_label}: {str(e)}")
                            model_data[metric_label] = "N/A"
                
                data.append(model_data)
                
            # Création du DataFrame
            df = pd.DataFrame(data)
            
            # Ajout du score global
            df['Score Global'] = df.apply(self._calculate_model_score, axis=1)
            
            # Tri par score global
            df = df.sort_values('Score Global', ascending=False)
            
            return df

        except Exception as e:
            logging.error(f"Erreur lors de la génération du tableau comparatif: {str(e)}")
            traceback.print_exc()
            return None
        
    def _calculate_scientific_score(self, metrics):
        """Calcule un score basé sur des critères objectifs"""
        score = 0
        
        # Densité (0-20 points)
        density = metrics['density']
        if 0.1 <= density <= 0.3:
            score += 20 * (1 - abs(0.2 - density) / 0.1)
        
        # Accessibilité (0-30 points)
        accessibility = metrics['pages_within_3_clicks']
        score += min(30, accessibility * 30)
        
        # Profondeur (0-15 points)
        depth = metrics['mean_depth']
        if 1.5 <= depth <= 2.5:
            score += 15 * (1 - abs(2 - depth) / 0.5)
        
        # Distribution PageRank (0-15 points)
        entropy = metrics['pagerank_entropy'] / 6  # Normalisé sur ~6 max
        score += min(15, entropy * 15)
        
        # Balance thématique (0-20 points)
        thematic_ratio = metrics['thematic_links'] / (metrics['thematic_links'] + metrics['cross_thematic_links'])
        if 0.6 <= thematic_ratio <= 0.8:  # 60-80% de liens thématiques
            score += 20 * (1 - abs(0.7 - thematic_ratio) / 0.1)
        
        return score

    def analyze_maillage_quality(self, metrics):
        """Analyse la qualité du maillage selon des critères objectifs"""
        return {
            'lisibilite': {
                'score': self._calculate_readability_score(metrics),
                'facteurs': {
                    'densite_optimale': 0.1 <= metrics['density'] <= 0.3,
                    'profondeur_optimale': 1.5 <= metrics['mean_depth'] <= 2.5,
                    'distribution_equilibree': metrics['pagerank_entropy'] >= 5.0
                }
            },
            'accessibilite': {
                'score': self._calculate_accessibility_score(metrics),
                'facteurs': {
                    'couverture_3_clics': metrics['pages_within_3_clicks'] >= 0.8,
                    'profondeur_moyenne': metrics['mean_depth'] <= 2.5
                }
            },
            'coherence': {
                'score': self._calculate_coherence_score(metrics),
                'facteurs': {
                    'equilibre_thematique': 0.6 <= metrics['thematic_links']/(metrics['thematic_links'] + metrics['cross_thematic_links']) <= 0.8,
                    'cohesion_clusters': metrics['cluster_density'] >= 0.4
                }
            }
        }

    def _calculate_model_score(self, row):
        """Calcule un score global pour chaque modèle"""
        try:
            score = 0
            try:
                # Densité (max 30 points)
                density_val = row['Densité globale'].split()[0]
                score += min(30, float(density_val) * 100)
            except:
                pass
                
            try:
                # Clustering (max 20 points)
                clustering_val = row['Clustering moyen'].split()[0]
                score += min(20, float(clustering_val) * 40)
            except:
                pass
                
            try:
                # Accessibilité (max 30 points)
                access_val = row['Pages à 3 clics'].split()[0]
                score += min(30, float(access_val) * 30)
            except:
                pass
                
            try:
                # Cohérence thématique (max 20 points)
                density_val = row['Densité des clusters'].split()[0]
                score += min(20, float(density_val) * 40)
            except:
                pass
                
            return round(score, 1)
        except Exception as e:
            logging.error(f"Erreur lors du calcul du score: {str(e)}")
            return 0.0
    

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
            
            # Export en différents formats
            if comparison_table is not None:
                comparison_table.to_csv('embedding_comparison.csv')
                with open('embedding_comparison.md', 'w') as f:
                    f.write(comparison_table.to_markdown())
                print("Exports générés avec succès")
            
            return comparison_table
            
        except Exception as e:
            print(f"Erreur lors de la comparaison des crawls: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
    def _generate_comparison_plots(self, comparison_table):
        """Génère des visualisations des comparaisons"""
        plt.figure(figsize=(15, 8))
        
        # Plot des scores globaux
        scores = comparison_table[['Modèle', 'Score Global']].set_index('Modèle')
        ax = scores.plot(kind='bar')
        plt.title('Comparaison des Scores Globaux par Modèle')
        plt.tight_layout()
        plt.savefig('scores_comparison.png')
        
        # Plot des métriques clés
        metrics_to_plot = ['Densité globale', 'Clustering moyen', 'Densité des clusters']
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 6))
        
        for i, metric in enumerate(metrics_to_plot):
            data = comparison_table[['Modèle', metric]].copy()
            data[metric] = data[metric].apply(lambda x: float(x.split()[0]))
            sns.barplot(data=data, x='Modèle', y=metric, ax=axes[i])
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
            
        plt.tight_layout()
        plt.savefig('metrics_comparison.png')

def main():
    # Récupération du port Redis
    redis_port = subprocess.check_output(
        "ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", 
        shell=True
    ).decode('utf-8').strip()
    
    redis_client = redis.Redis(host='localhost', port=int(redis_port), db=0)
    
    # Configuration des modèles à comparer
    base_crawl = "chadyagamma_fr_guide-sonotherapie___4a6f6253-3e87-45dc-aee6-4e5b6ce32f43_clustered_graph"
    
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
        ("0_0_0_0_8000___1192b5da-775f-431d-86cf-db1a3f890d81", "minilm", 60),
        ("0_0_0_0_8000___99db2848-fa45-4dc6-9c77-6893944425df", "minilm", 80),
        ("0_0_0_0_8000___bb99a2fc-a475-4340-bfda-b00eac39d79d", "minilm", 75),

        # MiniLM
        ("0_0_0_0_8000___2c19db21-5c22-431d-adbe-c27c741571c7", "minilm", 60),
        ("0_0_0_0_8000___0bb7a293-c77f-41f9-b643-8989a15da831", "minilm", 40),
        ("0_0_0_0_8000___c859dffc-ad44-4e62-8875-9e3ab9f83c35", "minilm", 20),
        

    ]
    
    comparator = EmbeddingsComparator(redis_client)
    comparison_table = comparator.compare_multiple_crawls(base_crawl, model_configs)
    
    print("\nTableau comparatif des modèles d'embeddings:")
    print(comparison_table.to_markdown())

if __name__ == "__main__":
    main()