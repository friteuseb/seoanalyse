import redis
import sys
import subprocess
import pandas as pd
import numpy as np
import networkx as nx
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from tabulate import tabulate
from enhanced_analyzer import EnhancedCoconAnalyzer

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingsComparator:
    def __init__(self, redis_client):
        self.redis = redis_client
    
     # Définition détaillée des métriques avec leurs descriptions et seuils
        self.metrics_definition = {
            'Métriques Fondamentales': [
                {
                    'path': ('structural_metrics', 'density'),  
                    'label': 'Densité du maillage',
                    'description': 'Ratio entre liens existants et possibles',
                    'optimal_range': '0.1-0.3',
                    'interpretation': {
                        'low': '< 0.1 : Maillage insuffisant',
                        'optimal': '0.1-0.3 : Navigation optimale',
                        'high': '> 0.3 : Dilution du PageRank'
                    }
                },
                {
                    'path': ('accessibility_metrics', 'pages_within_3_clicks'),
                    'label': 'Accessibilité à 3 clics',
                    'description': 'Pourcentage de pages accessibles en 3 clics',
                    'optimal_range': '> 80%',
                    'interpretation': {
                        'low': '< 60% : Navigation complexe',
                        'optimal': '> 80% : Bonne accessibilité',
                        'high': '= 100% : Structure très plate'
                    }
                }
            ],
            'Flux de PageRank': [
                {
                    'path': ('pr_metrics', 'entropy'),
                    'label': 'Distribution PageRank',
                    'description': 'Entropie de la distribution du PageRank',
                    'optimal_range': '> 3.0',
                    'interpretation': {
                        'low': '< 2.0 : Concentration excessive',
                        'optimal': '> 3.0 : Bonne distribution',
                        'high': '> 4.0 : Distribution très uniforme'
                    }
                },
                {
                    'path': ('pr_metrics', 'transfer_efficiency'),
                    'label': 'Efficacité du transfert',
                    'description': 'Conservation du PageRank dans les transmissions',
                    'optimal_range': '> 0.7',
                    'interpretation': {
                        'low': '< 0.5 : Pertes importantes',
                        'optimal': '> 0.7 : Bon transfert',
                        'high': '> 0.9 : Transfert optimal'
                    }
                }
            ],
            'Cohérence Sémantique': [
                {
                    'path': ('semantic_metrics', 'semantic_ratio'),
                    'label': 'Ratio thématique',
                    'description': 'Pertinence sémantique des liens',
                    'optimal_range': '> 0.6',
                    'interpretation': {
                        'low': '< 0.4 : Faible cohérence',
                        'optimal': '> 0.6 : Bonne cohérence',
                        'high': '> 0.8 : Très forte cohérence'
                    }
                }
            ]
        }

    
    def compare_multiple_crawls(self, base_crawl_id, model_configs):
        """
        Compare plusieurs crawls d'embeddings avec un crawl de base.
        
        Args:
            base_crawl_id (str): ID du crawl de référence
            model_configs (list): Liste de tuples (crawl_id, model_name, threshold)
            
        Returns:
            tuple: (DataFrame des comparaisons, descriptions des métriques) ou (None, None) si erreur
        """
        try:
            # Analyse du crawl de base
            print(f"Analyse du crawl de base: {base_crawl_id}")
            base_analyzer = EnhancedCoconAnalyzer(self.redis, base_crawl_id)
            base_metrics = base_analyzer.calculate_scientific_metrics()
            
            if not base_metrics:
                logging.error("Impossible de calculer les métriques de base")
                return None, None
                
            print("✓ Métriques de base calculées avec succès")
            
            # Analyse des variations
            embedding_metrics = []
            model_names = []
            
            print("\nAnalyse des différentes variations:")
            for crawl_id, model, threshold in model_configs:
                try:
                    print(f"Traitement de {model} (threshold={threshold})")
                    analyzer = EnhancedCoconAnalyzer(self.redis, crawl_id)
                    metrics = analyzer.calculate_scientific_metrics()
                    
                    # Vérification des métriques calculées
                    if not metrics:
                        print(f"⚠️  Pas de métriques pour {model}-t{threshold}")
                        continue
                        
                    # Calcul des métriques PageRank
                    pr_metrics = self.calculate_pagerank_metrics(analyzer.graph)
                    metrics['pr_metrics'] = pr_metrics
                    
                    # Calcul du ratio sémantique
                    semantic_ratio = self.calculate_semantic_ratio(analyzer.graph, analyzer.pages)
                    metrics['semantic_metrics'] = {'semantic_ratio': semantic_ratio}
                    
                    embedding_metrics.append(metrics)
                    model_names.append(f"{model}-t{threshold}")
                    print(f"✓ {model}-t{threshold} analysé avec succès\n")
                    
                except Exception as e:
                    logging.error(f"Erreur lors de l'analyse de {model}-t{threshold}: {str(e)}")
                    print(f"❌ Échec pour {model}-t{threshold}: {str(e)}\n")
                    continue
            
            if not embedding_metrics:
                logging.error("Aucune métrique n'a pu être calculée")
                return None, None
            
            # Génération du tableau comparatif
            print("Génération du tableau comparatif...")
            try:
                comparison_results = self.generate_embeddings_comparison(
                    base_metrics, 
                    embedding_metrics,
                    model_names
                )
                
                if comparison_results is None:
                    return None, None
                    
                df, descriptions = comparison_results
                
                # Export et visualisations
                self._export_results((df, descriptions))
                self._generate_comparison_plots(df)
                
                return df, descriptions
                
            except Exception as e:
                logging.error(f"Erreur lors de la génération du tableau: {str(e)}")
                traceback.print_exc()
                return None, None
                
        except Exception as e:
            logging.error(f"Erreur globale lors de la comparaison: {str(e)}")
            traceback.print_exc()
            return None, None
        


    def generate_embeddings_comparison(self, base_metrics, embedding_metrics_list, model_names):
        try:
            data = []
            for i, metrics in enumerate(embedding_metrics_list):
                model_data = {
                    'Modèle': model_names[i],
                    'Threshold': int(model_names[i].split('-t')[1]),
                    'Liens totaux': metrics['structural_metrics'].get('number_of_edges', 0),
                }
                
                orphan_stats = metrics['structural_metrics'].get('orphan_pages', {})
                base_orphan_stats = base_metrics['structural_metrics'].get('orphan_pages', {})
                
                orphan_count = orphan_stats.get('count', 0)
                orphan_percentage = orphan_stats.get('percentage', 0.0)
                base_orphan_count = base_orphan_stats.get('count', 0)
                
                if base_orphan_count > 0:
                    orphan_variation = ((orphan_count - base_orphan_count) / base_orphan_count) * 100
                    status = "⚠️" if orphan_count > base_orphan_count else "✅"
                    model_data['Pages Orphelines'] = f"{orphan_count} ({orphan_percentage:.1f}%) [{orphan_variation:+.1f}%] {status}"
                else:
                    model_data['Pages Orphelines'] = f"{orphan_count} ({orphan_percentage:.1f}%)"
                    
                model_data['URLs Orphelines'] = orphan_stats.get('urls', [])[:5]

                for category, metric_list in self.metrics_definition.items():
                    for metric in metric_list:
                        metric_path, metric_name = metric['path']
                        try:
                            value = metrics.get(metric_path, {}).get(metric_name, 0)
                            value = float(value)
                            
                            base_value = base_metrics.get(metric_path, {}).get(metric_name, 0)
                            base_value = float(base_value)
                            
                            status = self._get_metric_status(value, metric)
                            
                            if metric_path == 'accessibility_metrics':
                                display_value = value * 100
                                if base_value > 0:
                                    variation = ((value - base_value) / base_value) * 100
                                    model_data[metric['label']] = f"{display_value:.1f}% ({variation:+.1f}%) {status}"
                                else:
                                    model_data[metric['label']] = f"{display_value:.1f}%"
                            else:
                                if base_value > 0:
                                    variation = ((value - base_value) / base_value) * 100
                                    model_data[metric['label']] = f"{value:.2f} ({variation:+.1f}%) {status}"
                                else:
                                    model_data[metric['label']] = f"{value:.2f}"
                                        
                        except Exception as e:
                            logging.error(f"Erreur pour la métrique {metric_name}: {str(e)}")
                            model_data[metric['label']] = "N/A"
                
                # Score calculé différemment pour le modèle de base
                if model_names[i] == "cocon-t0":
                    score = self._calculate_base_score(metrics)
                else:
                    score = self._calculate_objective_score(metrics, base_metrics)
                
                model_data['Score Global'] = f"{score:.1f}"
                data.append(model_data)

            df = pd.DataFrame(data)
            
            cols = ['Modèle', 'Threshold', 'Liens totaux', 'Pages Orphelines']
            remaining_cols = [col for col in df.columns if col not in cols + ['URLs Orphelines']]
            cols.extend(remaining_cols)
            df = df[cols]
            
            df['Score_Num'] = df['Score Global'].astype(float)
            df = df.sort_values('Score_Num', ascending=False)
            df = df.drop('Score_Num', axis=1)
            
            descriptions = self._generate_metric_descriptions()
            
            if any('URLs Orphelines' in model_data for model_data in data):
                descriptions += "\n\n=== DÉTAILS DES PAGES ORPHELINES ===\n"
                for model_data in data:
                    descriptions += f"\n{model_data['Modèle']} :\n"
                    descriptions += f"• {model_data['Pages Orphelines']}\n"
                    if model_data.get('URLs Orphelines'):
                        descriptions += "Exemples d'URLs orphelines :\n"
                        for url in model_data['URLs Orphelines']:
                            descriptions += f"- {url}\n"
            
            return df, descriptions

        except Exception as e:
            logging.error(f"Erreur lors de la génération du tableau comparatif: {str(e)}")
            traceback.print_exc()
            return None, None
        


    def _calculate_base_score(self, metrics):
        """Calcule le score pour le modèle de base"""
        try:
            score = 0
            
            # 1. Qualité du maillage (30 points)
            density = metrics['structural_metrics'].get('density', 0)
            clustering = metrics['structural_metrics'].get('average_clustering', 0)
            
            # Densité optimale entre 0.1 et 0.3
            if 0.1 <= density <= 0.3:
                score += 15 * (1 - abs(0.2 - density) / 0.1)
            else:
                score += max(0, 15 * (1 - abs(0.2 - density) / 0.2))
            
            # Clustering
            score += min(15, clustering * 15)
            
            # 2. Accessibilité du contenu (30 points)
            pages_within_3_clicks = metrics['accessibility_metrics'].get('pages_within_3_clicks', 0)
            mean_depth = metrics['accessibility_metrics'].get('mean_depth', float('inf'))
            
            # Pages accessibles en 3 clics
            score += min(20, pages_within_3_clicks * 20)
            
            # Profondeur moyenne
            if mean_depth < float('inf'):
                depth_score = max(0, 10 * (1 - (mean_depth - 2) / 4))
                score += depth_score
            
            # 3. Pages orphelines (20 points)
            orphan_stats = metrics['structural_metrics'].get('orphan_pages', {})
            orphan_percentage = orphan_stats.get('percentage', 100)
            
            if orphan_percentage <= 5:
                orphan_score = 20
            elif orphan_percentage <= 10:
                orphan_score = 15 * (1 - (orphan_percentage - 5) / 5)
            elif orphan_percentage <= 20:
                orphan_score = 10 * (1 - (orphan_percentage - 10) / 10)
            else:
                orphan_score = max(0, 5 * (1 - (orphan_percentage - 20) / 30))
            
            score += orphan_score
            
            # 4. Cohérence sémantique (20 points)
            semantic_metrics = metrics.get('semantic_metrics', {})
            cluster_coherence = semantic_metrics.get('cluster_coherence', 0)
            semantic_flow = semantic_metrics.get('semantic_flow_strength', 0)
            
            score += min(10, cluster_coherence * 10)
            score += min(10, semantic_flow * 10)
            
            # Log détaillé
            logging.info(f"\nDétail du calcul du score (base):")
            logging.info(f"- Maillage: {15 * (1 - abs(0.2 - density) / 0.1) + min(15, clustering * 15):.1f}/30")
            logging.info(f"- Accessibilité: {min(20, pages_within_3_clicks * 20) + depth_score:.1f}/30")
            logging.info(f"- Pages orphelines: {orphan_score:.1f}/20")
            logging.info(f"- Cohérence sémantique: {min(10, cluster_coherence * 10) + min(10, semantic_flow * 10):.1f}/20")
            
            return round(max(0, min(100, score)), 1)
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul du score de base: {str(e)}")
            return 0.0

    def _calculate_objective_score(self, metrics, base_metrics):
        """Calcule le score en comparaison avec le modèle de base en gardant la même structure"""
        try:
            score = 0
            
            # 1. Qualité du maillage (30 points)
            density = metrics['structural_metrics'].get('density', 0)
            clustering = metrics['structural_metrics'].get('average_clustering', 0)
            base_density = base_metrics['structural_metrics'].get('density', 0)
            base_clustering = base_metrics['structural_metrics'].get('average_clustering', 0)
            
            # Densité comparée
            if 0.1 <= density <= 0.3:
                score += 15 * (1 - abs(0.2 - density) / 0.1)
            else:
                score += max(0, 15 * (1 - abs(0.2 - density) / 0.2))
            
            # Clustering comparé
            clustering_improvement = ((clustering - base_clustering) / base_clustering) * 100 if base_clustering > 0 else 0
            score += min(15, max(0, clustering_improvement / 10))
            
            # 2. Accessibilité (30 points) - même structure que l'original
            pages_within_3_clicks = metrics['accessibility_metrics'].get('pages_within_3_clicks', 0)
            mean_depth = metrics['accessibility_metrics'].get('mean_depth', float('inf'))
            
            score += min(20, pages_within_3_clicks * 20)
            if mean_depth < float('inf'):
                depth_score = max(0, 10 * (1 - (mean_depth - 2) / 4))
                score += depth_score
            
            # 3. Pages orphelines (20 points) - même structure que l'original
            orphan_stats = metrics['structural_metrics'].get('orphan_pages', {})
            orphan_percentage = orphan_stats.get('percentage', 100)
            
            if orphan_percentage <= 5:
                orphan_score = 20
            elif orphan_percentage <= 10:
                orphan_score = 15 * (1 - (orphan_percentage - 5) / 5)
            elif orphan_percentage <= 20:
                orphan_score = 10 * (1 - (orphan_percentage - 10) / 10)
            else:
                orphan_score = max(0, 5 * (1 - (orphan_percentage - 20) / 30))
                
            score += orphan_score
            
            # 4. Cohérence sémantique (20 points)
            semantic_metrics = metrics.get('semantic_metrics', {})
            cluster_coherence = semantic_metrics.get('cluster_coherence', 0)
            semantic_flow = semantic_metrics.get('semantic_flow_strength', 0)
            
            score += min(10, cluster_coherence * 10)
            score += min(10, semantic_flow * 10)
            
            # Log détaillé
            logging.info(f"\nDétail du calcul du score (comparatif):")
            logging.info(f"- Maillage: {score - orphan_score - min(10, semantic_flow * 10) - min(10, cluster_coherence * 10):.1f}/30")
            logging.info(f"- Accessibilité: {min(20, pages_within_3_clicks * 20) + depth_score:.1f}/30")
            logging.info(f"- Pages orphelines: {orphan_score:.1f}/20")
            logging.info(f"- Cohérence sémantique: {min(10, cluster_coherence * 10) + min(10, semantic_flow * 10):.1f}/20")
            
            return round(max(0, min(100, score)), 1)
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul du score: {str(e)}")
            return 0.0


    def _get_score_interpretation(self, score):
        """
        Fournit une interprétation du score global
        """
        if score >= 90:
            return "Excellent maillage, très bien optimisé ✨"
        elif score >= 80:
            return "Très bon maillage, quelques optimisations possibles ✅"
        elif score >= 70:
            return "Bon maillage, des améliorations recommandées 👍"
        elif score >= 60:
            return "Maillage correct mais nécessite attention ⚠️"
        elif score >= 50:
            return "Maillage faible, optimisations importantes nécessaires ⚠️"
        else:
            return "Maillage problématique, restructuration recommandée 🚨"

    def _export_results(self, comparison_data, link_comparison_matrix=None):
        """Export les résultats avec les informations sur les pages orphelines"""
        try:
            df, descriptions = comparison_data

            # Préparer le CSV avec les détails des pages orphelines
            csv_data = []
            for index, row in df.iterrows():
                orphan_info = row['Pages Orphelines']
                # Extraire les informations des pages orphelines
                count = int(orphan_info.split()[0])
                percentage = float(orphan_info.split('(')[1].split('%')[0])
                
                csv_row = {
                    'Modèle': row['Modèle'],
                    'Threshold': row['Threshold'],
                    'Liens totaux': row['Liens totaux'],
                    'Pages Orphelines (nombre)': count,
                    'Pages Orphelines (%)': percentage,
                }
                
                # Ajouter les autres colonnes existantes
                for col in df.columns:
                    if col not in ['Modèle', 'Threshold', 'Liens totaux', 'Pages Orphelines']:
                        csv_row[col] = row[col]
                        
                csv_data.append(csv_row)

            # Créer un nouveau DataFrame pour le CSV
            csv_df = pd.DataFrame(csv_data)
            csv_df.to_csv('embedding_comparison_detailed.csv', index=False)

            # Export Markdown
            with open('embedding_comparison.md', 'w') as f:
                f.write("# Analyse Comparative des Modèles d'Embedding\n\n")
                f.write("## Métriques Détaillées\n")
                f.write(df.to_markdown())
                f.write("\n\n")
                
                if link_comparison_matrix is not None:
                    f.write("## Matrice de Comparaison des Liens\n")
                    f.write(link_comparison_matrix.to_markdown())
                    f.write("\n\n")
                
                f.write("## Guide d'Interprétation\n")
                f.write(descriptions)
                
                # Ajouter une section spécifique pour les pages orphelines
                f.write("\n\n## Détail des Pages Orphelines\n")
                for index, row in df.iterrows():
                    f.write(f"\n### {row['Modèle']}\n")
                    orphan_info = row['Pages Orphelines']
                    f.write(f"- {orphan_info}\n")
                    
            print("Exports générés avec succès")
                
        except Exception as e:
            logging.error(f"Erreur lors de l'export des résultats: {str(e)}")
            raise


    def _generate_comparison_plots(self, comparison_table):
        """Génère des visualisations des comparaisons mises à jour"""
        try:
            # Score global
            plt.figure(figsize=(15, 8))
            models = comparison_table['Modèle']
            scores = comparison_table['Score Global']
            
            plt.bar(range(len(models)), scores)
            plt.title('Scores Globaux par Modèle')
            plt.xticks(range(len(models)), models, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('scores_comparison.png')
            plt.close()

            # Métriques clés
            metrics_to_plot = [
                'Densité du maillage',
                'Distribution PageRank',
                'Efficacité du transfert'
            ]
            
            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 6))
            
            for i, metric in enumerate(metrics_to_plot):
                data = comparison_table[['Modèle', metric]].copy()
                data[metric] = data[metric].apply(
                    lambda x: float(x.split()[0]) if isinstance(x, str) else x
                )
                
                axes[i].bar(range(len(models)), data[metric])
                axes[i].set_title(metric)
                axes[i].set_xticks(range(len(models)))
                axes[i].set_xticklabels(models, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig('metrics_comparison.png')
            plt.close()
                
        except Exception as e:
            logging.error(f"Erreur visualisation: {str(e)}")



    def _calculate_label_similarity(self, label1, label2):
        """Calcule la similarité entre deux labels"""
        try:
            # Version simple : mots communs
            words1 = set(label1.lower().split())
            words2 = set(label2.lower().split())
            
            if not words1 or not words2:
                return 0.0
                
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Erreur calcul similarité labels: {str(e)}")
            return 0.0



    def _get_metric_status(self, value, metric_def):
        """Détermine le statut d'une métrique par rapport à sa plage optimale"""
        try:
            # Pour les métriques en pourcentage, la valeur est déjà multipliée par 100
            is_percentage = '%' in metric_def['optimal_range']
            
            if is_percentage:
                # Enlever le % pour la comparaison
                optimal_range = metric_def['optimal_range'].replace('%', '')
            else:
                optimal_range = metric_def['optimal_range']

            if '-' in optimal_range:
                min_val, max_val = map(float, optimal_range.split('-'))
                if value < min_val:
                    return '⚠️'
                elif value > max_val:
                    return '⚠️'
                else:
                    return '✅'
            else:
                threshold = float(optimal_range.replace('>', '').replace('<', '').strip())
                if '>' in optimal_range:
                    return '✅' if value > threshold else '⚠️'
                else:  # '<' in optimal_range
                    return '✅' if value < threshold else '⚠️'
        except Exception as e:
            logging.error(f"Erreur évaluation statut métrique: {str(e)}")
            return '⚠️'
        
        

    def _generate_metric_descriptions(self):
        """Génère une description détaillée de toutes les métriques, incluant les pages orphelines"""
        descriptions = []
        descriptions.append("\n=== GUIDE D'INTERPRÉTATION DES MÉTRIQUES ===\n")
        
        # Ajout de la section sur les pages orphelines
        descriptions.append("\n## Pages Orphelines")
        descriptions.append("Description: Pages qui n'ont aucun lien entrant (sauf la page d'accueil)")
        descriptions.append("Interprétation:")
        descriptions.append("• Format: nombre (pourcentage) [variation%] status")
        descriptions.append("• ✅ : Réduction ou stabilité du nombre de pages orphelines")
        descriptions.append("• ⚠️ : Augmentation du nombre de pages orphelines")
        descriptions.append("\nSeuils recommandés:")
        descriptions.append("• Optimal: < 5% du total des pages")
        descriptions.append("• Acceptable: 5-10% du total des pages")
        descriptions.append("• Problématique: > 10% du total des pages")
        
        # Descriptions existantes
        for category, metrics in self.metrics_definition.items():
            descriptions.append(f"\n## {category}")
            for metric in metrics:
                descriptions.append(f"\n### {metric['label']}")
                descriptions.append(f"Description: {metric['description']}")
                descriptions.append(f"Plage optimale: {metric['optimal_range']}")
                descriptions.append("Interprétation:")
                for level, desc in metric['interpretation'].items():
                    descriptions.append(f"- {desc}")
                    
        return '\n'.join(descriptions)



    def calculate_pagerank_metrics(self, graph):
        """Calcule les métriques liées au PageRank"""
        try:
            # Calcul du PageRank
            pr = nx.pagerank(graph)
            
            # Entropie
            pr_values = list(pr.values())
            entropy = -sum(p * np.log2(p) for p in pr_values if p > 0)
            
            # Efficacité de transfert
            transfer_efficiency = self._calculate_transfer_efficiency(graph, pr)
            
            return {
                'entropy': entropy,
                'transfer_efficiency': transfer_efficiency
            }
        except Exception as e:
            logging.error(f"Erreur calcul métriques PageRank: {str(e)}")
            return {'entropy': 0, 'transfer_efficiency': 0}

    def _calculate_transfer_efficiency(self, graph, pagerank):
        """Calcule l'efficacité du transfert de PageRank"""
        try:
            efficiency = 0
            for node in graph.nodes():
                pr_in = sum(pagerank[pred] for pred in graph.predecessors(node))
                pr_out = sum(pagerank[succ] for succ in graph.successors(node))
                if pr_in > 0:
                    efficiency += min(pr_out/pr_in, 1.0)
            return efficiency / len(graph.nodes()) if graph.nodes() else 0
        except Exception as e:
            logging.error(f"Erreur calcul efficacité transfert: {str(e)}")
            return 0

    def calculate_semantic_ratio(self, graph, pages):
        """Calcule le ratio de pertinence thématique des liens"""
        try:
            total_links = 0
            semantic_links = 0
            
            for source, target in graph.edges():
                if source in pages and target in pages:
                    total_links += 1
                    # Calcul basé sur la similarité des clusters et labels
                    source_cluster = pages[source].cluster
                    target_cluster = pages[target].cluster
                    
                    # Similarité de cluster
                    cluster_similarity = 1.0 if source_cluster == target_cluster else 0.5
                    
                    # Similarité de labels (à implémenter selon votre logique d'embeddings)
                    label_similarity = self._calculate_label_similarity(
                        pages[source].label,
                        pages[target].label
                    )
                    
                    semantic_score = (cluster_similarity + label_similarity) / 2
                    semantic_links += semantic_score
                    
            return semantic_links / total_links if total_links > 0 else 0
        except Exception as e:
            logging.error(f"Erreur calcul ratio sémantique: {str(e)}")
            return 0


    def compare_links_between_models(self, base_crawl_id, model_configs):
        """
        Crée une matrice de comparaison des liens entre tous les modèles
        """
        try:
            # Stockage des edges pour chaque modèle
            model_edges = {}
            
            # Récupération des liens pour le modèle de base
            base_analyzer = EnhancedCoconAnalyzer(self.redis, base_crawl_id)
            base_edges = set(base_analyzer.graph.edges())
            model_edges['cocon-base'] = base_edges
            
            # Récupération des liens pour chaque modèle
            for crawl_id, model, threshold in model_configs:
                try:
                    analyzer = EnhancedCoconAnalyzer(self.redis, crawl_id)
                    edges = set(analyzer.graph.edges())
                    model_name = f"{model}-t{threshold}"
                    model_edges[model_name] = edges
                except Exception as e:
                    logging.error(f"Erreur récupération liens pour {model}-t{threshold}: {str(e)}")
                    continue
            
            # Création de la matrice de comparaison
            models = list(model_edges.keys())
            comparison_matrix = pd.DataFrame(index=models, columns=models)
            
            # Remplissage de la matrice
            for model1 in models:
                edges1 = model_edges[model1]
                for model2 in models:
                    edges2 = model_edges[model2]
                    
                    # Calcul des métriques de comparaison
                    common_links = len(edges1.intersection(edges2))
                    total_links_model1 = len(edges1)
                    total_links_model2 = len(edges2)
                    
                    # Calcul du pourcentage de liens communs
                    if total_links_model1 > 0 and total_links_model2 > 0:
                        # Format : "% communs (communs/total1/total2)"
                        similarity = (common_links / min(total_links_model1, total_links_model2)) * 100
                        comparison_matrix.loc[model1, model2] = f"{similarity:.1f}% ({common_links}/{total_links_model1}/{total_links_model2})"
                    else:
                        comparison_matrix.loc[model1, model2] = "N/A"
            
            # Ajout d'informations détaillées sur les différences
            differences_analysis = self._analyze_link_differences(model_edges)
            
            return comparison_matrix, differences_analysis
            
        except Exception as e:
            logging.error(f"Erreur dans la comparaison des liens: {str(e)}")
            return None, None

    def _analyze_link_differences(self, model_edges):
        """
        Analyse détaillée des différences de liens entre les modèles
        """
        differences = {}
        
        # Pour chaque paire de modèles
        models = list(model_edges.keys())
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                edges1 = model_edges[model1]
                edges2 = model_edges[model2]
                
                # Liens uniques à chaque modèle
                unique_to_model1 = edges1 - edges2
                unique_to_model2 = edges2 - edges1
                common_links = edges1.intersection(edges2)
                
                differences[f"{model1}_vs_{model2}"] = {
                    'common_links': len(common_links),
                    'unique_to_model1': len(unique_to_model1),
                    'unique_to_model2': len(unique_to_model2),
                    'sample_differences': {
                        'model1_sample': list(unique_to_model1)[:5],
                        'model2_sample': list(unique_to_model2)[:5]
                    }
                }
        
        return differences



    def generate_link_comparison_report(self, comparison_matrix, differences_analysis):
        """
        Génère un rapport détaillé de la comparaison des liens
        """
        report = []
        report.append("\n=== COMPARAISON DES LIENS ENTRE MODÈLES ===\n")
        
        # 1. Matrice de comparaison
        report.append("Matrice de similarité des liens :")
        report.append(comparison_matrix.to_markdown())
        
        # 2. Analyse des différences
        report.append("\nAnalyse détaillée des différences :")
        for comparison, data in differences_analysis.items():
            model1, model2 = comparison.split('_vs_')
            report.append(f"\n{model1} vs {model2}:")
            report.append(f"- Liens communs : {data['common_links']}")
            report.append(f"- Liens uniques à {model1} : {data['unique_to_model1']}")
            report.append(f"- Liens uniques à {model2} : {data['unique_to_model2']}")
            
            # Exemples de liens différents
            report.append("\nExemples de liens différents :")
            report.append(f"{model1} uniquement : {', '.join(str(x) for x in data['sample_differences']['model1_sample'])}")
            report.append(f"{model2} uniquement : {', '.join(str(x) for x in data['sample_differences']['model2_sample'])}")
        
        return "\n".join(report)


    def compare_specific_links(self, base_crawl_id, model_configs):
        """
        Compare les liens spécifiques entre les modèles
        """
        try:
            # Récupération des liens du modèle de base
            base_analyzer = EnhancedCoconAnalyzer(self.redis, base_crawl_id)
            base_edges = set(base_analyzer.graph.edges())
            print(f"Modèle de base: {len(base_edges)} liens")
            
            # Structure pour stocker les résultats
            results = {}
            
            # Comparaison avec chaque modèle
            for crawl_id, model, threshold in model_configs:
                model_name = f"{model}-t{threshold}"
                try:
                    # Récupération des liens du modèle à comparer
                    analyzer = EnhancedCoconAnalyzer(self.redis, crawl_id)
                    model_edges = set(analyzer.graph.edges())
                    print(f"\nAnalyse de {model_name}: {len(model_edges)} liens")
                    
                    # Trouver les liens identiques et différents
                    identical_links = base_edges.intersection(model_edges)
                    unique_to_base = base_edges - model_edges
                    unique_to_model = model_edges - base_edges
                    
                    results[model_name] = {
                        'identical_links': list(identical_links),
                        'unique_to_base': list(unique_to_base),
                        'unique_to_model': list(unique_to_model),
                        'stats': {
                            'total_base': len(base_edges),
                            'total_model': len(model_edges),
                            'identical': len(identical_links),
                            'unique_base': len(unique_to_base),
                            'unique_model': len(unique_to_model)
                        }
                    }
                    
                    # Création de DataFrames pour une meilleure visualisation
                    identical_df = pd.DataFrame(list(identical_links), columns=['Source', 'Target'])
                    unique_base_df = pd.DataFrame(list(unique_to_base), columns=['Source', 'Target'])
                    unique_model_df = pd.DataFrame(list(unique_to_model), columns=['Source', 'Target'])
                    
                    # Export en CSV pour analyse détaillée
                    identical_df.to_csv(f'identical_links_{model_name}.csv', index=False)
                    unique_base_df.to_csv(f'unique_to_base_{model_name}.csv', index=False)
                    unique_model_df.to_csv(f'unique_to_model_{model_name}.csv', index=False)
                    
                except Exception as e:
                    logging.error(f"Erreur lors de l'analyse de {model_name}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logging.error(f"Erreur dans la comparaison des liens: {str(e)}")
            return None
        

    def generate_link_comparison_report(self, results):
        """Génère un rapport détaillé avec colonnes groupées par modèle"""
        headers = [
            'Modèle',
            'Base %', 'Base Détails',  # Groupé pour Base
            'Modèle %', 'Modèle Détails',  # Groupé pour Modèle
            'Liens Uniques Base', 'Liens Uniques Modèle'  # Métriques additionnelles
        ]
        
        rows = []
        for model_name, data in results.items():
            stats = data['stats']
            identical = stats['identical']
            total_base = stats['total_base']
            total_model = stats['total_model']
            
            # Calcul des pourcentages
            pct_base = (identical / total_base * 100) if total_base > 0 else 0
            pct_model = (identical / total_model * 100) if total_model > 0 else 0
            
            # Format détaillé : liens_identiques/total_liens/liens_base
            base_details = f"{identical}/{total_base}/{total_model}"
            model_details = f"{identical}/{total_model}/{total_base}"
            
            rows.append([
                model_name,
                f"{pct_base:.1f}%",
                base_details,
                f"{pct_model:.1f}%", 
                model_details,
                stats['unique_base'],
                stats['unique_model']
            ])
        
        return tabulate(rows, headers=headers, tablefmt='pipe')


def main():
    try:
        # Configuration Redis
        redis_port = subprocess.check_output(
            "ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", 
            shell=True
        ).decode('utf-8').strip()
        
        redis_client = redis.Redis(host='localhost', port=int(redis_port), db=0)

        """"
        # Configuration des modèles à comparer
        base_crawl = "chadyagamma_fr_guide-sonotherapie___4a6f6253-3e87-45dc-aee6-4e5b6ce32f43"
        model_configs = [
            # Base model
            ("chadyagamma_fr_guide-sonotherapie___4a6f6253-3e87-45dc-aee6-4e5b6ce32f43", "cocon", 00),

            # Ada3 -text-embedding-3-small
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
        """

        # Configuration des modèles à comparer
        base_crawl = "semantic-suggestion_ecotechlab_fr___e76081e0-f04b-4d04-8461-6c6bbe91c90e"
        model_configs = [
            # Base model
            ("semantic-suggestion_ecotechlab_fr___e76081e0-f04b-4d04-8461-6c6bbe91c90e", "cocon", 00),

            # Ada3 -text-embedding-3-small
            ("0_0_0_0_8000___ccc5c344-182f-4f97-871a-bcf224822003", "ada3", 60),

            # CamemBERT
            ("0_0_0_0_8000___e4dcdc31-de25-430d-9b49-c96e829f8c41", "camembert", 60),

            # Ada2
            ("0_0_0_0_8000___d7a09598-c17f-4cd7-8a86-9f4a39e528ba", "ada2", 60),

            # MiniLM
            ("0_0_0_0_8000___bd6d97bc-84f3-4551-bd05-dfe2bd040433", "minilm", 20),
        ]
        
     # Exécution de la comparaison
        comparator = EmbeddingsComparator(redis_client)
        comparison_results = comparator.compare_multiple_crawls(base_crawl, model_configs)
        
        if comparison_results is not None:
            df, descriptions = comparison_results
            
            # Matrice de comparaison des liens
            comparison_matrix, differences = comparator.compare_links_between_models(
                base_crawl, model_configs
            )
            
            
            # Affichage des résultats
            print("\n=== ANALYSE COMPARATIVE DES MODÈLES ===")
            print(df.to_markdown())
            print("\n=== MATRICE DE COMPARAISON DES LIENS ===")
            print(comparison_matrix.to_markdown())
            
            # Export de tous les résultats
            comparator._export_results(
                comparison_results, 
                link_comparison_matrix=comparison_matrix
            )
            
            # Generation des visualisations
            comparator._generate_comparison_plots(df)
        
   # 2. Analyse détaillée des liens
        specific_comparison_results = comparator.compare_specific_links(base_crawl, model_configs)
        if specific_comparison_results is not None:
            report = comparator.generate_link_comparison_report(specific_comparison_results)
            print("\n=== ANALYSE DÉTAILLÉE DES LIENS ===")
            print(report)

    except Exception as e:
        logging.error(f"Erreur dans le programme principal: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()