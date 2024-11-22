import redis
import json
import logging
import numpy as np
import networkx as nx
from dataclasses import dataclass
import subprocess
from collections import defaultdict
from urllib.parse import urlparse
from cocon_analyzer import CoconAnalyzer 

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class PageMetrics:
    url: str
    depth: int
    cluster: int
    label: str
    incoming_links: int
    outgoing_links: int
    internal_pagerank: float
    semantic_relevance: float
    content_length: int

class EnhancedCoconAnalyzer(CoconAnalyzer):
    def __init__(self, redis_client, crawl_id: str):
        super().__init__(redis_client, crawl_id)
        self.semantic_transition_matrix = None
        self.load_data()

    def calculate_scientific_metrics(self):
        try:
            metrics = {
                'structural_metrics': self._calculate_structural_metrics(),
                'semantic_metrics': self._calculate_semantic_metrics(),
                'accessibility_metrics': self._calculate_accessibility_metrics(),
                'cluster_metrics': self._calculate_cluster_metrics()
            }
            return metrics
        except Exception as e:
            logging.error(f"Erreur lors du calcul des métriques: {str(e)}")
            return {
                'structural_metrics': self._get_default_structural_metrics(),
                'semantic_metrics': self._get_default_semantic_metrics(),
                'accessibility_metrics': self._get_default_accessibility_metrics(),
                'cluster_metrics': self._get_default_cluster_metrics()
            }

    def _get_default_structural_metrics(self):
        return {
            'average_clustering': 0.0,
            'reciprocity': 0.0,
            'density': 0.0,
            'average_shortest_path': 0.0,
            'average_degree': 0.0,
            'number_of_nodes': len(self.pages),
            'number_of_edges': len(self.graph.edges()),
            'bilingual_links': 0,
            'thematic_links': 0,
            'cross_thematic_links': 0
        }

    def _get_default_semantic_metrics(self):
        return {
            'cluster_coherence': 0.0,
            'semantic_flow_strength': 0.0,
            'inter_cluster_density': 0.0,
            'language_balance': 0.0,
            'thematic_isolation': 0.0
        }

    def _get_default_accessibility_metrics(self):
        return {
            'mean_depth': 1.0,
            'depth_variance': 0.0,
            'max_depth': 1.0,
            'min_depth': 1.0,
            'pagerank_entropy': 0.0,
            'pages_within_3_clicks': 1.0
        }

    def _get_default_cluster_metrics(self):
        return {
            'number_of_clusters': len(self.clusters),
            'average_cluster_size': 0.0,
            'cluster_size_variance': 0.0,
            'cluster_density': 0.0,
            'inter_cluster_links': 0
        }

    def _calculate_structural_metrics(self):
        try:
            if not self.graph.nodes():
                return self._get_default_structural_metrics()
                
            metrics = {
                'average_clustering': float(nx.average_clustering(self.graph.to_undirected())),
                'reciprocity': float(nx.reciprocity(self.graph)),
                'density': float(nx.density(self.graph)),
                'average_degree': float(sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes())),
                'number_of_nodes': self.graph.number_of_nodes(),
                'number_of_edges': self.graph.number_of_edges(),
                'bilingual_links': self._count_bilingual_links(),
                'thematic_links': self._count_thematic_links(),
                'cross_thematic_links': self._count_cross_thematic_links()
            }
            
            try:
                metrics['average_shortest_path'] = float(nx.average_shortest_path_length(self.graph))
            except:
                metrics['average_shortest_path'] = float('inf')
                
            return metrics
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul des métriques structurelles: {str(e)}")
            return self._get_default_structural_metrics()

    def _calculate_accessibility_metrics(self):
        """Calcul des métriques d'accessibilité avec gestion d'erreur robuste"""
        try:
            if not self.graph.nodes():
                return self._get_default_accessibility_metrics()

            # Trouver la page racine
            if not self.root_url:
                out_degrees = self.graph.out_degree()
                if out_degrees:
                    self.root_url = max(out_degrees, key=lambda x: x[1])[0]
                else:
                    return self._get_default_accessibility_metrics()

            depths = []
            for node in self.graph.nodes():
                try:
                    if nx.has_path(self.graph, self.root_url, node):
                        depth = nx.shortest_path_length(self.graph, self.root_url, node)
                        depths.append(depth)
                except:
                    continue

            if not depths:
                return self._get_default_accessibility_metrics()

            metrics = {
                'mean_depth': float(np.mean(depths)),
                'depth_variance': float(np.var(depths)),
                'max_depth': float(max(depths)),
                'min_depth': float(min(depths)),
                'pages_within_3_clicks': float(sum(1 for d in depths if d <= 3) / len(self.graph.nodes())),
            }

            try:
                pagerank = nx.pagerank(self.graph)
                metrics['pagerank_entropy'] = float(self._calculate_entropy(list(pagerank.values())))
            except:
                metrics['pagerank_entropy'] = 0.0

            return metrics

        except Exception as e:
            logging.error(f"Erreur lors du calcul des métriques d'accessibilité: {str(e)}")
            return self._get_default_accessibility_metrics()
        

    def format_change(self, before, after, inverse=False):
        """Format les changements de manière lisible avec gestion d'erreur"""
        try:
            before = float(before)
            after = float(after)
            
            if before == 0:
                return f"{after:.2f} (nouveau)"
            
            # Calcul correct du pourcentage de changement
            change = ((after - before) / before) * 100
            if inverse:
                change = -change  # Inverse le signe uniquement si demandé
            
            # Détermination de l'indicateur
            if abs(change) < 30:
                indicator = "→"
            else:
                indicator = "↑" if change > 0 else "↓"
                if abs(change) > 100:
                    indicator = indicator * 2  # ↑↑ ou ↓↓
            
            # Si le changement est trop grand
            if abs(change) > 1000:
                return f"{after:.2f} ({indicator} >1000%)"
                
            return f"{after:.2f} ({indicator} {change:+.1f}%)"
            
        except Exception as e:
            return f"{after:.2f} (non comparable)"



    def _count_bilingual_links(self):
        """Compte les liens entre versions linguistiques"""
        count = 0
        for source, target in self.graph.edges():
            # Vérifie si les URLs correspondent à des versions différentes
            if ('_fr_' in source and '_fr_' not in target) or ('_fr_' in target and '_fr_' not in source):
                count += 1
        return count

    def _count_thematic_links(self):
        """Compte les liens entre pages du même thème"""
        count = 0
        for source, target in self.graph.edges():
            if source in self.pages and target in self.pages:
                if self.pages[source].cluster == self.pages[target].cluster:
                    count += 1
        return count

    def _count_cross_thematic_links(self):
        """Compte les liens entre différents thèmes"""
        count = 0
        for source, target in self.graph.edges():
            if source in self.pages and target in self.pages:
                if self.pages[source].cluster != self.pages[target].cluster:
                    count += 1
        return count

    def _calculate_language_balance(self):
        """Calcule l'équilibre entre les versions linguistiques"""
        try:
            fr_count = sum(1 for url in self.pages if '_fr_' in url)
            total = len(self.pages)
            if total == 0:
                return 0
            return min(fr_count, total - fr_count) / (total / 2)
        except Exception:
            return 0

    def _calculate_thematic_isolation(self):
        """Calcule l'isolation thématique des clusters"""
        try:
            isolation_scores = []
            for cluster_id, urls in self.clusters.items():
                internal_links = 0
                external_links = 0
                for url in urls:
                    for _, dest in self.graph.out_edges(url):
                        if dest in self.pages:
                            if self.pages[dest].cluster == cluster_id:
                                internal_links += 1
                            else:
                                external_links += 1
                if internal_links + external_links > 0:
                    isolation_scores.append(internal_links / (internal_links + external_links))
            return np.mean(isolation_scores) if isolation_scores else 0
        except Exception:
            return 0
        

    def _calculate_semantic_metrics(self):
        """Métriques sémantiques améliorées"""
        try:
            # Matrice de transition entre clusters
            n_clusters = len(self.clusters)
            flow_matrix = np.zeros((n_clusters, n_clusters))
            
            # Parcours des arêtes du graphe
            for source, destination in self.graph.edges():
                if source in self.pages and destination in self.pages:
                    source_cluster = self.pages[source].cluster
                    target_cluster = self.pages[destination].cluster
                    flow_matrix[int(source_cluster)][int(target_cluster)] += 1
            
            # Normalisation
            row_sums = flow_matrix.sum(axis=1)
            self.semantic_transition_matrix = np.divide(flow_matrix, row_sums[:, np.newaxis],
                                                    where=row_sums[:, np.newaxis] != 0)
            
            metrics = {
                'cluster_coherence': self._calculate_cluster_coherence(),
                'semantic_flow_strength': float(np.mean(flow_matrix[flow_matrix > 0])) if flow_matrix.any() else 0,
                'inter_cluster_density': float(np.count_nonzero(flow_matrix) / (n_clusters * n_clusters)) if n_clusters > 0 else 0,
                'language_balance': self._calculate_language_balance(),
                'thematic_isolation': self._calculate_thematic_isolation()
            }
            
            return metrics
        except Exception as e:
            print(f"Erreur lors du calcul des métriques sémantiques: {str(e)}")
            return {
                'cluster_coherence': 0,
                'semantic_flow_strength': 0,
                'inter_cluster_density': 0,
                'language_balance': 0,
                'thematic_isolation': 0
            }
        


    def _calculate_cluster_metrics(self):
        """Calcul des métriques de cluster"""
        try:
            cluster_sizes = [len(urls) for urls in self.clusters.values()]
            
            metrics = {
                'number_of_clusters': len(self.clusters),
                'average_cluster_size': float(np.mean(cluster_sizes)) if cluster_sizes else 0,
                'cluster_size_variance': float(np.var(cluster_sizes)) if cluster_sizes else 0,
                'cluster_density': self._calculate_cluster_density(),
                'inter_cluster_links': self._count_inter_cluster_links()
            }
            
            return metrics
        except Exception as e:
            print(f"Erreur lors du calcul des métriques de cluster: {str(e)}")
            return {}

    def _calculate_entropy(self, distribution):
        """Calcule l'entropie d'une distribution"""
        distribution = np.array(distribution)
        distribution = distribution / np.sum(distribution)
        return -np.sum(distribution * np.log2(distribution + 1e-10))

    def _calculate_cluster_coherence(self):
        """Calcule la cohérence moyenne des clusters"""
        coherence_scores = []
        
        for cluster_urls in self.clusters.values():
            if len(cluster_urls) < 2:
                continue
                
            internal_links = 0
            possible_links = len(cluster_urls) * (len(cluster_urls) - 1)
            
            for url in cluster_urls:
                for target in self.graph.successors(url):
                    if target in cluster_urls:
                        internal_links += 1
                        
            if possible_links > 0:
                coherence_scores.append(internal_links / possible_links)
                
        return float(np.mean(coherence_scores)) if coherence_scores else 0

    def _calculate_cluster_density(self):
        """Calcule la densité moyenne des clusters"""
        densities = []
        
        for cluster_urls in self.clusters.values():
            if len(cluster_urls) < 2:
                continue
                
            subgraph = self.graph.subgraph(cluster_urls)
            density = nx.density(subgraph)
            densities.append(density)
            
        return float(np.mean(densities)) if densities else 0

    def _count_inter_cluster_links(self):
        """Compte le nombre de liens entre clusters différents"""
        inter_cluster_links = 0
        
        for source, target in self.graph.edges():
            if (source in self.pages and target in self.pages and 
                self.pages[source].cluster != self.pages[target].cluster):
                inter_cluster_links += 1
                
        return inter_cluster_links


    def generate_scientific_report(self, metrics1, metrics2):
        """Génère un rapport scientifique comparatif détaillé avec gestion d'erreur robuste"""
        try:
            report = []
            report.append("\n=== ANALYSE COMPARATIVE DU MAILLAGE INTERNE ===\n")

            # Fonction helper pour accéder aux métriques de manière sécurisée
            def get_metric(metrics, category, key, default=0.0):
                try:
                    return float(metrics.get(category, {}).get(key, default))
                except (TypeError, ValueError):
                    return default

            # Résumé comparatif
            report.append("📊 ÉVOLUTION GLOBALE")
            report.append("AVANT :")
            report.append(f"• {get_metric(metrics1, 'structural_metrics', 'number_of_nodes', 0):.0f} pages")
            report.append(f"• {get_metric(metrics1, 'structural_metrics', 'number_of_edges', 0):.0f} liens internes")
            report.append(f"• {get_metric(metrics1, 'structural_metrics', 'density', 0):.3f} densité moyenne")
            report.append(f"• {get_metric(metrics1, 'cluster_metrics', 'number_of_clusters', 0):.0f} clusters")
            report.append(f"• {get_metric(metrics1, 'structural_metrics', 'bilingual_links', 0):.0f} liens bilingues")
            
            report.append("\nAPRÈS :")
            report.append(f"• {get_metric(metrics2, 'structural_metrics', 'number_of_nodes', 0):.0f} pages")
            report.append(f"• {get_metric(metrics2, 'structural_metrics', 'number_of_edges', 0):.0f} liens internes")
            report.append(f"• {get_metric(metrics2, 'structural_metrics', 'density', 0):.3f} densité moyenne")
            report.append(f"• {get_metric(metrics2, 'cluster_metrics', 'number_of_clusters', 0):.0f} clusters")
            report.append(f"• {get_metric(metrics2, 'structural_metrics', 'bilingual_links', 0):.0f} liens bilingues")

            # Qualité du maillage
            report.append("\n🔗 QUALITÉ DU MAILLAGE")
            maillage_metrics = [
                ('Clustering moyen', 'average_clustering', 'structural_metrics', 'Mesure de la cohésion des groupes de pages'),
                ('Réciprocité', 'reciprocity', 'structural_metrics', 'Proportion de liens bidirectionnels'),
                ('Densité', 'density', 'structural_metrics', 'Proportion de liens réalisés vs possibles'),
                ('Liens thématiques', 'thematic_links', 'structural_metrics', 'Liens entre pages de même thème'),
                ('Liens inter-thèmes', 'cross_thematic_links', 'structural_metrics', 'Liens entre thèmes différents')
            ]

            for name, key, category, description in maillage_metrics:
                before = get_metric(metrics1, category, key)
                after = get_metric(metrics2, category, key)
                report.append(f"• {name}: {self.format_change(before, after)}")
                report.append(f"  ℹ️  {description}")

            # Accessibilité
            report.append("\n🎯 ACCESSIBILITÉ DU CONTENU")
            accessibility_metrics = [
                ('Pages à 3 clics', 'pages_within_3_clicks', 'accessibility_metrics', 'Part du contenu facilement accessible'),
                ('Profondeur moyenne', 'mean_depth', 'accessibility_metrics', 'Nombre moyen de clics nécessaires'),
                ('Distribution PageRank', 'pagerank_entropy', 'accessibility_metrics', 'Équilibre de l\'autorité des pages'),
                ('Profondeur maximale', 'max_depth', 'accessibility_metrics', 'Nombre maximal de clics nécessaires')
            ]

            for name, key, category, description in accessibility_metrics:
                before = get_metric(metrics1, category, key)
                after = get_metric(metrics2, category, key)
                report.append(f"• {name}: {self.format_change(before, after)}")
                report.append(f"  ℹ️  {description}")

            # Structure des clusters
            report.append("\n📚 STRUCTURE DES CLUSTERS")
            cluster_metrics = [
                ('Nombre de clusters', 'number_of_clusters', 'cluster_metrics', 'Nombre de thématiques distinctes'),
                ('Taille moyenne', 'average_cluster_size', 'cluster_metrics', 'Pages par thématique'),
                ('Densité interne', 'cluster_density', 'cluster_metrics', 'Cohésion au sein des thématiques'),
                ('Liens inter-clusters', 'inter_cluster_links', 'cluster_metrics', 'Connexions entre thématiques')
            ]

            for name, key, category, description in cluster_metrics:
                before = get_metric(metrics1, category, key)
                after = get_metric(metrics2, category, key)
                report.append(f"• {name}: {self.format_change(before, after)}")
                report.append(f"  ℹ️  {description}")

            # Synthèse et recommandations
            report.append("\n📋 SYNTHÈSE ET RECOMMANDATIONS")
            improvements = []
            
            # Analyse des améliorations
            density_change = (get_metric(metrics2, 'structural_metrics', 'density') / 
                            max(0.001, get_metric(metrics1, 'structural_metrics', 'density')) - 1) * 100
            thematic_links_change = (get_metric(metrics2, 'structural_metrics', 'thematic_links') / 
                                    max(1, get_metric(metrics1, 'structural_metrics', 'thematic_links')))
            clustering_change = (get_metric(metrics2, 'structural_metrics', 'average_clustering') / 
                            max(0.001, get_metric(metrics1, 'structural_metrics', 'average_clustering')) - 1) * 100

            if density_change > 0:
                improvements.append(f"Densité du maillage améliorée de {density_change:.1f}%")
            if thematic_links_change > 1:
                improvements.append(f"Liens thématiques multipliés par {thematic_links_change:.1f}")
            if clustering_change > 0:
                improvements.append(f"Cohésion des groupes améliorée de {clustering_change:.1f}%")

            # Points forts
            if improvements:
                report.append("\n✅ Points forts :")
                for imp in improvements:
                    report.append(f"• {imp}")

            # Recommandations
            recommendations = []
            
            # Accessibilité
            if get_metric(metrics2, 'accessibility_metrics', 'pages_within_3_clicks') < get_metric(metrics1, 'accessibility_metrics', 'pages_within_3_clicks'):
                recommendations.append("Améliorer l'accessibilité en ajoutant des raccourcis depuis la page d'accueil")
            
            # Densité
            if get_metric(metrics2, 'structural_metrics', 'density') < 0.15:
                recommendations.append("Augmenter les liens contextuels entre pages thématiquement proches")
            
            # Cohésion des clusters
            if get_metric(metrics2, 'cluster_metrics', 'cluster_density') < 0.3:
                recommendations.append("Renforcer les liens entre pages d'un même thème")
            
            # Équilibre des clusters
            cluster_variance = get_metric(metrics2, 'cluster_metrics', 'cluster_size_variance')
            if cluster_variance > 5:
                recommendations.append("Équilibrer la taille des clusters thématiques")
                
            # Liens inter-clusters
            if get_metric(metrics2, 'structural_metrics', 'cross_thematic_links') < get_metric(metrics1, 'structural_metrics', 'cross_thematic_links') * 0.7:
                recommendations.append("Rétablir des liens stratégiques entre les différentes thématiques")

            # Réciprocité
            if get_metric(metrics2, 'structural_metrics', 'reciprocity') < 0.8:
                recommendations.append("Augmenter la réciprocité des liens entre les pages")

            if recommendations:
                report.append("\n💡 Recommandations prioritaires :")
                for rec in recommendations:
                    report.append(f"• {rec}")

            # Score global
            try:
                score = self._calculate_global_score(metrics2)
                report.append(f"\n🎯 SCORE GLOBAL : {score:.1f}/100")
                
                if score < 50:
                    status = "Structure à retravailler en profondeur"
                elif score < 70:
                    status = "Améliorations nécessaires"
                elif score < 90:
                    status = "Bon cocon avec optimisations possibles"
                else:
                    status = "Excellent cocon sémantique"
                    
                report.append(f"Diagnostic : {status}")
                
            except Exception as e:
                print(f"Erreur lors du calcul du score global: {str(e)}")

            return "\n".join(report)

        except Exception as e:
            print(f"Erreur générale lors de la génération du rapport: {str(e)}")
            print(f"Type d'erreur: {type(e)}")
            import traceback
            traceback.print_exc()
            return "Erreur lors de la génération du rapport"
    
    def _calculate_links_per_page(self, metrics):
        """Calcule le nombre moyen de liens par page"""
        nodes = metrics['structural_metrics']['number_of_nodes']
        edges = metrics['structural_metrics']['number_of_edges']
        return edges / nodes if nodes > 0 else 0

    def _calculate_global_score(self, metrics1, metrics2):
        """Calcule un score global sur 100"""
        weights = {
            'density': 20,
            'clustering': 20,
            'accessibility': 30,
            'cohesion': 30
        }
        
        scores = {
            'density': min(100, (metrics2['structural_metrics']['density'] / 0.2) * 100),
            'clustering': metrics2['structural_metrics']['average_clustering'] * 100,
            'accessibility': metrics2['accessibility_metrics']['pages_within_3_clicks'] * 100,
            'cohesion': self._calculate_cluster_ratio(metrics2) * 50  # Plafonné à 50 points
        }
        
        return sum(scores[k] * (weights[k]/100) for k in weights)

    def _generate_recommendations(self, metrics):
        """Génère des recommandations basées sur les métriques"""
        recommendations = []
        
        if metrics['accessibility_metrics']['pages_within_3_clicks'] < 0.9:
            recommendations.append("Améliorer l'accessibilité en ajoutant des raccourcis vers les pages profondes")
        
        if metrics['structural_metrics']['density'] < 0.15:
            recommendations.append("Augmenter le nombre de liens contextuels entre pages connexes")
        
        if self._calculate_cluster_ratio(metrics) < 1.5:
            recommendations.append("Renforcer les liens entre pages de même thématique")
        
        return recommendations


    def _evaluate_improvements(self, metrics1, metrics2):
        """Évalue les améliorations et génère des recommandations ciblées"""
        result = {
            'major_strengths': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Analyse de l'accessibilité
        if metrics2['accessibility_metrics']['pages_within_3_clicks'] < 0.8:
            result['warnings'].append(
                "La proportion de pages accessibles en 3 clics pourrait être améliorée"
            )
            result['recommendations'].append(
                "Envisager l'ajout de liens de navigation transversaux pour faciliter l'accès"
            )
            
        # Analyse de la densité
        if metrics2['structural_metrics']['density'] < 0.1:
            result['warnings'].append(
                "La densité globale du maillage reste relativement faible"
            )
            result['recommendations'].append(
                "Ajouter des liens contextuels entre les pages connexes"
            )
            
        # Analyse des clusters
        if metrics2['cluster_metrics']['cluster_density'] < 0.3:
            result['warnings'].append(
                "La cohésion au sein des clusters pourrait être renforcée"
            )
            result['recommendations'].append(
                "Renforcer les liens entre pages de même thématique"
            )
            
        # Points forts
        changes = self._calculate_significant_changes(metrics1, metrics2)
        result['major_strengths'].extend(changes['improvements'])
        
        return result

    def _calculate_significant_changes(self, metrics1, metrics2, threshold=30):
        """Calcule les changements significatifs"""
        changes = {
            'improvements': [],
            'degradations': []
        }
        
        # Analyse des métriques structurelles
        for key in ['density', 'reciprocity', 'average_clustering']:
            before = metrics1['structural_metrics'].get(key, 0)
            after = metrics2['structural_metrics'].get(key, 0)
            if before > 0:
                change = ((after - before) / before) * 100
                if change > threshold:
                    changes['improvements'].append(
                        f"Amélioration de {key}: +{change:.1f}%"
                    )
                elif change < -threshold:
                    changes['degradations'].append(
                        f"Dégradation de {key}: {change:.1f}%"
                    )
                    
        return changes

    def _calculate_global_improvements(self, metrics1, metrics2):
        """Calcule un score global d'amélioration et identifie les changements majeurs"""
        improvements = {
            'major_improvements': [],
            'areas_of_concern': [],
            'global_score': 0
        }
        
        # Définition des seuils
        SIGNIFICANT_IMPROVEMENT = 10  # 10% d'amélioration
        SIGNIFICANT_DEGRADATION = -5  # 5% de dégradation
        
        # Calcul des changements pour chaque métrique
        all_changes = []
        
        for category in ['structural_metrics', 'semantic_metrics', 'accessibility_metrics', 'cluster_metrics']:
            for key in metrics1[category]:
                before = metrics1[category].get(key, 0)
                after = metrics2[category].get(key, 0)
                
                if before == 0:
                    continue
                    
                change = ((after - before) / before) * 100
                all_changes.append(change)
                
                # Identification des changements significatifs
                if change > SIGNIFICANT_IMPROVEMENT:
                    improvements['major_improvements'].append(
                        f"{key}: amélioration de {change:.1f}%")
                elif change < SIGNIFICANT_DEGRADATION:
                    improvements['areas_of_concern'].append(
                        f"{key}: diminution de {abs(change):.1f}%")
        
        # Calcul du score global
        improvements['global_score'] = np.mean(all_changes) if all_changes else 0
        
        return improvements
    
    def generate_embeddings_comparison(self, base_metrics, embedding_metrics_list, model_names):
        """
        Génère un tableau comparatif des différents modèles d'embeddings
        
        Args:
            base_metrics: Métriques du crawl d'origine
            embedding_metrics_list: Liste des métriques pour chaque variation d'embedding
            model_names: Liste des noms des modèles (ex: "ada3-t60", "minilm-t75", etc.)
        """
        try:
            # Définition des métriques clés à comparer
            key_metrics = {
                'Densité & Connectivité': [
                    ('structural_metrics', 'density', 'Densité globale'),
                    ('structural_metrics', 'average_clustering', 'Clustering moyen'),
                    ('structural_metrics', 'reciprocity', 'Réciprocité'),
                ],
                'Qualité Thématique': [
                    ('structural_metrics', 'thematic_links', 'Liens thématiques'),
                    ('structural_metrics', 'cross_thematic_links', 'Liens inter-thèmes'),
                    ('cluster_metrics', 'cluster_density', 'Densité des clusters'),
                ],
                'Accessibilité': [
                    ('accessibility_metrics', 'pages_within_3_clicks', 'Pages à 3 clics'),
                    ('accessibility_metrics', 'mean_depth', 'Profondeur moyenne'),
                    ('accessibility_metrics', 'pagerank_entropy', 'Distribution PageRank'),
                ]
            }

            # Création d'un DataFrame pandas pour le tableau
            import pandas as pd
            
            data = []
            for i, metrics in enumerate(embedding_metrics_list):
                model_data = {
                    'Modèle': model_names[i],
                    'Threshold': int(model_names[i].split('-t')[1]),
                    'Liens totaux': metrics['structural_metrics']['number_of_edges'],
                }
                
                # Calcul des variations par rapport au crawl de base
                for category, metric_list in key_metrics.items():
                    for metric_path, metric_name, metric_label in metric_list:
                        base_value = base_metrics[metric_path][metric_name]
                        current_value = metrics[metric_path][metric_name]
                        if base_value > 0:
                            variation = ((current_value - base_value) / base_value) * 100
                        else:
                            variation = float('inf') if current_value > 0 else 0
                        
                        model_data[metric_label] = f"{current_value:.2f} ({variation:+.1f}%)"
                
                data.append(model_data)
                
            # Création du DataFrame
            df = pd.DataFrame(data)

            # Ajout d'une ligne de score global
            df['Score Global'] = df.apply(self._calculate_model_score, axis=1)
            
            # Tri par score global
            df = df.sort_values('Score Global', ascending=False)

            return df

        except Exception as e:
            print(f"Erreur lors de la génération du tableau comparatif: {str(e)}")
            return None

    def _calculate_model_score(self, row):
        """Calcule un score global pour chaque modèle"""
        try:
            score = 0
            # Densité (max 30 points)
            density_value = float(row['Densité globale'].split()[0])
            score += min(30, density_value * 100)
            
            # Clustering (max 20 points)
            clustering_value = float(row['Clustering moyen'].split()[0])
            score += min(20, clustering_value * 40)
            
            # Accessibilité (max 30 points)
            accessibility = float(row['Pages à 3 clics'].split()[0])
            score += min(30, accessibility * 30)
            
            # Cohérence thématique (max 20 points)
            cluster_density = float(row['Densité des clusters'].split()[0])
            score += min(20, cluster_density * 40)
            
            return round(score, 1)
        except Exception as e:
            print(f"Erreur lors du calcul du score: {str(e)}")
            return 0