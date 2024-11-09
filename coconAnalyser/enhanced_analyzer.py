from cocon_analyzer import CoconAnalyzer
import numpy as np
from collections import defaultdict
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs  # Changé de eigvals à eigs
import json

class EnhancedCoconAnalyzer(CoconAnalyzer):
    def __init__(self, redis_client, crawl_id: str):
        super().__init__(redis_client, crawl_id)
        self.semantic_transition_matrix = None
        self.load_data()



    def calculate_scientific_metrics(self):
        """Calcule les métriques scientifiques pour l'analyse du maillage"""
        metrics = {
            'structural_metrics': self._calculate_structural_metrics(),
            'semantic_metrics': self._calculate_semantic_metrics(),
            'accessibility_metrics': self._calculate_accessibility_metrics(),
            'cluster_metrics': self._calculate_cluster_metrics()
        }
        return metrics
        
    def _calculate_structural_metrics(self):
        """Calcul des métriques structurelles du graphe"""
        try:
            # Calcul des métriques de base
            adjacency_matrix = nx.adjacency_matrix(self.graph)
            # Utilisation de eigs au lieu de eigvals
            eigenvalues, _ = eigs(adjacency_matrix.asftype(float), k=2, which='LR')
            eigenvalues = np.real(eigenvalues)  # Ne garder que la partie réelle

            metrics = {
                'average_clustering': float(nx.average_clustering(self.graph.to_undirected())),
                'reciprocity': float(nx.reciprocity(self.graph)),
                'density': float(nx.density(self.graph)),
                'average_shortest_path': float(nx.average_shortest_path_length(self.graph)) if nx.is_strongly_connected(self.graph) else float('inf'),
                'average_degree': float(sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()),
                'number_of_nodes': self.graph.number_of_nodes(),
                'number_of_edges': self.graph.number_of_edges(),
                'algebraic_connectivity': float(abs(sorted(eigenvalues)[0])),  # Modifié pour utiliser eigs
                'spectral_gap': float(abs(eigenvalues[0] - eigenvalues[1])) if len(eigenvalues) > 1 else 0.0
            }
            
            return metrics
        except Exception as e:
            print(f"Erreur lors du calcul des métriques structurelles: {str(e)}")
            return {}
        

    def _calculate_semantic_metrics(self):
        """Calcul des métriques sémantiques"""
        try:
            # Création de la matrice de transition entre clusters
            n_clusters = len(self.clusters)
            flow_matrix = np.zeros((n_clusters, n_clusters))
            
            for source, targets in self.graph.edges():
                if source in self.pages and targets in self.pages:
                    source_cluster = self.pages[source].cluster
                    target_cluster = self.pages[target].cluster
                    flow_matrix[source_cluster][target_cluster] += 1
            
            # Normalisation
            row_sums = flow_matrix.sum(axis=1)
            self.semantic_transition_matrix = np.divide(flow_matrix, row_sums[:, np.newaxis],
                                                     where=row_sums[:, np.newaxis] != 0)
            
            # Calcul des métriques sémantiques
            metrics = {
                'cluster_coherence': self._calculate_cluster_coherence(),
                'semantic_flow_strength': float(np.mean(flow_matrix[flow_matrix > 0])) if flow_matrix.any() else 0,
                'inter_cluster_density': float(np.count_nonzero(flow_matrix) / (n_clusters * n_clusters)) if n_clusters > 0 else 0
            }
            
            return metrics
        except Exception as e:
            print(f"Erreur lors du calcul des métriques sémantiques: {str(e)}")
            return {}

    def _calculate_accessibility_metrics(self):
        """Calcul des métriques d'accessibilité"""
        try:
            depths = [self.pages[node].depth for node in self.graph.nodes()]
            pageranks = [self.pages[node].internal_pagerank for node in self.graph.nodes()]
            
            metrics = {
                'mean_depth': float(np.mean(depths)),
                'depth_variance': float(np.var(depths)),
                'max_depth': float(max(depths)),
                'min_depth': float(min(depths)),
                'pagerank_entropy': float(self._calculate_entropy(pageranks)),
                'pages_within_3_clicks': sum(1 for d in depths if d <= 3) / len(depths)
            }
            
            return metrics
        except Exception as e:
            print(f"Erreur lors du calcul des métriques d'accessibilité: {str(e)}")
            return {}

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
        """Génère un rapport comparatif détaillé"""
        def format_change(before, after):
            if before == 0:
                return f"{after:.4f} (N/A)"
            change = ((after - before) / before) * 100
            return f"{after:.4f} ({change:+.1f}%)"

        report = []
        report.append("=== RAPPORT D'ANALYSE COMPARATIVE ===\n")

        # Métriques structurelles
        report.append("1. MÉTRIQUES STRUCTURELLES")
        for key in metrics1['structural_metrics']:
            before = metrics1['structural_metrics'].get(key, 0)
            after = metrics2['structural_metrics'].get(key, 0)
            report.append(f"{key}: {format_change(before, after)}")

        # Métriques sémantiques
        report.append("\n2. MÉTRIQUES SÉMANTIQUES")
        for key in metrics1['semantic_metrics']:
            before = metrics1['semantic_metrics'].get(key, 0)
            after = metrics2['semantic_metrics'].get(key, 0)
            report.append(f"{key}: {format_change(before, after)}")

        # Métriques d'accessibilité
        report.append("\n3. MÉTRIQUES D'ACCESSIBILITÉ")
        for key in metrics1['accessibility_metrics']:
            before = metrics1['accessibility_metrics'].get(key, 0)
            after = metrics2['accessibility_metrics'].get(key, 0)
            report.append(f"{key}: {format_change(before, after)}")

        # Métriques de cluster
        report.append("\n4. MÉTRIQUES DE CLUSTER")
        for key in metrics1['cluster_metrics']:
            before = metrics1['cluster_metrics'].get(key, 0)
            after = metrics2['cluster_metrics'].get(key, 0)
            report.append(f"{key}: {format_change(before, after)}")

        # Synthèse
        report.append("\n5. SYNTHÈSE DES AMÉLIORATIONS")
        improvements = self._calculate_global_improvements(metrics1, metrics2)
        report.append(f"Score global d'amélioration: {improvements['global_score']:.2f}%")
        
        if improvements['major_improvements']:
            report.append("\nAméliorations majeures:")
            for imp in improvements['major_improvements']:
                report.append(f"- {imp}")
                
        if improvements['areas_of_concern']:
            report.append("\nPoints d'attention:")
            for concern in improvements['areas_of_concern']:
                report.append(f"- {concern}")

        return "\n".join(report)

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