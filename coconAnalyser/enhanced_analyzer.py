from cocon_analyzer import CoconAnalyzer
import numpy as np
import logging
import networkx as nx

class EnhancedCoconAnalyzer(CoconAnalyzer):
    def __init__(self, redis_client, crawl_id: str):
        super().__init__(redis_client, crawl_id)
        self.semantic_transition_matrix = None
        self.load_data()



    def _normalize_url(self, url):
        """Normalise une URL en extrayant uniquement le dernier segment"""
        try:
            # Retirer les éventuels slashes de fin
            url = url.rstrip('/')
            
            # Récupérer le dernier segment
            last_segment = url.split('/')[-1]
            
            # Retirer l'extension si elle existe (.html ou autre)
            if '.' in last_segment:
                last_segment = last_segment.split('.')[0]
                
            return last_segment
            
        except Exception as e:
            logging.error(f"Erreur de normalisation d'URL {url}: {str(e)}")
            return url

    # Exemple d'utilisation :
    # "https://chadyagamma.fr/guide-sonotherapie/sonotherapie-benefices-domicile/"
    #  -> "sonotherapie-benefices-domicile"
    #
    # "http://0.0.0.0:8000/sonotherapie-benefices-domicile.html"
    #  -> "sonotherapie-benefices-domicile"
    
    def calculate_scientific_metrics(self):
        """Calcule toutes les métriques scientifiques pour l'analyse"""
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
        
    def _count_inter_cluster_links(self):
        """Compte le nombre de liens entre clusters différents"""
        count = 0
        for source, target in self.graph.edges():
            if source in self.pages and target in self.pages:
                source_cluster = self.pages[source].cluster
                target_cluster = self.pages[target].cluster
                if source_cluster != target_cluster:
                    count += 1
        return count

    def _calculate_structural_metrics(self):
        """Analyse structurelle avancée du maillage"""
        try:
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

    def _calculate_semantic_metrics(self):
        """Analyse sémantique du maillage"""
        try:
            n_clusters = len(self.clusters)
            flow_matrix = np.zeros((n_clusters, n_clusters))
            
            for source, target in self.graph.edges():
                if source in self.pages and target in self.pages:
                    source_cluster = self.pages[source].cluster
                    target_cluster = self.pages[target].cluster
                    flow_matrix[int(source_cluster)][int(target_cluster)] += 1
            
            row_sums = flow_matrix.sum(axis=1)
            self.semantic_transition_matrix = np.divide(flow_matrix, row_sums[:, np.newaxis],
                                                     where=row_sums[:, np.newaxis] != 0)
            
            return {
                'cluster_coherence': self._calculate_cluster_coherence(),
                'semantic_flow_strength': float(np.mean(flow_matrix[flow_matrix > 0])) if flow_matrix.any() else 0,
                'inter_cluster_density': float(np.count_nonzero(flow_matrix) / (n_clusters * n_clusters)) if n_clusters > 0 else 0,
                'language_balance': self._calculate_language_balance(),
                'thematic_isolation': self._calculate_thematic_isolation()
            }
        except Exception as e:
            logging.error(f"Erreur lors du calcul des métriques sémantiques: {str(e)}")
            return self._get_default_semantic_metrics()

    def _calculate_accessibility_metrics(self):
        """Analyse de l'accessibilité du contenu"""
        try:
            depths = []
            if self.root_url:
                for node in self.graph.nodes():
                    if nx.has_path(self.graph, self.root_url, node):
                        depth = nx.shortest_path_length(self.graph, self.root_url, node)
                        depths.append(depth)

            metrics = {}
            if depths:
                metrics['mean_depth'] = float(np.mean(depths))
                metrics['depth_variance'] = float(np.var(depths))
                metrics['max_depth'] = float(max(depths))
                metrics['min_depth'] = float(min(depths))
                metrics['pages_within_3_clicks'] = float(sum(1 for d in depths if d <= 3) / len(self.graph.nodes()))
            else:
                return self._get_default_accessibility_metrics()

            try:
                pagerank = nx.pagerank(self.graph)
                metrics['pagerank_entropy'] = float(self._calculate_entropy(list(pagerank.values())))
            except:
                metrics['pagerank_entropy'] = 0.0

            return metrics

        except Exception as e:
            logging.error(f"Erreur lors du calcul des métriques d'accessibilité: {str(e)}")
            return self._get_default_accessibility_metrics()

    def _calculate_cluster_metrics(self):
        """Analyse détaillée des clusters"""
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
            logging.error(f"Erreur lors du calcul des métriques de cluster: {str(e)}")
            return self._get_default_cluster_metrics()

    # Méthodes de support pour les calculs
    def _count_bilingual_links(self):
        return sum(1 for source, target in self.graph.edges() 
                  if ('_fr_' in source) != ('_fr_' in target))

    def _count_thematic_links(self):
        return sum(1 for source, target in self.graph.edges()
                  if source in self.pages and target in self.pages
                  and self.pages[source].cluster == self.pages[target].cluster)

    def _count_cross_thematic_links(self):
        return sum(1 for source, target in self.graph.edges()
                  if source in self.pages and target in self.pages
                  and self.pages[source].cluster != self.pages[target].cluster)

    def _calculate_cluster_density(self):
        densities = []
        for cluster_urls in self.clusters.values():
            if len(cluster_urls) > 1:
                subgraph = self.graph.subgraph(cluster_urls)
                densities.append(nx.density(subgraph))
        return float(np.mean(densities)) if densities else 0

    def _calculate_cluster_coherence(self):
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

    def _calculate_language_balance(self):
        fr_count = sum(1 for url in self.pages if '_fr_' in url)
        total = len(self.pages)
        return min(fr_count, total - fr_count) / (total / 2) if total > 0 else 0

    def _calculate_thematic_isolation(self):
        try:
            isolation_scores = []
            for cluster_id, urls in self.clusters.items():
                internal_links = external_links = 0
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

    # Méthodes pour les valeurs par défaut
    def _get_default_structural_metrics(self):
        return {
            'average_clustering': 0.0,
            'reciprocity': 0.0,
            'density': 0.0,
            'average_degree': 0.0,
            'number_of_nodes': len(self.pages),
            'number_of_edges': 0,
            'bilingual_links': 0,
            'thematic_links': 0,
            'cross_thematic_links': 0,
            'average_shortest_path': float('inf')
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
            'pages_within_3_clicks': 1.0,
            'pagerank_entropy': 0.0
        }

    def _get_default_cluster_metrics(self):
        return {
            'number_of_clusters': len(self.clusters),
            'average_cluster_size': 0.0,
            'cluster_size_variance': 0.0,
            'cluster_density': 0.0,
            'inter_cluster_links': 0
        }