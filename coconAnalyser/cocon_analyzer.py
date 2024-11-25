import redis
import json
import logging
import numpy as np
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict
from urllib.parse import urlparse

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
    is_orphan: bool = False  

class CoconAnalyzer:
    def __init__(self, redis_client, crawl_id: str):
        self.redis = redis_client
        self.crawl_id = crawl_id
        self.pages = {}
        self.graph = nx.DiGraph()
        self.clusters = defaultdict(list)
        self.root_url = None
        self.orphan_pages = set()  # Ajout d'un set pour suivre les pages orphelines


    def load_data(self):
        """Charge et parse les données depuis Redis avec normalisation des URLs"""
        try:
            pattern = f"{self.crawl_id}:doc:*"
            
            for key in self.redis.scan_iter(pattern):
                try:
                    doc_data = self.redis.hgetall(key)
                    if not doc_data:
                        continue

                    # Normalisation de l'URL de la page
                    url = self._normalize_url(doc_data[b'url'].decode('utf-8'))
                    
                    # Normalisation des liens internes
                    internal_links = json.loads(doc_data.get(b'internal_links_out', b'[]').decode('utf-8'))
                    normalized_links = [self._normalize_url(link) for link in internal_links]
                    
                    # Construction du graphe avec les URLs normalisées
                    self.graph.add_node(url)
                    for target in normalized_links:
                        self.graph.add_edge(url, target)

                    cluster = int(doc_data.get(b'cluster', b'0').decode('utf-8'))
                    label = doc_data.get(b'label', b'').decode('utf-8')
                    links_count = int(doc_data.get(b'links_count', b'0').decode('utf-8'))
                    content_length = int(doc_data.get(b'content_length', b'0').decode('utf-8'))

                    self.pages[url] = PageMetrics(
                        url=url,
                        depth=-1,
                        cluster=cluster,
                        label=label,
                        incoming_links=0,
                        outgoing_links=links_count,
                        internal_pagerank=0.0,
                        semantic_relevance=0.0,
                        content_length=content_length
                    )

                    self.clusters[cluster].append(url)

                except Exception as e:
                    logging.error(f"Erreur lors du traitement de la clé {key}: {str(e)}")
                    continue

            # Identifier la racine et mettre à jour les métriques
            self._identify_root()
            self._update_metrics()
            
        except Exception as e:
            logging.error(f"Erreur lors du chargement des données : {str(e)}")
            raise

    def _identify_root(self):
        """Identifie la page racine du site"""
        if not self.pages:
            return
            
        max_outgoing = -1
        for url, metrics in self.pages.items():
            outgoing_count = len([edge for edge in self.graph.edges() if edge[0] == url])
            if outgoing_count > max_outgoing:
                max_outgoing = outgoing_count
                self.root_url = url


    def _detect_issues(self):
        """Détecte les problèmes potentiels dans le cocon"""
        try:
            issues = {
                "orphan_pages": [],
                "dead_ends": [],
                "deep_pages": [],
                "weak_clusters": []
            }

            # Utilisation de in_degree pour vérifier les liens entrants
            for node in self.graph.nodes():
                in_degree = self.graph.in_degree(node)
                out_degree = self.graph.out_degree(node)
                
                # Une page est orpheline si elle n'a aucun lien entrant ET n'est pas la page d'accueil
                if in_degree == 0 and node != self.root_url:
                    issues["orphan_pages"].append(node)
                    logging.info(f"Page orpheline détectée: {node} (0 liens entrants)")
                
                # Une page est un cul-de-sac si elle n'a aucun lien sortant
                if out_degree == 0:
                    issues["dead_ends"].append(node)
                
                # Pages trop profondes
                if self.pages[node].depth > 3:
                    issues["deep_pages"].append(node)

            # Logs de debug détaillés
            logging.info(f"=== Analyse de la structure des liens ===")
            logging.info(f"Nombre total de pages: {len(self.graph.nodes())}")
            logging.info(f"Nombre total de liens: {len(self.graph.edges())}")
            logging.info(f"Pages orphelines trouvées: {len(issues['orphan_pages'])}")
            
            # Afficher quelques exemples de liens pour vérification
            if len(self.graph.edges()) > 0:
                logging.info("\nExemples de liens existants:")
                for source, target in list(self.graph.edges())[:5]:
                    logging.info(f"{source} -> {target}")

            return issues

        except Exception as e:
            logging.error(f"Erreur lors de la détection des problèmes: {str(e)}")
            return {"orphan_pages": [], "dead_ends": [], "deep_pages": [], "weak_clusters": []}
        



    def _calculate_objective_score(self, metrics):
        """Mise à jour du calcul du score pour inclure les îlots"""
        try:
            # Score existant
            score = super()._calculate_objective_score(metrics)
            
            # Pénalité pour les îlots (retirer jusqu'à 10 points)
            islands_stats = metrics['structural_metrics'].get('islands', {})
            island_count = islands_stats.get('count', 0)
            main_coverage = islands_stats.get('main_island_coverage', 0)
            
            island_penalty = 0
            if island_count > 1:
                # Pénalité basée sur le nombre d'îlots et la couverture de l'îlot principal
                island_penalty = min(10, (island_count - 1) * 3 + (100 - main_coverage) * 0.1)
            
            final_score = max(0, score - island_penalty)
            
            # Log de la pénalité
            if island_penalty > 0:
                logging.info(f"Pénalité îlots: -{island_penalty:.1f} points")
                logging.info(f"• Nombre d'îlots: {island_count}")
                logging.info(f"• Couverture principale: {main_coverage:.1f}%")
            
            return round(final_score, 1)

        except Exception as e:
            logging.error(f"Erreur lors du calcul du score avec îlots: {str(e)}")
            return 0.0    

    def get_orphan_pages_stats(self):
        """Retourne les statistiques détaillées sur les pages orphelines"""
        try:
            issues = self._detect_issues()
            orphan_pages = issues["orphan_pages"]
            total_pages = self.graph.number_of_nodes()
            
            # Exclure la page d'accueil du total
            total_non_home = total_pages - (1 if self.root_url else 0)
            
            # Calcul du pourcentage
            percentage = (len(orphan_pages) / total_non_home * 100) if total_non_home > 0 else 0
            
            # Collecter des informations détaillées sur chaque page orpheline
            orphan_details = []
            for url in orphan_pages:
                detail = {
                    'url': url,
                    'out_links': list(self.graph.successors(url)),
                    'content_length': self.pages[url].content_length if url in self.pages else 0,
                    'cluster': self.pages[url].cluster if url in self.pages else -1
                }
                orphan_details.append(detail)
            
            # Log des statistiques détaillées
            logging.info("\n=== Statistiques des pages orphelines ===")
            logging.info(f"Total pages analysées: {total_pages}")
            logging.info(f"Pages orphelines: {len(orphan_pages)} ({percentage:.2f}%)")
            if orphan_pages:
                logging.info("\nDétail des 5 premières pages orphelines:")
                for detail in orphan_details[:5]:
                    logging.info(f"URL: {detail['url']}")
                    logging.info(f"- Liens sortants: {len(detail['out_links'])}")
                    logging.info(f"- Cluster: {detail['cluster']}")

            return {
                'count': len(orphan_pages),
                'percentage': round(percentage, 2),
                'urls': orphan_pages,
                'details': {
                    'total_pages': total_pages,
                    'total_links': self.graph.number_of_edges(),
                    'orphan_details': orphan_details,
                    'root_url': self.root_url
                }
            }

        except Exception as e:
            logging.error(f"Erreur lors du calcul des statistiques des pages orphelines: {str(e)}")
            return {'count': 0, 'percentage': 0.0, 'urls': [], 'details': {}}
        
    def _generate_orphan_details(self):
        """Génère des détails sur les pages orphelines"""
        if not self.orphan_pages:
            return "Aucune page orpheline détectée"
        
        details = []
        for url in self.orphan_pages:
            metrics = self.pages[url]
            details.append({
                'url': url,
                'outgoing_links': metrics.outgoing_links,
                'content_length': metrics.content_length,
                'cluster': metrics.cluster
            })
            
        return details


    def _calculate_entropy(self, distribution):
        """Calcule l'entropie d'une distribution"""
        distribution = np.array(distribution)
        distribution = distribution / np.sum(distribution)
        return -np.sum(distribution * np.log2(distribution + 1e-10))
    

    def _update_metrics(self):
        """Met à jour toutes les métriques des pages"""
        try:
            # Code existant pour les liens entrants/sortants
            for url in self.pages:
                self.pages[url].incoming_links = self.graph.in_degree(url)
                self.pages[url].outgoing_links = self.graph.out_degree(url)

            # Calcul du PageRank
            pagerank = nx.pagerank(self.graph)
            
            # Calcul des profondeurs
            depths = {}
            if self.root_url:
                try:
                    depths = nx.shortest_path_length(self.graph, self.root_url)
                except nx.NetworkXError:
                    depths = {url: 1 for url in self.pages}

            # Mise à jour des métriques
            for url, metrics in self.pages.items():
                metrics.internal_pagerank = pagerank.get(url, 0.0)
                metrics.depth = depths.get(url, -1)


        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour des métriques : {str(e)}")