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

class CoconAnalyzer:
    def __init__(self, redis_client, crawl_id: str):
        self.redis = redis_client
        self.crawl_id = crawl_id
        self.pages = {}
        self.graph = nx.DiGraph()
        self.clusters = defaultdict(list)
        self.root_url = None

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

    def _update_metrics(self):
        """Met à jour les métriques de base pour toutes les pages"""
        try:
            # Mise à jour des liens entrants/sortants
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

    def _calculate_entropy(self, distribution):
        """Calcule l'entropie d'une distribution"""
        distribution = np.array(distribution)
        distribution = distribution / np.sum(distribution)
        return -np.sum(distribution * np.log2(distribution + 1e-10))