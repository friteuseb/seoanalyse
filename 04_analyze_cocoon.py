import redis
import json
import logging
from collections import defaultdict
from urllib.parse import urlparse
import numpy as np
import networkx as nx
from dataclasses import dataclass
import subprocess

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
        """Charge et parse les données depuis Redis"""
        pattern = f"{self.crawl_id}:doc:*"
        
        # Premier passage : charger toutes les données
        for key in self.redis.scan_iter(pattern):
            doc_data = self.redis.hgetall(key)
            if not doc_data:
                continue

            url = doc_data[b'url'].decode('utf-8').rstrip('/')
            internal_links = json.loads(doc_data.get(b'internal_links_out', b'[]').decode('utf-8'))
            cluster = int(doc_data.get(b'cluster', b'0').decode('utf-8'))
            label = doc_data.get(b'label', b'').decode('utf-8')
            links_count = int(doc_data.get(b'links_count', b'0').decode('utf-8'))
            content_length = int(doc_data.get(b'content_length', b'0').decode('utf-8'))

            # Ajouter le nœud au graphe
            self.graph.add_node(url)

            # Ajouter les liens au graphe
            clean_links = [link.rstrip('/') for link in internal_links]
            for target in clean_links:
                self.graph.add_edge(url, target)

            # Stocker les métriques
            self.pages[url] = PageMetrics(
                url=url,
                depth=-1,
                cluster=cluster,
                label=label,
                incoming_links=0,  # Sera calculé plus tard
                outgoing_links=links_count,
                internal_pagerank=0.0,
                semantic_relevance=0.0,
                content_length=content_length
            )

            # Grouper par cluster
            self.clusters[cluster].append(url)

            # Identifier la racine (page avec le plus de liens sortants)
            if not self.root_url or links_count > self.pages.get(self.root_url, PageMetrics(url='', depth=0, cluster=0, label='', incoming_links=0, outgoing_links=0, internal_pagerank=0.0, semantic_relevance=0.0, content_length=0)).outgoing_links:
                self.root_url = url

        self._update_metrics()


    def _update_metrics(self):
        """Calcule les métriques basées sur le graphe"""
        # Construction du dictionnaire des liens entrants
        incoming_links = defaultdict(list)
        for url, metrics in self.pages.items():
            for target in self.graph.successors(url):
                incoming_links[target].append(url)

        # Mise à jour des métriques pour chaque page
        for url, metrics in self.pages.items():
            # Liens sortants (directement depuis Redis)
            metrics.outgoing_links = self.graph.out_degree(url)
            # Liens entrants (calculés à partir des liens sortants des autres pages)
            metrics.incoming_links = len(incoming_links[url])
            
            # Profondeur depuis la racine
            try:
                metrics.depth = nx.shortest_path_length(self.graph, self.root_url, url)
            except nx.NetworkXNoPath:
                metrics.depth = -1

        # Calcul du PageRank
        pageranks = self._calculate_pagerank()
        for url, score in pageranks.items():
            if url in self.pages:
                self.pages[url].internal_pagerank = score

    def _calculate_pagerank(self, alpha=0.85, max_iter=100, tol=1e-6):
        """
        Calcule le PageRank de chaque page avec l'algorithme original de Google.
        
        Args:
            alpha (float): Facteur d'amortissement (damping factor), par défaut 0.85
            max_iter (int): Nombre maximum d'itérations
            tol (float): Tolérance pour la convergence

        Returns:
            dict: URLs et leurs scores PageRank
        """
        n = self.graph.number_of_nodes()
        if n == 0:
            return {}

        # Initialisation : distribution uniforme
        pagerank = {node: 1/n for node in self.graph.nodes()}
        
        for _ in range(max_iter):
            prev_pagerank = pagerank.copy()
            
            # Calcul du nouveau PageRank pour chaque page
            for node in self.graph.nodes():
                incoming = self.graph.predecessors(node)
                incoming_pr = sum(prev_pagerank[in_node] / self.graph.out_degree(in_node)
                                for in_node in incoming
                                if self.graph.out_degree(in_node) > 0)
                
                # Formule du PageRank
                # PR(A) = (1-alpha)/N + alpha * sum(PR(Bi)/C(Bi))
                # où Bi sont les pages pointant vers A
                # et C(Bi) est le nombre de liens sortants de Bi
                pagerank[node] = (1 - alpha) / n + alpha * incoming_pr

            # Vérification de la convergence
            err = sum(abs(pagerank[node] - prev_pagerank[node]) for node in self.graph.nodes())
            if err < tol:
                break

        # Normalisation des scores
        total = sum(pagerank.values())
        return {node: score/total for node, score in pagerank.items()}


    def analyze_cocon(self):
        """Analyse complète du cocon sémantique"""
        if not self.pages:
            self.load_data()

        # Analyse globale
        total_pages = len(self.pages)
        avg_depth = np.mean([p.depth for p in self.pages.values() if p.depth >= 0])
        max_depth = max(p.depth for p in self.pages.values() if p.depth >= 0)

        # Analyse des clusters
        cluster_analysis = self._analyze_clusters()

        # Analyse des liens
        link_analysis = self._analyze_links()

        # Détection des problèmes
        issues = self._detect_issues()

        return {
            "general_metrics": {
                "total_pages": total_pages,
                "average_depth": round(avg_depth, 2),
                "max_depth": max_depth,
                "total_internal_links": self.graph.number_of_edges(),
                "average_links_per_page": round(self.graph.number_of_edges() / total_pages, 2)
            },
            "cluster_metrics": cluster_analysis,
            "link_metrics": link_analysis,
            "issues": issues,
            "quality_score": self._calculate_quality_score()
        }

    def _analyze_clusters(self):
        """Analyse détaillée des clusters"""
        cluster_metrics = {}
        
        for cluster_id, urls in self.clusters.items():
            internal_links = 0
            external_links = 0
            depths = []
            pageranks = []
            
            for url in urls:
                if url not in self.pages:
                    continue
                    
                metrics = self.pages[url]
                if metrics.depth >= 0:
                    depths.append(metrics.depth)
                pageranks.append(metrics.internal_pagerank)
                
                # Compter les liens internes au cluster vs externes
                for target in self.graph.successors(url):
                    if target in urls:
                        internal_links += 1
                    else:
                        external_links += 1
            
            cluster_metrics[cluster_id] = {
                "size": len(urls),
                "label": self.pages[urls[0]].label if urls else "Unknown",
                "internal_links": internal_links,
                "external_links": external_links,
                "cohesion": round(internal_links / len(urls) if len(urls) > 0 else 0, 2),
                "avg_depth": round(np.mean(depths) if depths else -1, 2),
                "avg_pagerank": round(np.mean(pageranks) if pageranks else 0, 4)
            }
            
        return cluster_metrics

    def _analyze_links(self):
        """Analyse détaillée des liens"""
        return {
            "connectivity": {
                "strongly_connected_components": len(list(nx.strongly_connected_components(self.graph))),
                "weakly_connected_components": len(list(nx.weakly_connected_components(self.graph)))
            },
            "link_distribution": {
                "min_outgoing": min(self.graph.out_degree(n) for n in self.graph.nodes()),
                "max_outgoing": max(self.graph.out_degree(n) for n in self.graph.nodes()),
                "avg_outgoing": round(np.mean([self.graph.out_degree(n) for n in self.graph.nodes()]), 2),
                "avg_incoming": round(np.mean([self.graph.in_degree(n) for n in self.graph.nodes()]), 2)
            },
            "top_linked_pages": self._get_top_pages()
        }

    def _get_top_pages(self, limit=5):
        """Identifie les pages les plus importantes"""
        pages_metrics = []
        for url, metrics in self.pages.items():
            pages_metrics.append({
                "url": url,
                "incoming": self.graph.in_degree(url),
                "outgoing": self.graph.out_degree(url),
                "pagerank": metrics.internal_pagerank
            })
            
        return sorted(pages_metrics, key=lambda x: x['pagerank'], reverse=True)[:limit]

    def _detect_issues(self):
        """Détecte les problèmes potentiels dans le cocon"""
        issues = {
            "orphan_pages": [],
            "dead_ends": [],
            "deep_pages": [],
            "weak_clusters": []
        }

        for url, metrics in self.pages.items():
            if metrics.incoming_links == 0 and url != self.root_url:
                issues["orphan_pages"].append(url)
            if metrics.outgoing_links == 0:
                issues["dead_ends"].append(url)
            if metrics.depth > 3:  # Pages trop profondes
                issues["deep_pages"].append(url)

        # Détection des clusters faiblement connectés
        for cluster_id, urls in self.clusters.items():
            internal_links = sum(1 for u in urls for v in self.graph.successors(u) if v in urls)
            if internal_links / len(urls) < 1:  # Moins d'un lien interne par page en moyenne
                issues["weak_clusters"].append({
                    "cluster": cluster_id,
                    "label": self.pages[urls[0]].label if urls else "Unknown",
                    "cohesion": round(internal_links / len(urls), 2)
                })

        return issues

    def _calculate_quality_score(self):
        """Calcule un score global de qualité du cocon"""
        # Facteurs positifs
        avg_internal_links = np.mean([p.outgoing_links for p in self.pages.values()])
        connectivity_ratio = len(list(nx.strongly_connected_components(self.graph))) / len(self.pages)
        avg_depth = np.mean([p.depth for p in self.pages.values() if p.depth >= 0])
        
        # Pénalités
        orphan_penalty = len(self._detect_issues()["orphan_pages"]) / len(self.pages)
        depth_penalty = len([p for p in self.pages.values() if p.depth > 3]) / len(self.pages)
        
        # Score final (0-100)
        base_score = (
            (min(avg_internal_links / 5, 1) * 40) +  # Liens internes
            ((1 - connectivity_ratio) * 30) +        # Connectivité
            (max(0, 1 - avg_depth/5) * 30)          # Profondeur
        )
        
        penalties = (orphan_penalty + depth_penalty) * 50
        
        return round(max(0, min(100, base_score - penalties)), 2)

    def generate_report(self, results):
        """Génère un rapport d'analyse concis et actionnable"""
        
        def section_header(title):
            return f"\n{'='*50}\n{title}\n{'='*50}"

        report = []
        report.append(section_header("📊 RAPPORT D'ANALYSE DU COCON SÉMANTIQUE"))

        # Métriques Essentielles
        report.append("\n🔍 MÉTRIQUES CLÉS")
        metrics = results["general_metrics"]
        report.append(f"""
    - Pages totales : {metrics['total_pages']}
    - Profondeur moyenne : {metrics['average_depth']:.1f} clics
    - Profondeur max : {metrics['max_depth']} clics
    - Liens internes : {metrics['total_internal_links']}
    - Moyenne liens/page : {metrics['average_links_per_page']:.1f}
    """)

        # Santé des Clusters
        report.append("\n📈 CLUSTERS THÉMATIQUES")
        for cluster_id, data in results["cluster_metrics"].items():
            if data['cohesion'] < 0.5:
                status = "⚠️  À renforcer"
            elif data['cohesion'] > 1:
                status = "✅ Excellent"
            else:
                status = "👍 Correct"
                
            report.append(f"""
    Cluster {cluster_id} - {data['label']} {status}
    - Pages : {data['size']}
    - Cohésion : {data['cohesion']:.2f}
    - Liens internes/externes : {data['internal_links']}/{data['external_links']}""")

        # Pages Critiques
        report.append("\n⭐ PAGES STRATÉGIQUES")
        top_pages = sorted(self.pages.items(), key=lambda x: x[1].internal_pagerank, reverse=True)[:3]
        for url, metrics in top_pages:
            report.append(f"""
    {url}
    - PageRank : {metrics.internal_pagerank:.4f}
    - Liens entrants/sortants : {metrics.incoming_links}/{metrics.outgoing_links}""")

        # Problèmes Détectés
        issues = results["issues"]
        critical_issues = []
        if len(issues['orphan_pages']) > 0:
            critical_issues.append(f"⚠️  {len(issues['orphan_pages'])} pages orphelines")
        if len(issues['dead_ends']) > 0:
            critical_issues.append(f"⚠️  {len(issues['dead_ends'])} pages en impasse")
        if len(issues['deep_pages']) > 0:
            critical_issues.append(f"⚠️  {len(issues['deep_pages'])} pages trop profondes")

        if critical_issues:
            report.append("\n⚠️  ALERTES")
            report.extend(critical_issues)
            if issues['orphan_pages']:
                report.append("\nPages orphelines prioritaires:")
                for url in issues['orphan_pages'][:3]:
                    report.append(f"• {url}")

        # Score et Recommandations
        report.append(f"\n🎯 SCORE GLOBAL : {results['quality_score']}/100")
        
        if results['quality_score'] < 50:
            status = "Structure à retravailler en profondeur"
        elif results['quality_score'] < 70:
            status = "Améliorations nécessaires"
        elif results['quality_score'] < 90:
            status = "Bon cocon avec optimisations possibles"
        else:
            status = "Excellent cocon sémantique"
            
        report.append(f"Diagnostic : {status}")

        # Actions Prioritaires
        report.append("\n📝 ACTIONS PRIORITAIRES")
        recommendations = []
        if metrics['average_depth'] > 3:
            recommendations.append("• Réduire la profondeur moyenne (créer des raccourcis)")
        if issues['orphan_pages']:
            recommendations.append("• Ajouter des liens vers les pages orphelines listées")
        if any(c['cohesion'] < 0.5 for c in results["cluster_metrics"].values()):
            recommendations.append("• Renforcer les liens entre pages de même thématique")
        
        report.extend(recommendations if recommendations else ["✅ Aucune action critique requise"])

        return "\n".join(report)

def main(crawl_id):
    redis_port = subprocess.check_output(
        "ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", 
        shell=True
    )
    redis_client = redis.Redis(host='localhost', port=int(redis_port), db=0)
    
    analyzer = CoconAnalyzer(redis_client, crawl_id)
    results = analyzer.analyze_cocon()
    
    # Génération et affichage du rapport
    print(analyzer.generate_report(results))
    
    # Optionnel : sauvegarder le rapport dans un fichier
    with open(f"cocon_analysis_{crawl_id[:30]}.txt", "w") as f:
        f.write(analyzer.generate_report(results))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_cocon.py <crawl_id>")
        sys.exit(1)
    
    main(sys.argv[1])