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
        try:
            pattern = f"{self.crawl_id}:doc:*"
        
            # Premier passage : charger toutes les données
            for key in self.redis.scan_iter(pattern):
                try:
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

                except Exception as e:
                    logging.error(f"Erreur lors du traitement de la clé {key}: {str(e)}")
                    continue

            # Identifier la racine
            max_outgoing = -1
            for url, metrics in self.pages.items():
                outgoing_count = len([edge for edge in self.graph.edges() if edge[0] == url])
                if outgoing_count > max_outgoing:
                    max_outgoing = outgoing_count
                    self.root_url = url

            self._update_metrics()

        except Exception as e:
            logging.error(f"Erreur lors du chargement des données : {str(e)}")
            raise
        
    def _update_metrics(self):
        # Identifie la racine comme étant la page avec le plus de liens sortants
        max_links = 0
        for url, metrics in self.pages.items():
            doc_data = self.redis.hgetall(f"{self.crawl_id}:doc:{url.split('/')[-1]}")
            links_count = int(doc_data.get(b'links_count', b'0').decode('utf-8'))
            if links_count > max_links:
                max_links = links_count
                self.root_url = url

        # Ne pas reconstruire le graphe, utiliser celui qui existe déjà
        incoming_links = defaultdict(set)
        
        # Calculer les liens entrants à partir du graphe existant
        for edge in self.graph.edges():
            source, target = edge
            incoming_links[target].add(source)

        # Mise à jour des métriques pour chaque page
        for url, metrics in self.pages.items():
            # Mise à jour des liens
            metrics.incoming_links = len(incoming_links[url])
            metrics.outgoing_links = self.graph.out_degree(url)

        # Calcul du PageRank sur le graphe existant
        pagerank_scores = nx.pagerank(self.graph)
        
        # Calcul des profondeurs
        try:
            depth_dict = dict(nx.shortest_path_length(self.graph, self.root_url))
        except nx.NetworkXError:
            depth_dict = {url: 1 for url in self.pages}

        # Mise à jour des métriques finales
        for url, metrics in self.pages.items():
            metrics.depth = depth_dict.get(url, 1)
            metrics.internal_pagerank = pagerank_scores.get(url, 0.0)

    def calculate_semantic_coherence(self):
        """Analyse la cohérence sémantique des clusters"""
        cluster_coherence = {}
        
        for cluster_id, urls in self.clusters.items():
            # Ignorer le cluster qui contient la page d'accueil si c'est un cluster à une seule page
            if len(urls) == 1 and urls[0] == self.root_url:
                continue
                
            labels = []
            internal_links = 0
            external_links = 0
            
            # Compter les liens entre pages du même cluster
            for url in urls:
                doc_data = self.redis.hgetall(f"{self.crawl_id}:doc:{url.split('/')[-1]}")
                labels.append(self.pages[url].label)
                outgoing_links = json.loads(doc_data.get(b'internal_links_out', b'[]').decode('utf-8'))
                
                for target in outgoing_links:
                    if target in self.pages:  # Ne compter que les liens vers des pages existantes
                        if target in urls:
                            internal_links += 1
                        else:
                            external_links += 1

            # Calculer les métriques de cohérence
            total_links = internal_links + external_links
            coherence_score = internal_links / total_links if total_links > 0 else 0
            internal_density = internal_links / len(urls) if urls else 0
            
            cluster_coherence[cluster_id] = {
                'coherence_score': round(coherence_score, 2),
                'common_terms': self._extract_common_terms(labels),
                'internal_links_density': round(internal_density, 2),
                'internal_links': internal_links,
                'external_links': external_links,
                'size': len(urls)
            }
                
        return cluster_coherence

    def _analyze_clusters(self):
        """Analyse détaillée des clusters"""
        cluster_metrics = {}
        
        for cluster_id, urls in self.clusters.items():
            internal_links = 0
            external_links = 0
            depths = []
            pageranks = []
            
            # Vérifier si c'est le cluster de la page d'accueil
            is_home_cluster = len(urls) == 1 and urls[0] == self.root_url
            
            # Compter les liens réels entre pages
            for url in urls:
                if url not in self.pages:
                    continue
                    
                metrics = self.pages[url]
                if metrics.depth >= 0:
                    depths.append(metrics.depth)
                pageranks.append(metrics.internal_pagerank)
                
                # Compter les liens réels à partir du graphe
                for target in self.graph.successors(url):
                    if target in urls:
                        internal_links += 1
                    else:
                        external_links += 1
            
            # Adapter les métriques selon le type de cluster
            if is_home_cluster:
                status = "✅ Page d'accueil"
                cohesion = 1.0
            else:
                if len(urls) > 0:
                    cohesion = internal_links / (len(urls) * (len(urls) - 1)) if len(urls) > 1 else 0
                else:
                    cohesion = 0
                
                if cohesion < 0.5:
                    status = "⚠️  À renforcer"
                elif cohesion > 1:
                    status = "✅ Excellent"
                else:
                    status = "👍 Correct"
            
            cluster_metrics[cluster_id] = {
                "size": len(urls),
                "label": self.pages[urls[0]].label if urls else "Unknown",
                "internal_links": internal_links,
                "external_links": external_links,
                "cohesion": round(cohesion, 2),
                "avg_depth": round(np.mean(depths) if depths else -1, 2),
                "avg_pagerank": round(np.mean(pageranks) if pageranks else 0, 4),
                "is_home": is_home_cluster,
                "status": status
            }
                    
        return cluster_metrics


    def _detect_issues(self):
        """Détecte les problèmes potentiels dans le cocon"""
        issues = {
            "orphan_pages": [],
            "dead_ends": [],
            "deep_pages": [],
            "weak_clusters": []
        }

        for url, metrics in self.pages.items():
            # Une page est orpheline si elle n'a aucun lien entrant ET n'est pas la page d'accueil
            if metrics.incoming_links == 0 and url != self.root_url:
                issues["orphan_pages"].append(url)
            
            # Une page est un cul-de-sac si elle n'a aucun lien sortant vers d'autres pages du site
            if metrics.outgoing_links == 0:
                issues["dead_ends"].append(url)
                
            if metrics.depth > 3:
                issues["deep_pages"].append(url)

        # Analyse des clusters reste inchangée...
        return issues
    
    def verify_links(self):
        """Vérifie la cohérence des liens"""
        for url, metrics in self.pages.items():
            incoming = list(self.graph.predecessors(url))
            outgoing = list(self.graph.successors(url))
            print(f"Page: {url}")
            print(f"Liens entrants: {len(incoming)} ({incoming})")
            print(f"Liens sortants: {len(outgoing)} ({outgoing})")


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
            
            # Si c'est le cluster de la page d'accueil (un seul url avec beaucoup de liens sortants)
            is_home_cluster = len(urls) == 1 and urls[0] == self.root_url
            
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
            
            # Adaptation des métriques pour la page d'accueil
            if is_home_cluster:
                status = "✅ Page d'accueil"
                cohesion = 1.0  # La cohésion n'a pas de sens pour une seule page
            else:
                if len(urls) > 0:
                    cohesion = internal_links / len(urls)
                else:
                    cohesion = 0
                
                if cohesion < 0.5:
                    status = "⚠️  À renforcer"
                elif cohesion > 1:
                    status = "✅ Excellent"
                else:
                    status = "👍 Correct"
            
            cluster_metrics[cluster_id] = {
                "size": len(urls),
                "label": self.pages[urls[0]].label if urls else "Unknown",
                "internal_links": internal_links,
                "external_links": external_links,
                "cohesion": round(cohesion, 2),
                "avg_depth": round(np.mean(depths) if depths else -1, 2),
                "avg_pagerank": round(np.mean(pageranks) if pageranks else 0, 4),
                "is_home": is_home_cluster,
                "status": status
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
        general_metrics = results["general_metrics"] 
        report.append(f"""
        - Pages totales : {general_metrics['total_pages']}
        - Profondeur moyenne : {general_metrics['average_depth']:.1f} clics
        - Profondeur max : {general_metrics['max_depth']} clics
        - Liens internes : {general_metrics['total_internal_links']}
        - Moyenne liens/page : {general_metrics['average_links_per_page']:.1f}
        """)

        # Rapport Cluster
        report.append("\n📈 CLUSTERS THÉMATIQUES")
        for cluster_id, data in results["cluster_metrics"].items():
            if data['is_home']:
                report.append(f"""
        Cluster {cluster_id} - {data['label']} {data['status']}
        - Type : Page d'accueil du guide
        - Liens sortants : {data['external_links']}
        - PageRank : {data['avg_pagerank']:.4f}""")
            else:
                report.append(f"""
        Cluster {cluster_id} - {data['label']} {data['status']}
        - Pages : {data['size']}
        - Cohésion : {data['cohesion']:.2f}
        - Liens internes/externes : {data['internal_links']}/{data['external_links']}
        - PageRank moyen : {data['avg_pagerank']:.4f}""")

        # Pages Stratégiques avec vérification détaillée des liens
        report.append("\n⭐ PAGES STRATÉGIQUES")
        top_pages = sorted(self.pages.items(), key=lambda x: x[1].internal_pagerank, reverse=True)[:3]
        for url, metrics in top_pages:
            if url == self.root_url:
                role = "Page d'accueil"
            else:
                role = f"Cluster {metrics.cluster}"
            
            # Compte détaillé des liens
            incoming = {source.rstrip('/') for source in self.graph.predecessors(url)}
            outgoing = {target.rstrip('/') for target in self.graph.successors(url)}
            
            # Calcul sécurisé du ratio entrée/sortie
            ratio = len(incoming)/len(outgoing) if len(outgoing) > 0 else 0
            
            report.append(f"""
        {url}
        - Rôle : {role}
        - PageRank : {metrics.internal_pagerank:.4f}
        - Liens entrants uniques : {len(incoming)} sources uniques
        - Liens sortants uniques : {len(outgoing)} destinations uniques
        - Ratio entrée/sortie : {ratio:.2f}""")

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
        if general_metrics['average_depth'] > 3:
            recommendations.append("• Réduire la profondeur moyenne (créer des raccourcis)")
        if issues['orphan_pages']:
            recommendations.append("• Ajouter des liens vers les pages orphelines listées")
        if any(c['cohesion'] < 0.5 for c in results["cluster_metrics"].values() if not c.get('is_home', False)):
            recommendations.append("• Renforcer les liens entre pages de même thématique")

        report.extend(recommendations if recommendations else ["✅ Aucune action critique requise"])
        return "\n".join(report)
        
    def _verify_links(self):
        """Vérifie la cohérence des liens pour debug"""
        for url in self.pages:
            incoming = {source.rstrip('/') for source in self.graph.predecessors(url)}
            outgoing = {target.rstrip('/') for target in self.graph.successors(url)}
            if len(incoming) != self.pages[url].incoming_links or len(outgoing) != self.pages[url].outgoing_links:
                logging.warning(f"Incohérence pour {url}:")
                logging.warning(f"Stocké : in={self.pages[url].incoming_links}, out={self.pages[url].outgoing_links}")
                logging.warning(f"Calculé : in={len(incoming)}, out={len(outgoing)}")


# En dehors de la classe, à la fin du fichier :
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