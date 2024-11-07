from cocon_analyzer import CoconAnalyzer  
from collections import defaultdict
import json

class EnhancedCoconAnalyzer(CoconAnalyzer):
    def __init__(self, redis_client, crawl_id: str):
        # Appel du constructeur parent
        super().__init__(redis_client, crawl_id)
        self.semantic_groups = {}
        self.topic_flow = {}


    def generate_enhanced_report(self):
        """Génère un rapport enrichi avec les nouvelles analyses"""
        # D'abord, obtenir les résultats de base
        results = self.analyze_cocon()
        
        # Générer le rapport de base
        base_report = super().generate_report(results)
        
        # Ajouter les analyses supplémentaires
        topic_flow = self.analyze_topic_flow()
        semantic_coherence = self.calculate_semantic_coherence()
        
        enhanced_sections = []
        
        # Analyse du flux thématique
        enhanced_sections.append("\n🔄 FLUX THÉMATIQUE")
        for source_cluster, targets in topic_flow['flow_matrix'].items():
            cluster_name = topic_flow['cluster_names'].get(source_cluster, f"Cluster {source_cluster}")
            enhanced_sections.append(f"\n{cluster_name}:")
            for target_cluster, count in targets.items():
                target_name = topic_flow['cluster_names'].get(target_cluster, f"Cluster {target_cluster}")
                enhanced_sections.append(f"  → {target_name}: {count} liens")
        
        # Cohérence sémantique
        enhanced_sections.append("\n🎯 COHÉRENCE SÉMANTIQUE")
        for cluster_id, coherence in semantic_coherence.items():
            enhanced_sections.append(f"""
        Cluster {cluster_id}:
        - Score de cohérence: {coherence['coherence_score']}
        - Termes communs: {', '.join(coherence['common_terms'])}
        - Densité de liens internes: {coherence['internal_links_density']}
        - Liens internes: {coherence['internal_links']}
        - Liens externes: {coherence['external_links']}
        - Taille du cluster: {coherence['size']}""")
        
        # Combiner le rapport de base avec les sections enrichies
        return base_report + "\n" + "\n".join(enhanced_sections)


    # Le reste de vos méthodes restent inchangées...
    def analyze_topic_flow(self):
        """Analyse le flux thématique entre les clusters"""
        flow_matrix = defaultdict(lambda: defaultdict(int))
        cluster_names = {}
        
        # Construire la matrice des flux entre clusters
        for source_url, metrics in self.pages.items():
            source_cluster = metrics.cluster
            if source_cluster not in cluster_names:
                cluster_names[source_cluster] = metrics.label
                
            doc_data = self.redis.hgetall(f"{self.crawl_id}:doc:{source_url.split('/')[-1]}")
            outgoing_links = json.loads(doc_data.get(b'internal_links_out', b'[]').decode('utf-8'))
            
            for target_url in outgoing_links:
                if target_url in self.pages:
                    target_cluster = self.pages[target_url].cluster
                    flow_matrix[source_cluster][target_cluster] += 1

        return {
            'flow_matrix': dict(flow_matrix),
            'cluster_names': cluster_names
        }

    def calculate_semantic_coherence(self):
        """Analyse la cohérence sémantique des clusters"""
        cluster_coherence = {}
        
        for cluster_id, urls in self.clusters.items():
            # Ignorer le cluster de la page d'accueil
            if len(urls) == 1 and urls[0] == self.root_url:
                continue
                
            labels = []
            internal_links = 0
            external_links = 0
            
            # Collecter les labels et compter les liens
            for url in urls:
                if url not in self.pages:
                    continue
                    
                metrics = self.pages[url]
                labels.append(metrics.label)
                
                for target in self.graph.successors(url):
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

    def _extract_common_terms(self, labels):
        """Extrait les termes communs des labels d'un cluster"""
        if not labels:
            return []
            
        # Séparer les mots de tous les labels
        word_sets = [set(label.lower().split()) for label in labels]
        
        # Trouver l'intersection de tous les ensembles
        common_terms = set.intersection(*word_sets)
        
        return list(common_terms)