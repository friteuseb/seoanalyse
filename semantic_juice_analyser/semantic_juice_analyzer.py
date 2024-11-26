from enhanced_analyzer import EnhancedCoconAnalyzer
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import logging
import json

class SemanticJuiceAnalyzer(EnhancedCoconAnalyzer):
    def __init__(self, redis_client, crawl_id: str):
        super().__init__(redis_client, crawl_id)
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.embeddings = {}
        self.semantic_transition_matrix = None
        self.semantic_pagerank = None
        self.thematic_vectors = {}
        
    def analyze_semantic_juice(self):
        """Analyse complète du transfert de jus sémantique"""
        try:
            # Vérification des données
            if not self.pages:
                logging.error("Aucune page trouvée dans le crawl")
                return None
                
            logging.info(f"Analyse de {len(self.pages)} pages...")

            # 1. Calcul des embeddings
            logging.info("Calcul des embeddings...")
            self._compute_page_embeddings()
            
            if not self.embeddings:
                logging.error("Aucun embedding n'a pu être calculé")
                return None
                
            logging.info(f"Embeddings calculés pour {len(self.embeddings)} pages")

            # 2. Construction matrice de transition
            logging.info("Construction de la matrice de transition...")
            self._build_semantic_transition_matrix()
            
            if self.semantic_transition_matrix is None:
                logging.error("Échec de la construction de la matrice de transition")
                return None

            # 3. Calcul PageRank sémantique
            logging.info("Calcul du PageRank sémantique...")
            semantic_pagerank = self._compute_semantic_pagerank()
            
            if semantic_pagerank is None:
                logging.error("Échec du calcul du PageRank sémantique")
                return None

            # 4. Analyse thématique
            logging.info("Analyse des thématiques...")
            themes_before = self._analyze_raw_themes()
            themes_after = self._analyze_themes_with_juice()

            # 5. Métriques d'efficacité
            logging.info("Calcul des métriques...")
            metrics = self._calculate_juice_metrics()

            return {
                'themes_before': themes_before,
                'themes_after': themes_after,
                'metrics': metrics,
                'semantic_pagerank': semantic_pagerank
            }

        except Exception as e:
            logging.error(f"Erreur lors de l'analyse sémantique: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None

    def _compute_page_embeddings(self):
        """Calcule les embeddings pour chaque page"""
        try:
            for url, page in self.pages.items():
                doc_key = f"{self.crawl_id}:doc:{url.split('/')[-1]}"
                content = self.redis.hget(doc_key, 'content')
                
                if content:
                    text = content.decode('utf-8')
                    if text.strip():  # Vérifier que le contenu n'est pas vide
                        self.embeddings[url] = self.model.encode(text)
                        logging.debug(f"Embedding calculé pour {url}")
                    else:
                        logging.warning(f"Contenu vide pour {url}")
                else:
                    logging.warning(f"Pas de contenu trouvé pour {url}")
                    
            logging.info(f"Embeddings calculés pour {len(self.embeddings)}/{len(self.pages)} pages")
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul des embeddings: {str(e)}")

    def _build_semantic_transition_matrix(self):
        """Construit la matrice de transition basée sur les similarités sémantiques"""
        try:
            if not self.embeddings:
                raise ValueError("Aucun embedding disponible")

            urls = list(self.embeddings.keys())
            n = len(urls)
            
            if n == 0:
                raise ValueError("Aucune URL avec embedding")

            matrix = np.zeros((n, n))
            
            for i, source in enumerate(urls):
                for j, target in enumerate(urls):
                    # Calcul de similarité cosinus
                    embedding_source = self.embeddings[source]
                    embedding_target = self.embeddings[target]
                    
                    norm_source = np.linalg.norm(embedding_source)
                    norm_target = np.linalg.norm(embedding_target)
                    
                    if norm_source == 0 or norm_target == 0:
                        similarity = 0
                    else:
                        similarity = np.dot(embedding_source, embedding_target) / (norm_source * norm_target)
                    
                    # Renforcement si lien existant
                    if self.graph.has_edge(source, target):
                        similarity *= 1.5
                        
                    matrix[i][j] = max(0, similarity)

            # Normalisation avec vérification
            row_sums = matrix.sum(axis=1)
            non_zero_rows = row_sums > 0
            
            if not any(non_zero_rows):
                raise ValueError("Toutes les lignes sont nulles dans la matrice")
                
            normalized_matrix = np.zeros_like(matrix)
            normalized_matrix[non_zero_rows] = matrix[non_zero_rows] / row_sums[non_zero_rows, np.newaxis]
            
            self.semantic_transition_matrix = normalized_matrix
            logging.info("Matrice de transition construite avec succès")
            
        except Exception as e:
            logging.error(f"Erreur lors de la construction de la matrice: {str(e)}")
            self.semantic_transition_matrix = None

    def _compute_semantic_pagerank(self, alpha=0.85, max_iter=100):
        """Calcule le PageRank sémantique avec gestion d'erreurs"""
        try:
            if self.semantic_transition_matrix is None:
                raise ValueError("Matrice de transition non disponible")

            n = len(self.semantic_transition_matrix)
            if n == 0:
                raise ValueError("Matrice de transition vide")

            # Initialisation uniforme
            pagerank = np.ones(n) / n
            
            # Itérations de l'algorithme
            for _ in range(max_iter):
                pagerank_next = (1 - alpha) / n + alpha * self.semantic_transition_matrix.T.dot(pagerank)
                
                # Vérification de la convergence
                if np.allclose(pagerank, pagerank_next):
                    break
                    
                pagerank = pagerank_next

            # Normalisation
            if pagerank.sum() > 0:
                pagerank = pagerank / pagerank.sum()
            else:
                raise ValueError("Somme du PageRank nulle")

            # Stockage dans un dictionnaire
            urls = list(self.embeddings.keys())
            self.semantic_pagerank = {urls[i]: float(pagerank[i]) for i in range(n)}
            
            return self.semantic_pagerank

        except Exception as e:
            logging.error(f"Erreur lors du calcul du PageRank sémantique: {str(e)}")
            return None

    def _analyze_raw_themes(self, n_themes=5):
        """Analyse les thématiques avant maillage avec gestion d'erreurs"""
        try:
            if not self.embeddings:
                raise ValueError("Aucun embedding disponible")
                
            embeddings_matrix = np.array(list(self.embeddings.values()))
            
            from sklearn.cluster import KMeans
            n_themes = min(n_themes, len(embeddings_matrix))
            
            if n_themes < 2:
                raise ValueError("Pas assez de données pour le clustering")
            
            kmeans = KMeans(n_clusters=n_themes, random_state=42)
            clusters = kmeans.fit_predict(embeddings_matrix)
            
            themes = {}
            for i in range(n_themes):
                cluster_embeddings = embeddings_matrix[clusters == i]
                if len(cluster_embeddings) > 0:
                    themes[i] = cluster_embeddings.mean(axis=0)
            
            return themes
            
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse des thématiques brutes: {str(e)}")
            return {}

    def _analyze_themes_with_juice(self):
        """Analyse les thématiques après transfert de jus avec gestion d'erreurs"""
        try:
            if self.semantic_pagerank is None:
                raise ValueError("PageRank sémantique non disponible")
                
            weighted_embeddings = {}
            urls = list(self.embeddings.keys())
            
            for i, url in enumerate(urls):
                if url in self.embeddings:
                    if url in self.semantic_pagerank:
                        weight = self.semantic_pagerank[url]
                        weighted_embeddings[url] = self.embeddings[url] * weight
            
            if not weighted_embeddings:
                raise ValueError("Aucun embedding pondéré calculé")
            
            matrix = np.array(list(weighted_embeddings.values()))
            
            if matrix.shape[0] < 2:
                raise ValueError("Pas assez de données pour l'analyse thématique")
            
            from sklearn.decomposition import NMF
            n_components = min(5, matrix.shape[0])
            
            # Utilisation de la valeur absolue pour NMF
            matrix_abs = np.abs(matrix)
            nmf = NMF(n_components=n_components, random_state=42)
            themes = nmf.fit_transform(matrix_abs)
            
            return {i: themes[:,i] for i in range(themes.shape[1])}
            
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse des thématiques avec jus: {str(e)}")
            return {}

    def _calculate_juice_metrics(self):
        """Calcule les métriques avec gestion d'erreurs"""
        try:
            return {
                'semantic_coherence': self._compute_semantic_coherence(),
                'juice_efficiency': self._compute_juice_efficiency(),
                'theme_preservation': self._compute_theme_preservation(),
                'link_relevance': self._compute_link_relevance()
            }
        except Exception as e:
            logging.error(f"Erreur lors du calcul des métriques: {str(e)}")
            return {
                'semantic_coherence': 0,
                'juice_efficiency': 0,
                'theme_preservation': 0,
                'link_relevance': 0
            }