import os
import redis
import numpy as np
import json
import nltk
import sys
import logging
import warnings
import asyncio
import subprocess
import anthropic
import torch
import traceback
from dotenv import load_dotenv
from transformers import CamembertModel, CamembertTokenizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chargement des variables d'environnement
load_dotenv()

# Configuration initiale
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Suppression des avertissements non pertinents
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")


# Au début du fichier, avec les autres fonctions utilitaires
def convert_to_serializable(obj):
    """Convertit les types numpy en types Python standards."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class SemanticAnalyzer:
    def __init__(self):
        # Initialisation de CamemBERT
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        self.model = CamembertModel.from_pretrained('camembert-base')
        self.model.eval()
        
        # Initialisation des stop words
        try:
            nltk.download('stopwords', quiet=True)
            self.french_stop_words = list(stopwords.words('french'))
        except Exception as e:
            logging.warning(f"Erreur lors du chargement des stop words: {e}")
            self.french_stop_words = None
        
        # Initialisation du client Anthropic si la clé est disponible
        self.llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        if not self.llm_client:
            logging.warning("Anthropic API non configurée, analyse simplifiée uniquement")

    def compute_embeddings(self, texts, batch_size=8):
        """Génère les embeddings par lots pour optimiser la mémoire."""
        all_embeddings = []
        total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size > 0 else 0)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logging.info(f"Traitement du batch {i//batch_size + 1}/{total_batches}")
            
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  max_length=512, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.numpy())
        
        return np.vstack(all_embeddings)


    def cluster_documents(self, embeddings):
        """Clustering optimisé avec DBSCAN et gestion des cas limites."""
        try:
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            # Calcul de la matrice des distances avec plus de précautions
            distances = np.linalg.norm(embeddings_scaled[:, None] - embeddings_scaled, axis=2)
            
            # Calcul plus robuste des valeurs d'eps
            distance_matrix = distances[~np.eye(distances.shape[0], dtype=bool)]
            if len(distance_matrix) == 0:
                logging.warning("Pas assez de données pour le clustering DBSCAN")
                return np.zeros(len(embeddings)), {'method': 'single_cluster'}
                
            min_dist = np.min(distance_matrix[distance_matrix > 0])
            if min_dist <= 0 or np.isnan(min_dist):
                min_dist = 0.1  # Valeur par défaut sûre
                
            # Générer une gamme d'eps plus sûre
            percentiles = np.percentile(distance_matrix, [10, 25, 50, 75, 90])
            eps_range = np.array([min_dist] + list(percentiles))
            eps_range = eps_range[eps_range > 0]  # Garantir des valeurs positives
            
            best_score = -1
            best_labels = None
            best_params = None
            
            logging.info(f"Test de {len(eps_range)} valeurs d'eps entre {eps_range[0]:.3f} et {eps_range[-1]:.3f}")
            
            for eps in eps_range:
                for min_samples in [2, 3, 4]:
                    try:
                        dbscan = DBSCAN(eps=float(eps), min_samples=min_samples)
                        labels = dbscan.fit_predict(embeddings_scaled)
                        
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        if n_clusters < 1:
                            continue
                            
                        outliers_ratio = np.sum(labels == -1) / len(labels)
                        if outliers_ratio > 0.5:  # Plus de 50% sont des outliers
                            continue
                            
                        # Calcul du score silhouette uniquement pour les points non-outliers
                        mask = labels != -1
                        if np.sum(mask) > 1:
                            score = silhouette_score(embeddings_scaled[mask], labels[mask])
                            current_score = score * (1 - outliers_ratio)
                            
                            if current_score > best_score:
                                best_score = current_score
                                best_labels = labels
                                best_params = {'eps': eps, 'min_samples': min_samples}
                                
                                logging.info(f"""
                                Meilleure configuration trouvée:
                                - Epsilon: {eps:.3f}
                                - Min samples: {min_samples}
                                - Clusters: {n_clusters}
                                - Outliers: {outliers_ratio*100:.1f}%
                                - Score: {current_score:.3f}
                                """)
                                
                    except Exception as e:
                        logging.debug(f"Configuration échouée (eps={eps}, min_samples={min_samples}): {str(e)}")
                        continue
            
            # Fallback vers KMeans si DBSCAN échoue
            if best_labels is None:
                logging.warning("DBSCAN n'a pas trouvé de configuration valide, utilisation de KMeans")
                n_clusters = min(5, len(embeddings))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                best_labels = kmeans.fit_predict(embeddings_scaled)
                best_params = {'method': 'kmeans', 'n_clusters': n_clusters}
                
            return best_labels, best_params
            
        except Exception as e:
            logging.error(f"Erreur dans le clustering: {str(e)}")
            # En cas d'erreur, retourner un cluster unique
            return np.zeros(len(embeddings)), {'method': 'fallback_single_cluster'}

    # 1. Corriger la méthode d'appel à l'API Anthropic
    async def enrich_cluster_with_llm(self, cluster_summary):
        """Enrichit la description du cluster avec Claude."""
        if not self.llm_client:
            return {"description": f"Groupe de {cluster_summary['size']} pages sur {', '.join(cluster_summary['key_terms'][:3])}"}
        
        prompt = f"""
        Analysez ce groupe de {cluster_summary['size']} pages web.
        Mots-clés principaux: {', '.join(cluster_summary['key_terms'])}
        Générez une description courte et naturelle du thème principal de ces pages.
        La description doit être en français et faire moins de 100 caractères.
        """
        
        try:
            message = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            # La réponse de Claude est dans message.content[0].text
            return {"description": message.content[0].text.strip()}
        except Exception as e:
            logging.error(f"Erreur LLM: {e}")
            return {"description": f"Groupe de pages sur {', '.join(cluster_summary['key_terms'][:3])}"}


    def prepare_cluster_summary(self, texts, labels, cluster_id):
        """Prépare le résumé d'un cluster pour l'enrichissement."""
        cluster_texts = [text for i, text in enumerate(texts) if labels[i] == cluster_id]
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words=self.french_stop_words
        )
        
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        top_indices = avg_tfidf.argsort()[-5:][::-1]
        
        return {
            "size": len(cluster_texts),
            "key_terms": [feature_names[i] for i in top_indices],
            "sample_texts": cluster_texts[:3]
        }

# Fonctions Redis
def get_redis_port():
    """Récupère le port Redis dynamique de DDEV."""
    try:
        port = subprocess.check_output(
            "ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", 
            shell=True
        )
        return int(port.strip())
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du port Redis : {e}")
        sys.exit(1)

def get_redis_connection():
    """Établit la connexion Redis avec DDEV."""
    try:
        port = get_redis_port()
        logging.info(f"Connexion à Redis sur le port {port}")
        return redis.Redis(host='localhost', port=port, db=0)
    except Exception as e:
        logging.error(f"Erreur de connexion Redis: {e}")
        sys.exit(1)

def get_documents_from_redis(r, crawl_id):
    """Récupère les documents depuis Redis."""
    documents = []
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        try:
            doc_data = r.hgetall(key)
            documents.append({
                "doc_id": key.decode('utf-8'),
                "url": doc_data[b'url'].decode('utf-8'),
                "content": doc_data[b'content'].decode('utf-8'),
                "internal_links_out": json.loads(doc_data.get(b'internal_links_out', b'[]').decode('utf-8'))
            })
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du document {key}: {e}")
    
    return documents

    
def save_analysis_results(r, crawl_id, documents, labels, cluster_infos):
    """Sauvegarde les résultats de l'analyse dans Redis."""
    base_crawl_id = crawl_id.split(':')[0]
    
    # Création du graphe simple (juste les liens)
    simple_graph = {
        "nodes": [],
        "links": []
    }
    
    # Création du graphe avec clusters
    clustered_graph = {
        "nodes": [],
        "links": []
    }
    
    # Construction des graphes
    for idx, doc in enumerate(documents):
        url = doc['url']
        cluster = labels[idx]
        
        # Gérer les liens internes qui peuvent être une liste ou une chaîne JSON
        internal_links = doc.get('internal_links_out', [])
        if isinstance(internal_links, str):
            internal_links = json.loads(internal_links)
        
        # Ajout des noeuds
        node = {
            "id": url,
            "label": url.split('/')[-1] or url.split('/')[-2],
            "internal_links_count": len(internal_links),
            "group": cluster
        }
        
        simple_graph["nodes"].append(node)
        clustered_graph["nodes"].append({**node, "title": cluster_infos.get(str(cluster), {}).get("description", "")})
        
        # Ajout des liens
        for target in internal_links:
            link = {
                "source": url,
                "target": target,
                "value": 1
            }
            simple_graph["links"].append(link)
            clustered_graph["links"].append(link)
    
    # Sauvegarde des graphes dans Redis
    r.set(f"{base_crawl_id}_simple_graph", json.dumps(simple_graph))
    r.set(f"{base_crawl_id}_clustered_graph", json.dumps(clustered_graph))
    
    # Sauvegarde des infos de cluster
    r.set(f"{base_crawl_id}_cluster_info", json.dumps(cluster_infos))
    
    logging.info(f"""
    💾 Données sauvegardées :
    • Nœuds : {len(simple_graph['nodes'])}
    • Liens : {len(simple_graph['links'])}
    • Clusters : {len(cluster_infos)}
    """)

def convert_numpy_types(obj):
    """Convertit les types numpy en types Python standards."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def generate_and_save_graphs(redis_client, crawl_id, documents_list, labels):
    """Génère et sauvegarde les graphes simple et clustered dans Redis."""
    try:
        # Structure commune pour les deux types de graphes
        base_nodes = []
        base_links = []
        
        # Créer un mapping des URLs vers les IDs de nœuds
        url_to_id = {doc['url']: f"node_{i}" for i, doc in enumerate(documents_list)}
        
        # Générer les nœuds de base
        for i, doc in enumerate(documents_list):
            node = {
                'id': url_to_id[doc['url']],
                'label': doc['url'],
                'internal_links_count': len(doc.get('internal_links_out', [])),
                'group': convert_numpy_types(labels[i]),  # Conversion du type numpy
                'title': f"Cluster {convert_numpy_types(labels[i])}"
            }
            base_nodes.append(node)
            
            # Générer les liens
            if 'internal_links_out' in doc:
                for target_url in doc['internal_links_out']:
                    if target_url in url_to_id:
                        link = {
                            'source': url_to_id[doc['url']],
                            'target': url_to_id[target_url],
                            'value': 1
                        }
                        base_links.append(link)
        
        # Graphe simple (sans clustering)
        simple_graph = {
            'nodes': base_nodes,
            'links': base_links
        }
        
        # Graphe clustered (avec informations de cluster)
        clustered_nodes = base_nodes.copy()
        for node in clustered_nodes:
            # Récupérer les informations de cluster
            cluster_info = redis_client.get(f"{crawl_id}_cluster_info")
            if cluster_info:
                try:
                    cluster_data = json.loads(cluster_info)
                    cluster_desc = cluster_data.get(str(node['group']), {}).get('description', f"Cluster {node['group']}")
                    node['title'] = cluster_desc
                except json.JSONDecodeError:
                    node['title'] = f"Cluster {node['group']}"
        
        clustered_graph = {
            'nodes': clustered_nodes,
            'links': base_links
        }
        
        # Sauvegarder les graphes dans Redis avec la conversion des types numpy
        simple_json = json.dumps(simple_graph, default=convert_numpy_types)
        clustered_json = json.dumps(clustered_graph, default=convert_numpy_types)
        
        redis_client.set(f"{crawl_id}_simple_graph", simple_json)
        redis_client.set(f"{crawl_id}_clustered_graph", clustered_json)
        
        logging.info(f"""
        ✅ Graphes générés et sauvegardés avec succès:
        - Nœuds: {len(base_nodes)}
        - Liens: {len(base_links)}
        - Clusters: {len(set(convert_numpy_types(label) for label in labels))}
        """)
        
    except Exception as e:
        logging.error(f"Erreur lors de la génération des graphes: {str(e)}")
        raise

async def main():
    if len(sys.argv) < 2:
        logging.error("Usage: python3 02_analyse.py <crawl_id> [--no-cluster]")
        return
    
    crawl_id = sys.argv[1]
    disable_clustering = "--no-cluster" in sys.argv
    
    r = get_redis_connection()
    documents = get_documents_from_redis(r, crawl_id)
    
    if not documents:
        logging.error("Aucun document trouvé")
        return
        
    if disable_clustering:
        logging.info("🔄 Clusterisation désactivée - Attribution d'un groupe unique")
        # Pas besoin de calculer les embeddings ni d'initialiser l'analyzer
        labels = [0] * len(documents)
        cluster_infos = {
            "0": {
                "size": len(documents),
                "description": "Groupe unique (clusterisation désactivée)",
                "key_terms": []
            }
        }
    else:
        # Code pour la clusterisation active
        analyzer = SemanticAnalyzer()
        contents = [doc["content"] for doc in documents]
        
        logging.info("Génération des embeddings...")
        embeddings = analyzer.compute_embeddings(contents)
        
        logging.info("Clustering des documents...")
        labels, params = analyzer.cluster_documents(embeddings)
        
        logging.info("Enrichissement des clusters...")
        cluster_infos = {}
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            
            summary = analyzer.prepare_cluster_summary(contents, labels, cluster_id)
            enriched = await analyzer.enrich_cluster_with_llm(summary)
            cluster_infos[str(cluster_id)] = {
                **summary,
                "description": enriched["description"]
            }
    
    logging.info("Sauvegarde des résultats...")
    save_analysis_results(r, crawl_id, documents, labels, cluster_infos)
    
    logging.info(f"""
    ✅ Analyse terminée avec succès:
    - Documents analysés: {len(documents)}
    - Mode clusterisation: {'Activé' if not disable_clustering else 'Désactivé'}
    - {'Clusters trouvés: ' + str(len(cluster_infos)) if not disable_clustering else 'Groupe unique'}
    - Outliers: {sum(1 for l in labels if l == -1) if not disable_clustering else 0}
    """)
        
if __name__ == "__main__":
    asyncio.run(main())