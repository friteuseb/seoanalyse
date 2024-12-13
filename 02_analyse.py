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
from dotenv import load_dotenv
from transformers import CamembertModel, CamembertTokenizer
from sklearn.cluster import DBSCAN, KMeans
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
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        best_score = -1
        best_labels = None
        best_params = None
        
        # Calcul de la matrice des distances
        distances = np.linalg.norm(embeddings_scaled[:, None] - embeddings_scaled, axis=2)
        
        # Assurer des valeurs minimales sûres pour eps
        min_eps = max(0.1, np.min(distances[distances > 0]))
        eps_range = np.percentile(distances[distances > 0], [5, 10, 15, 20, 25, 30])
        eps_range = np.append(eps_range, min_eps)  # Ajouter la valeur minimale
        eps_range = np.unique(eps_range)  # Supprimer les doublons
        eps_range = eps_range[eps_range > 0]  # Supprimer les valeurs nulles
        
        min_samples_range = [2, 3, 4]
        
        logging.info("Recherche des meilleurs paramètres de clustering...")
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddings_scaled)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    outliers_ratio = np.sum(labels == -1) / len(labels)
                    
                    if n_clusters >= 2 and outliers_ratio < 0.3:
                        mask = labels != -1
                        if np.sum(mask) > 1:
                            score = silhouette_score(embeddings_scaled[mask], labels[mask])
                            current_score = score * (1 - outliers_ratio) * (n_clusters / 5)
                            
                            if current_score > best_score:
                                best_score = current_score
                                best_labels = labels
                                best_params = {'eps': eps, 'min_samples': min_samples}
                                
                                logging.info(f"""
                                Nouvelle meilleure configuration:
                                - Epsilon: {eps:.3f}
                                - Min samples: {min_samples}
                                - Clusters: {n_clusters}
                                - Outliers: {outliers_ratio*100:.1f}%
                                - Score: {current_score:.3f}
                                """)
                except Exception as e:
                    logging.debug(f"Échec configuration eps={eps}, min_samples={min_samples}: {str(e)}")
                    continue
        
        # Si aucune bonne configuration n'est trouvée, utiliser KMeans comme fallback
        if best_labels is None:
            logging.warning("DBSCAN échoué, utilisation de KMeans comme fallback")
            n_clusters = min(5, len(embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            best_labels = kmeans.fit_predict(embeddings_scaled)
            best_params = {'method': 'kmeans', 'n_clusters': n_clusters}
        
        return best_labels, best_params

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
    
    # Sauvegarder les informations de clustering
    for i, doc in enumerate(documents):
        cluster_id = int(labels[i])  # Conversion explicite en int Python standard
        info = cluster_infos.get(str(cluster_id), {"description": "Contenu non classé"})
        
        r.hset(doc["doc_id"], mapping={
            "cluster": cluster_id,
            "cluster_description": info["description"]
        })
    
    # Convertir les clés numpy.int64 en str avant la sérialisation
    serializable_infos = {
        str(k): v for k, v in cluster_infos.items()
    }
    
    serialized_info = json.dumps(serializable_infos, default=convert_to_serializable)
    r.set(f"{base_crawl_id}_cluster_info", serialized_info)
    
    logging.info(f"Résultats sauvegardés pour {len(documents)} documents dans {len(cluster_infos)} clusters")



async def main():
    if len(sys.argv) < 2:
        logging.error("Usage: python3 02_analyse.py <crawl_id>")
        return
    
    crawl_id = sys.argv[1]
    r = get_redis_connection()
    documents = get_documents_from_redis(r, crawl_id)
    
    if not documents:
        logging.error("Aucun document trouvé")
        return
    
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
        cluster_infos[str(cluster_id)] = {  # Convertir en str ici aussi
            **summary,
            "description": enriched["description"]
        }
    
    logging.info("Sauvegarde des résultats...")
    save_analysis_results(r, crawl_id, documents, labels, cluster_infos)
    
    logging.info(f"""
    ✅ Analyse terminée avec succès:
    - Documents analysés: {len(documents)}
    - Clusters trouvés: {len(cluster_infos)}
    - Outliers: {sum(1 for l in labels if l == -1)}
    """)

if __name__ == "__main__":
    asyncio.run(main())