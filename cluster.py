import redis
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mpld3

# Connexion à Redis
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
except redis.ConnectionError as e:
    print(f"Erreur de connexion à Redis: {e}")
    exit(1)

def get_documents_from_redis(crawl_id):
    """Récupère les documents d'un crawl spécifique à partir de Redis."""
    try:
        documents = []
        for key in r.scan_iter(f"{crawl_id}:doc:*"):
            doc_data = r.hgetall(key)
            doc_id = key.decode('utf-8')
            url = doc_data[b'url'].decode('utf-8')
            content = doc_data[b'content'].decode('utf-8')
            labels = json.loads(doc_data[b'labels'].decode('utf-8'))
            cluster = int(doc_data[b'cluster'].decode('utf-8'))
            documents.append({
                "doc_id": doc_id,
                "url": url,
                "content": content,
                "labels": labels,
                "cluster": cluster
            })
        return documents
    except Exception as e:
        print(f"Erreur lors de la récupération des documents de Redis: {e}")
        return []

def visualize_clusters(documents):
    """Visualise les clusters des documents."""
    try:
        # Extraction des embeddings et clusters
        embeddings = [list(doc["labels"].values()) for doc in documents]
        clusters = [doc["cluster"] for doc in documents]
        urls = [doc["url"] for doc in documents]

        # Réduction des dimensions à 2D pour la visualisation
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Création du graphique
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
        plt.title('Clusterisation des URLs')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.colorbar(scatter, label='Cluster')

        # Ajouter des labels aux points
        for i, url in enumerate(urls):
            plt.annotate(url, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.7)

        # Sauvegarde du graphique en HTML
        html_filename = "visualizations/cluster_visualization.html"
        mpld3.save_html(plt.gcf(), html_filename)
        print(f"Graphique de clusterisation sauvegardé dans {html_filename}")
    except Exception as e:
        print(f"Erreur lors de la visualisation des clusters: {e}")

def main():
    crawl_id = "www_reseau-travaux_fr__6509c71d-fab6-489d-bfff-6c5c54280cd9"
    documents = get_documents_from_redis(crawl_id)
    
    if not documents:
        print("No documents found for the given crawl ID.")
        return
    
    visualize_clusters(documents)

if __name__ == "__main__":
    main()
