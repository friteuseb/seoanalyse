import redis
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def embedding_from_redis(redis_embedding):
    return [float(x) for x in redis_embedding.decode('utf-8').strip('[]').split(', ')]

def retrieve_embeddings_from_redis():
    embeddings = []
    clusters = []
    urls = []

    for key in r.scan_iter("doc:*"):
        doc_data = r.hgetall(key)
        embedding = embedding_from_redis(doc_data[b'embedding'])
        cluster = doc_data.get(b'cluster')
        if cluster:
            cluster = int(cluster.decode('utf-8'))
        else:
            cluster = -1  # Utiliser -1 pour indiquer les documents non assignés
        url = doc_data[b'url'].decode('utf-8')
        
        embeddings.append(embedding)
        clusters.append(cluster)
        urls.append(url)

    return np.array(embeddings), clusters, urls

def visualize_clusters_with_plotly(embeddings, clusters, urls):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    df = pd.DataFrame({
        'PCA1': reduced_embeddings[:, 0],
        'PCA2': reduced_embeddings[:, 1],
        'Cluster': clusters,
        'URL': urls
    })

    fig = px.scatter(
        df, x='PCA1', y='PCA2', color='Cluster', hover_data=['URL'],
        title="Visualisation des Clusters"
    )
    fig.show()

embeddings, clusters, urls = retrieve_embeddings_from_redis()
visualize_clusters_with_plotly(embeddings, clusters, urls)

