import redis
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import webbrowser

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def embedding_from_redis(redis_embedding):
    return [float(x) for x in redis_embedding.decode('utf-8').strip('[]').split(', ')]

def visualize_clusters(crawl_id):
    embeddings = []
    urls = []
    clusters = []
    labels = []

    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        embedding = embedding_from_redis(doc_data[b'embedding'])
        embeddings.append(embedding)
        urls.append(doc_data[b'url'].decode('utf-8'))
        clusters.append(int(doc_data[b'cluster'].decode('utf-8')))
        labels.append(doc_data[b'label'].decode('utf-8'))

    embeddings = np.array(embeddings)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    df = pd.DataFrame({
        'PCA1': reduced_embeddings[:, 0],
        'PCA2': reduced_embeddings[:, 1],
        'Cluster': clusters,
        'URL': urls,
        'Label': labels
    })

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis', picker=True)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    for i, txt in enumerate(df['URL']):
        short_url = txt.split('/')[-1]
        ax.annotate(short_url, (df['PCA1'][i], df['PCA2'][i]), fontsize=8, alpha=0.7)

    def onpick(event):
        ind = event.ind[0]
        url = df['URL'].iloc[ind]
        print('URL:', url)
        webbrowser.open(url)

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.title("Visualisation des clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

if __name__ == "__main__":
    crawl_id = input("Entrez l'ID du crawl à analyser: ")
    visualize_clusters(crawl_id)
