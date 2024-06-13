import redis
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import webbrowser
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Utiliser un modèle de transformation en français
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

french_stop_words = [
    'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 
    'eux', 'il', 'je', 'la', 'le', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 
    'moi', 'mon', 'ne', 'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pas', 'pour', 
    'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son', 'sur', 'ta', 'te', 'tes', 'toi', 
    'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd', 'j', 'l', 'à', 'm', 
    'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant', 'suis', 'es', 'est', 
    'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 
    'serais', 'serait', 'serions', 'seriez', 'seraient', 'étais', 'était', 'étions', 
    'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 
    'soyons', 'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 
    'fussent', 'ayant', 'ayante', 'ayantes', 'ayants', 'eu', 'eue', 'eues', 'eus', 
    'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 
    'auront', 'aurais', 'aurait', 'aurions', 'auriez', 'auraient', 'avais', 'avait', 
    'avions', 'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 
    'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 
    'eussent', 'ceci', 'cela', 'celà', 'cet', 'cette', 'ici', 'ils', 'les', 'leurs', 
    'quel', 'quels', 'quelle', 'quelles', 'sans', 'soi'
]

def get_documents_from_redis(crawl_id):
    urls = []
    contents = []
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        url = doc_data[b'url'].decode('utf-8')
        content = doc_data[b'content'].decode('utf-8')
        urls.append(url)
        contents.append(content)
    return urls, contents

def compute_embeddings(contents):
    return model.encode(contents)

def cluster_embeddings(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, kmeans

def reduce_dimensions(embeddings):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def determine_cluster_labels(contents, clusters, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words=french_stop_words)
    X = vectorizer.fit_transform(contents)
    terms = vectorizer.get_feature_names_out()
    
    labels = []
    for i in range(n_clusters):
        cluster_center = X[clusters == i].mean(axis=0).A.flatten()
        sorted_indices = cluster_center.argsort()[::-1]
        top_terms = [terms[idx] for idx in sorted_indices[:5]]
        labels.append(' '.join(top_terms))
    
    return labels

def save_results_to_redis(crawl_id, urls, clusters, labels):
    for i, url in enumerate(urls):
        doc_id = f"{crawl_id}:doc:{i+1}"
        r.hset(doc_id, "cluster", int(clusters[i]))  # Conversion en int
        r.hset(doc_id, "label", labels[clusters[i]])

def visualize_clusters(urls, reduced_embeddings, clusters, labels):
    print(f"Number of URLs: {len(urls)}")
    print(f"Number of Reduced Embeddings: {reduced_embeddings.shape}")
    print(f"Number of Clusters: {len(clusters)}")
    print(f"Labels: {labels}")
    
    if len(urls) != reduced_embeddings.shape[0] or len(clusters) != reduced_embeddings.shape[0]:
        print("Inconsistent lengths found:")
        print(f"URLs: {len(urls)}")
        print(f"Reduced Embeddings: {reduced_embeddings.shape[0]}")
        print(f"Clusters: {len(clusters)}")
        return

    df = pd.DataFrame({
        'PCA1': reduced_embeddings[:, 0],
        'PCA2': reduced_embeddings[:, 1],
        'Cluster': clusters,
        'URL': urls,
        'Label': [labels[cluster] for cluster in clusters]
    })

    fig, ax = plt.subplots(figsize=(15, 8))
    scatter = ax.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis', picker=True)
    
    cluster_label_map = {cluster: label for cluster, label in enumerate(labels)}
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {k} - {v}', 
                                  markerfacecolor=plt.cm.viridis(i / len(cluster_label_map)), markersize=10) 
                       for i, (k, v) in enumerate(cluster_label_map.items())]
    
    ax.legend(handles=legend_elements, title="Clusters")
    
    for i, txt in enumerate(df['URL']):
        short_url = txt.split('/')[-1]
        ax.annotate(f"{short_url} ({df['Label'][i]})", (df['PCA1'][i], df['PCA2'][i]), fontsize=8, alpha=0.7)

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

def main():
    crawl_id = input("Entrez l'ID du crawl à analyser: ")
    urls, contents = get_documents_from_redis(crawl_id)
    
    if not contents:
        print("No content found for the given crawl ID.")
        return
    
    print("Computing embeddings...")
    embeddings = compute_embeddings(contents)
    
    print("Clustering embeddings...")
    clusters, kmeans = cluster_embeddings(embeddings, n_clusters=5)
    
    print("Reducing dimensions for visualization...")
    reduced_embeddings = reduce_dimensions(embeddings)
    
    print("Determining cluster labels...")
    labels = determine_cluster_labels(contents, clusters, n_clusters=5)
    
    print("Saving results to Redis...")
    save_results_to_redis(crawl_id, urls, clusters, labels)
    
    print("Visualizing clusters...")
    visualize_clusters(urls, reduced_embeddings, clusters, labels)

if __name__ == "__main__":
    main()
