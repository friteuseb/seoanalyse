import redis
import uuid
from trafilatura.spider import focused_crawler
from trafilatura import fetch_url, extract
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords français
nltk.download('stopwords')
nltk.download('punkt')

# Configuration du modèle d'embedding
model_name = "sentence-transformers/LaBSE"  # Utilisation de LaBSE pour le multilingue, dont le français
model = SentenceTransformer(model_name)

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Utiliser les stopwords en français de NLTK
french_stop_words = set(stopwords.words('french')).union(["et", "à", "de", "pour", "dans", "sur", "avec", "le", "la", "les"])

def save_embedding(crawl_id, id, url, content, embedding, internal_links, cluster=None, label=None):
    data = {
        'url': url,
        'content': content,
        'embedding': str(embedding),
        'internal_links': str(internal_links)
    }
    if cluster is not None:
        data['cluster'] = str(cluster)
    if label is not None:
        data['label'] = label
    r.hset(f'{crawl_id}:doc:{id}', mapping=data)

def extract_internal_links(html_content, base_url):
    soup = BeautifulSoup(html_content, 'html.parser')
    internal_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(base_url, href)
        if base_url in full_url:
            internal_links.append(full_url)
    return internal_links

def assign_cluster_labels(contents, clusters):
    vectorizer = TfidfVectorizer(stop_words=french_stop_words)
    X = vectorizer.fit_transform(contents)
    terms = vectorizer.get_feature_names_out()
    
    labels = []
    for i in range(5):  # Assuming 5 clusters
        cluster_indices = [j for j, cluster in enumerate(clusters) if cluster == i]
        cluster_contents = X[cluster_indices]
        summed_tfidf = np.asarray(cluster_contents.sum(axis=0)).flatten()
        top_term_index = summed_tfidf.argsort()[-1]
        labels.append(terms[top_term_index])
    
    cluster_labels = [labels[cluster] for cluster in clusters]
    return cluster_labels

def crawl_site(url):
    crawl_id = str(uuid.uuid4())
    to_visit, known_links = focused_crawler(url, max_seen_urls=100, max_known_urls=1000)
    
    if not known_links:
        print("Erreur : Impossible d'extraire le contenu du site.")
        return

    embeddings = []
    contents = []
    urls = []
    for idx, link in enumerate(known_links):
        print(f"Crawling: {link}")
        html_content = fetch_url(link)
        content = extract(html_content)
        if content:
            internal_links = extract_internal_links(html_content, url)
            embedding = model.encode(content)
            save_embedding(crawl_id, idx, link, content, embedding, internal_links)
            embeddings.append(embedding)
            contents.append(content)
            urls.append(link)

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(embeddings)
    clusters = kmeans.labels_
    labels = assign_cluster_labels(contents, clusters)

    for idx in range(len(urls)):
        save_embedding(crawl_id, idx, urls[idx], contents[idx], embeddings[idx], extract_internal_links(fetch_url(urls[idx]), url), cluster=clusters[idx], label=labels[idx])

    print(f"Crawl terminé avec l'ID {crawl_id}")
    return crawl_id

if __name__ == "__main__":
    url = input("Entrez l'URL du site à crawler: ")
    crawl_id = crawl_site(url)
    print(f"Crawl terminé. ID du Crawl: {crawl_id}")
