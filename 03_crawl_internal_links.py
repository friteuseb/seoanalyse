import redis
import trafilatura
from bs4 import BeautifulSoup
import sys
from urllib.parse import urljoin, urlparse
import json

# Connexion à Redis
r = redis.Redis(host='localhost', port=32768, db=0)

def list_crawls():
    crawls = {}
    for key in r.scan_iter("*:doc_count"):
        crawl_id = key.decode('utf-8').split(':')[0]
        crawls[crawl_id] = r.get(key).decode('utf-8')
    return crawls

def select_crawl():
    crawls = list_crawls()
    print("Available crawls:")
    for i, (crawl_id, count) in enumerate(crawls.items(), 1):
        print(f"{i}. {crawl_id} (Documents: {count})")
    selected = int(input("Select the crawl number to analyze: ")) - 1
    return list(crawls.keys())[selected]

def get_documents_from_redis(crawl_id):
    documents = {}
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        doc_id = key.decode('utf-8')
        url = doc_data[b'url'].decode('utf-8')
        label = doc_data.get(b'label', b'').decode('utf-8')
        cluster = doc_data.get(b'cluster', b'0').decode('utf-8')
        documents[doc_id] = {'url': url, 'label': label, 'cluster': int(cluster)}
    return documents

def extract_internal_links(base_url, content, selector):
    soup = BeautifulSoup(content, 'html.parser')

    # Utilisation de soup.select pour trouver les éléments correspondants au sélecteur CSS
    content_area = soup.select(selector)
    if not content_area:
        print(f"No content found for selector {selector}")
        return []

    links = set()
    for section in content_area:
        for link in section.find_all('a', href=True):
            href = link['href']
            if urlparse(href).netloc == '':
                full_url = urljoin(base_url, href)
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    links.add(full_url)
    
    return list(links)

def save_internal_links_to_redis(crawl_id, documents, selector):
    for doc_id, doc_info in documents.items():
        url = doc_info['url']
        print(f"Processing {url} for internal links...")
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            print(f"Failed to download {url}")
            continue
        internal_links_out = extract_internal_links(url, downloaded, selector)
        if internal_links_out:
            print(f"Internal links found for {url}: {internal_links_out}")
            r.hset(doc_id, "internal_links_out", json.dumps(internal_links_out))
        else:
            print(f"No internal links found for {url}")

def assign_cluster_colors(documents):
    cluster_colors = {}
    color_list = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    for doc_id, doc_info in documents.items():
        cluster = doc_info['cluster']
        if cluster not in cluster_colors:
            cluster_colors[cluster] = color_list[len(cluster_colors) % len(color_list)]
        doc_info['color'] = cluster_colors[cluster]
    return documents

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 03_crawl_internal_links.py <crawl_id> <CSS Selector>")
        sys.exit(1)
    
    crawl_id = sys.argv[1]
    selector = sys.argv[2]
    
    documents = get_documents_from_redis(crawl_id)
    if not documents:
        print("No documents found for the given crawl ID.")
        return

    documents = assign_cluster_colors(documents)
    save_internal_links_to_redis(crawl_id, documents, selector)
    print("Internal links crawling complete.")

if __name__ == "__main__":
    main()