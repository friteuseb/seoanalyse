import redis
import trafilatura
from bs4 import BeautifulSoup
import sys
from urllib.parse import urljoin, urlparse
import json
import os

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

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

def get_internal_links(crawl_id):
    internal_links = {}
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        if b'internal_links_out' in doc_data:
            internal_links[key.decode('utf-8')] = json.loads(doc_data[b'internal_links_out'].decode('utf-8'))
    return internal_links

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

def is_valid_url(url):
    return url.startswith('http://') or url.startswith('https://')

def update_json_with_links(json_file, internal_links, documents):
    if not os.path.exists(json_file):
        print(f"Error: {json_file} does not exist.")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    for node in nodes:
        url = node['id']
        if url in internal_links:
            node['internal_links'] = [link for link in internal_links[url] if is_valid_url(link)]
        if url in documents:
            node['label'] = documents[url]['label']
            node['group'] = documents[url]['cluster']
            node['color'] = documents[url]['color']

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

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
    
    internal_links = get_internal_links(crawl_id)
    update_json_with_links(f'{crawl_id}_simple_graph.json', internal_links, documents)
    update_json_with_links(f'{crawl_id}_clustered_graph.json', internal_links, documents)
    print("JSON files updated with internal links and labels.")

if __name__ == "__main__":
    main()