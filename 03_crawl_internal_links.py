import redis
import trafilatura
from bs4 import BeautifulSoup
import sys

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def list_crawls():
    crawls = {}
    for key in r.scan_iter("*:doc_count"):
        crawl_id = key.decode('utf-8').split(':')[0]
        crawls[crawl_id] = r.get(key).decode('utf-8')
    return crawls

def select_crawl(crawl_id):
    crawls = list_crawls()
    if crawl_id in crawls:
        return crawl_id
    print("Invalid crawl ID.")
    sys.exit(1)

def get_documents_from_redis(crawl_id):
    documents = {}
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        doc_id = key.decode('utf-8')
        url = doc_data[b'url'].decode('utf-8')
        documents[doc_id] = url
    return documents

def extract_internal_links(url, selector):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print(f"Failed to download {url}")
        return []
    soup = BeautifulSoup(downloaded, 'html.parser')

    links = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/'):
            href = url.rstrip('/') + href
        if url in href:
            links.add(href)

    if selector.startswith('.'):
        content = soup.find_all(class_=selector[1:])
    elif selector.startswith('#'):
        content = soup.find_all(id=selector[1:])
    else:
        print("Invalid selector")
        return []

    filtered_links = set()
    for section in content:
        for link in section.find_all('a', href=True):
            href = link['href']
            if href.startswith('/'):
                href = url.rstrip('/') + href
            if href in links:
                filtered_links.add(href)
    
    return list(filtered_links)

def save_internal_links_to_redis(doc_id, internal_links):
    r.hset(doc_id, "internal_links", ','.join(internal_links))
    print(f"Internal links saved in {doc_id}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 03_crawl_internal_links.py <Crawl_ID> <Selector>")
        sys.exit(1)

    crawl_id = sys.argv[1]
    selector = sys.argv[2]

    crawl_id = select_crawl(crawl_id)
    documents = get_documents_from_redis(crawl_id)

    for doc_id, url in documents.items():
        print(f"Processing {url} for internal links...")
        internal_links = extract_internal_links(url, selector)
        if internal_links:
            print(f"Internal links found for {url}: {internal_links}")
            save_internal_links_to_redis(doc_id, internal_links)
        else:
            print(f"No internal links found for {url}")

    print("Internal links crawling complete.")

if __name__ == "__main__":
    main()