import redis
import trafilatura
from bs4 import BeautifulSoup

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
        documents[doc_id] = url
    return documents

def extract_internal_links(url, selector):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print(f"Failed to download {url}")
        return []
    soup = BeautifulSoup(downloaded, 'html.parser')

    if selector.startswith('.'):
        content = soup.find_all(class_=selector[1:])
    elif selector.startswith('#'):
        content = soup.find_all(id=selector[1:])
    else:
        print("Invalid selector")
        return []

    links = set()
    for section in content:
        for link in section.find_all('a', href=True):
            href = link['href']
            if href.startswith('/'):
                href = url + href
            if url in href:
                links.add(href)
    
    return list(links)

def save_internal_links_to_redis(doc_id, internal_links):
    r.hset(doc_id, "internal_links", ','.join(internal_links))
    print(f"Internal links saved in {doc_id}")

def main():
    crawl_id = select_crawl()
    documents = get_documents_from_redis(crawl_id)
    
    selector = input("Enter the CSS selector (class or ID) of the content area (e.g., .content or #main): ")

    for doc_id, url in documents.items():
        print(f"Processing {url} for internal links...")
        internal_links = extract_internal_links(url, selector)
        if internal_links:
            print(f"Extracted internal links: {internal_links}")
            save_internal_links_to_redis(doc_id, internal_links)
        else:
            print(f"No internal links found for {url}")

    print("Internal links crawling complete.")

if __name__ == "__main__":
    main()