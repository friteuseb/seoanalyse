import redis
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests

# Connexion à Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def fetch_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch {url}")
        return None

def extract_internal_links(html_content, base_url, selector, selector_type):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Debug: Afficher le sélecteur et le type de sélecteur
    print(f"Using selector: {selector} with type: {selector_type}")

    # Sélection de la zone de contenu en fonction du type de sélecteur
    if selector_type == 'class':
        content_area = soup.find_all(class_=selector)
    elif selector_type == 'id':
        content_area = soup.find_all(id=selector)
    elif selector_type == 'xpath':
        # Note: BeautifulSoup ne supporte pas XPath directement, l'utilisation de lxml ou similaire serait nécessaire
        print("XPath not supported directly by BeautifulSoup")
        return []

    # Debug: Afficher le nombre de zones de contenu trouvées
    print(f"Found {len(content_area)} content areas")

    if content_area:
        # Debug: Afficher le contenu de la première zone trouvée pour vérifier
        print(f"Content area example: {content_area[0].prettify()}")

    links = []
    # Extraction des liens internes de la zone de contenu
    for area in content_area:
        for link in area.find_all('a', href=True):
            href = link['href']
            # Conversion des liens relatifs en liens absolus
            if href.startswith('/'):
                href = urljoin(base_url, href)
            # Vérification que le lien est interne
            if href.startswith(base_url):
                links.append(href)
    
    # Debug: Afficher les liens internes trouvés
    print(f"Extracted internal links: {links}")
    return links

def save_internal_links(doc_id, internal_links):
    if internal_links:
        internal_links_str = ','.join(internal_links)
        r.hset(doc_id, "internal_links", internal_links_str)
        print(f"Internal links saved for {doc_id}: {internal_links_str}")
    else:
        print(f"No internal links found for {doc_id}")

def process_documents(crawl_id, selector, selector_type):
    for key in r.scan_iter(f"{crawl_id}:doc:*"):
        doc_data = r.hgetall(key)
        url = doc_data[b'url'].decode('utf-8')
        print(f"Processing {url} for internal links...")

        html_content = fetch_html(url)
        if html_content:
            internal_links = extract_internal_links(html_content, url, selector, selector_type)
            print(f"Extracted {len(internal_links)} internal links from {url}: {internal_links}")
            save_internal_links(key, internal_links)
        else:
            print(f"Failed to extract content for {url}")

if __name__ == "__main__":
    crawl_id = input("Entrez l'ID du crawl: ")
    selector = input("Entrez la classe, l'ID ou le XPath de la zone de contenu (sans le point pour les classes): ")
    selector_type = input("Entrez le type de sélecteur ('class', 'id', 'xpath'): ")
    
    process_documents(crawl_id, selector, selector_type)
    print("Crawl des liens internes terminé.")
