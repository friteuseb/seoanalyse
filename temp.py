import sys
import subprocess
import redis
from bs4 import BeautifulSoup
import logging
import uuid
import time
from datetime import datetime
import trafilatura
import json

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_redis_port():
    try:
        port = subprocess.check_output(
            "ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", 
            shell=True
        )
        return int(port.strip())
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du port Redis : {e}")
        sys.exit(1)

def extract_urls_from_list(html_content):
    """Extrait les URLs de la liste fournie"""
    soup = BeautifulSoup(html_content, 'html.parser')
    urls = []
    for item in soup.select('li.item a'):
        if item and item.get('href'):
            urls.append('https://www.gold.fr' + item['href'])
    return urls

def save_page_to_redis(redis_client, url, content, crawl_id):
    """Sauvegarde une page dans Redis"""
    try:
        doc_id = f"{crawl_id}:doc:{redis_client.incr(f'{crawl_id}:doc_count')}"
        redis_client.hset(doc_id, mapping={
            "url": url,
            "content": content,
            "crawl_date": datetime.now().isoformat(),
            "links_count": 0,
            "internal_links_out": "[]"
        })
        return True
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde dans Redis: {e}")
        return False

def crawl_url(url):
    """Crawl une URL avec trafilatura"""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        content = trafilatura.extract(downloaded, include_links=True)
        if content:
            # Extraire spécifiquement le contenu de .pageItem si présent
            soup = BeautifulSoup(downloaded, 'html.parser')
            page_item = soup.select_one('.pageItem')
            if page_item:
                return page_item.get_text(strip=True)
            return content
    except Exception as e:
        logging.error(f"Erreur lors du crawl de {url}: {e}")
    return None

def main():
    # Connexion à Redis
    redis_client = redis.Redis(host='localhost', port=get_redis_port(), db=0)

    # 1. Lire le fichier paste.txt
    try:
        with open('paste.txt', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logging.error("❌ Le fichier paste.txt n'a pas été trouvé.")
        return
    except Exception as e:
        logging.error(f"❌ Erreur lors de la lecture du fichier: {e}")
        return

    # 2. Extraire les URLs
    urls = extract_urls_from_list(content)
    if not urls:
        logging.error("❌ Aucune URL n'a été extraite du fichier.")
        return

    # 3. Créer un seul ID de crawl pour tout le processus
    crawl_id = f"gold_fr_guide__{str(uuid.uuid4())}"
    logging.info(f"🔄 Démarrage du crawl de {len(urls)} URLs avec ID: {crawl_id}")
    
    successful = 0
    failed = 0

    # 4. Crawler et sauvegarder chaque URL
    for i, url in enumerate(urls, 1):
        try:
            content = crawl_url(url)
            if content and save_page_to_redis(redis_client, url, content, crawl_id):
                successful += 1
                logging.info(f"✅ Page sauvegardée: {url}")
            else:
                failed += 1
                logging.error(f"❌ Échec pour {url}")
        except Exception as e:
            failed += 1
            logging.error(f"❌ Erreur pour {url}: {e}")

        # Log de progression
        if i % 5 == 0 or i == len(urls):
            logging.info(f"""
            📊 Progression: {i}/{len(urls)} ({(i/len(urls))*100:.1f}%)
            ✅ Succès: {successful}
            ❌ Échecs: {failed}
            """)

        # Petit délai entre les requêtes
        time.sleep(2)

    # 5. Lancer l'analyse des liens internes
    if successful > 0:
        logging.info("🔄 Analyse des liens internes...")
        process = subprocess.run(
            ["python3", "03_crawl_internal_links.py", crawl_id, ".pageItem"],
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            logging.error(f"❌ Erreur lors de l'analyse des liens: {process.stderr}")
            return

        # 6. Lancer l'analyse finale
        logging.info("🔄 Génération du graphe...")
        process = subprocess.run(
            ["python3", "02_analyse.py", crawl_id, "--no-cluster"],
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            logging.error(f"❌ Erreur lors de l'analyse: {process.stderr}")
            return

        logging.info(f"""
        ✨ Analyse terminée avec succès !
        
        ID du crawl: {crawl_id}
        URLs analysées: {len(urls)}
        Succès: {successful}
        Échecs: {failed}
        
        Vous pouvez maintenant visualiser le graphe avec cet ID.
        """)
    else:
        logging.error("❌ Aucune page n'a pu être traitée avec succès.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n⛔ Processus interrompu par l'utilisateur")
    except Exception as e:
        logging.error(f"❌ Une erreur inattendue est survenue: {e}")
