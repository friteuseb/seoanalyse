import subprocess
import sys
import os
import json
from urllib.parse import urlparse

def load_config():
    config_path = 'config.json'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        print("Please copy 'config.json.example' to 'config.json' and fill in your API token.")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)
    return config

def validate_url(url):
    result = urlparse(url)
    return all([result.scheme, result.netloc])

def run_crawl(url):
    try:
        result = subprocess.run(['python3', '01_crawl.py', url], capture_output=True, text=True, check=True)
        print("Crawl output:", result.stdout)
        crawl_id_line = [line for line in result.stdout.split('\n') if line.startswith('Crawl terminé. ID du crawl:')]
        if crawl_id_line:
            return crawl_id_line[0].split(': ')[1]
    except subprocess.CalledProcessError as e:
        print("Crawl errors:", e.stderr)
    return None

def run_internal_links_crawl(crawl_id, selector):
    try:
        result = subprocess.run(['python3', '03_crawl_internal_links.py', crawl_id, selector], capture_output=True, text=True, check=True)
        print("Internal links crawl output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Internal links crawl errors:", e.stderr)

def run_analysis(crawl_id):
    try:
        result = subprocess.run(['python3', '02_analyse.py', crawl_id], capture_output=True, text=True, check=True)
        print("Analysis output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Analysis errors:", e.stderr)

def main(url, selector):
    if not validate_url(url):
        print(f"Invalid URL: {url}")
        print("Please provide a valid URL (e.g., https://www.example.com)")
        return

    config = load_config()

    if "HUGGINGFACEHUB_API_TOKEN" not in config:
        print("Error: HUGGINGFACEHUB_API_TOKEN not found in config.json")
        return

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]

    print("Starting crawl process...")
    crawl_id = run_crawl(url)
    if not crawl_id:
        print("Crawl failed.")
        return

    print(f"Crawl completed with ID: {crawl_id}")
    print("Starting internal links crawl process...")
    run_internal_links_crawl(crawl_id, selector)

    print("Internal links crawl completed.")
    print("Starting analysis process...")
    run_analysis(crawl_id)

    print("Analysis completed.")

if __name__ == "__main__":
    print("""
    Bienvenue sur l'application de Visualisation Sémantique !

    Cette application vous permet de :
    1. Lancer un crawl d'un site web pour en extraire le contenu.
    2. Analyser les liens internes du site.
    3. Réaliser une analyse sémantique pour déterminer les thématiques dominantes.
    4. Visualiser les résultats sous forme de graphes interactifs.
    

    Étapes pour utiliser l'application :
    1. Fournissez l'URL du site web à crawler.
    2. Fournissez le sélecteur CSS pour extraire les liens internes.
    3. Patientez pendant que l'application effectue les analyses.
    4. Visualisez les résultats via les serveurs web et Flask.
    """)
    
    if len(sys.argv) < 3:
        print("Usage: python3 main.py <URL> <CSS Selector>")
        print("\nExample: python3 main.py https://www.example-domain.com .content")
        sys.exit(1)
    
    url = sys.argv[1]
    selector = sys.argv[2]
    main(url, selector)