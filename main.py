import subprocess
import sys
import os
import json
import logging
from urllib.parse import urlparse

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    config_path = 'config.json'
    if not os.path.exists(config_path):
        logging.error(f"Error: {config_path} not found.")
        logging.info("Please copy 'config.json.example' to 'config.json' and fill in your API token.")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)
    return config

def validate_url(url):
    result = urlparse(url)
    return all([result.scheme, result.netloc])

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    output = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')  # Affiche la ligne en temps réel
        output.append(line)  # Stocke la ligne pour un traitement ultérieur si nécessaire
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return ''.join(output)

def run_crawl(url):
    try:
        output = run_command(['python3', '01_crawl.py', url])
        crawl_id_line = [line for line in output.split('\n') if line.startswith('Crawl terminé. ID du crawl:')]
        if crawl_id_line:
            return crawl_id_line[0].split(': ')[1]
    except subprocess.CalledProcessError as e:
        logging.error(f"Crawl failed with error code {e.returncode}")
    return None

def run_internal_links_crawl(crawl_id, selector):
    try:
        run_command(['python3', '03_crawl_internal_links.py', crawl_id, selector])
    except subprocess.CalledProcessError as e:
        logging.error(f"Internal links crawl failed with error code {e.returncode}")

def run_analysis(crawl_id):
    try:
        run_command(['python3', '02_analyse.py', crawl_id])
    except subprocess.CalledProcessError as e:
        logging.error(f"Analysis failed with error code {e.returncode}")

def main(url, selector):
    if not validate_url(url):
        logging.error(f"Invalid URL: {url}")
        logging.info("Please provide a valid URL (e.g., https://www.example.com)")
        return

    config = load_config()

    if "HUGGINGFACEHUB_API_TOKEN" not in config:
        logging.error("Error: HUGGINGFACEHUB_API_TOKEN not found in config.json")
        return

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]

    logging.info("Starting crawl process...")
    crawl_id = run_crawl(url)
    if not crawl_id:
        logging.error("Crawl failed.")
        return

    logging.info(f"Crawl completed with ID: {crawl_id}")
    logging.info("Starting internal links crawl process...")
    run_internal_links_crawl(crawl_id, selector)

    logging.info("Internal links crawl completed.")
    logging.info("Starting analysis process...")
    run_analysis(crawl_id)

    logging.info("Analysis completed.")

def cleanup():
    # Ajoutez ici le code pour nettoyer les fichiers temporaires ou libérer les ressources
    logging.info("Cleaning up resources...")

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
    
    try:
        main(url, selector)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        cleanup()