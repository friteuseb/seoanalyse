import subprocess
import sys
import os
import json
import logging
import argparse
from urllib.parse import urlparse

def print_banner():
    print("""
    🌐 Visualisation Sémantique Web
    ==============================

    Cette application permet d'analyser la structure sémantique d'un site web :
    1. Crawl du contenu du site
    2. Analyse des liens internes
    3. Analyse sémantique des thématiques
    4. Visualisation interactive des résultats

    """)

def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description="Outil d'analyse sémantique et de visualisation de sites web",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
----------------------
1. Analyser avec un sélecteur de classe:
   python3 main.py https://example.com/ ".content"

2. Analyser avec un ID:
   python3 main.py https://example.com/ "#main-content"

3. Analyser plusieurs zones:
   python3 main.py https://example.com/ "#main-content, .article-content"

Note: Les sélecteurs CSS doivent être entre guillemets pour éviter les problèmes d'interprétation.
      """
    )
    
    parser.add_argument("url", 
                       help="URL du site à analyser (ex: https://example.com/)")
    parser.add_argument("selector", 
                       help="Sélecteur CSS pour cibler les zones à analyser (ex: '#content' ou '.main-content')")
    return parser

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
  
def main(url=None, selector=None):
    parser = setup_argument_parser()
    
    # Si les arguments ne sont pas fournis directement, les prendre des args du parser
    if url is None or selector is None:
        args = parser.parse_args()
        url = args.url
        selector = args.selector
    
    if not validate_url(url):
        logging.error(f"URL invalide: {url}")
        logging.info("L'URL doit commencer par http:// ou https://")
        return

    logging.info(f"🎯 Analyse de {url} avec le sélecteur: {selector}")

    config = load_config()
    if "HUGGINGFACEHUB_API_TOKEN" not in config:
        logging.error("❌ Erreur: HUGGINGFACEHUB_API_TOKEN non trouvé dans config.json")
        return

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]

    logging.info("🔄 Démarrage du crawl...")
    crawl_id = run_crawl(url)
    if not crawl_id:
        logging.error("❌ Échec du crawl")
        return

    logging.info(f"✅ Crawl terminé avec l'ID: {crawl_id}")
    logging.info("🔍 Analyse des liens internes...")
    run_internal_links_crawl(crawl_id, selector)

    logging.info("✅ Analyse des liens terminée")
    logging.info("🧠 Démarrage de l'analyse sémantique...")
    run_analysis(crawl_id)

    logging.info(f"""
    ✨ Analyse terminée avec succès !
    
    Pour visualiser les résultats :
    1. Ouvrez votre navigateur
    2. Accédez au tableau de bord de visualisation
    3. Sélectionnez le crawl avec l'ID: {crawl_id}
    """)

def cleanup():
    logging.info("🧹 Nettoyage des ressources...")

if __name__ == "__main__":
    # Afficher la bannière une seule fois au début
    print_banner()
    
    # Vérifier les arguments
    if len(sys.argv) < 3:
        parser = setup_argument_parser()
        parser.print_help()
        sys.exit(1)
    
    url = sys.argv[1]
    selector = sys.argv[2]
    
    try:
        main(url, selector)
    except KeyboardInterrupt:
        logging.info("\n⛔ Analyse interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Une erreur inattendue est survenue: {e}")
    finally:
        cleanup()