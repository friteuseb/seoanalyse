import subprocess
import sys
import os
import json
import logging
import argparse
from urllib.parse import urlparse
from termcolor import colored



def print_banner():
    print(colored("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            Visualisation Sémantique Web v1.0                 ║
    ║            Analyse du Maillage Interne                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """, "cyan", attrs=["bold"]))

    print(colored("📊 FONCTIONNALITÉS", "yellow", attrs=["bold"]))
    print(colored("""
    1. Crawl du contenu des zones spécifiées
    2. Analyse des liens internes éditoriaux
    3. Analyse sémantique des thématiques
    4. Visualisation interactive des résultats
    """, "white"))

    print(colored("🎯 ZONES D'ANALYSE", "yellow", attrs=["bold"]))
    print(colored("""
    Utilisez les sélecteurs CSS pour cibler précisément les zones :
    """, "white"))
    print(colored("    • ", "green") + "Classes : " + colored(".content, .article, .post", "cyan"))
    print(colored("    • ", "green") + "IDs : " + colored("#main-content, #article", "cyan"))
    print(colored("    • ", "green") + "Exclusions : " + colored("#content:not(.menu):not(.footer)", "cyan"))
    print()

def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description=colored("Outil d'analyse sémantique et visualisation du maillage interne éditorial", "cyan", attrs=["bold"]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=colored("""
╔══════════════════════════════════════════════════════════════╗
║                   EXEMPLES D'UTILISATION                     ║
╚══════════════════════════════════════════════════════════════╝
""", "yellow", attrs=["bold"]) + """
1. Analyser la page entière:
   """ + colored('python3 main.py https://example.com/', "cyan") + """

2. Analyser une zone par classe CSS:
   """ + colored('python3 main.py https://example.com/ ".content"', "cyan") + """

3. Analyser une zone par ID:
   """ + colored('python3 main.py https://example.com/ "#main-content"', "cyan") + """

4. Analyser plusieurs zones:
   """ + colored('python3 main.py https://example.com/ "#main-content, .article-content"', "cyan") + """

5. Analyser en excluant des zones:
   """ + colored('python3 main.py https://example.com/ "#main-content:not(.navigation)"', "cyan") + """
   """ + colored('python3 main.py https://example.com/ ".content:not(#menu):not(.sidebar)"', "cyan") + """

""" + colored("📝 NOTES:", "yellow", attrs=["bold"]) + """
""" + colored("•", "green") + """ Sans sélecteur CSS spécifié, l'analyse portera sur toute la page
""" + colored("•", "green") + """ Les sélecteurs CSS doivent être entre guillemets
""" + colored("•", "green") + """ Utilisez :not() pour exclure les zones non pertinentes
""" + colored("•", "green") + """ L'analyse exclut automatiquement les liens externes
""")
    
    parser.add_argument("url", 
                       help=colored("URL du site à analyser (ex: https://example.com/)", "cyan"))
    parser.add_argument("selector", 
                       nargs='?',  # Rend l'argument optionnel
                       default=None,  # Valeur par défaut si non spécifié
                       help=colored("Sélecteur CSS pour cibler les zones à analyser (optionnel)", "cyan"))
    parser.add_argument("--exclude-patterns", "-e",
                       nargs='+',
                       default=[],
                       help=colored("Patterns à exclure (ex: -e sku cart checkout)", "cyan"))
    parser.add_argument("--no-cluster",
                       action="store_true",
                       help=colored("Désactive la clusterisation sémantique", "cyan"))
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

def run_crawl(url, exclude_patterns=None):
    try:
        # Construire la commande avec les patterns d'exclusion
        command = ['python3', '01_crawl.py', url]
        if exclude_patterns:
            command.extend(exclude_patterns)
        output = run_command(command)
        crawl_id_line = [line for line in output.split('\n') if line.startswith('Crawl terminé. ID du crawl:')]
        if crawl_id_line:
            return crawl_id_line[0].split(': ')[1]
    except subprocess.CalledProcessError as e:
        logging.error(f"Crawl failed with error code {e.returncode}")
    return None

def run_internal_links_crawl(crawl_id, selector, exclude_patterns=None):
    try:
        command = ['python3', '03_crawl_internal_links.py', crawl_id, selector]
        if exclude_patterns:
            command.extend(['-e'] + exclude_patterns)
        run_command(command)
    except subprocess.CalledProcessError as e:
        logging.error(f"Internal links crawl failed with error code {e.returncode}")

def run_analysis(crawl_id, disable_clustering=False):
    try:
        command = ['python3', '02_analyse.py', crawl_id]
        if disable_clustering:
            command.append('--no-cluster')
        run_command(command)
    except subprocess.CalledProcessError as e:
        logging.error(f"Analysis failed with error code {e.returncode}")



def main(url=None, selector=None):
    parser = setup_argument_parser()
    
    # Initialiser les variables avec des valeurs par défaut
    exclude_patterns = []
    disable_clustering = False
    
    # Si les arguments ne sont pas fournis directement, les prendre des args du parser
    if url is None:
        args = parser.parse_args()
        url = args.url
        selector = args.selector  # Peut être None si non spécifié
        exclude_patterns = args.exclude_patterns
        disable_clustering = args.no_cluster
    else:
        # Si url est fourni directement, on prend les arguments de sys.argv
        if len(sys.argv) > 3:  # Si des arguments supplémentaires sont fournis
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == '-e' or sys.argv[i] == '--exclude-patterns':
                    i += 1
                    while i < len(sys.argv) and not sys.argv[i].startswith('-'):
                        exclude_patterns.append(sys.argv[i])
                        i += 1
                elif sys.argv[i] == '--no-cluster':
                    disable_clustering = True
                    i += 1
                else:
                    i += 1

    if not validate_url(url):
        logging.error(f"URL invalide: {url}")
        logging.info("L'URL doit commencer par http:// ou https://")
        return

    # Message adaptatif selon la présence ou non du sélecteur
    if selector:
        logging.info(f"🎯 Analyse de {url} avec le sélecteur: {selector}")
    else:
        logging.info(f"🎯 Analyse complète de {url}")

    config = load_config()
    if "HUGGINGFACEHUB_API_TOKEN" not in config:
        logging.error("❌ Erreur: HUGGINGFACEHUB_API_TOKEN non trouvé dans config.json")
        return

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]

    logging.info("🔄 Démarrage du crawl...")
    crawl_id = run_crawl(url, exclude_patterns)  # Passage des patterns d'exclusion
    if not crawl_id:
        logging.error("❌ Échec du crawl")
        return

    logging.info(f"✅ Crawl terminé avec l'ID: {crawl_id}")

    # Message adaptatif pour l'analyse des liens
    if selector:
        logging.info(f"🔍 Analyse des liens internes dans la zone sélectionnée...")
    else:
        logging.info(f"🔍 Analyse des liens internes sur toute la page...")
    
    # Si des patterns sont à exclure, on le signale dans les logs
    if exclude_patterns:
        logging.info(f"🔍 Filtrage des URLs contenant : {exclude_patterns}")

    # Un seul appel à run_internal_links_crawl avec les patterns d'exclusion
    run_internal_links_crawl(crawl_id, selector, exclude_patterns)
    logging.info("✅ Analyse des liens terminée")

    logging.info("🧠 Démarrage de l'analyse sémantique...")
    if disable_clustering:
        logging.info("ℹ️ Clusterisation désactivée")
    run_analysis(crawl_id, disable_clustering)

    # Message de fin avec information sur la portée de l'analyse
    scope = "la zone sélectionnée" if selector else "toute la page"
    logging.info(f"""
    ✨ Analyse terminée avec succès !
    
    Résumé :
    • URL analysée : {url}
    • Portée : {scope}
    • ID du crawl : {crawl_id}
    • Patterns exclus : {exclude_patterns if exclude_patterns else "Aucun"}
    
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