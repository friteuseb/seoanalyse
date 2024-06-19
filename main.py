import subprocess
import sys
import os
import json

def load_config():
    with open('config.json') as f:
        config = json.load(f)
    return config

def run_crawl(url):
    result = subprocess.run(['python3', '01_crawl.py', url], capture_output=True, text=True)
    print("Crawl output:", result.stdout)
    print("Crawl errors:", result.stderr)
    crawl_id_line = [line for line in result.stdout.split('\n') if line.startswith('Crawl terminé. ID du crawl:')]
    if crawl_id_line:
        crawl_id = crawl_id_line[0].split(': ')[1]
        return crawl_id
    return None

def run_internal_links_crawl(crawl_id, selector):
    result = subprocess.run(['python3', '03_crawl_internal_links.py', crawl_id, selector], capture_output=True, text=True)
    print("Internal links crawl output:", result.stdout)
    print("Internal links crawl errors:", result.stderr)

def run_analysis(crawl_id):
    result = subprocess.run(['python3', '02_analyse.py', crawl_id], capture_output=True, text=True)
    print("Analysis output:", result.stdout)
    print("Analysis errors:", result.stderr)

def start_http_server():
    print("Starting server for visualization...")
    os.chdir('visualizations')
    subprocess.Popen(['python3', '-m', 'http.server', '8000'])

def start_flask_server():
    print("Starting Flask server for JSON files...")
    subprocess.Popen(['python3', 'flask_server.py'])

def main(url, selector):
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
    start_http_server()
    start_flask_server()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 main.py <URL> <CSS Selector>")
        sys.exit(1)
    
    url = sys.argv[1]
    selector = sys.argv[2]
    main(url, selector)
