import redis
import sys
import subprocess
from enhanced_analyzer import EnhancedCoconAnalyzer

def main(crawl_id):
    # Récupération du port Redis comme dans votre code original
    redis_port = subprocess.check_output(
        "ddev describe -j | jq -r '.raw.services[\"redis-1\"].host_ports | split(\",\")[0]'", 
        shell=True
    )
    redis_client = redis.Redis(host='localhost', port=int(redis_port), db=0)
    
    # Utilisation de l'analyseur enrichi
    analyzer = EnhancedCoconAnalyzer(redis_client, crawl_id)
    
    # Génération du rapport enrichi
    enhanced_report = analyzer.generate_enhanced_report()
    
    # Affichage du rapport
    print(enhanced_report)
    
    # Sauvegarde dans un fichier
    with open(f"enhanced_cocon_analysis_{crawl_id[:30]}.txt", "w") as f:
        f.write(enhanced_report)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_cocon_enhanced.py <crawl_id>")
        sys.exit(1)
    
    main(sys.argv[1])