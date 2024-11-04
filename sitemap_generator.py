import os
import datetime
from urllib.parse import urljoin

def generate_sitemap(base_url, directory):
    """
    Génère un sitemap.xml pour un site statique.
    
    Args:
        base_url: URL de base (ex: http://0.0.0.0:8000)
        directory: Chemin vers le dossier contenant les fichiers HTML
    """
    
    urls = []
    # Parcourir le dossier récursivement
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.html'):
                # Créer le chemin relatif
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                # Convertir les backslashes en forward slashes pour les URLs
                rel_path = rel_path.replace('\\', '/')
                # Créer l'URL complète
                url = urljoin(base_url, rel_path)
                # Obtenir la date de dernière modification
                lastmod = datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).strftime('%Y-%m-%d')
                urls.append((url, lastmod))

    # Générer le XML
    xml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml_content.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
    
    for url, lastmod in urls:
        xml_content.append('  <url>')
        xml_content.append(f'    <loc>{url}</loc>')
        xml_content.append(f'    <lastmod>{lastmod}</lastmod>')
        xml_content.append('    <changefreq>daily</changefreq>')
        xml_content.append('    <priority>0.8</priority>')
        xml_content.append('  </url>')
    
    xml_content.append('</urlset>')
    
    # Écrire le fichier
    output_path = os.path.join(directory, 'sitemap.xml')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml_content))
    
    print(f"Sitemap généré avec {len(urls)} URLs: {output_path}")
    return output_path

if __name__ == "__main__":
    BASE_URL = "http://0.0.0.0:8000"
    PAGES_DIR = "pages"  # Ajustez selon votre structure
    
    generate_sitemap(BASE_URL, PAGES_DIR)