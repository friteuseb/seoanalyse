import sqlite3
import trafilatura

# Fonction pour créer une base de données et une table
def create_database(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Fonction pour insérer les données extraites dans la base de données
def insert_content(db_name, url, content):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO extracted_content (url, content)
        VALUES (?, ?)
    ''', (url, content))
    conn.commit()
    conn.close()

# Exemple d'utilisation
def main():
    db_name = 'extracted_content.db'
    create_database(db_name)
    
    # Exemple d'URL à extraire (à remplacer par vos propres URLs)
    urls = [
        'https://example.com/page1',
        'https://example.com/page2'
    ]
    
    for url in urls:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            content = trafilatura.extract(downloaded)
            if content:
                insert_content(db_name, url, content)
                print(f"Content from {url} inserted into the database.")
            else:
                print(f"Failed to extract content from {url}")
        else:
            print(f"Failed to fetch {url}")

if __name__ == '__main__':
    main()
