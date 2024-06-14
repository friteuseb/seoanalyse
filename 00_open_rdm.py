import subprocess
import os

def open_rdm():
    rdm_path = '/snap/bin/another-redis-desktop-manager'
    if not os.path.exists(rdm_path):
        print(f"Erreur : Impossible de trouver RDM à l'emplacement spécifié : {rdm_path}")
        return

    try:
        subprocess.Popen([rdm_path])
    except Exception as e:
        print(f"Erreur : Impossible d'ouvrir RDM : {str(e)}")

if __name__ == "__main__":
    open_rdm()
