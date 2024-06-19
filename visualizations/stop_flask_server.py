import os
import signal
import subprocess

def stop_flask_server():
    # Trouver le processus Flask en utilisant le nom du fichier
    result = subprocess.run(['pgrep', '-f', 'flask_server.py'], capture_output=True, text=True)
    pids = result.stdout.strip().split('\n')

    if not pids:
        print("No Flask server running.")
        return

    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"Stopped Flask server with PID: {pid}")
        except OSError as e:
            print(f"Error stopping Flask server with PID: {pid}: {e}")

if __name__ == "__main__":
    stop_flask_server()
