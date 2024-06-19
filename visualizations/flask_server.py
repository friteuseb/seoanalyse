from flask import Flask, jsonify, send_from_directory
import os

app = Flask(__name__)

CRAWLS_DIR = 'visualizations/crawls'

@app.route('/api/crawls', methods=['GET'])
def list_crawls():
    try:
        crawls = [f for f in os.listdir(CRAWLS_DIR) if f.endswith('.json')]
        return jsonify({'crawls': crawls})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualizations/crawls/<path:filename>', methods=['GET'])
def serve_crawl_file(filename):
    try:
        return send_from_directory(CRAWLS_DIR, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(debug=False, port=5000)
