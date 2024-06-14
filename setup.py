from setuptools import setup, find_packages

setup(
    name='webcrawler',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'aiohttp==3.9.5',
        'beautifulsoup4==4.12.3',
        'huggingface-hub==0.23.3',
        'nltk==3.8.1',
        'numpy==1.26.4',
        'redis==5.0.5',
        'requests==2.25.1',
        'scikit-learn==1.5.0',
        'sentence-transformers==2.2.2',
        'tld==0.13',
        'trafilatura==1.10.0'
    ],
    entry_points={
        'console_scripts': [
            'webcrawler=crawl:main',
            'analyse=analyse:main',
            'visualize=3d-force-graph:main',
        ],
    },
)

# Post-installation script to download NLTK stopwords
import nltk
nltk.download('stopwords')
