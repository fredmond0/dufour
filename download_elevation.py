#!/usr/bin/env python3
"""
SwissALTI3D Elevation Data Downloader
Downloads elevation tiles from the SwissALTI3D dataset
"""

import os
import csv
import requests
from urllib.parse import urlparse
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('elevation_download.log'),
        logging.StreamHandler()
    ]
)

# Configuration
CSV_FILE = 'ch.swisstopo.swissalti3d-W9EvkmpW.csv'
DOWNLOAD_DIR = 'data/elevation/swissalti3d'
MAX_WORKERS = 4  # Number of concurrent downloads
CHUNK_SIZE = 8192  # Download chunk size

def create_directories():
    """Create necessary directories"""
    Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directory: {DOWNLOAD_DIR}")

def get_filename_from_url(url):
    """Extract filename from URL"""
    parsed = urlparse(url)
    return os.path.basename(parsed.path)

def download_tile(url, output_dir):
    """Download a single elevation tile"""
    try:
        filename = get_filename_from_url(url)
        output_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            logging.info(f"Skipping {filename} - already exists")
            return True, filename
        
        # Download the file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(output_path)
        logging.info(f"Downloaded {filename} ({file_size / 1024 / 1024:.1f} MB)")
        return True, filename
        
    except Exception as e:
        logging.error(f"Failed to download {url}: {str(e)}")
        return False, url

def download_all_tiles():
    """Download all elevation tiles from CSV"""
    create_directories()
    
    # Read URLs from CSV
    urls = []
    try:
        with open(CSV_FILE, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].startswith('http'):
                    urls.append(row[0].strip())
    except Exception as e:
        logging.error(f"Failed to read CSV file: {str(e)}")
        return
    
    logging.info(f"Found {len(urls)} URLs to download")
    
    # Download tiles with progress tracking
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all download tasks
        future_to_url = {executor.submit(download_tile, url, DOWNLOAD_DIR): url for url in urls}
        
        # Process completed downloads
        for future in as_completed(future_to_url):
            success, result = future.result()
            if success:
                successful += 1
            else:
                failed += 1
            
            # Progress update every 10 downloads
            if (successful + failed) % 10 == 0:
                logging.info(f"Progress: {successful + failed}/{len(urls)} (Success: {successful}, Failed: {failed})")
    
    logging.info(f"Download complete! Success: {successful}, Failed: {failed}")

def download_sample_tiles(num_tiles=10):
    """Download a small sample of tiles for testing"""
    create_directories()
    
    # Read first few URLs from CSV
    urls = []
    try:
        with open(CSV_FILE, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= num_tiles:
                    break
                if row and row[0].startswith('http'):
                    urls.append(row[0].strip())
    except Exception as e:
        logging.error(f"Failed to read CSV file: {str(e)}")
        return
    
    logging.info(f"Downloading sample of {len(urls)} tiles")
    
    # Download tiles
    successful = 0
    failed = 0
    
    for url in urls:
        success, result = download_tile(url, DOWNLOAD_DIR)
        if success:
            successful += 1
        else:
            failed += 1
        time.sleep(0.1)  # Small delay between downloads
    
    logging.info(f"Sample download complete! Success: {successful}, Failed: {failed}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download SwissALTI3D elevation data')
    parser.add_argument('--sample', action='store_true', help='Download only a sample of tiles for testing')
    parser.add_argument('--num-sample', type=int, default=10, help='Number of sample tiles to download')
    
    args = parser.parse_args()
    
    if args.sample:
        download_sample_tiles(args.num_sample)
    else:
        download_all_tiles()
