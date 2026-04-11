#!/usr/bin/env python3
"""Parallel download of WSI files for 3-modal intersection patients."""
import os
import csv
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WSI_DIR = "/data/data/Drug_Pred/05_morphology/wsi"
TARGET_CSV = "/data/data/Drug_Pred/05_morphology/wsi_target_3modal.csv"
BASE = "https://api.gdc.cancer.gov"
N_WORKERS = 6  # parallel threads

def create_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=N_WORKERS + 2))
    return session

def download_one(file_id, file_name, file_size_mb):
    out_path = os.path.join(WSI_DIR, file_name)
    if os.path.exists(out_path):
        actual = os.path.getsize(out_path) / (1024**2)
        if actual > file_size_mb * 0.9:  # allow 10% tolerance
            return file_name, "skip", actual

    session = create_session()
    try:
        resp = session.get(f"{BASE}/data/{file_id}", timeout=600, stream=True)
        resp.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=131072):
                f.write(chunk)
        actual = os.path.getsize(out_path) / (1024**2)
        return file_name, "ok", actual
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        return file_name, f"error: {e}", 0

def main():
    # Load target list
    with open(TARGET_CSV) as f:
        reader = csv.DictReader(f)
        files = list(reader)

    # Filter to not-yet-downloaded
    existing = set(os.listdir(WSI_DIR))
    todo = [f for f in files if f['file_name'] not in existing]
    print(f"Total target: {len(files)}, already downloaded: {len(files)-len(todo)}, remaining: {len(todo)}")
    print(f"Estimated remaining size: {sum(float(f['file_size_MB']) for f in todo)/1024:.1f} GB")
    print(f"Using {N_WORKERS} parallel workers")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()

    completed = 0
    total_mb = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(download_one, f['file_id'], f['file_name'], float(f['file_size_MB'])): f
            for f in todo
        }

        for future in as_completed(futures):
            fname, status, size_mb = future.result()
            completed += 1
            total_mb += size_mb

            elapsed = time.time() - start_time
            speed = total_mb / elapsed if elapsed > 0 else 0
            remaining_mb = sum(float(f['file_size_MB']) for f in todo) - total_mb
            eta_min = remaining_mb / speed / 60 if speed > 0 else 0

            if completed % 10 == 0 or status != "ok":
                print(f"[{completed}/{len(todo)}] {fname[:50]:50s} | {status:5s} | "
                      f"{size_mb:.0f}MB | {speed:.1f}MB/s | ETA: {eta_min:.0f}min")
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\nDone! {completed} files, {total_mb/1024:.1f} GB in {elapsed/3600:.1f} hours")
    print(f"Final count: {len(os.listdir(WSI_DIR))} files in {WSI_DIR}")

if __name__ == '__main__':
    main()
