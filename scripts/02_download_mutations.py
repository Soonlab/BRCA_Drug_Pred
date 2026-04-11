#!/usr/bin/env python3
"""Download TCGA-BRCA somatic mutation data (MAF) from GDC."""
import requests, json, os, tarfile, gzip, shutil

OUT_DIR = "/data/data/Drug_Pred/02_genomic/mutations"
BASE = "https://api.gdc.cancer.gov"

# Query for open-access MAF files (Masked Somatic Mutation)
filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "data_category", "value": "Simple Nucleotide Variation"}},
        {"op": "=", "content": {"field": "data_type", "value": "Masked Somatic Mutation"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]
}

# Get file list
params = {
    "filters": json.dumps(filters),
    "fields": "file_id,file_name,file_size,data_type,analysis.workflow_type",
    "size": 100,
    "format": "JSON"
}
resp = requests.get(f"{BASE}/files", params=params)
resp.raise_for_status()
hits = resp.json()["data"]["hits"]
print(f"Found {len(hits)} MAF files")

for h in hits:
    print(f"  {h['file_name']} ({h['file_size']/1024/1024:.1f} MB) - {h.get('analysis',{}).get('workflow_type','N/A')}")

# Download each MAF file
for h in hits:
    fid = h["file_id"]
    fname = h["file_name"]
    out_path = os.path.join(OUT_DIR, fname)
    
    if os.path.exists(out_path) or os.path.exists(out_path.replace('.gz','')):
        print(f"  {fname} already exists, skipping")
        continue
    
    print(f"Downloading {fname}...")
    resp = requests.get(f"{BASE}/data/{fid}", timeout=300)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"  Saved ({len(resp.content)/1024/1024:.1f} MB)")

# Also download aggregated mutation data via MC3 or similar
print("\n=== Downloading TCGA MC3 public MAF (pan-cancer, will filter BRCA) ===")
# GDC aggregated somatic mutations for TCGA-BRCA
agg_filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "data_category", "value": "Simple Nucleotide Variation"}},
        {"op": "=", "content": {"field": "data_type", "value": "Aggregated Somatic Mutation"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]
}
params["filters"] = json.dumps(agg_filters)
resp = requests.get(f"{BASE}/files", params=params)
agg_hits = resp.json()["data"]["hits"]
print(f"Found {len(agg_hits)} aggregated MAF files")
for h in agg_hits:
    print(f"  {h['file_name']} ({h['file_size']/1024/1024:.1f} MB)")
    fid = h["file_id"]
    fname = h["file_name"]
    out_path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(out_path):
        resp = requests.get(f"{BASE}/data/{fid}", timeout=600)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        print(f"  Saved ({len(resp.content)/1024/1024:.1f} MB)")

print("\nDone!")
