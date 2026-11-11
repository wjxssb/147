#!/usr/bin/env python3
"""
Downloads the 18 HDF5 sessions for subject 89335547 from the official UCLA Box share:
https://ucla.box.com/s/3xc4nwpfjfpo6ydjs94t0v2kuq37d5eg
Saves files directly into data/{session_name}.hdf5 as expected by emg2qwerty.
"""

import os
import re
import sys
import json
import urllib.request
from pathlib import Path

BOX_SHARED_URL = "https://ucla.box.com/s/3xc4nwpfjfpo6ydjs94t0v2kuq37d5eg"
SHARED_NAME = "3xc4nwpfjfpo6ydjs94t0v2kuq37d5eg"

def fetch_file_list():
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    req = urllib.request.Request(BOX_SHARED_URL, headers=headers)
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    
    idx = html.find("Box.postStreamData = ")
    if idx == -1:
        raise RuntimeError("Could not find Box.postStreamData in HTML")
    end_idx = html.find(";</script>", idx)
    data = json.loads(html[idx + len("Box.postStreamData = "):end_idx])
    folder_data = data.get("/app-api/enduserapp/shared-folder", {})
    items = folder_data.get("items", [])
    print(f"Found {len(items)} files in Box shared folder.")
    return items

def download_file(item, dest_dir: Path):
    file_id = item["id"]
    name = item["name"]
    dest_path = dest_dir / name
    expected_size = item.get("itemSize")

    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if expected_size is None or actual_size == expected_size:
            print(f"[SKIP] {name} already exists ({actual_size} bytes).")
            return
        else:
            print(f"[RESUME] {name} partial ({actual_size}/{expected_size} bytes).")

    download_url = (
        f"https://ucla.app.box.com/index.php?rm=box_download_shared_file"
        f"&shared_name={SHARED_NAME}&file_id=f_{file_id}"
    )
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    req = urllib.request.Request(download_url, headers=headers)

    print(f"[DOWNLOADING] {name} ({expected_size / 1e6:.1f} MB)...")
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        bytes_copied = 0
        chunk_size = 1024 * 1024  # 1MB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            bytes_copied += len(chunk)
            if expected_size:
                pct = bytes_copied / expected_size * 100
                sys.stdout.write(f"\r  -> {bytes_copied / 1e6:.1f} / {expected_size / 1e6:.1f} MB ({pct:.1f}%)")
                sys.stdout.flush()
        print()
    print(f"[DONE] {name}")

def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    items = fetch_file_list()
    for item in items:
        download_file(item, data_dir)
    print("All files downloaded successfully.")

if __name__ == "__main__":
    main()
