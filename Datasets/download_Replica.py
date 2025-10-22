import os
import subprocess
import requests
from pathlib import Path

TARGET = r"E:\replica"
os.makedirs(TARGET, exist_ok=True)
os.chdir(TARGET)

merged_file = "merge.tar.gz"
if os.path.exists(merged_file):
    print(f"{merged_file} already exists. Please delete it before running again.")
    raise SystemExit(1)

print("Downloading parts...")
base_url = "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.part"
parts = [f"a{chr(c)}" for c in range(ord("a"), ord("q") + 1)]
session = requests.Session()
timeout = 30

def get_remote_size(url):
    resp = session.head(url, allow_redirects=True, timeout=timeout)
    resp.raise_for_status()
    cl = resp.headers.get("Content-Length")
    return int(cl) if cl is not None else None

for p in parts:
    url = f"{base_url}{p}"
    local_name = f"replica_v1_0.tar.gz.part{p}"
    tmp_name = local_name + ".download"

    try:
        remote_size = get_remote_size(url)
    except requests.HTTPError as e:
        print(f"  HEAD failed for {url}: {e}")
        # If HEAD fails (404 etc.), stop trying further parts
        break

    if os.path.exists(local_name):
        local_size = os.path.getsize(local_name)
        if remote_size is not None and local_size == remote_size:
            print(f"  Skipping {local_name} (already complete)")
            continue
        if remote_size is not None and local_size > remote_size:
            print(f"  Local file larger than remote; re-downloading {local_name}")
            os.remove(local_name)
            local_size = 0
    else:
        local_size = 0

    # If we have a partial file, try to resume
    if local_size > 0 and remote_size is not None and local_size < remote_size:
        headers = {"Range": f"bytes={local_size}-"}
        print(f"  Attempting resume for {local_name} from byte {local_size}...")
        with session.get(url, headers=headers, stream=True, timeout=timeout) as r:
            if r.status_code == 206:
                # Server supports range -> append remaining bytes
                with open(local_name, "ab") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"  Resumed and completed {local_name}")
                continue
            elif r.status_code == 200:
                # Server ignored Range; fall through to full re-download logic
                print(f"  Server ignored Range (status 200). Re-downloading {local_name} to temp file...")
            else:
                r.raise_for_status()

    # Full (re)download to temp file, then atomically replace
    print(f"  Downloading {local_name} ...")
    with session.get(url, stream=True, timeout=timeout) as r:
        if r.status_code == 404:
            print(f"  Part {p} not found ({url})")
            break
        r.raise_for_status()
        with open(tmp_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    # Replace existing partial file (if any) with the downloaded temp file
    os.replace(tmp_name, local_name)
    # sanity check: verify size if remote_size is known
    if remote_size is not None and os.path.getsize(local_name) != remote_size:
        print(f"  Warning: downloaded size mismatch for {local_name} (expected {remote_size}, got {os.path.getsize(local_name)})")

print("Merging parts...")
with open(merged_file, "wb") as outfile:
    for p in parts:
        part_file = f"replica_v1_0.tar.gz.part{p}"
        if not os.path.exists(part_file):
            print(f"  Missing {part_file}, aborting merge.")
            raise SystemExit(1)
        with open(part_file, "rb") as infile:
            outfile.write(infile.read())

print("Extracting merge.tar.gz...")
result = subprocess.run(["tar", "-xvzf", merged_file], capture_output=True, text=True)
if result.returncode != 0:
    print("Failed to extract:", result.stderr)
    raise SystemExit(1)

print("Deleting merge.tar.gz...")
os.remove(merged_file)
print("Finished.")
print("(Consider deleting replica_v1_0.tar.gz.part* to save disk space.)")
