import os
import subprocess
import requests

TARGET = r"C:\replica"
os.makedirs(TARGET, exist_ok=True)
os.chdir(TARGET)

merged_file = "merge.tar.gz"
if os.path.exists(merged_file):
    print(f"{merged_file} already exists. Please delete it before running again.")
    exit(1)

print("Downloading parts...")
base_url = "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.part"
# parts aa–aq
parts = [f"a{chr(c)}" for c in range(ord("a"), ord("q") + 1)]

for p in parts:
    url = f"{base_url}{p}"
    local_name = f"replica_v1_0.tar.gz.part{p}"
    if os.path.exists(local_name):
        print(f"  Skipping existing part {local_name}")
        continue
    print(f"  Downloading {local_name} ...")
    with requests.get(url, stream=True) as r:
        if r.status_code == 404:
            print(f"  Part {p} not found ({url})")
            break
        r.raise_for_status()
        with open(local_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

print("Merging parts...")
with open(merged_file, "wb") as outfile:
    for p in parts:
        part_file = f"replica_v1_0.tar.gz.part{p}"
        if not os.path.exists(part_file):
            print(f"  Missing {part_file}, skipping merge.")
            exit(1)
        with open(part_file, "rb") as infile:
            outfile.write(infile.read())

print("Extracting merge.tar.gz...")
result = subprocess.run(["tar", "-xvzf", merged_file], capture_output=True, text=True)
if result.returncode != 0:
    print("Failed to extract:", result.stderr)
    exit(1)

print("Deleting merge.tar.gz...")
os.remove(merged_file)

print("Finished.")
print("(Consider deleting replica_v1_0.tar.gz.part* to save disk space.)")
