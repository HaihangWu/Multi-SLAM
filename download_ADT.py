import os
import json
import requests
from tqdm import tqdm
from pathlib import Path

# === Configuration ===
base_dir = Path(r"E:\ADT")  # Base directory for storing downloaded files
cdn_file = base_dir / "ADT_download_urls.json"  # JSON file with all sequences

# === Sequences you need ===
DATASET_DICT = {
    "room0": {
        "agent_0": "Apartment_release_decoration_seq136_M1292",
        "agent_1": "Apartment_release_decoration_seq137_M1292",
        "agent_2": "Apartment_release_decoration_seq139_M1292"
    },
    "room1": {
        "agent_0": "Apartment_release_decoration_seq134_M1292",
        "agent_1": "Apartment_release_decoration_skeleton_seq133_M1292",
        "agent_2": "Apartment_release_decoration_skeleton_seq135_M1292"
    }
}

TEST_DATASET_DICT = {
    "room0": "Apartment_release_decoration_seq133_M1292",
    "room1": "Apartment_release_decoration_seq139_M1292"
}

# === Resume-capable downloader ===
def download_with_resume(url, dest_path):
    dest_path = Path(dest_path)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    max_retries = 3

    # Skip if already fully downloaded
    if dest_path.exists() and not temp_path.exists():
        print(f"✔ Already downloaded: {dest_path.name}")
        return

    headers = {}
    if temp_path.exists():
        downloaded = temp_path.stat().st_size
        headers["Range"] = f"bytes={downloaded}-"
    else:
        downloaded = 0

    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, headers=headers, timeout=(10, 1800)) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + downloaded
                mode = "ab" if downloaded > 0 else "wb"
                with open(temp_path, mode) as f, tqdm(
                    total=total,
                    initial=downloaded,
                    unit="B",
                    unit_scale=True,
                    desc=dest_path.name,
                    ncols=80,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            # Safely rename
            if dest_path.exists():
                dest_path.unlink()  # delete old file if exists
            temp_path.rename(dest_path)
            print(f"✅ Finished: {dest_path.name}")
            return

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}/{max_retries} failed for {dest_path.name}: {e}")
            import time
            time.sleep(5)

    print(f"❌ Failed after {max_retries} attempts: {dest_path.name}")


# === Main logic ===
def main():
    if not cdn_file.exists():
        print(f"❌ CDN file not found: {cdn_file}")
        return

    # Collect only the sequences you need
    required_scenes = set()
    for room in DATASET_DICT.values():
        required_scenes.update(room.values())
    required_scenes.update(TEST_DATASET_DICT.values())

    urls_to_download = []
    with open(cdn_file, "r") as f:
        data = json.load(f)
        sequences = data.get("sequences", {})

        for scene_name, files_dict in sequences.items():
            if scene_name in required_scenes:
                for data_group, files in files_dict.get("data_groups", {}).items():
                    # Check and collect all files to download
                    for file_name in files:
                        if "download_url" in file_name and "filename" in file_name:
                            url = file_name["download_url"]
                            filename = file_name["filename"]
                            urls_to_download.append((url, filename))

    print(f"Found {len(urls_to_download)} files to download for selected sequences.")

    # Download all files
    for url, filename in urls_to_download:
        dest_path = base_dir / filename
        download_with_resume(url, dest_path)


if __name__ == "__main__":
    main()