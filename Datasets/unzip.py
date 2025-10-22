import os
import zipfile
import tarfile


def unzip_all_zip_files(directory):
    # Walk through all subdirectories and files in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".tar.gz", ".tgz")):
                tgz_path = os.path.join(root, file)
                extract_to_dir = root  # Extract to the same directory
                print(f"Extracting {tgz_path} to {extract_to_dir}")

                try:
                    with tarfile.open(tgz_path, "r:gz") as tar_ref:
                        tar_ref.extractall(extract_to_dir)
                except Exception as e:
                    print(f"Failed to extract {tgz_path}: {e}")

            # if file.endswith(".zip"):
            #     zip_file_path = os.path.join(root, file)
            #     extract_to_dir = root  # Set the extraction directory to the folder where the zip file is located
            #
            #     # Extract the zip file if not already extracted
            #     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            #         print(f"Extracting {zip_file_path} to {extract_to_dir}")
            #         zip_ref.extractall(extract_to_dir)


if __name__ == "__main__":
    base_directory = r"C:\Users\hthh1\Downloads\TUM RGB-D"  # Replace with the directory you want to scan
    unzip_all_zip_files(base_directory)
