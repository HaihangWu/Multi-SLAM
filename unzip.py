import os
import zipfile


def unzip_all_zip_files(directory):
    # Walk through all subdirectories and files in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".zip"):
                zip_file_path = os.path.join(root, file)
                extract_to_dir = root  # Set the extraction directory to the folder where the zip file is located

                # Extract the zip file if not already extracted
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    print(f"Extracting {zip_file_path} to {extract_to_dir}")
                    zip_ref.extractall(extract_to_dir)


if __name__ == "__main__":
    base_directory = r"E:\ADT"  # Replace with the directory you want to scan
    unzip_all_zip_files(base_directory)
