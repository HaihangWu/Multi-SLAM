import zipfile
import os

# Paths to the folders you want to zip
folder1_path = r'C:\Users\hthh1\Downloads\MA_ADT'
folder2_path = r'C:\Users\hthh1\Downloads\ADT'

# Path where the zip file will be saved
output_path = r'C:\Users\hthh1\Downloads\ADT.zip'

# Create a zip file
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add folder 1 to the zip file
    for root, dirs, files in os.walk(folder1_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Add the file to the zip, maintaining folder structure
            zipf.write(file_path, os.path.relpath(file_path, folder1_path))

    # Add folder 2 to the zip file
    for root, dirs, files in os.walk(folder2_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Add the file to the zip, maintaining folder structure
            zipf.write(file_path, os.path.relpath(file_path, folder2_path))

print(f"Zip file created at: {output_path}")
