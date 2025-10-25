#!/bin/bash

dest="/data/gpfs/projects/punim0512/data/7-scenes"
mkdir -p "$dest"

urls=(
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip"
)

for url in "${urls[@]}"; do
    file_name=$(basename "$url")
    scene_name="${file_name%.*}"
    scene_path="$dest/$scene_name"

    echo "Unzipping $file_name..."
    unzip "$dest/$file_name" -d "$dest"
    rm -f "$dest/$file_name"  # Delete main scene zip after unzip

    # Unzip each sequence inside the scene folder and delete its zip
    for seq_zip in "$scene_path"/seq-*.zip; do
        [ -f "$seq_zip" ] || continue  # Skip if no seq zip
        echo "Unzipping $(basename "$seq_zip")..."
        unzip "$seq_zip" -d "$scene_path"
        rm -f "$seq_zip"  # Delete sequence zip after unzip
    done
done

