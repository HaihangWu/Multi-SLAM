#!/bin/bash

dest="/data/gpfs/projects/punim0512/data/7-scenes/"
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

    # Skip if scene already exists
    if [ -d "$scene_path" ]; then
        echo "$scene_name already unzipped, skipping."
        continue
    fi

    # Uncomment to download if needed
    # wget "$url" -O "$dest/$file_name"

    echo "Unzipping $file_name..."
    unzip "$dest/$file_name" -d "$dest"
done

