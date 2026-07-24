#!/bin/bash
SOURCE_DIR="/home/meow/cesnet_data/PEG/Colorado/Source"

mkdir -p "$SOURCE_DIR/Pollen_production"
mkdir -p "$SOURCE_DIR/Pollen_deposition"

# For each species directory in Source
for SPECIES_DIR in "$SOURCE_DIR"/*; do
    if [ -d "$SPECIES_DIR" ]; then
        SPECIES=$(basename "$SPECIES_DIR")
        # Skip if it's already one of the target folders or starting with 2607
        if [ "$SPECIES" = "Pollen_production" ] || [ "$SPECIES" = "Pollen_deposition" ] || [[ "$SPECIES" == 2607* ]]; then
            continue
        fi

        echo "Processing $SPECIES..."
        
        # Look at files in the species directory
        for FILE in "$SPECIES_DIR"/*.czi; do
            if [ -f "$FILE" ]; then
                FILENAME=$(basename "$FILE")
                if [[ "$FILENAME" == *"pol_pro"* ]]; then
                    # Production
                    TARGET_DIR="$SOURCE_DIR/Pollen_production/$SPECIES"
                else
                    # Deposition
                    TARGET_DIR="$SOURCE_DIR/Pollen_deposition/$SPECIES"
                fi
                
                mkdir -p "$TARGET_DIR"
                mv "$FILE" "$TARGET_DIR/"
            fi
        done
        
        # Remove original dir if empty
        rmdir "$SPECIES_DIR" 2>/dev/null
    fi
done

echo "Reorganization complete."
