#!/bin/bash

sta_idx=$1
end_idx=$2

# Create the base directory if it doesn't exist
# rm -rf bee
mkdir -p bee/data

unzip -o mabc-apa0-face0.zip -d bee
unzip -o mabc-apa1-face1.zip -d bee

for apa in apa0 apa1; do
    # Loop from 0 to 9
    for x in $(seq ${sta_idx} ${end_idx}); do
        # Create the destination directory if it doesn't exist
        mkdir -p bee/data/$x
        
        # Copy the file
        cp tru-$apa-$x.json bee/data/$x/$x-tru-$apa.json
        cp rec-op-$apa-$x.json bee/data/$x/$x-rec-op-$apa.json
        cp mc-$x.json bee/data/$x/$x-mc.json
        
        # Print status
        echo "Copied tru-$apa-$x.json to bee/data/$x/"
    done
done

echo "All files copied successfully."

rm -f upload.zip
cd bee
zip -r ../upload.zip data
cd ..
/exp/sbnd/app/users/yuhw/dl-clus/script/upload-to-bee.sh upload.zip