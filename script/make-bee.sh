#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 <start_idx> <end_idx> [input_dir] [output_dir]" >&2
    exit 1
fi

sta_idx=$1
end_idx=$2
input_dir=${3:-$PWD}
output_dir=${4:-$PWD}

if [[ ! -d $input_dir ]]; then
    echo "Input directory not found: $input_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"

input_dir=$(cd "$input_dir" && pwd)
output_dir=$(cd "$output_dir" && pwd)
bee_dir="$output_dir/bee"
data_dir="$bee_dir/data"
zip_path="$output_dir/upload.zip"

rm -rf "$bee_dir"
mkdir -p "$data_dir"

for zip_name in mabc-apa0-face0.zip mabc-apa1-face1.zip; do
    unzip -o "$input_dir/$zip_name" -d "$bee_dir"
done

for apa in apa0 apa1; do
    for x in $(seq "$sta_idx" "$end_idx"); do
        target_dir="$data_dir/$x"
        mkdir -p "$target_dir"

        cp "$input_dir/tru-$apa-$x.json" "$target_dir/$x-tru-$apa.json"
        cp "$input_dir/rec-op-$apa-$x.json" "$target_dir/$x-rec-op-$apa.json"
        cp "$input_dir/mc-$x.json" "$target_dir/$x-mc.json"

        echo "Copied files for $apa index $x to $target_dir/"
    done
done

rm -f "$zip_path"
(
    cd "$bee_dir"
    zip -r "$zip_path" data
)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$script_dir/upload-to-bee.sh" "$zip_path"
