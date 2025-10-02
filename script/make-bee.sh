#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <input_dir> <output_dir> [start_idx] [end_idx]" >&2
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
    usage
    exit 1
fi

input_dir=$1
output_dir=$2
sta_idx=${3-}
end_idx=${4-}

if [[ ! -d $input_dir ]]; then
    echo "Input directory not found: $input_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"

input_dir=$(cd "$input_dir" && pwd)
output_dir=$(cd "$output_dir" && pwd)

if [[ -n ${sta_idx} && ! $sta_idx =~ ^[0-9]+$ ]]; then
    echo "start_idx must be numeric: $sta_idx" >&2
    exit 1
fi

if [[ -n ${end_idx} && ! $end_idx =~ ^[0-9]+$ ]]; then
    echo "end_idx must be numeric: $end_idx" >&2
    exit 1
fi

if [[ -z ${sta_idx} || -z ${end_idx} ]]; then
    shopt -s nullglob
    idx_candidates=()
    for f in "$input_dir"/mc-*.json; do
        fname=${f##*/}
        idx=${fname%.json}
        idx=${idx#mc-}
        if [[ $idx =~ ^[0-9]+$ ]]; then
            idx_candidates+=("$idx")
        fi
    done
    shopt -u nullglob

    if [[ ${#idx_candidates[@]} -eq 0 ]]; then
        echo "Unable to infer indices from $input_dir" >&2
        exit 1
    fi

    mapfile -t sorted_indices < <(printf '%s\n' "${idx_candidates[@]}" | sort -n | uniq)

    if [[ -z ${sta_idx} ]]; then
        sta_idx=${sorted_indices[0]}
    fi
    if [[ -z ${end_idx} ]]; then
        end_idx=${sorted_indices[${#sorted_indices[@]}-1]}
    fi
fi

if [[ -z ${sta_idx} || -z ${end_idx} ]]; then
    echo "Both start_idx and end_idx could not be determined." >&2
    exit 1
fi

if (( sta_idx > end_idx )); then
    echo "start_idx ($sta_idx) cannot be greater than end_idx ($end_idx)" >&2
    exit 1
fi

echo "make-bee.sh: Processing indices from $sta_idx to $end_idx"

bee_dir="$output_dir/bee"
data_dir="$bee_dir/data"
zip_path="$output_dir/upload.zip"

rm -rf "$bee_dir"
mkdir -p "$data_dir"

for zip_name in mabc-apa0-face0.zip mabc-apa1-face1.zip; do
    unzip -oq "$input_dir/$zip_name" -d "$bee_dir"
done

for apa in apa0 apa1; do
    for x in $(seq "$sta_idx" "$end_idx"); do
        target_dir="$data_dir/$x"
        mkdir -p "$target_dir"

        cp "$input_dir/tru-$apa-$x.json" "$target_dir/$x-tru-$apa.json"
        cp "$input_dir/rec-op-$apa-$x.json" "$target_dir/$x-rec-op-$apa.json"
        cp "$input_dir/mc-$x.json" "$target_dir/$x-mc.json"

        # echo "Copied files for $apa index $x to $target_dir/"
    done
done

rm -f "$zip_path"
(
    cd "$bee_dir"
    zip -qr "$zip_path" data
)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$script_dir/upload-to-bee.sh" "$zip_path"
