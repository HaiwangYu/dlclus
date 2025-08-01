#!/bin/bash

name=$2
base_name="${name%.jsonnet}"


# Split $WIRECELL_PATH on colon into an array of directories
IFS=':' read -ra cfg_dirs <<< "$WIRECELL_PATH"

# Build the -J arguments from the array
J_ARGS=""
for (( idx=${#cfg_dirs[@]}-1; idx>=0; idx-- )); do
    J_ARGS="$J_ARGS -J ${cfg_dirs[$idx]}"
done
# Print the generated J_ARGS for debugging purposes
# echo "J_ARGS: $J_ARGS"

if [[ $1 == "json" || $1 == "all" ]]; then
    jsonnet \
      --ext-str reality="data" \
      --ext-code recobwire_tags="[\"simtpc2d:gauss\", \"simtpc2d:wiener\"]" \
      --ext-code trace_tags="[\"gauss\", \"wiener\"]" \
      --ext-code summary_tags="[\"\", \"simtpc2d:wienersummary\"]" \
      --ext-code input_mask_tags="[\"simtpc2d:badmasks\"]" \
      --ext-code output_mask_tags="[\"bad\"]" \
      $J_ARGS \
      ${base_name}.jsonnet \
      -o ${base_name}.json
fi

if [[ $1 == "pdf" || $1 == "all" ]]; then
    # wirecell-pgraph dotify --jpath -1 --no-services --no-params ${base_name}.json ${base_name}.pdf
    wirecell-pgraph dotify --jpath -1 ${3} ${base_name}.json ${base_name}.pdf
fi
