#!/usr/bin/env bash
# For every model folder that has exactly 112 files, delete all JSON files
# except all_results.json and summary.json.
# Incomplete folders (< 112 files) are left untouched.

RESULTS_DIR="$(dirname "$0")/results"
KEEP=("all_results.json" "summary.json")

for model_dir in "$RESULTS_DIR"/*/; do
    count=$(ls "$model_dir" | wc -l)
    if [ "$count" -ne 112 ]; then
        echo "SKIP  $model_dir  ($count files — not a full directory)"
        continue
    fi

    echo "CLEAN $model_dir  ($count files)"
    for f in "$model_dir"*.json; do
        fname=$(basename "$f")
        if [[ "$fname" == "all_results.json" || "$fname" == "summary.json" ]]; then
            echo "  keep  $fname"
        else
            echo "  delete $fname"
            rm "$f"
        fi
    done
done

echo ""
echo "Done. Remaining file counts:"
for model_dir in "$RESULTS_DIR"/*/; do
    echo "  $(ls "$model_dir" | wc -l)  $model_dir"
done
