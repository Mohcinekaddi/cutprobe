#!/bin/bash
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
RUNS=runs
STATUS=$RUNS/full_pipeline_status.txt
mkdir -p "$RUNS"

run_model() {
    local cfg=$1 batch=$2 dir=$RUNS/$1 conf=configs/$1.yaml
    local summary=$dir/attack/attack_summary_gpu.csv
    mkdir -p "$dir"
    if [ ! -f "$dir/search_results.json" ]; then
        echo "$(date '+%F %T') $cfg search start" >> $STATUS
        if $PY cli.py search -c $conf > "$dir/pipeline_search.log" 2>&1; then
            echo "$(date '+%F %T') $cfg search done" >> $STATUS
        else
            echo "$(date '+%F %T') $cfg search FAILED (see $dir/pipeline_search.log)" >> $STATUS
            return
        fi
    else
        echo "$(date '+%F %T') $cfg search already done" >> $STATUS
    fi
    if [ ! -f "$summary" ] || [ "$summary" -ot "$dir/search_results.json" ] || [ "$summary" -ot "$conf" ]; then
        if [ -f "$summary" ]; then
            local prev=$dir/attack_prev_$(date -r "$summary" '+%Y%m%d_%H%M')
            mv "$dir/attack" "$prev" && echo "$(date '+%F %T') $cfg previous attack kept in $prev" >> $STATUS
        fi
        echo "$(date '+%F %T') $cfg attack start" >> $STATUS
        if $PY cli.py attack -c $conf --set attack.batch=$batch > "$dir/pipeline_attack.log" 2>&1; then
            echo "$(date '+%F %T') $cfg attack done" >> $STATUS
        else
            echo "$(date '+%F %T') $cfg attack FAILED (see $dir/pipeline_attack.log)" >> $STATUS
        fi
    else
        echo "$(date '+%F %T') $cfg attack already done" >> $STATUS
    fi
}

echo "$(date '+%F %T') pipeline start" >> $STATUS
run_model vgg19_cifar10 256
run_model yolo11n_coco 32
run_model yolo26n_coco 32
run_model resnet152_imagenet 64
run_model vgg19_imagenet_caffe2 64
run_model densenet_imagenet 64
echo "$(date '+%F %T') pipeline end" >> $STATUS
