#!/bin/bash
dataset_path="/data/gpfs/projects/punim0512/data/MA_ADT/"
datasets=(
    room0_agent_0
    room0_agent_1
#    room0_agent_2
#    room1_agent_0
#    room1_agent_1
#    room1_agent_2
)

no_calib=false
print_only=false
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-calib)
            no_calib=true
            ;;
        --print)
            print_only=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$print_only" = false ]; then
    for dataset in ${datasets[@]}; do
#        dataset_name="$dataset_path""$dataset"/
        scene=$(echo "$dataset" | cut -d'_' -f1)
        agent=$(echo "$dataset" | cut -d'_' -f2-)

        # Reconstruct the new folder structure: room0/agent_0/
        full_dataset_path="${dataset_path}${scene}/${agent}/results/"
        echo "Processing dataset: $full_dataset_path"
        if [ "$no_calib" = true ]; then
            python main.py --dataset $full_dataset_path --no-viz --save-as MA_ADT/no_calib/$dataset --config config/eval_no_calib.yaml
        else
            python main.py --dataset $full_dataset_path --no-viz --save-as MA_ADT/calib/$dataset --config config/eval_calib.yaml
        fi
    done
fi

for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path""$dataset"/
    echo ${dataset_name}
    if [ "$no_calib" = true ]; then
        evo_ape tum groundtruths/MA_ADT/$dataset.txt logs/MA_ADT/no_calib/$dataset/results.txt -as
    else
        evo_ape tum groundtruths/MA_ADT/$dataset.txt logs/MA_ADT/calib/$dataset/results.txt -as
    fi

done
