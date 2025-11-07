#!/bin/bash
# base_dataset_path="/data/gpfs/projects/punim0512/data/MA_ADT/"
#datasets=(
#    room0_agent_0
#    room0_agent_1
#    room0_agent_2
#    room1_agent_0
#    room1_agent_1
#    room1_agent_2
#)

base_dataset_path="/data/gpfs/projects/punim0512/data/MA_Replica/"
datasets=(
#    office0_agent_0
#    office0_agent_1
#    apart0_agent_0
#    apart0_agent_1
#    apart1_agent_0
#    apart1_agent_1
    apart2_agent_0
    apart2_agent_1
)

MA_dir=$(basename "$base_dataset_path")
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

#if [ "$print_only" = false ]; then
#        if [ "$no_calib" = true ]; then
#            python main.py --base_dataset_path "$base_dataset_path"  --dataset "${datasets[@]}" --no-viz --save-as ${MA_dir}/no_calib/ --config config/eval_no_calib.yaml
#        else
#            python main.py --base_dataset_path "$base_dataset_path" --dataset "${datasets[@]}" --no-viz --save-as ${MA_dir}/calib/ --config config/eval_calib.yaml
#        fi
#fi

chmod +x ./mast3r_slam/evaluation_reconstruction.py
for dataset in ${datasets[@]}; do
    dataset_name="$base_dataset_path""$dataset"/
    echo ${dataset_name}
    if [ "$no_calib" = true ]; then
        evo_ape tum groundtruths/${MA_dir}/$dataset.txt logs/${MA_dir}/no_calib/$dataset/results.txt -as
    else
      evo_ape tum groundtruths/${MA_dir}/$dataset.txt logs/${MA_dir}/calib/$dataset/results.txt -as
      python ./mast3r_slam/evaluation_reconstruction.py --base_dataset_path "$base_dataset_path" \
            --dataset "${dataset}" --GT groundtruths/${MA_dir}/$dataset.txt \
             --ResDir logs/${MA_dir}/no_calib/$dataset
    fi

done
