#!/bin/bash

# Function to run the Python script with given arguments
run_experiment() {
    local twins=$1
    local sft_1=$2
    local lora64=$3
    local gpu=$4
    
    echo "Running experiment: twins=$twins, sft_1=$sft_1, lora64=$lora64 on GPU $gpu"
    start_time=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu python scripts/orpo_lat.py --twins $twins --sft_1 $sft_1 --lora64 $lora64
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    echo "Experiment completed in $runtime seconds"
    return $runtime
}

# Run a single experiment
gpu=0
twins=true
sft_1=true
lora64=true

run_experiment $twins $sft_1 $lora64 $gpu
single_runtime=$?

# Calculate total runtime estimate
total_combinations=8  # 2 * 2 * 2
total_estimate=$((single_runtime * total_combinations))

echo "Estimated total runtime for all combinations: $total_estimate seconds ($(($total_estimate / 60)) minutes or $(($total_estimate / 3600)) hours)"