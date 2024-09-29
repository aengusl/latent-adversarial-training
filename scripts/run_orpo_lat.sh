#!/bin/bash

# Array of GPUs
gpus=(0 1 2 3 4 5 6 7)

# Function to run the Python script with given arguments
run_experiment() {
    local twins=$1
    local dpo=$2
    local lora64=$3
    local gpu=$4
    
    echo "Running experiment: twins=$twins, dpo=$dpo, lora64=$lora64 on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/jailbreaks_script.py --twins $twins --dpo $dpo --lora64 $lora64 &
}

# Counter for GPU assignment
gpu_counter=0

# Run experiments for all combinations
for twins in true false; do
    for dpo in true false; do
        for lora64 in true false; do
            run_experiment $twins $dpo $lora64 ${gpus[$gpu_counter]}
            
            # Move to next GPU, wrapping around if necessary
            gpu_counter=$(( (gpu_counter + 1) % ${#gpus[@]} ))
            
            # Optional: add a small delay to stagger the starts
            sleep 0
        done
    done
done

# Wait for all background processes to finish
wait

echo "All experiments completed."