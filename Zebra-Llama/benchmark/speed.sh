
# batches=(8 16 32 64 128 256 512 1024)
# model="8B"

# # Loop through models and generation lengths

# for batch in "${batches[@]}"; do
#     echo "Running benchmark for model $model with batchsize $batch"
#     python benchmark_generation_speed.py --model-size "$model" --genlen 8192 --batch $batch --promptlen 1 --num_attn 8 --repeats 1
# done



batches=(2048 4096)
model="8B"

# Loop through models and generation lengths

for batch in "${batches[@]}"; do
    echo "Running benchmark for model $model with batchsize $batch"
    python benchmark_generation_speed.py --model-size "$model" --genlen $batch --batch 8 --promptlen 1 --num_attn 32 --repeats 1
done 