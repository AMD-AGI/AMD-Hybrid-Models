
# batches=(8 16 32 64 128 256 512 1024)
# model="8B"

# # Loop through models and generation lengths

# for batch in "${batches[@]}"; do
#     echo "Running benchmark for model $model with batchsize $batch"
#     python benchmark_generation_speed.py --model-size "$model" --genlen 8192 --batch $batch --promptlen 1 --num_attn 8 --repeats 1
# done



# batches=(48)
# model="8B"

# # Loop through models and generation lengths

# for batch in "${batches[@]}"; do
#     echo "Running benchmark for model $model with batchsize $batch"
#     python benchmark_generation_speed.py --model-size "$model" --genlen 8192 --batch $batch --promptlen 1 --num_attn 8 --repeats 1
# done



# contexts=(4096 8192 16384 32768 65536 131072 262144)
# model="8B"

# # Loop through models and generation lengths

# for context in "${contexts[@]}"; do
#     echo "Running benchmark for model $model with batchsize $batch"
#     python benchmark_generation_speed.py --model-size "$model" --genlen 1024 --batch 48 --promptlen $context --num_attn 8 --repeats 1 
# done


# contexts=(4096 8192 16384 32768 65536 131072 262144)
# model="8B"

# # Loop through models and generation lengths

# for context in "${contexts[@]}"; do
#     echo "Running benchmark for model $model with batchsize $batch"
#     python benchmark_generation_speed.py --model-size "$model" --genlen 1024 --batch 48 --promptlen $context --num_attn 16 --repeats 1 
# done


contexts=(131072)
model="8B"

# Loop through models and generation lengths

for context in "${contexts[@]}"; do
    echo "Running benchmark for model $model with batchsize $batch"
    python benchmark_generation_speed.py --model-size "$model" --genlen 1024 --batch 48 --promptlen $context --num_attn 32 --repeats 1 
done


