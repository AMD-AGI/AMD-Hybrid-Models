
This code is adapted from [here](https://github.com/Leooyii/LCEG/tree/main/needle). If you use this code, please consider citing the original paper.

### How to Run Needle Test

```
export MODEL_NAME=/home/guihonli@amd.com/project/amd-mla-main/kv_128_q_1344_dim32_8b_teacher-dpo
export RESULT_SAVE_PATH=kv_128_q_1344_dim32_8b_teacher-dpo-needle
python -u benchmark/needle/needle_in_haystack.py --s_len 0 --e_len 65536\
    --model_provider MLA \
    --model_path ${MODEL_NAME} \
    --test_name ${RESULT_SAVE_PATH} 
```

Notice that, during the distillation, we only train model with 2k context.

Here is the results

<img src="img/needle.png" alt="needle">
