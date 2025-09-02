## Evaluations

To run evaluations of models in the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor) library:

1. Clone the `lm-evaluation-harness` repository by running `git clone https://github.com/EleutherAI/lm-evaluation-harness.git`. Use the `big-refactor` branch by running `git checkout big-refactor`.
2. Follow the number of shots specified in `https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard`. For example, for the teacher model HuggingFaceH4/zephyr-7b-beta, their configuration is [here](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta).
3. Run the evaluation with the following command (more documentation is available in the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor) repository):

```
python evals/lm_harness_eval.py --model mla_hybrid --model_args pretrained=[MODEL_PATH] --tasks mmlu --num_fewshot 5 --device cuda --batch_size 16
```

```
python benchmark/llm_eval/lm_harness_eval.py --model mla_hybrid --model_args pretrained=PATH_TO_MLA_MODEL --tasks mmlu,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,pubmedqa,race --num_fewshot 0 --device cuda --batch_size 16
```

For teacher model [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), use this command to run

```
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct --tasks mmlu,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,pubmedqa,race --num_fewshot 0 --device cuda --batch_size 16
```

To evaluate few-shot results on the OpenLLM leaderboard, please refer to the following script.

```
python benchmark/llm_eval/lm_harness_eval.py --model mla_hybrid \
    --model_args pretrained=PATH_TO_MLA_MODEL \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --device cuda \
    --batch_size 4
```

