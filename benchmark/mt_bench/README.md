**MT-Bench**

[MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench): a multi-turn benchmark spanning 80 dialogues and 10 domains.

### Install

```
cd FastChat
pip install -e ".[model_worker,llm_judge]"
```

You can use `python show_result.py` to display our results.

Note that the OpenAI requirements for MT-bench and AlpacaEval are different. If you use the same conda environment, please ensure you use `openai==0.28.0`.

### Models
TBD