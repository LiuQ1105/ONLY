# AD-HH + ONLY on LLaVA-v1.5-7B

This integration keeps ONLY's selected-layer `hidden_states_cd` path and adds
AD-HH to the regular LLaMA attention output. The intervention runs after
softmax and before the attention/value product. It checks only the final query
row, which is the newly generated token during cached decoding.

## Head list

Pass the AD-HH heads as comma-separated `layer:head` pairs, for example
`20:3,23:11`. Replace this example with the actual head list for the checkpoint
being evaluated. Layer and head indices are zero-based.

## POPE

The POPE script expects one image per example. `--adhh_img_length` defaults to
576, the standard single-image LLaVA-v1.5-7B visual sequence length. Confirm
this against the actual multimodal sequence in your checkout before reporting
results. The image start position is taken from the `IMAGE_TOKEN_INDEX`
sentinel in the prompt for each example.

```bash
python eval_bench/pope_eval_llava.py \
  --model_path /path/to/llava-v1.5-7b \
  --pope_path /path/to/coco_pope_adversarial.json \
  --data_path /path/to/coco/val2014 \
  --log_path ./logs/pope/only_adhh \
  --type adversarial \
  --temperature 0 \
  --use_only true \
  --enhance_layer_index 16 \
  --use_adhh true \
  --adhh_threshold 0.4 \
  --adhh_heads '20:3,23:11,27:4' \
  --adhh_img_length 576
```

Set `--use_only false` for AD-HH-only, or `--use_adhh false` for ONLY-only.
Set both to false for the baseline.

## MME

Use the LLaVA MME evaluator and pass the same intervention options. The shell
script in this repository defaults to another model family, so invoke the
LLaVA evaluator directly or update the shell script's model selection.

```bash
python experiments/eval/mme_llava.py \
  --model-path /path/to/llava-v1.5-7b \
  --question-file /path/to/MME_Benchmark_release_version/mme_hallucination.jsonl \
  --image-folder /path/to/MME_Benchmark_release_version \
  --answers-file ./experiments/output/llava_only_adhh_mme_answers.jsonl \
  --use_only true \
  --enhance_layer_index 16 \
  --use_adhh true \
  --adhh_threshold 0.4 \
  --adhh_heads '20:3,23:11,27:4' \
  --adhh_img_length 576
```

## Initial ablation

Run four configurations with identical checkpoint, data order, prompts,
decoding settings, and seeds:

| Run | `--use_only` | `--use_adhh` |
|---|---:|---:|
| Baseline | false | false |
| ONLY | true | false |
| AD-HH | false | true |
| Joint | true | true |

Report POPE accuracy, precision, recall, F1, and yes ratio; report the MME
hallucination score and its subtask results. Also record the model/repository
revision, AD-HH head list, threshold, ONLY layer, image token interval, decoding
settings, and attention backend. A random-head control should use the same
number of heads as the AD-HH list.
