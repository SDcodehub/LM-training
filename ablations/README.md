### Ablations folder

This folder holds small, reproducible shell scripts for ablation experiments. Each script:
- sets a WandB `GROUP` to collect its runs together
- defines a baseline and one or more variants
- calls `LM_training/scripts/train.py` with the same shared args for fairness

How to use:
- Update the `DATA_ARGS` paths in the script(s) to your `.npy` dataset files.
- Optionally adjust the `PROJECT` name to your WandB project.
- Run a script, e.g.:

```bash
bash ablations/ablation_lr.sh
```

In the WandB UI:
- Filter by `group:<GROUP_NAME>` or enable grouping by "Group"
- Add a Parallel Coordinates chart to compare `hyperparameters -> metrics`

Scripts included:
- `ablation_template.sh`: copy, edit, and reuse for custom ablations
- `ablation_lr.sh`: compares a few learning rates
- `ablation_depth.sh`: compares `num_layers`
- `ablation_width.sh`: compares width (`d_model`, `num_heads`, `d_ff`)
- `ablation_context.sh`: compares `context_length`


