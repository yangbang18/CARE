num_runs=$1
gpu=$2

# by default, we run 5 times for every experiment
num_runs=${num_runs:-5}
gpu=${gpu:-0}


base_cmd="python train.py \
--dataset MSVD \
--arch base \
--method Transformer"

# CARE with different feature combinations
cmd="$base_cmd --task CARE --feats R101 --decoder_modality_flags V --predictor_modality_flags VT"
bash scripts/run.sh "$cmd" $num_runs $gpu

cmd="$base_cmd --task CARE --feats IRv2 --decoder_modality_flags V --predictor_modality_flags VT"
bash scripts/run.sh "$cmd" $num_runs $gpu

cmd="$base_cmd --task CARE --feats IRv2 --decoder_modality_flags I --predictor_modality_flags IT"
bash scripts/run.sh "$cmd" $num_runs $gpu

cmd="$base_cmd --task CARE --feats ViT --decoder_modality_flags V --predictor_modality_flags VT"
bash scripts/run.sh "$cmd" $num_runs $gpu

# CA-Baseline
cmd="$base_cmd --task CABase --feats ViT --decoder_modality_flags V"
bash scripts/run.sh "$cmd" $num_runs $gpu

# Baseline
## `--modality mi` is equivalent to `--decoder_modality_flags V`
## but the definition of the task `Base` requires `--modality`
cmd="$base_cmd --task Base --feats ViT --modality mi"
bash scripts/run.sh "$cmd" $num_runs $gpu
