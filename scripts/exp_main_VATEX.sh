num_runs=$1
gpu=$2

# by default, we run 5 times for every experiment
num_runs=${num_runs:-5}
gpu=${gpu:-0}

# Note: 
# The official splits of VATEX is 25,991: 3,000: 6,000. 
# However, some video clips are no longer available, 
# resulting in the splits 25,006: 2,893: 5,792 (in our case).
# Therefore, we complete the predictions of the missing 208 testing videos 
# with the predictions yielded by the model trained with officially released I3D features.

# ./data/VATEX_I3D_preds.json is generated as follows:
# python train.py --method Transformer --devices 0 --seed 0 --feats I3D --task Base --dataset VATEX --arch large
# python translate.py --checkpoint_paths ./exps/VATEX/Transformer/Base/large_I3D_m/best.ckpt --json_path ./data --json_name VATEX_I3D_preds.json

base_cmd="python train.py \
--dataset VATEX \
--method Transformer \
--VATEX_I3D_preds_json ./data/VATEX_I3D_preds.json"


# CARE with different architectures and feature combinations
cmd="$base_cmd --task CARE --arch median --feats IRv2 --decoder_modality_flags V --predictor_modality_flags VT"
bash scripts/run.sh "$cmd" $num_runs $gpu

cmd="$base_cmd --task CARE --arch median --feats ViT --decoder_modality_flags VA --predictor_modality_flags VAT"
bash scripts/run.sh "$cmd" $num_runs $gpu

cmd="$base_cmd --task CARE --arch large --feats ViT --decoder_modality_flags VA --predictor_modality_flags VAT"
bash scripts/run.sh "$cmd" $num_runs $gpu

# CA-Baseline
cmd="$base_cmd --task CABase --arch median --feats ViT --decoder_modality_flags VA"
bash scripts/run.sh "$cmd" $num_runs $gpu

# Baseline
## `--modality ami` is equivalent to `--decoder_modality_flags VA`
## but the definition of the task `Base` requires `--modality`
cmd="$base_cmd --task Base --arch median --feats ViT --modality ami"
bash scripts/run.sh "$cmd" $num_runs $gpu
