num_runs=$1
gpu=$2

# by default, we run 5 times for every experiment
num_runs=${num_runs:-5}
gpu=${gpu:-0}


base_cmd="python train.py \
--dataset MSRVTT \
--arch base \
--method Transformer \
--modality ami \
--decoder_modality_flags VA"

# Using ImageNet Pre-Trained Image Encoder
## V + A + T for concept detection, with GSG, with LSG
cmd="$base_cmd --task Concept --feats R101 --predictor_modality_flags VAT --use_attr_flags G1Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu
## V + T for concept detection, with GSG, with LSG
cmd="$base_cmd --task Concept --feats R101 --predictor_modality_flags VT --use_attr_flags G1Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu
## V + A for concept detection, with GSG, with LSG
cmd="$base_cmd --task Concept --feats R101 --predictor_modality_flags VA --use_attr_flags G1Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu
## V for concept detection, with GSG, with LSG
cmd="$base_cmd --task Concept --feats R101 --predictor_modality_flags V --use_attr_flags G1Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu

## V + A + T for concept detection, without GSG, with LSG
cmd="$base_cmd --task Concept --feats R101 --predictor_modality_flags VAT --use_attr_flags G0Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu
## V + A + T for concept detection, with GSG, without LSG
cmd="$base_cmd --task Concept --feats R101 --predictor_modality_flags VAT --use_attr_flags G1L0"
bash scripts/run.sh "$cmd" $num_runs $gpu
## V + A + T for concept detection, without GSG, without LSG
cmd="$base_cmd --task Concept --feats R101 --predictor_modality_flags VAT --use_attr_flags G0L0"
bash scripts/run.sh "$cmd" $num_runs $gpu

## Baseline
cmd="$base_cmd --task Base --feats R101"
bash scripts/run.sh "$cmd" $num_runs $gpu


# Using CLIP's Image Encoder
## V + A + T for concept detection, with GSG, with LSG
cmd="$base_cmd --task Concept --feats ViT --predictor_modality_flags VAT --use_attr_flags G1Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu
## V + T for concept detection, with GSG, with LSG
cmd="$base_cmd --task Concept --feats ViT --predictor_modality_flags VT --use_attr_flags G1Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu
## V + A for concept detection, with GSG, with LSG
cmd="$base_cmd --task Concept --feats ViT --predictor_modality_flags VA --use_attr_flags G1Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu
## V for concept detection, with GSG, with LSG
cmd="$base_cmd --task Concept --feats ViT --predictor_modality_flags V --use_attr_flags G1Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu

## V + A + T for concept detection, without GSG, with LSG
cmd="$base_cmd --task Concept --feats ViT --predictor_modality_flags VAT --use_attr_flags G0Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu
## V + A + T for concept detection, with GSG, without LSG
cmd="$base_cmd --task Concept --feats ViT --predictor_modality_flags VAT --use_attr_flags G1L0"
bash scripts/run.sh "$cmd" $num_runs $gpu
## V + A + T for concept detection, without GSG, without LSG
cmd="$base_cmd --task Concept --feats ViT --predictor_modality_flags VAT --use_attr_flags G0L0"
bash scripts/run.sh "$cmd" $num_runs $gpu

## Baseline
cmd="$base_cmd --task Base --feats ViT"
bash scripts/run.sh "$cmd" $num_runs $gpu
