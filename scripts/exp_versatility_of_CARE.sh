num_runs=$1
gpu=$2

# by default, we run 5 times for every experiment
num_runs=${num_runs:-5}
gpu=${gpu:-0}

base_cmd_on_msvd="python train.py \
--dataset MSVD \
--arch base \
--feats ViT \
--modality mi \
--decoder_modality_flags V \
--predictor_modality_flags VT"

base_cmd_on_msrvtt="python train.py \
--dataset MSRVTT \
--arch base \
--feats ViT \
--modality ami \
--decoder_modality_flags VA \
--predictor_modality_flags VAT"

########## SALSTM
cmd="$base_cmd_on_msvd --method SALSTM --task Base" 
bash scripts/run.sh "$cmd" $num_runs $gpu
cmd="$base_cmd_on_msrvtt --method SALSTM --task Base" 
bash scripts/run.sh "$cmd" $num_runs $gpu

########## SALSTM + Our CARE
cmd="$base_cmd_on_msvd --method SALSTM --task CARE" 
bash scripts/run.sh "$cmd" $num_runs $gpu
cmd="$base_cmd_on_msrvtt --method SALSTM --task CARE" 
bash scripts/run.sh "$cmd" $num_runs $gpu

########## TopDown
cmd="$base_cmd_on_msvd --method TopDown --task Base" 
bash scripts/run.sh "$cmd" $num_runs $gpu
cmd="$base_cmd_on_msrvtt --method TopDown --task Base" 
bash scripts/run.sh "$cmd" $num_runs $gpu

########## TopDown + Our CARE
cmd="$base_cmd_on_msvd --method TopDown --task CARE" 
bash scripts/run.sh "$cmd" $num_runs $gpu
cmd="$base_cmd_on_msrvtt --method TopDown --task CARE" 
bash scripts/run.sh "$cmd" $num_runs $gpu

########## NACF
# Note: we need to train an auto-regressive teacher first
cmd="$base_cmd_on_msvd --method ARB --task Base" 
bash scripts/run.sh "$cmd" 1 $gpu
cmd="$base_cmd_on_msrvtt --method ARB --task Base" 
bash scripts/run.sh "$cmd" 1 $gpu

cmd="$base_cmd_on_msvd --method NACF --task Base --with_teacher_during_training" 
bash scripts/run.sh "$cmd" $num_runs $gpu
cmd="$base_cmd_on_msrvtt --method NACF --task Base --with_teacher_during_training" 
bash scripts/run.sh "$cmd" $num_runs $gpu

########## NACF + Our CARE
# Note: we need to train an auto-regressive teacher first
cmd="$base_cmd_on_msvd --method ARB --task CARE" 
bash scripts/run.sh "$cmd" 1 $gpu
cmd="$base_cmd_on_msrvtt --method ARB --task CARE" 
bash scripts/run.sh "$cmd" 1 $gpu

cmd="$base_cmd_on_msvd --method NACF --task CARE --with_teacher_during_training" 
bash scripts/run.sh "$cmd" $num_runs $gpu
cmd="$base_cmd_on_msrvtt --method NACF --task CARE --with_teacher_during_training" 
bash scripts/run.sh "$cmd" $num_runs $gpu

########## PGN
cmd="$base_cmd_on_msvd --method PointerGen --task Base" 
bash scripts/run.sh "$cmd" $num_runs $gpu
cmd="$base_cmd_on_msrvtt --method PointerGen --task Base" 
bash scripts/run.sh "$cmd" $num_runs $gpu

########## PGN + Our CARE
cmd="$base_cmd_on_msvd --method PointerGen --task CARE" 
bash scripts/run.sh "$cmd" $num_runs $gpu
cmd="$base_cmd_on_msrvtt --method PointerGen --task CARE" 
bash scripts/run.sh "$cmd" $num_runs $gpu

########## SwinBERT
# Note: we do not train the model in an end-to-end manner like SwinBERT
# Instead, we conduct experiments on features extracted from fine-tuned SwinBERT checkpoints
cmd="$base_cmd_on_msvd --method Transformer --task Base --feats SwinBERTDense" 
bash scripts/run.sh "$cmd" $num_runs $gpu
cmd="$base_cmd_on_msrvtt --method Transformer --task Base --feats SwinBERTDense" 
bash scripts/run.sh "$cmd" $num_runs $gpu

########## SwinBERT + Our CARE
cmd="$base_cmd_on_msvd --method Transformer --task CARE --feats SwinBERTDense" 
bash scripts/run.sh "$cmd" $num_runs $gpu
cmd="$base_cmd_on_msrvtt --method Transformer --task CARE --feats SwinBERTDense" 
bash scripts/run.sh "$cmd" $num_runs $gpu
