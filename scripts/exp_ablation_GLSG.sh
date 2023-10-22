num_runs=$1
gpu=$2

# by default, we run 5 times for every experiment
num_runs=${num_runs:-5}
gpu=${gpu:-0}


base_cmd="python train.py \
--dataset MSRVTT \
--arch base \
--method Transformer \
--task Concept \
--feats ViT \
--decoder_modality_flags VA \
--predictor_modality_flags VAT"

######################################
## GSG: None;       LSG: None
cmd="$base_cmd --use_attr_flags G0L0"
bash scripts/run.sh "$cmd" $num_runs $gpu

######################################
## GSG: Emb-Add;    LSG: None
cmd="$base_cmd --use_attr_flags G1L0"
bash scripts/run.sh "$cmd" $num_runs $gpu

## GSG: Semantic Composition (SC); LSG: None
cmd="$base_cmd --use_attr_flags G0L0 --compositional_intra --compositional_ffn --scope SC"
bash scripts/run.sh "$cmd" $num_runs $gpu

######################################
## GSG: Emb-Add;    LSG: Hybrid Attention
cmd="$base_cmd --use_attr_flags G1Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu

## GSG: Semantic Composition (SC); LSG: None
cmd="$base_cmd --use_attr_flags G0Lc --compositional_intra --compositional_ffn --scope SC --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu

######################################
## GSG: None;       LSG: Hybrid Attention
cmd="$base_cmd --use_attr_flags G0Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu

## GSG: None;       LSG: Hybrid Attention w/o Biases
cmd="$base_cmd --use_attr_flags G0Lc"
bash scripts/run.sh "$cmd" $num_runs $gpu

## GSG: None;       LSG: Cross -> Semantic
cmd="$base_cmd --use_attr_flags G0L1 --attr_layer_pos cross2attr --scope cross2semantic"
bash scripts/run.sh "$cmd" $num_runs $gpu

## GSG: None;       LSG: Semantic -> Cross
cmd="$base_cmd --use_attr_flags G0L1 --attr_layer_pos attr2cross --scope semantic2cross"
bash scripts/run.sh "$cmd" $num_runs $gpu

## GSG: None;       LSG: Cross || Semantic
cmd="$base_cmd --use_attr_flags G0L1 --attr_layer_pos parallel --scope parallel"
bash scripts/run.sh "$cmd" $num_runs $gpu

######################################
## GSG: Emb-Add;    LSG: Hybrid Attention
cmd="$base_cmd --use_attr_flags G1Lc --add_hybrid_attention_bias"
bash scripts/run.sh "$cmd" $num_runs $gpu

## GSG: Emb-Add;    LSG: Hybrid Attention w/o Biases
cmd="$base_cmd --use_attr_flags G1Lc"
bash scripts/run.sh "$cmd" $num_runs $gpu

## GSG: Emb-Add;    LSG: Cross -> Semantic
cmd="$base_cmd --use_attr_flags G1L1 --attr_layer_pos cross2attr --scope cross2semantic"
bash scripts/run.sh "$cmd" $num_runs $gpu

## GSG: Emb-Add;    LSG: Semantic -> Cross
cmd="$base_cmd --use_attr_flags G1L1 --attr_layer_pos attr2cross --scope semantic2cross"
bash scripts/run.sh "$cmd" $num_runs $gpu

## GSG: Emb-Add;    LSG: Cross || Semantic
cmd="$base_cmd --use_attr_flags G1L1 --attr_layer_pos parallel --scope parallel"
bash scripts/run.sh "$cmd" $num_runs $gpu
