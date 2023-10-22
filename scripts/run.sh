cmd=$1
num_runs=$2
gpu=$3

# by default, we run 5 times for every experiment
num_runs=${num_runs:-5}
gpu=${gpu:-0}

for ((seed=0; seed<$num_runs; seed++))
do
    this_cmd="$cmd --seed $seed --devices $gpu"
    echo cmd: $this_cmd
    eval $this_cmd
done
