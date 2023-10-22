# refer to notebooks/retrieval_robustness.ipynb for an example

path=$1
gpu=$2

for ratio in 0.1 1 10
do
echo $ratio
    for name in 'best.ckpt' 'best-v1.ckpt' 'best-v2.ckpt' 'best-v3.ckpt' 'best-v4.ckpt'
    do
        CUDA_VISIBLE_DEVICES=$gpu python translate.py -cp $path/$name --retrieval_db_ratio $ratio --save_csv --csv_name "retrieval_db_ratio_${ratio}.csv" --mode test
    done
done
