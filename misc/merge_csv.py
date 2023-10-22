import argparse
import glob
import os
from pprint import pprint

import pandas
pandas.set_option("display.precision", 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="MSVD", choices=['MSVD', 'MSRVTT', 'VATEX'])
    parser.add_argument("-ss", "--skip_scopes", nargs='+', type=str, default=['test_'],
                        help="skip scopes whose names are exact one of these")
    parser.add_argument("-sm", "--skip_models", nargs='+', type=str, default=[])
    parser.add_argument("-tasks", "--tasks", nargs='+', type=str, default=[])
    parser.add_argument("-sv", "--sort_values", nargs='+', type=str, default=['model_name', 'task_name', 'scope_name'])
    parser.add_argument("-name", "--output_name", type=str, default="",
                        help="output file name.")
    parser.add_argument("--output_path", type=str, default='./results')
    parser.add_argument("--csv_name", type=str, default="test_result.csv", )
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--base_path", type=str, default='./exps')

    parser.add_argument('-a', '--average', action='store_true')

    parser.add_argument("-ok", "--only_keep", type=str, nargs='+', default=[])
    parser.add_argument('-isin', '--seed_is_in', type=int, nargs='+', default=[])
    args = parser.parse_args()

    if not args.output_name:
        if len(args.only_keep):
            args.output_name = '_'.join(args.only_keep)
        else:
            args.output_name = 'merged_all_csv'

    BASE_PATH = os.path.join(args.base_path, args.dataset)
    # find path
    path = os.path.join(BASE_PATH, f"*/*/*/{args.csv_name}")
    models_paths = glob.glob(path)
    models_paths = sorted(models_paths)

    # skip some file
    new_paths = []
    for path in models_paths:
        ps = path.split("/")
        model_name, task_name, scope_name = ps[3:6]
        if model_name in args.skip_models:
            continue
        if scope_name in args.skip_scopes:
            continue
        if len(args.tasks) and task_name not in args.tasks:
            continue
        if len(args.only_keep):
            flags = [1 for item in args.only_keep if item in scope_name]
            if sum(flags) != len(args.only_keep):
                continue
        print(ps)
        new_paths.append(path)
    models_paths = new_paths

    # merge
    csv_data = []
    for i, path in enumerate(models_paths):
        ps = path.split("/")
        model_name, task_name, scope_name = ps[3:6]
        csv_df = pandas.read_csv(path)
        if args.average:
            if len(args.seed_is_in):
                csv_df = csv_df[csv_df['seed'].isin(args.seed_is_in)]

            csv_df[["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "Sum", "novel", "unique"]] *= 100

            if 'mAP' in csv_df.columns:
                csv_df[['mAP']] *= 100
                
            n_runs = len(csv_df)
            mean_values = csv_df.mean().round(1).astype(str)
            std_values = csv_df.std().round(2).astype(str)

            csv_df = mean_values + ' (' + std_values + ')'
            csv_df = pandas.DataFrame(csv_df)
            csv_df = pandas.DataFrame(csv_df.values.T, index=csv_df.columns, columns=csv_df.index)
            csv_df.insert(0, "n_runs", str(n_runs))    

        csv_df.insert(0, "model_name", model_name)
        csv_df.insert(1, "task_name", task_name)
        csv_df.insert(2, "scope_name", scope_name)

        csv_data.append(csv_df)
        # print(csv_df)
    assert len(csv_data) > 0, f"No test data in `experiments` dir for dataset `{args.dataset}`"
    all_df = pandas.concat(csv_data).sort_values(args.sort_values)
    
    if not args.average:
        all_df = all_df.round(args.round)
        all_df[["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "Sum", "novel", "unique"]] *= 100
        all_df["Sum"] = all_df["Bleu_4"] + all_df["METEOR"] + all_df["ROUGE_L"] + all_df["CIDEr"]
    
    pprint(all_df)
    
    output_file_name = args.output_name if ".csv" in args.output_name else args.output_name + ".csv"
    
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        output_path = args.output_path
    else:
        output_path = BASE_PATH

    output_path = os.path.join(output_path, f"{output_file_name}")
    all_df.to_csv(output_path, index=False)

'''
python misc/merge_csv.py --base_path ./exps --output_path ~/merge_results -tasks MCD_SG -a -name VA_diff_use_attr
'''
