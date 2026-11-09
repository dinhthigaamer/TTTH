# Read instance from file in instances folder
import json
from common import Instance, UAV, Solution, evaluate
from svrec import svrec
from na_greedy import na_greedy
from heu_greedy import heu_greedy
from zigzag import zigzag
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PAR_PATH = {
    '../../instances': 'summary_under_50.csv',
    '../../rescuenet_50x50/test': 'summary_50x50.csv',
    '../../rescuenet_100x100/test': 'summary_100x100.csv',
}


def read_instance(file_path):
    with open(file_path, 'r') as f:
        instance = json.load(f)
    return instance


def extract_instance_data(data) -> Instance:
    # Extract relevant data from the instance
    grid = data['grid']

    instance = Instance(
        w=grid["w"],
        u=grid["u"],
        D=data["D"],
        eta=data["eta"],
        uavs=[UAV(depot=tuple(uav["depot"]), B=uav["B"], v_h=uav["v_h"],
                  e_h=uav["e_h"], tau=uav["tau"], eps=uav["eps"]) for uav in data["uavs"]]
    )
    return instance


def eval(instance: Instance, solution: Solution):
    return evaluate(instance, solution.routes)


def summarize_solution():
    files = []
    res = {
        "svrec": [],
        "na_greedy": [],
        "heu_greedy": [],
        "zigzag": []
    }

    for path, summary_file in PAR_PATH.items():
        for file in tqdm(Path(path).rglob('*.json'), desc=f"Processing files in {path}"):
            file_path = file
            instance = extract_instance_data(read_instance(file_path))

            sol_svrec = svrec(instance)
            res_svrec = eval(instance, sol_svrec)

            sol_na_greedy = na_greedy(instance)
            res_na_greedy = eval(instance, sol_na_greedy)

            sol_heu_greedy = heu_greedy(instance)
            res_heu_greedy = eval(instance, sol_heu_greedy)

            sol_zigzag = zigzag(instance)
            res_zigzag = eval(instance, sol_zigzag)

            files.append(file)
            res['svrec'].append(res_svrec['f'])
            res['na_greedy'].append(res_na_greedy['f'])
            res['heu_greedy'].append(res_heu_greedy['f'])
            res['zigzag'].append(res_zigzag['f'])
            # print(f"Instance: {file}, Evaluation Result: {res}")

        df = pd.DataFrame({'Instance': files, 'svrec': res['svrec'], 'na_greedy': res['na_greedy'],
                           'heu_greedy': res['heu_greedy'], 'zigzag': res['zigzag']})
        df.to_csv(summary_file, index=False)


def main():
    # Example usage
    # Replace with your instance file path
    # file_path = '../../instances/rescuenet_10836_30x40_K3_b10_s3.json'
    # instance = extract_instance_data(read_instance(file_path))

    # solution = svrec(instance)
    # res = eval(instance, solution)
    # print(res)
    summarize_solution()


if __name__ == '__main__':
    main()
