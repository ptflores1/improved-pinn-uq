import os
import subprocess
import argparse
import json
from multiprocessing import Pool
from itertools import product

parser = argparse.ArgumentParser(description='Run a method on a given equation.')

parser.add_argument("--ignore-snapshot", action="store_true", help="Whether to ignore the snapshot of the experiments.")
cli_args = parser.parse_args()

methods = ["fcnn", "nlm", "bbb", "hmc"]
equations = ["lcdm", "cpl", "quintessence", "hs", "lab", "cocoa"]
bundle_options = ["-b", ""]
errorbounds_options = ["-eb", ""]
ignore_snapshot = cli_args.ignore_snapshot

snapshot = None
if os.path.exists("logs/snapshot.json"):
    with open("logs/snapshot.json", "r") as f:
        snapshot = json.load(f)

def valid_experiment(x):
    if x[0] in ["quintessence", "hs", "lab", "cocoa"] and x[3] == "-eb":
        return False
    if x[0] == "cocoa" and x[2] == "-b":
        return False
    if x[1] == "fcnn" and x[3] == "-eb":
        return False
    return True

args = list(map(list, filter(valid_experiment, product(equations, methods, bundle_options, errorbounds_options))))
for i, arg in enumerate(args):
    arg.append(i + 1)

def run(command_args):
    n = command_args[-1]
    command_args = list(filter(bool, command_args))[:-1]
    inverse = ["-i=1"] if "-b" in command_args else ["-i=0"]
    command_args += inverse

    if not ignore_snapshot and snapshot is not None and snapshot.get(" ".join(command_args), False): # Ignore successful experiments
        return ' '.join(command_args)

    with open("logs/" + f"{n:02d}_" + " ".join(command_args) + ".log", "w") as f:
        res = subprocess.run(["python3", "main.py"] + command_args, stdout=f, stderr=f)
    return res

if __name__ == "__main__":
    with Pool() as p:
        results = p.map(run, args)
    
    snapshot = {}
    for i, result in enumerate(results):
        if isinstance(result, str):
            print(f"{i+1:02d}", f"Skipping {result}")
            snapshot[result] = True
        else:
            print(f"{i+1:02d}", "ERROR" if result.returncode != 0 else "SUCCESS", " ".join(result.args))
            snapshot[" ".join(result.args[2:])] = result.returncode == 0

    with open("logs/snapshot.json", "w") as f:
        json.dump(snapshot, f)
