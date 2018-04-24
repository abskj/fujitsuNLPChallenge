import sys
import json
import numpy as np
from scipy.stats import rankdata
from functools import reduce

def print_help():
    print("Usage: python evaluation.py <json>")
    print("Calculate MRR(Mean Reciprocal Rank) about given json lines file.")
    print("The json lines file need to follow the format specified by Fujitsu.")

def calc_mrr(rank):
    rank = list(map(lambda x: 1./x, rank))
    return np.mean(rank)

def main(fname):
    data = []
    with open(fname) as f:
        for line in f.readlines():
            data += [json.loads(line)]
    rank = []
    for elem in data:
        cand = elem['candidates']
        rank += rankdata(elem['results'])[cand].tolist()
    mrr = calc_mrr(rank)
    print("MRR: {}".format(mrr))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_help()
        sys.exit()
    fname = sys.argv[1]
    main(fname)
