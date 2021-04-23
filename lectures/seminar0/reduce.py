#!/opt/conda/envs/dsenv/bin/python

import sys 
res = {}
for line in sys.stdin:
    tmp = line.split('\t')
    res[tmp[0]] = res.get(tmp[0], 0) + int(tmp[1])
for el in res.items():
    print(f"{el[0]}\t{el[1]}")