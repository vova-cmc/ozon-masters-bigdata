#!/opt/conda/envs/dsenv/bin/python

import sys 
import re

for line in sys.stdin:
    res = re.findall('\w+', line.lower())
    for el in res:
        print(el + '\t1')