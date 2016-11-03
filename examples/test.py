# import pulp
# pulp.pulpTestAll()

import matplotlib.pyplot as plt
import numpy as np
import os
readfile = './data/dynamic_agents_partial_cooporation.txt'
print(readfile.split("/")[-1])
print(readfile.strip(".txt"))
print(readfile)
print(os.path.splitext(readfile)[0])