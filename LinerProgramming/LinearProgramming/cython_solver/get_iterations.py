# Get iterative values for each variables, this is recorded for training.  
# The result is saved in path: ./data/xiter/
# For instance, given 500 variables and 8000 iterations, the file will give a iterative matrix with dimension 500 X 8000. 

import lpbox 
import time
import numpy as np 
import argparse

def fix(i,j,k):
    solver = lpbox.PyLPboxADMMsolver(2)  # 0-donot print fix info; 1-print fix info; 2-get xiters.
    solver.read_File(i, j, k)
    solver.solve_init()
    time_begin = time.time()
    solver.solve_iter(0,1e4)
    obj = solver.cal_Obj()
    time_end = time.time()
    t = time_end-time_begin
    print(f'Objective: {-obj}. time elapsed in python: {time_end-time_begin}\n')
    return t 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1, help="number of instances")
    parser.add_argument('-j', type=int, default=1, help="number of items")
    parser.add_argument('-k', type=int, default=1, help="number of bids")
    args = parser.parse_args()

    for i in range(args.n):
        fix(i+1, args.j, args.k)
