import lpbox 
import time
import numpy as np 

def fix(i):
    solver = lpbox.PyLPboxADMMsolver(0)  # 0-donot print fix info; 1-print fix info; 2-get xiters.
    solver.read_File(i, 100, 500)
    solver.solve_init()
    time_begin = time.time()
    solver.solve_iter(0,1e4)
    obj = solver.cal_Obj()
    time_end = time.time()
    t = time_end-time_begin
    print(f'Objective: {-obj}. time elapsed in python: {time_end-time_begin}\n')
    return t 

if __name__ == "__main__":
    for i in range(1,3):
        fix(i)
