# distutils: language = c++

from LPboxADMMsolver cimport LPboxADMMsolver
cimport numpy as np 
import numpy as np

cdef class PyLPboxADMMsolver:
    cdef LPboxADMMsolver solver 

    def __cinit__(self, int consistency, double fix_threshold):
        self.solver = LPboxADMMsolver(consistency, fix_threshold)

    def __cinit__(self, int print_info):
        self.solver = LPboxADMMsolver(print_info)

    def read_File(self, int i, int k, int j):
        return self.solver.readFile(i,k,j)

    def solve_init(self):
        return self.solver.ADMM_lp_iters_init()

    def solve_iter(self, int i, int j):
        return self.solver.ADMM_lp_iters(i, j)

    def cal_Obj(self):
        return self.solver.cal_obj()

    def get_curBinObj(self):
        return self.solver.get_curBinObj()

    def solve_iter_l2f(self, int i, int j, np.ndarray[np.double_t, ndim=1] vec, int num):
        return self.solver.ADMM_lp_iters_l2f(i, j, &vec[0], num)


    def get_x_iters_1d(self,ws):
        cdef double* x = self.solver.get_x_iters_d(ws)
        cdef int n = self.solver.get_n()
        cdef np.ndarray ret = np.zeros([n*20,1], dtype=np.double)
        for i in range(n*20):
            ret[i] = x[i]
        return ret
    
    def get_x_iters_2d(self, ws):
        cdef double* x = self.solver.get_x_iters_d(ws)
        cdef int n = self.solver.get_n()
        cdef np.ndarray ret = np.zeros([n,ws], dtype=np.double)
        for i in range(n):
            for j in range(ws):
                ret[i][j] = x[i*ws + j]
        return ret

    def get_n(self):
        return self.solver.get_n()

    def get_iter(self):
        return self.solver.get_iter()

    def get_x_sol(self, n):
        cdef double* x = self.solver.get_x_sol()
        cdef np.ndarray ret = np.zeros([n,1], dtype=np.double)
        for i in range(n):
            ret[i] = x[i]
        return ret 

    def get_final_x_sol(self, n):
        cdef double* x = self.solver.get_final_x_sol()
        cdef np.ndarray ret = np.zeros([n,1], dtype=np.double)
        for i in range(n):
            ret[i] = x[i]
        return ret 

    def check_infeasible_lpbox(self):
        return self.solver.check_infeasible_lpbox()

    def check_infeasible_l2f(self):
        return self.solver.check_infeasible_l2f()
  


    