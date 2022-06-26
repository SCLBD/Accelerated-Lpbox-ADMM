# distutils: language = c++
# cython: language_level=3

from LPboxADMMsolver cimport LPboxADMMsolver
cimport numpy as np 
import numpy as np

cdef class PyLPboxADMMsolver:
    cdef LPboxADMMsolver solver 

    def __cinit__(self, int print_info):
        self.solver = LPboxADMMsolver(print_info)

    def __cinit__(self, int print_info, int numNodes, int problem):
        self.solver = LPboxADMMsolver(print_info, numNodes, problem)

    def solve_init(self):
        return self.solver.ADMM_bqp_unconstrained_init()

    def solve_iter(self):
        return self.solver.ADMM_bqp_unconstrained_legacy()

    def solve_iter_l2f(self, int i, int j, np.ndarray[np.double_t, ndim=1] vec, int num):
        return self.solver.ADMM_bqp_unconstrained_l2f(i, j, &vec[0], num)

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

    def get_org_n(self):
        return self.solver.get_org_n()

    def get_obj(self):
        return self.solver.get_final_obj()

    def get_x_sol(self):
        cdef double* x = self.solver.get_x_sol()
        cdef int n = self.solver.get_org_n()
        cdef np.ndarray ret = np.zeros([n,1], dtype=np.double)
        for i in range(n):
            ret[i] = x[i]
        return ret 
    
    def save_img(self):
        return self.solver.save_img()


  


    