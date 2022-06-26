cdef extern from "LPboxADMMsolver.cpp":
    pass

cdef extern from "LPboxADMMsolver.h":
    cdef cppclass LPboxADMMsolver:
        LPboxADMMsolver() except +
        LPboxADMMsolver(int) except +
        LPboxADMMsolver(int, double) except +
        void readFile(int, int, int)
        int ADMM_lp_iters_init()
        int ADMM_lp_iters(int, int)
        double cal_obj()
        int ADMM_lp_iters_l2f(int, int, double*, int);
        double* get_x_iters_d(int);
        int get_n();
        int get_iter();
        double* get_x_sol();
        double* get_final_x_sol();
        double get_curBinObj();
        int check_infeasible_lpbox();
        int check_infeasible_l2f();
        
        