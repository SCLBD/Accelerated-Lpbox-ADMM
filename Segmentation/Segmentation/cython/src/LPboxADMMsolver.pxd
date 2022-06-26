cdef extern from "LPboxADMMsolver.cpp":
    pass

cdef extern from "LPboxADMMsolver.h":
    cdef cppclass LPboxADMMsolver:
        LPboxADMMsolver() except +
        LPboxADMMsolver(int) except +
        LPboxADMMsolver(int, int, int) except +
        void ADMM_bqp_unconstrained_init()
        int ADMM_bqp_unconstrained_legacy()
        int ADMM_bqp_unconstrained_l2f(int, int, double*, int);
        double* get_x_iters_d(int);
        int get_n();
        int get_org_n();
        double get_final_obj();
        double* get_x_sol();
        void save_img();
    

        
        