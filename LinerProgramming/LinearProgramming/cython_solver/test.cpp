#include <vector>
#include <Eigen4/Sparse>
#include <Eigen4/Dense>
#include <iostream>
#include <iomanip>
#include "LPboxADMMsolver.h"
#include <chrono>


void testlp(int i, int k, int j){
	LPboxADMMsolver solv(0);
	solv.readFile(i,k,j);
    solv.ADMM_lp_iters_init(); // for initialization 
	solv.ADMM_lp_iters(0, 2e4);
	int tmp = solv.check_infeasible_l2f();
	printf("this is feasiblibity: %d", tmp);
	printf("\n");
}

int main(int argc, char **argv){
	int i = atoi(argv[1]);
	int j = atoi(argv[2]);
	int k = atoi(argv[3]);

	std::chrono::steady_clock::time_point start, end;
	start = std::chrono::steady_clock::now();
	testlp(i,j,k); 
	end = std::chrono::steady_clock::now();
	double time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //micro
	std::cout << "Time elapsed: " << 1.0*time_elapsed/1000 << "s;" << std::endl;
    return 0;
}




