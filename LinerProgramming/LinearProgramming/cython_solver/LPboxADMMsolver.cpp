//
// Created by squintingsmile on 2020/9/22.
//

#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <Eigen4/Dense>
#include <Eigen4/Sparse>

#include <fstream>
#include "LPboxADMMsolver.h"
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>


// using namespace cv;
using namespace Eigen::internal;
using std::abs;
using std::sqrt;
Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
		", ", ", ", "", "", " << ", ";");

/* The original function is imported from the Eigen library. Added clocks to monitor times */
void _conjugate_gradient(const SparseMatrix& mat, const DenseVector& rhs, DenseVector& x,
		const Eigen::DiagonalPreconditioner<_double_t>& precond, int& iters, typename DenseVector::RealScalar& tol_error)
{

	typedef typename DenseVector::RealScalar RealScalar;
	typedef typename DenseVector::Scalar Scalar;
	typedef Eigen::Matrix<Scalar,Dynamic,1> VectorType;

	RealScalar tol = tol_error;
	int maxIters = iters;

	int n = mat.cols();

	VectorType residual = rhs - mat * x; //initial residual

	RealScalar rhsNorm2 = rhs.squaredNorm();

	if(rhsNorm2 == 0)
	{
		x.setZero();
		iters = 0;
		tol_error = 0;
		return;
	}

	const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();
	RealScalar threshold = Eigen::numext::maxi(RealScalar(tol*tol*rhsNorm2),considerAsZero);
	RealScalar residualNorm2 = residual.squaredNorm();

	if (residualNorm2 < threshold)
	{
		iters = 0;
		tol_error = sqrt(residualNorm2 / rhsNorm2);
		return;
	}

	VectorType p(n);
	p = precond.solve(residual);      // initial search direction

	VectorType z(n), tmp(n);
	RealScalar absNew = Eigen::numext::real(residual.dot(p));  // the square of the absolute value of r scaled by invM
	int i = 0;
	while(i < maxIters)
	{
		tmp.noalias() = mat * p;                    // the bottleneck of the algorithm

		Scalar alpha = absNew / p.dot(tmp);         // the amount we travel on dir

		x += alpha * p;                             // update solution

		residual -= alpha * tmp;                    // update residual

		residualNorm2 = residual.squaredNorm();

		if(residualNorm2 < threshold) {
			i++;
			break;
		}

		z = precond.solve(residual);                // approximately solve for "A z = residual"

		RealScalar absOld = absNew;
		absNew = Eigen::numext::real(residual.dot(z));     // update the absolute value of r
		RealScalar beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
		p = z + beta * p;                           // update search direction
		i++;
	}
	tol_error = sqrt(residualNorm2 / rhsNorm2);
	iters = i;
	return;
}

/* The .noalias() function explicity states that result of the matrix multiplication can be written
 * on lhs directly without allocating an empty vector first and store the result, then assigning
 * it to lhs. This hopefully will save some time
 */
void mat_mul_vec(const SparseMatrix &mat, const DenseVector &vec, DenseVector& res) {
	if (mat.cols() == mat.rows() || &vec != &res) {
		res.noalias() = mat * vec;
	} else {
		res = mat * vec;
	}
}

/* This is used to avoid direct matrix multiplication. Suppose that we want to calculate A^TAx where A is large
 * so calculating A^TAx is intractable. Then we calculate Ax then calculate A^T(Ax) to reduce the number of operations.
 * In this vector mat_expression, each entry is a vector of sparse matrices. For each entry, the result is calculated by
 * entry[n] ... entry[2] * (entry[1] * (entry[0] * x)) and the results calculated by different entries are sum together
 */
void calculate_mat_expr_multiplication(const std::vector<std::vector<const SparseMatrix*>> &mat_expressions,
		DenseVector &x, DenseVector &result, DenseVector &temp_vec) {

	if (mat_expressions.size() == 0) {
		printf("Empty matrix expression when doing multiplication\n");
		return;
	}

	DenseVector *original_x;
	if (&x == &result && mat_expressions.size() > 1) {
		original_x = new DenseVector(x);
	} else {
		original_x = &x;
	}

	{
		const std::vector<const SparseMatrix*> &expression = mat_expressions[0];
		int counter = 0;
		for (const SparseMatrix *mat : expression) {
			if (counter == 0) {
				mat_mul_vec(*mat, x, result);
			} else {
				mat_mul_vec(*mat, result, result);
			}
			counter += 1;
		}
	}

	if (mat_expressions.size() > 1) {
		for (auto iter = std::next(mat_expressions.begin(), 1); iter != mat_expressions.end(); iter++) {
			int counter = 0;
			for (const SparseMatrix *mat : *iter) {
				if (counter == 0) {
					mat_mul_vec(*mat, *original_x, temp_vec);
				} else {
					mat_mul_vec(*mat, temp_vec, temp_vec);
				}
				counter += 1;
			}

			result += temp_vec;
		}

		if (&x == &result) {
			delete original_x;
		}
	}
}


/* The original function is imported from the Eigen library. Passing the two temporary vectors to save the time of
 * allocating vectors, this is the one for using.
 */

void _conjugate_gradient(const std::vector<std::vector<const SparseMatrix*>> &mat_expressions, const DenseVector& rhs,
		DenseVector& x, const Eigen::DiagonalPreconditioner<_double_t>& precond, 
		int& iters,
		typename DenseVector::RealScalar& tol_error, 
		DenseVector &temp_vec_for_cg, DenseVector &temp_vec_for_mat_mul, const char* log) {

	FILE *fp;
	fp = fopen(log, "a+");

	typedef typename DenseVector::RealScalar RealScalar;
	typedef typename DenseVector::Scalar Scalar;
	typedef Eigen::Matrix<Scalar,Dynamic,1> VectorType;

	RealScalar tol = tol_error;// == 1e-3
	int maxIters = iters;      //1e3

	int n = mat_expressions[0][0]->cols();


	calculate_mat_expr_multiplication(mat_expressions, x, temp_vec_for_cg, temp_vec_for_mat_mul);
	VectorType residual = rhs - temp_vec_for_cg; //initial residual

	RealScalar rhsNorm2 = rhs.squaredNorm();

	if(rhsNorm2 == 0) {
		x.setZero();
		iters = 0;
		tol_error = 0;
		return;
	}

	const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();
	RealScalar threshold = Eigen::numext::maxi(RealScalar(tol*tol*rhsNorm2),considerAsZero);
	RealScalar residualNorm2 = residual.squaredNorm();

	if (residualNorm2 < threshold)
	{
		iters = 0;
		tol_error = sqrt(residualNorm2 / rhsNorm2);
		return;
	}
	VectorType p(n);
	p = precond.solve(residual);      // initial search direction

	VectorType z(n), tmp(n);
	RealScalar absNew = Eigen::numext::real(residual.dot(p));  // the square of the absolute value of r scaled by invM
	int i = 0;
	while(i < maxIters)
	{
		calculate_mat_expr_multiplication(mat_expressions, p, tmp, temp_vec_for_mat_mul);

		Scalar alpha = absNew / p.dot(tmp);         // the amount we travel on dir

		x += alpha * p;                             // update solution

		residual -= alpha * tmp;                    // update residual
		residualNorm2 = residual.squaredNorm();
		fprintf(fp, "In PCG: Iteration: %d; x norm: %f; p: %f; tmp: %f, alpha: %f, residual: %f\n", 
			i, x.norm(),p.norm(),tmp.norm(), alpha, residualNorm2);


		if(residualNorm2 < threshold) {
			i++;
			break;
		}

		z = precond.solve(residual);                // approximately solve for "A z = residual"

		RealScalar absOld = absNew;
		absNew = Eigen::numext::real(residual.dot(z));     // update the absolute value of r
		RealScalar beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
		p = z + beta * p;                           // update search direction
		i++;
	}
	tol_error = sqrt(residualNorm2 / rhsNorm2);
	iters = i;

	fclose(fp);

	return;
}

int _conjugate_gradient(const std::vector<std::vector<const SparseMatrix*>> &mat_expressions, const DenseVector& rhs,
		DenseVector& x, const Eigen::DiagonalPreconditioner<_double_t>& precond, 
		int& iters,
		typename DenseVector::RealScalar& tol_error, 
		DenseVector &temp_vec_for_cg, DenseVector &temp_vec_for_mat_mul, 
		const char* log, int iter_num) {

	// FILE *fp;
	// if(iter_num==0) fp = fopen("src/log/log_pcg.txt", "w+");
	// else fp = fopen("src/log/log_pcg.txt", "a+");
	// fprintf(fp, "Iterations: %d\n",iter_num);

	typedef typename DenseVector::RealScalar RealScalar;
	typedef typename DenseVector::Scalar Scalar;
	typedef Eigen::Matrix<Scalar,Dynamic,1> VectorType;

	RealScalar tol = tol_error;// == 1e-3
	int maxIters = iters;      //1e3

	int n = mat_expressions[0][0]->cols();


	calculate_mat_expr_multiplication(mat_expressions, x, temp_vec_for_cg, temp_vec_for_mat_mul);
	VectorType residual = rhs - temp_vec_for_cg; //initial residual
	// fprintf(fp, "Before PCG: Iteration: %d; x norm: %f; tmp: %f, rhs: %f, residual: %f\n", 
	// 		iter_num, x.norm(),temp_vec_for_cg.norm(), rhs.norm(), residual.norm());
	RealScalar rhsNorm2 = rhs.squaredNorm();

	if(rhsNorm2 == 0) {
		x.setZero();
		iters = 0;
		tol_error = 0;
		return 1;
	}

	const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();
	RealScalar threshold = Eigen::numext::maxi(RealScalar(tol*tol*rhsNorm2),considerAsZero);
	RealScalar residualNorm2 = residual.squaredNorm();

	if (residualNorm2 < threshold)
	{
		iters = 0;
		tol_error = sqrt(residualNorm2 / rhsNorm2);
		return 1;
	}
	VectorType p(n);
	p = precond.solve(residual);      // initial search direction

	VectorType z(n), tmp(n);
	RealScalar absNew = Eigen::numext::real(residual.dot(p));  // the square of the absolute value of r scaled by invM
	int i = 0;
	while(i < maxIters)
	{
		calculate_mat_expr_multiplication(mat_expressions, p, tmp, temp_vec_for_mat_mul);

		Scalar alpha = absNew / p.dot(tmp);         // the amount we travel on dir
		if(alpha < 0) return -1;
		x += alpha * p;                             // update solution

		residual -= alpha * tmp;                    // update residual
		residualNorm2 = residual.squaredNorm();
		// fprintf(fp, "In PCG: Iteration: %d; x norm: %f; p: %f; tmp: %f, alpha: %f, residual: %f\n", 
		// 	i, x.norm(),p.norm(),tmp.norm(), alpha, residualNorm2);

		if(residualNorm2 < threshold) {
			i++;
			break;
		}

		z = precond.solve(residual);                // approximately solve for "A z = residual"

		RealScalar absOld = absNew;
		absNew = Eigen::numext::real(residual.dot(z));     // update the absolute value of r
		RealScalar beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
		p = z + beta * p;                           // update search direction
		i++;
	}
	tol_error = sqrt(residualNorm2 / rhsNorm2);
	iters = i;

	// fprintf(fp, "End of iter\n");
	// fclose(fp);

	return 1;
}


void print_vector(DenseVector d){
	int len = d.rows();
	printf("Test 105: Print DenseVector with rows=%d: [", len);
	for(int i = 0 ; i < len; i++){
		printf("%f,",d.array()[i]);
	}
	printf("]\n");
}

void print_index(Eigen::ArrayXi d){
	int len = d.rows();
	printf("Test 105: Print DenseVector with rows=%d: [", len);
	for(int i = 0 ; i < len; i++){
		printf("%d,",d.array()[i]);
	}
	printf("]\n");
}



_double_t LPboxADMMsolver::std_dev(std::vector<_double_t>& arr, size_t begin, size_t end) {
	_double_t mean = 0;
	_double_t std_deviation = 0;
	size_t size = end - begin;
	for (int i = begin; i < end; i++) {
		mean += arr[i];
	}

	mean /= size;
	for (int i = 0; i < size; i++) {
		std_deviation += (arr[begin + i] - mean) * (arr[begin + i] - mean);
	}

	std_deviation /= size - 1;
	if (std_deviation == 0) {
		return 0;
	}

	return std::pow(std_deviation, 1.0 / 2);
}

void LPboxADMMsolver::project_vec_greater_than(DenseVector &x, DenseVector &res, const int &greater_val, const int &set_val) {
	int len = x.size();
	for (int i = 0; i < len; i++) {
		res[i] = x[i] > greater_val ? set_val : x[i];
	}
}

void LPboxADMMsolver::project_vec_less_than(DenseVector &x, DenseVector &res, const int &less_val, const int &set_val) {
	int len = x.size();
	for (int i = 0; i < len; i++) {
		res(i) = x(i) < less_val ? set_val : x(i);
	}
}

void LPboxADMMsolver::project_box(int n, const double *x, double *y) {
	y = new double[n];
	while (n--) {
		if (x[n] > 1) {
			y[n] = 1;
		} else {
			if (x[n] < 0) {
				y[n] = 0;
			} else {
				y[n] = x[n];
			}
		}
	}
}

/* TODO: Utilize eigen .array() functions to speed up this operation */
void LPboxADMMsolver::project_box(int n, const DenseVector &x, DenseVector &y) {
	while (n--) {
		if (x[n] > 1) {
			y[n] = 1;
		} else {
			if (x[n] < 0) {
				y[n] = 0;
			} else {
				y[n] = x[n];
			}
		}
	}
}

void LPboxADMMsolver::project_shifted_Lp_ball(int n, const DenseVector &x, int p, DenseVector &y) {
	y.array() = x.array() - 0.5;
	_double_t normp_shift = y.norm();

	y.array() = y.array() * std::pow(n, 1.0 / p) / (2 * normp_shift) + 0.5;
}

_double_t LPboxADMMsolver::compute_cost(const DenseVector& x, const SparseMatrix& A, 
		const DenseVector& b, DenseVector &temp_vec_for_mat_mul) {
	mat_mul_vec(A, x, temp_vec_for_mat_mul);
	double val = x.transpose().dot(temp_vec_for_mat_mul);
	double val2 = b.dot(x);
	return val + val2;
}

_double_t LPboxADMMsolver::compute_cost(const DenseVector& x, const SparseMatrix& A, const DenseVector& b) {
	double val = x.transpose().dot(A * x);
	double val2 = b.dot(x);
	return val + val2;
}

_double_t LPboxADMMsolver::compute_cost_lp( DenseVector& x,  DenseVector& b) {
	// int len = x.rows();
	// _double_t ret;
	// for(int i = 0; i < len; i++){
	// 	if(std::isnan(x.array()[i])) printf("######## Error 102: x_%d=nan",i);
	// 	if(std::isnan(b.array()[i])) printf("######## Error 102: b_%d=nan",i);
		
		
	// 	ret += x.array()[i] * b.array()[i];
	// }
	// return ret;
	double val2 = b.dot(x);
	return val2;
}

_double_t LPboxADMMsolver::compute_std_obj(std::vector<_double_t> obj_list, int history_size) {
	size_t s = obj_list.size();
	_double_t std_obj;
	if (s <= history_size) {
		std_obj = std_dev(obj_list, 0, s);
	} else {
		std_obj = std_dev(obj_list, s - history_size, s);
	}

	return std_obj / std::abs(obj_list[s-1]);
}


LPboxADMMsolver::LPboxADMMsolver(){
	// printf("Object is created!\n");
	this->print_fix_info = 0;
}

LPboxADMMsolver::LPboxADMMsolver(int print_fix_info){
	printf("Object with fix_info is created!\n");
	this->print_fix_info = print_fix_info;
}

LPboxADMMsolver::LPboxADMMsolver(int consistency, _double_t fix_threshold){
	// printf("Object with input is created!\n");
	this->consistency = consistency;
	this->fix_threshold  = fix_threshold;
}

// solve lpbox ADMM. initialization. 
int LPboxADMMsolver::ADMM_lp_iters_init(){

	stop_threshold = 1e-4;
	std_threshold = 1e-6;
	gamma_val = 1.6;
	gamma_factor = 0.95;
	rho_change_step = 25;  //5
	max_iters = 2e4;  //2e3;
	initial_rho = 25; //
	history_size = 3;
	learning_fact = 1 + 1.0 / 100;
	pcg_tol = 1e-3; // 1e-4
	pcg_maxiters = 1e3;
	rel_tol = 5e-5;
	projection_lp = 2;


	std_threshold = 1e-12;
	history_size = 10;



	// SolverInstruction solver_instruction;
	// MatrixInfo matrix_info;

	// matrix_info.x0 = &x0;
	// // matrix_info.A = &_A;
	// matrix_info.b = &_b;
	// matrix_info.n = n;
	// matrix_info.E = &_E;
	// matrix_info.f = &_f; 
	// matrix_info.l = l;

	instruction.problem_type = inequality;
	instruction.update_y3 = 1;
	instruction.update_z3 = 0;
	instruction.update_z4 = 1;
	instruction.update_rho3 = 0;
	instruction.update_rho4 = 1;
	
	// const SparseMatrix *E_ptr;
	// const DenseVector *b_ptr, *f_ptr;

	// b_ptr = matrix_info.b;

	// SparseMatrix C_transpose, E_transpose, _2A_plus_rho1_rho2, rho3_C_transpose, rho4_E_transpose;
	// DenseVector y3, z3, z4, Csq_diag, Esq_diag;

	l = (*E_ptr).rows();   // rows
	// int m = matrix_info.m;
	n = (*E_ptr).cols();   // cols

	

	FILE *fp;
	if (does_log) {
		fp = fopen(log_file_path.c_str(), "w+");
	}

	// printf("longkang test 0\n"); 

	x_sol    = DenseVector(n);
	y1       = DenseVector(n);
	y2       = DenseVector(n);
	z1       = DenseVector(n);
	z2       = DenseVector(n);
	prev_idx = DenseVector(n);
	best_sol = DenseVector(n);
	temp_vec = DenseVector(n);
	cur_idx  = DenseVector(n);
	temp_mat = SparseMatrix(n, n);

	x_sol.setZero();

	// printf("longkang test 0223, E col: %ld, row: %ld\n",E_ptr->cols(),E_ptr->rows());
	// printf("longkang test 0223, x col: %ld, row: %ld\n",x_sol.cols(),x_sol.rows());
	// printf("longkang test 0223, f col: %ld, row: %ld\n",f_ptr->cols(),f_ptr->rows());

	/*
		Early Fix
	*/
	fix_threshold = 5e-4; //5e-4;
	consistency = 5;
	x_prev = DenseVector::Zero(n);
	x_det = DenseVector::Zero(n);
	x_count = DenseVector::Zero(n);
	x_flag = DenseVector::Zero(n);

	// make it work for indset 
	x_sol = DenseVector::Zero(n);

	// x_iters = DenseMatrix(25,n);

	left_idx = DenseVector::Zero(n);
	for(int i = 0 ; i < n; i++){
		left_idx.array()[i] = i;
		x_sol.array()[i] = 1;
	}

	org_n = n;
	fix_n = 0;
	fix_sum = 0;
	ret_idx = -1 * DenseVector::Ones(n);
	ret_val = -1 * DenseVector::Ones(n);
	fix_obj = 0;
	sum_fix_obj = 0;

	// fixed_idx_org = DenseVector::Zero(n);
	// val = DenseVector::Zero(n);
	// x_fixed_idx = DenseVector::Zero(n);
	// x_fixed_val = DenseVector::Zero(n); 


	// printf("longkang test 1\n"); 

	/* The temp_vec_for_cg and temp_vec_for_mat_mat are vector that used to store the temporary vectors
	 * in the algorithm to prevent allocating new vectos. Adding this shall has around 5% of performance
	 * improvement (although might reduce the performance under some tasks by 10% since it increases the cost 
	 * of other operations besides matrix multiplication, such as vector addition and vector multiplication
	 * for reasons unknown)
	 */
	temp_vec_for_cg = DenseVector(n);
	temp_vec_for_mat_mul = DenseVector(n);

	// _double_t cur_obj;
	// bool rhoUpdated = true;

	// x_sol = //*(matrix_info.x0);

	z1.array() = 0;
	z2.array() = 0;
	cur_idx.array() = 0;


	// Eigen::DiagonalPreconditioner<_double_t> diagonalPreconditioner;

	rho1 = initial_rho;
	rho2 = initial_rho;
	rho3 = initial_rho;
	rho4 = initial_rho;
	prev_rho1 = rho1;
	prev_rho2 = rho2;
	prev_rho3 = rho3;
	prev_rho4 = rho4;

	// std::vector<_double_t> obj_list; /* Stores the objective value calculated during each iteration */
	// _double_t std_obj = 1;

	// _double_t cvg_test1;
	// _double_t cvg_test2;
	// _double_t rho_change_ratio;

	if (instruction.update_y3) {
		y3 = DenseVector(l);
	}

	if (instruction.update_z4) {
		z4 = DenseVector(l);
		// z4.array() = 0;
		z4.setZero();
	}

	// printf("longkang test 2\n");
	/* If this task contains inequality constraints, initialize E and f and relevant matrices */
	// if (instruction.problem_type & inequality) {
	// 	E_ptr = matrix_info.E;
	// 	// E_transpose = (*E_ptr).transpose();
	// 	// rho4_E_transpose = rho4 * E_transpose;
	// 	f_ptr = matrix_info.f;
	// }

	// /* Storing the matrix 2 * A + (rho1 + rho2) * I to save calculation time */
	// // _2A_plus_rho1_rho2 = 2 * (*A_ptr);
	// // _2A_plus_rho1_rho2.diagonal().array() += rho1 + rho2;
	// DenseMatrix tmp(n,n);
	// tmp.diagonal().array() = rho1 + rho2;
	// _2A_plus_rho1_rho2 = tmp.sparseView();

	// SparseMatrix preconditioner_diag_mat(n, n); /* The diagonal matrix used by the preconditioner */
	// preconditioner_diag_mat.reserve(n);
	// std::vector<Triplet> preconditioner_diag_mat_triplets;
	// preconditioner_diag_mat_triplets.reserve(n);
	// /* The diagonal elements in the original expression evaluated by the preconditioner is given by the
	//  * diagonal elements of 2 * _A + (rho1 + rho2) * I + rho3 * _C^T * _C
	//  */
	// for (int i = 0; i < n; i++) {
	// 	preconditioner_diag_mat_triplets.push_back(Triplet(i, i, 0));
	// }
	// preconditioner_diag_mat.setFromTriplets(preconditioner_diag_mat_triplets.begin(), preconditioner_diag_mat_triplets.end());
	// preconditioner_diag_mat.diagonal().array() = _2A_plus_rho1_rho2.diagonal().array();
	// preconditioner_diag_mat.makeCompressed();


	// /* The matrix expression for the unconstraint case is 2 * _A + (rho1 + rho2) * I */
	// // std::vector<std::vector<const SparseMatrix*>> matrix_expressions;
	// matrix_expressions.emplace_back();
	// matrix_expressions.back().push_back(&_2A_plus_rho1_rho2);


	// /* If the problem has equality constraint, add rho4 * E^T * E to the matrix expression */
	// if (instruction.problem_type &inequality) {
	// 	matrix_expressions.emplace_back();
	// 	matrix_expressions.back().push_back(E_ptr);
	// 	matrix_expressions.back().push_back(&rho4_E_transpose);

	// 	/* Calculating the diagonal elements of Esq for preconditioner */
	// 	Esq_diag = DenseVector(n);
	// 	Esq_diag.setZero();
	// 	for(int j = 0; j < E_transpose.outerSize(); ++j) {
	// 		typename SparseMatrix::InnerIterator it(E_transpose, j);
	// 		while (it) {
	// 			if(it.value() != 0.0) {
	// 				Esq_diag[j] += it.value() * it.value();
	// 			}
	// 			++it;
	// 		}
	// 	}
	// 	preconditioner_diag_mat.diagonal().array() += rho4 * Esq_diag.array();
	// }

	// this->preconditioner_diag_mat = preconditioner_diag_mat; 
	// this->preconditioner_diag_mat_triplets = preconditioner_diag_mat_triplets;
	// // this->matrix_expressions = matrix_expressions;

	// printf("longkang test 222\n");
	y1 = x_sol;
	y2 = x_sol;
	if (instruction.update_y3) {
		// printf("longkang test 223, E col: %ld, row: %ld\n",E_ptr->cols(),E_ptr->rows());
		// printf("longkang test 223, x col: %ld, row: %ld\n",x_sol.cols(),x_sol.rows());
		// printf("longkang test 223, f col: %ld, row: %ld\n",f_ptr->cols(),f_ptr->rows());
		y3 = *f_ptr - *E_ptr * x_sol;
		// printf("longkang test 224\n");
	}

	prev_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
	best_sol = x_sol;

	best_bin_obj = compute_cost_lp(x_sol, *b_ptr);
	// printf("longkang test 3\n");
	// if (does_log) {
	// 	fprintf(fp, "Initial state\n");
    //     fprintf(fp, "norm of x_sol: %lf; size %ld\n", x_sol.norm(),x_sol.rows());
    //     fprintf(fp, "norm of b: %lf, size: %ld\n", (*b_ptr).norm(), (*b_ptr).rows());
    //     fprintf(fp, "norm of y1: %lf, size: %ld\n", y1.norm(),y1.rows());
    //     fprintf(fp, "norm of y2: %lf, size: %ld\n", y2.norm(), y2.rows());
    //     if (instruction.update_y3) {
    //         fprintf(fp, "norm of y3: %lf, size: %ld\n", y3.norm(), y3.rows());
    //     }

    //     fprintf(fp, "norm of z1: %lf, size: %ld\n", z1.norm(), z1.rows());
    //     fprintf(fp, "norm of z2: %lf, size: %ld\n", z2.norm(), z2.rows());

    //     // if (instruction.update_z3) {
    //     //     fprintf(fp, "norm of z3: %lf\n", z3.norm());
    //     // }

    //     if (instruction.update_z4) {
    //         fprintf(fp, "norm of z4: %lf, size: %ld\n", z4.norm(), z4.rows());
    //     }

    //     fprintf(fp, "norm of cur_idx: %lf\n", cur_idx.norm());
    //     fprintf(fp, "rho1: %lf\n", rho1);
    //     fprintf(fp, "rho2: %lf\n", rho2);
    //     fprintf(fp, "rho3: %lf\n", rho3);
    //     fprintf(fp, "rho4: %lf\n", rho4);
    //     fprintf(fp, "-------------------------------------------------\n");
	// }

	if (does_log) {
		fclose(fp);
	}

	return 1;
}

// solve lobox ADMM. iterations.
int LPboxADMMsolver::ADMM_lp_iters(int iter_start, int iter_end){
	int ret = 0; // determine converge or not. 
	std::chrono::steady_clock::time_point start, end;
	start = std::chrono::steady_clock::now();
	// printf("Longkang test 0\n");
	FILE *fp;
	if (does_log) {
		fp = fopen(log_file_path.c_str(), "w+");
	}

	FILE *xi;
	if(print_fix_info==2 || print_fix_info==3){
		xi = fopen(path_xiter.c_str(), "w+");
		// fprintf(xi, "This is the start!!!\n");
	}

	FILE *al;
	al = fopen(csv.c_str(), "a+");

	_double_t obj_val;
	int iter;
	double time_elapsed = 0;
	// printf("test: start:%d, end: %d\n", iter_start, iter_end);

	// E_transpose = (*E_ptr).transpose();
	// rho4_E_transpose = rho4 * E_transpose;
	// esq = E_transpose * (*E_ptr);
	// DenseMatrix tmp(n,n);
	// tmp.diagonal().array() = rho1 + rho2;
	// _2A_plus_rho1_rho2 = tmp.sparseView();
	for (iter = iter_start; iter < iter_end; iter++) {

		// printf("Longkang test 1.\n");
		if (does_log) {
			fprintf(fp, "Iteration: %d\n", iter+1);
		}

		/*
			update y1	
		*/
		temp_vec = x_sol + z1 / rho1;

		/* Project vector on [0, 1] box */
		project_box(n, temp_vec, y1);


		/*
			update y2	
		*/
		temp_vec = x_sol + z2 / rho2;

		/* Project vector on shifted lp box */
		project_shifted_Lp_ball(n, temp_vec, projection_lp, y2);


		/*
			update y3	
		*/
		if (instruction.update_y3) {
			mat_mul_vec(*E_ptr, x_sol, temp_vec_for_mat_mul);
			y3 = *f_ptr - temp_vec_for_mat_mul - z4 / rho4;
			project_vec_less_than(y3, y3, 0, 0); 
		}

		// printf("Longkang test 2.\n");
		if(iter==0){
			// printf("Longkang test 20.\n");
			update_expression(0);
			// printf("Longkang test 21.\n");
			// printf("Test: size matrix_expression: %ld\n",matrix_expressions.size());
			// for(int i = 0 ;i < matrix_expressions.size(); i++){
			// 	printf("in %d, size: %ld\n",i,matrix_expressions[i].size());
			// }
			// // std::cout << matrix_expressions << std::endl;
			// printf("Test: size preconditiner: %ld, %ld\n",preconditioner_diag_mat.rows(),preconditioner_diag_mat.cols());
			// // std::cout << preconditioner_diag_mat << std::endl;
		}
		// /*
		// 	update x	
		// */


		// /* If the iteration is nonzero and it divides rho_change_step, it means
		//  * that the rho updated in the last iteration
		//  */
		if (iter != 0 && rhoUpdated) {
			/* Note that we need the previous rho to update in order to get the difference between
			 * the updated matrix and the not updated matrix. Another possible calculation is by
			 * calculating rho * rho_change_ration / learning_fact
			 */
			// printf("Longkang test 21.\n");
			_2A_plus_rho1_rho2.diagonal().array() += rho_change_ratio * (prev_rho1 + prev_rho2);
			if (instruction.problem_type != unconstrained) {
				preconditioner_diag_mat.diagonal().array() += rho_change_ratio * (prev_rho1 + prev_rho2);
			}			
			// printf("Longkang test 22.\n");
			if (instruction.update_rho4) {
				preconditioner_diag_mat.diagonal().array() += rho_change_ratio * prev_rho4 * Esq_diag.array();
				rho4_E_transpose = learning_fact * rho4_E_transpose;
			}
		}
		// printf("Longkang test 3.\n");

		/* If the problem in equality, the rhs vector is 
		 * rho1 * y1 + rho2 * y2 + rho4 * E^T * (f - y3) - (b + z1 + z2 + E^T * z4)
		 */
		if (instruction.problem_type == inequality) {
			temp_vec = rho1 * y1 + rho2 * y2 - (*b_ptr + z1 + z2);
			mat_mul_vec(rho4_E_transpose, *f_ptr - y3, temp_vec_for_mat_mul);
			temp_vec += temp_vec_for_mat_mul;
			mat_mul_vec(E_transpose, z4, temp_vec_for_mat_mul);
			temp_vec -= temp_vec_for_mat_mul;
		}


		// printf("Longkang test 31.\n");
		/* Explicit version of conjugate gradient used for profiling */
		if (rhoUpdated) {
			if (instruction.problem_type != unconstrained) {
				diagonalPreconditioner.compute(preconditioner_diag_mat);
			} else {
				diagonalPreconditioner.compute(_2A_plus_rho1_rho2);
			}
			rhoUpdated = false;
		}
		_double_t tol = pcg_tol;
		x_sol = y1;
		int maxiter = pcg_maxiters;
		_conjugate_gradient(matrix_expressions, temp_vec, x_sol, diagonalPreconditioner, maxiter, tol, 
				temp_vec_for_cg, temp_vec_for_mat_mul,log_file_path.c_str(),iter); //

		if (does_log) {
			fprintf(fp, "Conjugate gradient stops after %d iterations\n", maxiter);
			fprintf(fp, "Conjugate gradient stops with residual %lf\n", tol);
		}


		if(print_fix_info==2){
			fprintf(xi, "Iter%d,", iter+1);
			for(int i = 0 ; i < n-1; i++){
				fprintf(xi, "%lf,",x_sol.array()[i]);
			}
			fprintf(xi, "%lf\n",x_sol.array()[n-1]);
		}
		
		
		
		// printf("Longkang test 4.\n");
		/*
			update z1, z2, z4	
		*/
		z1 = z1 + gamma_val * rho1 * (x_sol - y1);
		z2 = z2 + gamma_val * rho2 * (x_sol - y2);
		if (instruction.update_z4) {
			if(iter==iter_start)
				z4 = gamma_val * rho4 * ((*E_ptr) * x_sol + y3 - (*f_ptr));
			else
				z4 = z4 + gamma_val * rho4 * ((*E_ptr) * x_sol + y3 - (*f_ptr));
		}

		// printf("Longkang test 5.\n");

		/*
			Evaluate 	
		*/
		_double_t temp0 = std::max(x_sol.norm(), _double_t(2.2204e-16));
		cvg_test1 = (x_sol - y1).norm() / temp0;
		cvg_test2 = (x_sol - y2).norm() / temp0;
		if (cvg_test1 <= stop_threshold && cvg_test2 <= stop_threshold && iter!=iter_start) {
			printf("Stop because y1_y2. iter: %d, stop_threshold: %.6f\n", iter, std::max(cvg_test1, cvg_test2));
			if (does_log) {
				fprintf(fp, "iter: %d, stop_threshold: %.6f\n", iter, std::max(cvg_test1, cvg_test2));
			}

			if(print_fix_info==3){
				fprintf(xi, "Iter%d,", iter+1);
				for(int i = 0 ; i < n-1; i++){
					fprintf(xi, "%lf,",x_sol.array()[i]);
				}
				fprintf(xi, "%lf\n",x_sol.array()[n-1]);
			}

			break;
		}

		if ((iter+1) % rho_change_step == 0) {
			prev_rho1 = rho1;
			prev_rho2 = rho2;
			rho1 = learning_fact * rho1;
			rho2 = learning_fact * rho2;

			if (instruction.update_rho3) {
				prev_rho3 = rho3;
				rho3 = learning_fact * rho3;
			}

			if (instruction.update_rho4) {
				prev_rho4 = rho4;
				rho4 = learning_fact * rho4;
			}

			gamma_val = std::max(gamma_val * gamma_factor, _double_t(1.0));
			rhoUpdated = true;
			rho_change_ratio = learning_fact - 1.0;
		}

		obj_val = compute_cost_lp(x_sol, *b_ptr);
		obj_list.push_back(obj_val);
		if (obj_list.size() >= history_size) {
			std_obj = compute_std_obj(obj_list, history_size);
		}
		if (std_obj <= std_threshold) {
			ret = 1;
			if (does_log) {
				fprintf(fp, "iter: %d, std_threshold: %.6f\n", iter, std_obj);
				// printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter,x_sol.norm(),obj_val,cur_obj);
		
			}
			printf("Stop because obj_std. iter: %d, std_threshold: %.6f\n", iter, std_obj);
			// printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter,x_sol.norm(),obj_val,cur_obj);
			if(print_fix_info==3){
				fprintf(xi, "Iter%d,", iter+1);
				for(int i = 0 ; i < n-1; i++){
					fprintf(xi, "%lf,",x_sol.array()[i]);
				}
				fprintf(xi, "%lf\n",x_sol.array()[n-1]);
			}
		
			break;
		}

		// if(iter%100==0) 
		// 	printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter,x_sol.norm(),obj_val,cur_obj);
		

		cur_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
		prev_idx = cur_idx;
		cur_obj = compute_cost_lp(prev_idx, *b_ptr);
		// printf("Longkang test 2\n");
		// printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter+1,x_sol.norm(),obj_val,cur_obj);
		// printf("Longkang test 3\n");

		if (best_bin_obj >= cur_obj) {
			best_bin_obj = cur_obj;
			best_sol = x_sol;
		}
		// printf("Longkang test 4\n");
		if (does_log) {
			// fprintf(fp, "current objective: %lf\n", obj_val);
			// fprintf(fp, "current binary objective: %lf\n", cur_obj);

			// if (instruction.problem_type == equality || instruction.problem_type == equality_and_inequality) {
			// 	fprintf(fp, "equality constraint violation: %lf\n", (*matrix_info.C * cur_idx - *matrix_info.d).norm() / x_sol.rows());
			// }

			// if (instruction.problem_type == inequality || instruction.problem_type == equality_and_inequality) {
			// 	DenseVector diff = *matrix_info.E * cur_idx - *matrix_info.f;
			// 	project_vec_less_than(diff, diff, 0, 0);
			// 	fprintf(fp, "inequality constraint violation: %lf\n", diff.norm() / x_sol.rows());
			// }

			fprintf(fp, "norm of x_sol: %.9lf\n", x_sol.norm());
			fprintf(fp, "norm of y1: %.9lf\n", y1.norm());
			fprintf(fp, "norm of y2: %.9lf\n", y2.norm());
			if (instruction.update_y3) {
				fprintf(fp, "norm of y3: %.9lf\n", y3.norm());
			}

			fprintf(fp, "norm of z1: %.9lf\n", z1.norm());
			fprintf(fp, "norm of z2: %.9lf\n", z2.norm());

			// if (instruction.update_z3) {
			// 	fprintf(fp, "norm of z3: %lf\n", z3.norm());
			// }

			if (instruction.update_z4) {
				// z4 = z4 + gamma_val * rho4 * ((*E_ptr) * x_sol + y3 - (*f_ptr));
				fprintf(fp, "For z4\nnorm of z4: %.9lf\n", z4.norm());
				// fprintf(fp, "norm of gamma_val: %.9lf\n", gamma_val);
				// fprintf(fp, "norm of rho4: %.9lf\n", rho4);
				// fprintf(fp, "norm of (*E_ptr): %.9lf\n", (*E_ptr).norm());
				// fprintf(fp, "norm of x_sol: %.9lf\n", x_sol.norm());
				// fprintf(fp, "norm of y3: %.9lf\n", y3.norm());
				// fprintf(fp, "norm of (*f_ptr): %.9lf\n\n", (*f_ptr).norm());
			}

			// fprintf(fp, "norm of cur_idx: %lf\n", cur_idx.norm());
			// fprintf(fp, "rho1: %lf\n", rho1);
			// fprintf(fp, "rho2: %lf\n", rho2);
			// if (instruction.update_rho3) {
			// 	fprintf(fp, "rho3: %lf\n", rho3);
			// }
			// if (instruction.update_rho4) {
			// 	fprintf(fp, "rho4: %lf\n", rho4);
			// }
			fprintf(fp,"LongkangIter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter+1,x_sol.norm(),obj_val,cur_obj);
			end = std::chrono::steady_clock::now();
			time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //micro
			time_elapsed = 1.0*time_elapsed/1000;
			fprintf(fp, "Time elapsed: %lfs\n", time_elapsed);
			fprintf(fp, "-------------------------------------------------\n");
		}
	}

	// printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter+1,x_sol.norm(),obj_val,cur_obj);
	
		
	// this->sol.x_sol = new DenseVector(x_sol);
	// this->sol.y1 = new DenseVector(y1);
	// this->sol.y2 = new DenseVector(y2);
	// this->sol.best_sol = new DenseVector(best_sol);

	end = std::chrono::steady_clock::now();
	time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //micro
	time_elapsed = 1.0*time_elapsed/1000;
	fprintf(al, "%d,%f,%d,%f\n", file_idx,-cur_obj,iter+1,time_elapsed);

	// std::cout << "Time elapsed: " << time_elapsed << "us//" << 1.0*time_elapsed/1000 << "s;" << std::endl;
	if (does_log) {
		fprintf(fp, "Time elapsed: %lfs\n", time_elapsed);
		fclose(fp);
	} 
	
	if(print_fix_info==2) fclose(xi);
	fclose(al);

	// this->sol.time_elapsed = time_elapsed;

	return ret;
}

// lpbox with Learning to Early Fix. 
int LPboxADMMsolver::ADMM_lp_iters_l2f(int iter_start, int iter_end, double* vec, int fix_num){
	int ret = 0; // determine converge or not. 
	std::chrono::steady_clock::time_point start, end;
	start = std::chrono::steady_clock::now();
	// printf("Longkang test 0\n");
	FILE *fp;
	if (does_log) {
		fp = fopen(log_file_path.c_str(), "a+");
	}
	// printf("longkang test 1\n");
	_double_t obj_val;
	// int iter;
	long time_elapsed = 0;
	// x_count = DenseVector::Zero(n);
	// x_flag = DenseVector::Zero(n);
	x_iters = DenseMatrix::Zero(n-fix_num,500); 
	int step = (iter_end - iter_start + 1) / 20; //1000 / 20 = 50, 500 / 20 = 25; 
	int cc = 0;
	// x_iters = DenseMatrix(25,n);

	// printf("Test for vec\n");
	// for(int i = 0; i < 10; i++){
	// 	std::cout << vec[i] << ", ";
	// }
	// std::cout <<  std::endl;

	if(fix_num!=0)
	{ 
			fix_sum += fix_num;
			Eigen::ArrayXi fix_idx(fix_num);         //fixed index 
			Eigen::ArrayXi non_fix_idx(n-fix_num);   //left index
			DenseVector org_fix_val(fix_num);     // fixed value

	
			std::vector<Triplet> efix; // E elements for fixing.
			std::vector<Triplet> enonfix; // E elements for not fixing.
			int j=0, k=0;
			for(int i = 0; i < n; i++){
				if(vec[i] == 1){  // 1 - fix to 1
					fix_idx[j] = i;
					org_fix_val[j] = 1;
					j++;
				}					
				else if(vec[i] == 0){  // 0 - fix to 0
					fix_idx[j] = i;
					org_fix_val[j] = 0;
					j++;
				}
				else{  // -1 - not fix
					non_fix_idx[k] = i;
					k++;
				}


				if(vec[i] == -1){
					for(Eigen::SparseMatrix<_double_t, Eigen::ColMajor>::InnerIterator it(*E_ptr,i); it; ++it){
						enonfix.push_back(Triplet(it.row(), k-1, it.value()));
					}
				}
				else{
					for(Eigen::SparseMatrix<_double_t, Eigen::ColMajor>::InnerIterator it(*E_ptr,i); it; ++it){
						// it.row(); //row index
						// it.col(); // column index, == k 
						// it.value(); // value 
						efix.push_back(Triplet(it.row(), j-1, it.value()));
					}
				}


			}

			// printf("LK test: E size: %ld, %ld\n",E_ptr->rows(), E_ptr->cols());
			// printf("Test 99\n");
			E1 = SparseMatrix((*E_ptr).rows(), k);
			E1.setFromTriplets(enonfix.begin(), enonfix.end()); 
			E1.makeCompressed();
			// printf("LK test: E1 size: %ld, %ld\n",E1.rows(), E1.cols());
			

			// printf("Test 100\n");
			// E2 = dE(Eigen::all,fix_idx).sparseView(); 
			E2 = SparseMatrix((*E_ptr).rows(), j);
			E2.setFromTriplets(efix.begin(), efix.end());
			E2.makeCompressed();
			// printf("LK test: E2 size: %ld, %ld\n",E2.rows(), E2.cols());



			if(j!=fix_num) printf("######## Error in iter: j!=fix_num");

			if(print_fix_info==1){
			printf("LK test: fixed_idx size: %ld\n",fix_idx.rows());
			printf("LK test: non_fixed_idx size: %ld\n",non_fix_idx.rows());}
			
			org_fix_idx = left_idx(fix_idx);
			org_non_fix_idx = left_idx(non_fix_idx);
			left_idx = org_non_fix_idx;

			if(print_fix_info==1){
			printf("LK test: fix_val size: %ld\n",org_fix_val.rows());
			printf("LK test: fix_idx size: %ld\n",org_fix_idx.rows());
			printf("LK test: left_idx size: %ld\n",left_idx.rows());}

			ret_idx.resize(ret_idx_prev.rows()+org_fix_idx.rows(), 1);
			ret_idx << ret_idx_prev, org_fix_idx;
			ret_val.resize(ret_val_prev.rows()+org_fix_val.rows(), 1);
			ret_val << ret_val_prev, org_fix_val;
			ret_idx_prev = ret_idx;
			ret_val_prev = ret_val;

			if(print_fix_info==1){
			printf("LK test: ret_idx size: %ld\n",ret_idx.rows());
			printf("LK test: ret_val size: %ld\n",ret_val.rows());}

			if(n-fix_num==0){
				ret = 1;
				n = 0;
				iter_end = iter_start;
				// x_sol = 0 * x_sol;
			}

			else
		{

			x_sol = x_sol(non_fix_idx); // what is x_sol is empty
			if(x_sol.norm()<1e-3) ret=1;

			x_prev = x_sol;
			prev_idx = (x_prev.array() >= 0.5).matrix().cast<_double_t>();

			y1 = y1(non_fix_idx); y2 = y2(non_fix_idx); //y3 = y3(non_fix_idx);
			z1 = z1(non_fix_idx); z2 = z2(non_fix_idx); 

			b1 = (*b_ptr)(non_fix_idx); b2 = (*b_ptr)(fix_idx);
			
			if(print_fix_info==1){
			printf("LK test: b1 size: %ld\n",b1.rows());
			printf("LK test: b2 size: %ld\n",b2.rows());}

			x2 = org_fix_val;//(org_fix_val.array() >= 0.5).matrix().cast<_double_t>(); //1-non_fixed; 2-fixed;
			// printf("Test 101: x2.rows=%ld; b2.rows=%ld\n",x2.rows(),b2.rows());
			fix_obj = compute_cost_lp(x2, b2);
			// printf("Test 102\n");
			if(std::isnan(fix_obj)) 
			{
				printf("############ return error in computation~\n");
				printf("############ Print Before b_ptr\n");
				print_vector(*b_ptr);
			}

			prev_sum = sum_fix_obj;
			sum_fix_obj += fix_obj;

			prev_obj = cur_obj;		
			// printf("Test 103\n");



			// avoid go sparse from dense. 
			// dE = Eigen::MatrixXd(*E_ptr); //from sparse to dense. 
			// SparseMatrix sm_tmp = *E_ptr; 
			// printf("this is the outersize of E: %ld\n",(*E_ptr).outerSize());
			// printf("this is the innersize of E: %ld\n",(*E_ptr).innerSize());
			// printf("LK test: dE size: %ld, %ld\n",dE.rows(),dE.cols());


			
			// printf("Test 103-1\n");
			// dE = Eigen::MatrixXd(*E_ptr); //from sparse to dense.
			// E1 = dE(Eigen::all,non_fix_idx).sparseView();
			// E2 = dE(Eigen::all,fix_idx).sparseView();


			// printf("LK test: E1 size: %ld, %ld\n",E1.rows(), E1.cols());
			// printf("LK test: E2 size: %ld, %ld\n",E2.rows(), E2.cols());

			// printf("Test 103-3\n");

			mat_mul_vec(E2, x2, temp_vec_for_mat_mul);

			f1 = (*f_ptr) - temp_vec_for_mat_mul;
			// printf("Test 104\n");
			if(print_fix_info==1){
			printf("LK test: dE size: %ld, %ld\n",dE.rows(),dE.cols());
			printf("LK test: E1 size: %ld, %ld\n",E1.rows(), E1.cols());
			printf("LK test: E2 size: %ld, %ld\n",E2.rows(), E2.cols());
			printf("LK test: E size: %ld, %ld\n",(*E_ptr).rows(), (*E_ptr).cols());
			printf("LK test: f1 size: %ld\n",f1.rows());
			printf("LK test: F norm: %f\n",f1.norm());
			printf("LK test: E norm: %f\n",E1.norm());}
			
			// printf("Test 105\n");

			// E_ptr = &E1;
			// f_ptr = &f1;
			// b_ptr = &b1;

			n = n - fix_num;

			E_ptr = new SparseMatrix(E1.rows(), n);
			*E_ptr = E1;
			f_ptr = new DenseVector(E1.rows());
			*f_ptr = f1;
			b_ptr = new DenseVector(n);
			*b_ptr = b1;


			// printf("Test 106\n");


			if(std::isnan(fix_obj)) 
			{
				printf("############ Print After b_ptr (should = b1)\n");
				print_vector(*b_ptr);
				printf("############ Print Non-Fixed Index\n");
				print_index(non_fix_idx);
				printf("############ Print b1 (non_fixed)\n");
				print_vector(b1);
				printf("############ Print b2 (fixed)\n");
				print_vector(b2);
				printf("############ Print Fixed Index\n");
				print_index(fix_idx);

			}

			if(print_fix_info==1){
			printf("LK test: E_ptr size: %ld, %ld\n",E_ptr->rows(), E_ptr->cols());
			printf("LK test: f_ptr size: %ld\n",f_ptr->rows());
			}

			// printf("Test 107\n");
			update_expression(iter);	

		}

			printf("Iter: %d; Fixed %ld Elements; Totally Fixed %d Elements; Left %ld Elements; Sum_fix_obj: %f\n",
					iter_start,fix_idx.rows(),fix_sum,non_fix_idx.rows(),sum_fix_obj);
	}

	



	for (iter = iter_start; iter < iter_end; iter++) {

		if (does_log) {
			fprintf(fp, "Iteration: %d\n", iter);
		}

		/*
			update y1	
		*/
		temp_vec = x_sol + z1 / rho1;

		/* Project vector on [0, 1] box */
		project_box(n, temp_vec, y1);


		/*
			update y2	
		*/
		temp_vec = x_sol + z2 / rho2;

		/* Project vector on shifted lp box */
		project_shifted_Lp_ball(n, temp_vec, projection_lp, y2);
		// y2 = y1;


		/*
			update y3	
		*/
		if (instruction.update_y3) {
			mat_mul_vec(*E_ptr, x_sol, temp_vec_for_mat_mul);
			y3 = *f_ptr - temp_vec_for_mat_mul - z4 / rho4;
			project_vec_less_than(y3, y3, 0, 0); 
		}
		// printf("Longkang test 2.\n");
// printf("Error out 1: sum: %f; obj: %f\n",sum_fix_obj,cur_obj);
		/*
			update x	
		*/

		/* If the iteration is nonzero and it divides rho_change_step, it means
		 * that the rho updated in the last iteration
		 */

		/*	update: 
				E_transpose, rho4_E_transpose;
				_2A_plus_rho1_rho2, preconditioner_diag_mat, Esq_diag, 
				matrix_expressions, 
		*/      
		if(iter==0)
			update_expression(0);

		if (iter != 0 && rhoUpdated) {
			/* Note that we need the previous rho to update in order to get the difference between
			 * the updated matrix and the not updated matrix. Another possible calculation is by
			 * calculating rho * rho_change_ration / learning_fact
			 */
			_2A_plus_rho1_rho2.diagonal().array() += rho_change_ratio * (prev_rho1 + prev_rho2);
			if (instruction.problem_type != unconstrained) {
				preconditioner_diag_mat.diagonal().array() += rho_change_ratio * (prev_rho1 + prev_rho2);
			}			

			if (instruction.update_rho4) {
				preconditioner_diag_mat.diagonal().array() += rho_change_ratio * prev_rho4 * Esq_diag.array();
				rho4_E_transpose = learning_fact * rho4_E_transpose;
			}
		}
		// printf("Longkang test 3.\n");


		/* If the problem in equality, the rhs vector is 
		 * rho1 * y1 + rho2 * y2 + rho4 * E^T * (f - y3) - (b + z1 + z2 + E^T * z4)
		 */
		if (instruction.problem_type == inequality) {
			temp_vec = rho1 * y1 + rho2 * y2 - (*b_ptr + z1 + z2);
			// printf("Longkang test 31.\n");
			mat_mul_vec(rho4_E_transpose, *f_ptr - y3, temp_vec_for_mat_mul);
			temp_vec += temp_vec_for_mat_mul;
			// printf("Longkang test 32.\n");
			mat_mul_vec(E_transpose, z4, temp_vec_for_mat_mul);
			temp_vec -= temp_vec_for_mat_mul;
		}
		// printf("Longkang test 4.\n");

// printf("Error out 2: sum: %f; obj: %f\n",sum_fix_obj,cur_obj);
		/* Explicit version of conjugate gradient used for profiling */

		if (rhoUpdated) {
			if (instruction.problem_type != unconstrained) { // inequal 
				// printf("Longkang test 41.\n");
				diagonalPreconditioner.compute(preconditioner_diag_mat);
			} else {
				diagonalPreconditioner.compute(_2A_plus_rho1_rho2);
			}
			rhoUpdated = false;
		}
		// printf("Longkang test 42.\n");
		_double_t tol = pcg_tol;
		// x_sol = y1;
		x_sol_try = y1;
		int maxiter = pcg_maxiters;
		// if(x_sol.norm()<1e-3){
		// 	ret = 1;
		// 	break;
		// }

		// x_sol_try = x_sol;
		int cg = _conjugate_gradient(matrix_expressions, temp_vec, x_sol_try, diagonalPreconditioner, maxiter, tol, 
				temp_vec_for_cg, temp_vec_for_mat_mul,log_file_path.c_str(), iter); //
		
		if(cg == -1){ //p<0, wrong in pcg
			// x_sol = x_prev;
			// prev_obj_bin;
			printf("Early exit cg=-1 out: sum: %f; obj: %f; ALL: %f\n",prev_obj,prev_sum, prev_obj+prev_sum);
			return 1;
			// SparseMatrix esq = (*E_ptr).transpose() * (*E_ptr);
			// esq += _2A_plus_rho1_rho2;
			// printf("Longkang Test: Now in BiCGSTAB\n");
			// Eigen::BiCGSTAB<Eigen::SparseMatrix<double> > solver;
			// // printf("Longkang Test: Now in Conjucate Gradient\n");
			// // Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > solver;
			// // printf("Longkang Test: Now in GMRES\n");
			// // Eigen::GMRES<Eigen::SparseMatrix<double> > solver;
			// // printf("Longkang Test: Now in MINRES\n");
			// // Eigen::MINRES<Eigen::SparseMatrix<double> > solver;
			// solver.compute(esq);
			// x_sol = solver.solve(temp_vec);
		}
		else{
			x_sol = x_sol_try;
		}

		if(1){ //(iter-iter_start) % step == 10
			x_iters.block(0,cc,n,1) = x_sol;
			cc++;
		}

		
		if (does_log) {
			fprintf(fp, "Conjugate gradient stops after %d iterations\n", maxiter);
			fprintf(fp, "Conjugate gradient stops with residual %lf\n", tol);
		}

		// printf("Longkang test 5.\n");
		/*
			update z1, z2, z4	
		*/
		z1 = z1 + gamma_val * rho1 * (x_sol - y1);
		z2 = z2 + gamma_val * rho2 * (x_sol - y2);
		if (instruction.update_z4) {
			z4 = z4 + gamma_val * rho4 * ((*E_ptr) * x_sol + y3 - (*f_ptr));
		}

		// printf("Longkang test 6.\n");

// printf("Error out 3: sum: %f; obj: %f\n",sum_fix_obj,cur_obj);
		

		/*
			Evaluate 	
		*/
		_double_t temp0 = std::max(x_sol.norm(), _double_t(2.2204e-16));
		cvg_test1 = (x_sol - y1).norm() / temp0;
		cvg_test2 = (x_sol - y2).norm() / temp0;
		if (cvg_test1 <= stop_threshold && cvg_test2 <= stop_threshold) {
			ret = 1;
			printf("Stop becuase y1_y2. iter: %d, stop_threshold: %.6f\n", iter, std::max(cvg_test1, cvg_test2));
			if (does_log) {
				fprintf(fp, "iter: %d, stop_threshold: %.6f\n", iter, std::max(cvg_test1, cvg_test2));
			}
			break;
		}
		// printf("Longkang test 8.\n");


		if ((iter+1) % rho_change_step == 0) {
			prev_rho1 = rho1;
			prev_rho2 = rho2;
			rho1 = learning_fact * rho1;
			rho2 = learning_fact * rho2;

			if (instruction.update_rho3) {
				prev_rho3 = rho3;
				rho3 = learning_fact * rho3;
			}

			if (instruction.update_rho4) {
				prev_rho4 = rho4;
				rho4 = learning_fact * rho4;
			}

			gamma_val = std::max(gamma_val * gamma_factor, _double_t(1.0));
			rhoUpdated = true;
			rho_change_ratio = learning_fact - 1.0;
		}

		obj_val = compute_cost_lp(x_sol, *b_ptr);
		obj_list.push_back(obj_val);
		if (obj_list.size() >= history_size) {
			std_obj = compute_std_obj(obj_list, history_size);
		}
		if (std_obj <= std_threshold) {
			ret = 1;
			if (does_log) {
				fprintf(fp, "iter: %d, std_threshold: %.6f\n", iter, std_obj);
				// printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter,x_sol.norm(),obj_val,cur_obj);
		
			}
			printf("Stop because std_obj. iter: %d, std_threshold: %.6f\n", iter, std_obj);
			// printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter,x_sol.norm(),obj_val,cur_obj);		
			break;
		}

		// printf("Longkang test 9.\n");
// printf("Error out 4: sum: %f; obj: %f\n",sum_fix_obj,cur_obj);
		cur_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
		prev_idx = cur_idx;
		cur_obj = compute_cost_lp(prev_idx, *b_ptr);

		if (best_bin_obj >= cur_obj) {
			best_bin_obj = cur_obj;
			best_sol = x_sol;
		}
	
	}

	// printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter,x_sol.norm(),obj_val,cur_obj);

	if (does_log) {
		// fprintf(fp, "Time elapsed: %ldus\n", time_elapsed);
		fclose(fp);
	} 
	
	return ret;
}

// check constraints feasibility. return the number of infeasible constraints.
int LPboxADMMsolver::check_infeasible_lpbox(){
	int inf=0;
	int fea=0;
	mat_mul_vec(*E_ptr, x_sol, temp_vec_for_mat_mul);
	// y3 = *f_ptr - temp_vec_for_mat_mul - z4 / rho4;
	int len = (*E_ptr).rows();
	for(int i = 0; i < len; i++){
		if(temp_vec_for_mat_mul.array()[i]<=1.0) 
			fea++;
		else
			inf++;
	}
	printf("Total constraints: [%d], Feasible: [%d], Infeasible: [%d]\n", len, fea, inf);
	return inf;
}

int LPboxADMMsolver::check_infeasible_l2f(){ //////////////
	int inf=0;
	int fea=0;
	double* org_sol = new double[org_n];
	org_sol = get_x_sol();
	DenseVector tmp_sol = DenseVector::Zero(org_n);
	for(int i = 0; i < org_n; i++){
		tmp_sol.array()[i] = org_sol[i];
	}
	int len = (*org_E_ptr).rows();
	mat_mul_vec(*org_E_ptr, tmp_sol, temp_vec_for_mat_mul);
	for(int i = 0; i < len; i++){
		if(temp_vec_for_mat_mul.array()[i]<=1.0) 
			fea++;
		else
			inf++;
	}
	printf("Total constraints: [%d], Feasible: [%d], Infeasible: [%d]\n", len, fea, inf);
	return inf;
}


// get x_iterations for Learning to Early Fix. 
double* LPboxADMMsolver::get_x_iters_d(int ws){ 
	int rows = x_iters.rows(); // n_num
	int cols = ws; // 500/  100
	// printf("In get_x_iters_d: rows:%d, cols:%d, n: %d.\n",rows, cols, n);
	double* ret = new double[rows*cols]; 
	for(int i = 0; i < rows; i++){
		for(int j = 0 ; j < cols; j++){
			ret[i*cols + j] = x_iters(i,j);
		}
	}
	return ret;
}

// get objective. 
double LPboxADMMsolver::cal_obj(){
	int len = x_sol.rows();
	if(n!=0){
		// printf("Calculate objective:\n");
		// printf("fixed sum: %d; x_sol size: %d; n=%d; \n",fix_sum, len, n);
		// printf("Final Objective: %f\n",sum_fix_obj+cur_obj);
		return sum_fix_obj+cur_obj;
	}
	else{
		printf("Final Objective: %f\n",sum_fix_obj);
		return sum_fix_obj;
	}
}

double LPboxADMMsolver::get_curBinObj(){
	return cur_obj;
}

double* LPboxADMMsolver::get_x_sol(){
	if(n!=0){
		ret_idx.resize(ret_idx_prev.rows()+left_idx.rows(), 1);
		ret_val.resize(ret_val_prev.rows()+x_sol.rows(), 1);
		ret_idx << ret_idx_prev, left_idx;
		cur_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
		ret_val << ret_val_prev, cur_idx;
	}
	int len = ret_idx.rows();
	double* ret_sol = new double[len];
	for(int i = 0; i < len; i++){
		// int index = static_cast<int>(ret_idx[i]);
		int index = int(ret_idx[i]);
		double val = ret_val[i];
		ret_sol[index] = val; 
	}
	return ret_sol;
}


double* LPboxADMMsolver::get_final_x_sol(){
	// if(n!=0){
	// 	ret_idx.resize(ret_idx_prev.rows()+left_idx.rows(), 1);
	// 	ret_val.resize(ret_val_prev.rows()+x_sol.rows(), 1);
	// 	ret_idx << ret_idx_prev, left_idx;
	// 	cur_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
	// 	ret_val << ret_val_prev, cur_idx;
	// }
	int len = x_sol.rows();
	double* ret_sol = new double[len];
	for(int i = 0; i < len; i++){
		// int index = static_cast<int>(ret_idx[i]);
		// int index = int(ret_idx[i]);
		// double val = ret_val[i];
		ret_sol[i] = x_sol.array()[i]; 
	}
	return ret_sol;
}


// rule based fix
int LPboxADMMsolver::ADMM_lp_iters_fix(int iter_start, int iter_end){
	int ret = 0; // determine converge or not. 
	std::chrono::steady_clock::time_point start, end;
	start = std::chrono::steady_clock::now();
	// printf("Longkang test 0\n");
	FILE *fp;
	if (does_log) {
		fp = fopen(log_file_path.c_str(), "a+");
	}
	// printf("longkang test 1\n");
	_double_t obj_val;
	int iter;
	long time_elapsed = 0;
	x_count = DenseVector::Zero(n);
	x_flag = DenseVector::Zero(n);
	for (iter = iter_start; iter < iter_end; iter++) {

		// printf("Longkang test 1.\n");
		if (does_log) {
			fprintf(fp, "Iteration: %d\n", iter);
		}

		/*
			update y1	
		*/
		temp_vec = x_sol + z1 / rho1;

		/* Project vector on [0, 1] box */
		project_box(n, temp_vec, y1);


		/*
			update y2	
		*/
		temp_vec = x_sol + z2 / rho2;

		/* Project vector on shifted lp box */
		project_shifted_Lp_ball(n, temp_vec, projection_lp, y2);


		/*
			update y3	
		*/
		if (instruction.update_y3) {
			mat_mul_vec(*E_ptr, x_sol, temp_vec_for_mat_mul);
			y3 = *f_ptr - temp_vec_for_mat_mul - z4 / rho4;
			project_vec_less_than(y3, y3, 0, 0); 
		}
		// printf("Longkang test 2.\n");
// printf("Error out 1: sum: %f; obj: %f\n",sum_fix_obj,cur_obj);
		/*
			update x	
		*/

		/* If the iteration is nonzero and it divides rho_change_step, it means
		 * that the rho updated in the last iteration
		 */

		/*	update: 
				E_transpose, rho4_E_transpose;
				_2A_plus_rho1_rho2, preconditioner_diag_mat, Esq_diag, 
				matrix_expressions, 
		*/      
		if(iter==0)
			update_expression(0);

		if (iter != 0 && rhoUpdated) {
			/* Note that we need the previous rho to update in order to get the difference between
			 * the updated matrix and the not updated matrix. Another possible calculation is by
			 * calculating rho * rho_change_ration / learning_fact
			 */
			_2A_plus_rho1_rho2.diagonal().array() += rho_change_ratio * (prev_rho1 + prev_rho2);
			if (instruction.problem_type != unconstrained) {
				preconditioner_diag_mat.diagonal().array() += rho_change_ratio * (prev_rho1 + prev_rho2);
			}			

			if (instruction.update_rho4) {
				preconditioner_diag_mat.diagonal().array() += rho_change_ratio * prev_rho4 * Esq_diag.array();
				rho4_E_transpose = learning_fact * rho4_E_transpose;
			}
		}
		// printf("Longkang test 3.\n");


		/* If the problem in equality, the rhs vector is 
		 * rho1 * y1 + rho2 * y2 + rho4 * E^T * (f - y3) - (b + z1 + z2 + E^T * z4)
		 */
		if (instruction.problem_type == inequality) {
			temp_vec = rho1 * y1 + rho2 * y2 - (*b_ptr + z1 + z2);
			// printf("Longkang test 31.\n");
			mat_mul_vec(rho4_E_transpose, *f_ptr - y3, temp_vec_for_mat_mul);
			temp_vec += temp_vec_for_mat_mul;
			// printf("Longkang test 32.\n");
			mat_mul_vec(E_transpose, z4, temp_vec_for_mat_mul);
			temp_vec -= temp_vec_for_mat_mul;
		}
		// printf("Longkang test 4.\n");

// printf("Error out 2: sum: %f; obj: %f\n",sum_fix_obj,cur_obj);
		/* Explicit version of conjugate gradient used for profiling */

		if (rhoUpdated) {
			if (instruction.problem_type != unconstrained) { // inequal 
				// printf("Longkang test 41.\n");
				diagonalPreconditioner.compute(preconditioner_diag_mat);
			} else {
				diagonalPreconditioner.compute(_2A_plus_rho1_rho2);
			}
			rhoUpdated = false;
		}
		// printf("Longkang test 42.\n");
		_double_t tol = pcg_tol;
		// x_sol = y1;
		x_sol_try = y1;
		int maxiter = pcg_maxiters;
		// if(x_sol.norm()<1e-3){
		// 	ret = 1;
		// 	break;
		// }

		// x_sol_try = x_sol;
		int cg = _conjugate_gradient(matrix_expressions, temp_vec, x_sol_try, diagonalPreconditioner, maxiter, tol, 
				temp_vec_for_cg, temp_vec_for_mat_mul,log_file_path.c_str(), iter); //
		
		if(cg == -1){ //p<0, wrong in pcg
			// x_sol = x_prev;
			// prev_obj_bin;
			printf("Early exit cg=-1 out: sum: %f; obj: %f; ALL: %f\n",prev_obj,prev_sum, prev_obj+prev_sum);
			return 1;


			// SparseMatrix esq = (*E_ptr).transpose() * (*E_ptr);
			// esq += _2A_plus_rho1_rho2;
			// printf("Longkang Test: Now in BiCGSTAB\n");
			// Eigen::BiCGSTAB<Eigen::SparseMatrix<double> > solver;
			// // printf("Longkang Test: Now in Conjucate Gradient\n");
			// // Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > solver;
			// // printf("Longkang Test: Now in GMRES\n");
			// // Eigen::GMRES<Eigen::SparseMatrix<double> > solver;
			// // printf("Longkang Test: Now in MINRES\n");
			// // Eigen::MINRES<Eigen::SparseMatrix<double> > solver;
			// solver.compute(esq);
			// x_sol = solver.solve(temp_vec);
		}
		else{
			x_sol = x_sol_try;
		}

		
		if (does_log) {
			fprintf(fp, "Conjugate gradient stops after %d iterations\n", maxiter);
			fprintf(fp, "Conjugate gradient stops with residual %lf\n", tol);
		}

		// printf("Longkang test 5.\n");
		/*
			update z1, z2, z4	
		*/
		z1 = z1 + gamma_val * rho1 * (x_sol - y1);
		z2 = z2 + gamma_val * rho2 * (x_sol - y2);
		if (instruction.update_z4) {
			z4 = z4 + gamma_val * rho4 * ((*E_ptr) * x_sol + y3 - (*f_ptr));
		}

		// printf("Longkang test 6.\n");

// printf("Error out 3: sum: %f; obj: %f\n",sum_fix_obj,cur_obj);
		/*
			Early Fix
		*/
		consistency = 5;
		fix_threshold = 1e-3; 
		x_det = (abs(x_sol.array() - x_prev.array())<=fix_threshold).matrix().cast<_double_t>();
		for(int i = 0 ; i < n; i++){
			if(x_det.array()[i]){
				x_count.array()[i] += 1;
				if(x_count.array()[i] >= consistency) 
					x_flag.array()[i] = 1; 
			}
			else
				x_count.array()[i] = 0;
		}
		x_prev = x_sol; 
		// printf("Longkang test 7.\n");

		/*
			Evaluate 	
		*/
		_double_t temp0 = std::max(x_sol.norm(), _double_t(2.2204e-16));
		cvg_test1 = (x_sol - y1).norm() / temp0;
		cvg_test2 = (x_sol - y2).norm() / temp0;
		if (cvg_test1 <= stop_threshold && cvg_test2 <= stop_threshold) {
			printf("iter: %d, stop_threshold: %.6f\n", iter, std::max(cvg_test1, cvg_test2));
			if (does_log) {
				fprintf(fp, "iter: %d, stop_threshold: %.6f\n", iter, std::max(cvg_test1, cvg_test2));
			}
			break;
		}
		// printf("Longkang test 8.\n");


		if ((iter+1) % rho_change_step == 0) {
			prev_rho1 = rho1;
			prev_rho2 = rho2;
			rho1 = learning_fact * rho1;
			rho2 = learning_fact * rho2;

			if (instruction.update_rho3) {
				prev_rho3 = rho3;
				rho3 = learning_fact * rho3;
			}

			if (instruction.update_rho4) {
				prev_rho4 = rho4;
				rho4 = learning_fact * rho4;
			}

			gamma_val = std::max(gamma_val * gamma_factor, _double_t(1.0));
			rhoUpdated = true;
			rho_change_ratio = learning_fact - 1.0;
		}

		obj_val = compute_cost_lp(x_sol, *b_ptr);
		obj_list.push_back(obj_val);
		if (obj_list.size() >= history_size) {
			std_obj = compute_std_obj(obj_list, history_size);
		}
		if (std_obj <= std_threshold) {
			ret = 1;
			if (does_log) {
				fprintf(fp, "iter: %d, std_threshold: %.6f\n", iter, std_obj);
				// printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter,x_sol.norm(),obj_val,cur_obj);
		
			}
			// printf("iter: %d, std_threshold: %.6f\n", iter, std_obj);
			// printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n",iter,x_sol.norm(),obj_val,cur_obj);		
			break;
		}

		// printf("Longkang test 9.\n");
// printf("Error out 4: sum: %f; obj: %f\n",sum_fix_obj,cur_obj);
		cur_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
		prev_idx = cur_idx;
		cur_obj = compute_cost_lp(prev_idx, *b_ptr);
		// prev_obj_bin = cur_obj;

		if (best_bin_obj >= cur_obj) {
			best_bin_obj = cur_obj;
			best_sol = x_sol;
		}
		// printf("Longkang test 10.\n");

		/*
			Early Fix
		*/
		fix_n = x_flag.sum();
		printf("LK test: new_n: %d\n",fix_n);
		// printf("Longkang test 101.\n");
		if(iter==iter_end-1){
			printf("LK test: new_n: %d\n",fix_n);
		}
		if(fix_n<=10) fix_n=0;
// printf("Error out 5: sum: %f; obj: %f\n",sum_fix_obj,cur_obj);
		int tmp = 1;
		if(tmp==1 && fix_n!=0 ){ 
			// printf("Longkang test 102.\n");			
			// printf("LK test: new_n: %d\n",fix_n);
			// if (!fix_n) continue;
			fix_sum += fix_n;
			Eigen::ArrayXi fix_idx(fix_n);         //fixed index 
			Eigen::ArrayXi non_fix_idx(n-fix_n);   //left index
			int j=0, k=0;
			for(int i = 0; i < n; i++){
				if(x_flag.array()[i]){
					fix_idx[j] = i;
					j++;
				}					
				else{
					non_fix_idx[k] = i;
					k++;
				}
			}
			printf("LK test: fixed_idx size: %ld\n",fix_idx.rows());
			printf("LK test: non_fixed_idx size: %ld\n",non_fix_idx.rows());
			org_fix_val = x_sol(fix_idx);
			org_fix_idx = left_idx(fix_idx);
			org_non_fix_idx = left_idx(non_fix_idx);
			printf("LK test: fix_val size: %ld\n",org_fix_val.rows());
			printf("LK test: fix_idx size: %ld\n",org_fix_idx.rows());

			left_idx = org_non_fix_idx;
			printf("LK test: left_idx size: %ld\n",left_idx.rows());

			ret_idx.resize(ret_idx_prev.rows()+org_fix_idx.rows(), 1);
			ret_idx << ret_idx_prev, org_fix_idx;
			ret_val.resize(ret_val_prev.rows()+org_fix_val.rows(), 1);
			ret_val << ret_val_prev, org_fix_val;
			ret_idx_prev = ret_idx;
			ret_val_prev = ret_val;
			printf("LK test: ret_idx size: %ld\n",ret_idx.rows());
			printf("LK test: ret_val size: %ld\n",ret_val.rows());


			if(n-fix_n==0){
				ret = 1;
				n = 0;
				// x_sol = 0 * x_sol;
			}
			else{

			x_sol = x_sol(non_fix_idx); // what is x_sol is empty
			if(x_sol.norm()<1e-3) ret=1;

			x_prev = x_sol;
			prev_idx = (x_prev.array() >= 0.5).matrix().cast<_double_t>();

			y1 = y1(non_fix_idx); y2 = y2(non_fix_idx); //y3 = y3(non_fix_idx);
			z1 = z1(non_fix_idx); z2 = z2(non_fix_idx); 

			b1 = (*b_ptr)(non_fix_idx); b2 = (*b_ptr)(fix_idx);
			printf("LK test: b1 size: %ld\n",b1.rows());
			printf("LK test: b2 size: %ld\n",b2.rows());

			x2 = (org_fix_val.array() >= 0.5).matrix().cast<_double_t>(); //1-non_fixed; 2-fixed;
			fix_obj = compute_cost_lp(x2, b2);
			prev_sum = sum_fix_obj;
			sum_fix_obj += fix_obj;

			prev_obj = cur_obj;

		

			dE = Eigen::MatrixXd(*E_ptr); //from sparse to dense. 
			printf("LK test: dE size: %ld, %ld\n",dE.rows(),dE.cols());

			E1 = dE(Eigen::all,non_fix_idx).sparseView(); 
			E2 = dE(Eigen::all,fix_idx).sparseView(); 
			f1 = (*f_ptr) - E2 * x2;
			printf("LK test: E1 size: %ld, %ld\n",E1.rows(), E1.cols());
			printf("LK test: E2 size: %ld, %ld\n",E2.rows(), E2.cols());
			printf("LK test: E size: %ld, %ld\n",(*E_ptr).rows(), (*E_ptr).cols());
			printf("LK test: f1 size: %ld\n",f1.rows());
			printf("LK test: F norm: %f\n",f1.norm());
			printf("LK test: E norm: %f\n",E1.norm());
			E_ptr = &E1;
			f_ptr = &f1;
			b_ptr = &b1;
			n = n - fix_n;
			printf("LK test: E_ptr size: %ld, %ld\n",E_ptr->rows(), E_ptr->cols());
			printf("LK test: f_ptr size: %ld\n",f_ptr->rows());

			// update_expression(iter);	

			}

			// printf("Iter: %d; Fixed %ld Elements; Totally Fixed %d Elements; Left %ld Elements; Sum_fix_obj: %f\n",
					// iter,fix_idx.rows(),fix_sum,non_fix_idx.rows(),sum_fix_obj);
		}

		int fix_num = fix_n;
		if(tmp==2 && fix_num!=0)
		{ 
			fix_sum += fix_num;
			Eigen::ArrayXi fix_idx(fix_num);         //fixed index 
			Eigen::ArrayXi non_fix_idx(n-fix_num);   //left index
			DenseVector org_fix_val(fix_num);     // fixed value

	
			std::vector<Triplet> efix; // E elements for fixing.
			std::vector<Triplet> enonfix; // E elements for not fixing.
			int j=0, k=0;
			for(int i = 0; i < n; i++){
				if(x_flag.array()[i] == 1){  // 1 - fix to 1
					fix_idx[j] = i;
					org_fix_val[j] = 1;
					j++;
				}					
				// else if(vec[i] == 0){  // 0 - fix to 0
				// 	fix_idx[j] = i;
				// 	org_fix_val[j] = 0;
				// 	j++;
				// }
				else{  // -1 - not fix
					non_fix_idx[k] = i;
					k++;
				}


				if(x_flag.array()[i] == 0){
					for(Eigen::SparseMatrix<_double_t, Eigen::ColMajor>::InnerIterator it(*E_ptr,i); it; ++it){
						enonfix.push_back(Triplet(it.row(), k-1, it.value()));
					}
				}
				else{
					for(Eigen::SparseMatrix<_double_t, Eigen::ColMajor>::InnerIterator it(*E_ptr,i); it; ++it){
						// it.row(); //row index
						// it.col(); // column index, == k 
						// it.value(); // value 
						efix.push_back(Triplet(it.row(), j-1, it.value()));
					}
				}


			}

			// printf("LK test: E size: %ld, %ld\n",E_ptr->rows(), E_ptr->cols());
			// printf("Test 99\n");
			E1 = SparseMatrix((*E_ptr).rows(), k);
			E1.setFromTriplets(enonfix.begin(), enonfix.end()); 
			E1.makeCompressed();
			// printf("LK test: E1 size: %ld, %ld\n",E1.rows(), E1.cols());
			

			// printf("Test 100\n");
			// E2 = dE(Eigen::all,fix_idx).sparseView(); 
			E2 = SparseMatrix((*E_ptr).rows(), j);
			E2.setFromTriplets(efix.begin(), efix.end());
			E2.makeCompressed();
			// printf("LK test: E2 size: %ld, %ld\n",E2.rows(), E2.cols());



			if(j!=fix_num) printf("######## Error in iter: j!=fix_num");

			if(print_fix_info==1){
			printf("LK test: fixed_idx size: %ld\n",fix_idx.rows());
			printf("LK test: non_fixed_idx size: %ld\n",non_fix_idx.rows());}
			
			org_fix_idx = left_idx(fix_idx);
			org_non_fix_idx = left_idx(non_fix_idx);
			left_idx = org_non_fix_idx;

			if(print_fix_info==1){
			printf("LK test: fix_val size: %ld\n",org_fix_val.rows());
			printf("LK test: fix_idx size: %ld\n",org_fix_idx.rows());
			printf("LK test: left_idx size: %ld\n",left_idx.rows());}

			ret_idx.resize(ret_idx_prev.rows()+org_fix_idx.rows(), 1);
			ret_idx << ret_idx_prev, org_fix_idx;
			ret_val.resize(ret_val_prev.rows()+org_fix_val.rows(), 1);
			ret_val << ret_val_prev, org_fix_val;
			ret_idx_prev = ret_idx;
			ret_val_prev = ret_val;

			if(print_fix_info==1){
			printf("LK test: ret_idx size: %ld\n",ret_idx.rows());
			printf("LK test: ret_val size: %ld\n",ret_val.rows());}

			if(n-fix_num==0){
				ret = 1;
				n = 0;
				iter_end = iter_start;
				// x_sol = 0 * x_sol;
			}

			else
		{

			x_sol = x_sol(non_fix_idx); // what is x_sol is empty
			if(x_sol.norm()<1e-3) ret=1;

			x_prev = x_sol;
			prev_idx = (x_prev.array() >= 0.5).matrix().cast<_double_t>();

			y1 = y1(non_fix_idx); y2 = y2(non_fix_idx); //y3 = y3(non_fix_idx);
			z1 = z1(non_fix_idx); z2 = z2(non_fix_idx); 

			b1 = (*b_ptr)(non_fix_idx); b2 = (*b_ptr)(fix_idx);
			
			if(print_fix_info==1){
			printf("LK test: b1 size: %ld\n",b1.rows());
			printf("LK test: b2 size: %ld\n",b2.rows());}

			x2 = org_fix_val;//(org_fix_val.array() >= 0.5).matrix().cast<_double_t>(); //1-non_fixed; 2-fixed;
			// printf("Test 101: x2.rows=%ld; b2.rows=%ld\n",x2.rows(),b2.rows());
			fix_obj = compute_cost_lp(x2, b2);
			// printf("Test 102\n");
			if(std::isnan(fix_obj)) 
			{
				printf("############ return error in computation~\n");
				printf("############ Print Before b_ptr\n");
				print_vector(*b_ptr);
			}

			prev_sum = sum_fix_obj;
			sum_fix_obj += fix_obj;

			prev_obj = cur_obj;		
			// printf("Test 103\n");



			// avoid go sparse from dense. 
			// dE = Eigen::MatrixXd(*E_ptr); //from sparse to dense. 
			// SparseMatrix sm_tmp = *E_ptr; 
			// printf("this is the outersize of E: %ld\n",(*E_ptr).outerSize());
			// printf("this is the innersize of E: %ld\n",(*E_ptr).innerSize());
			// printf("LK test: dE size: %ld, %ld\n",dE.rows(),dE.cols());


			
			// printf("Test 103-1\n");
			// dE = Eigen::MatrixXd(*E_ptr); //from sparse to dense.
			// E1 = dE(Eigen::all,non_fix_idx).sparseView();
			// E2 = dE(Eigen::all,fix_idx).sparseView();


			// printf("LK test: E1 size: %ld, %ld\n",E1.rows(), E1.cols());
			// printf("LK test: E2 size: %ld, %ld\n",E2.rows(), E2.cols());

			// printf("Test 103-3\n");

			mat_mul_vec(E2, x2, temp_vec_for_mat_mul);

			f1 = (*f_ptr) - temp_vec_for_mat_mul;
			// printf("Test 104\n");
			if(print_fix_info==1){
			printf("LK test: dE size: %ld, %ld\n",dE.rows(),dE.cols());
			printf("LK test: E1 size: %ld, %ld\n",E1.rows(), E1.cols());
			printf("LK test: E2 size: %ld, %ld\n",E2.rows(), E2.cols());
			printf("LK test: E size: %ld, %ld\n",(*E_ptr).rows(), (*E_ptr).cols());
			printf("LK test: f1 size: %ld\n",f1.rows());
			printf("LK test: F norm: %f\n",f1.norm());
			printf("LK test: E norm: %f\n",E1.norm());}
			
			// printf("Test 105\n");

			// E_ptr = &E1;
			// f_ptr = &f1;
			// b_ptr = &b1;

			n = n - fix_num;

			E_ptr = new SparseMatrix(E1.rows(), n);
			*E_ptr = E1;
			f_ptr = new DenseVector(E1.rows());
			*f_ptr = f1;
			b_ptr = new DenseVector(n);
			*b_ptr = b1;


			// printf("Test 106\n");


			if(std::isnan(fix_obj)) 
			{
				printf("############ Print After b_ptr (should = b1)\n");
				print_vector(*b_ptr);
				printf("############ Print Non-Fixed Index\n");
				print_index(non_fix_idx);
				printf("############ Print b1 (non_fixed)\n");
				print_vector(b1);
				printf("############ Print b2 (fixed)\n");
				print_vector(b2);
				printf("############ Print Fixed Index\n");
				print_index(fix_idx);

			}

			if(print_fix_info==1){
			printf("LK test: E_ptr size: %ld, %ld\n",E_ptr->rows(), E_ptr->cols());
			printf("LK test: f_ptr size: %ld\n",f_ptr->rows());
			}

			// printf("Test 107\n");
			update_expression(iter);	

		}

			printf("Iter: %d; Fixed %ld Elements; Totally Fixed %d Elements; Left %ld Elements; Sum_fix_obj: %f\n",
					iter_start,fix_idx.rows(),fix_sum,non_fix_idx.rows(),sum_fix_obj);
	}

	
	}

	printf("Longkang- iter: %d;  x_sol: %lf; dou_obj:%lf; bin_obj: %lf\n\n",iter,x_sol.norm(),obj_val,cur_obj);

	// if(ret) 
	// 	printf("obj: %f",sum_fix_obj+cur_obj);
		

	// this->sol.x_sol = new DenseVector(x_sol);
	// this->sol.y1 = new DenseVector(y1);
	// this->sol.y2 = new DenseVector(y2);
	// this->sol.best_sol = new DenseVector(best_sol);

	// end = std::chrono::steady_clock::now();
	// time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //micro
	// // std::cout << "Time elapsed: " << time_elapsed << "us//" << 1.0*time_elapsed/1000 << "s;" << std::endl;
	if (does_log) {
		// fprintf(fp, "Time elapsed: %ldus\n", time_elapsed);
		fclose(fp);
	} 
	// this->sol.time_elapsed = time_elapsed;

	return ret;
}

// update matrix_expressions, preconditioner_diag_mat, preconditioner_diag_mat_triplets
void LPboxADMMsolver::update_expression(int iter){
// printf("Longkang test 10.\n");
	if (instruction.problem_type & inequality) {
		E_transpose = (*E_ptr).transpose();
		rho4_E_transpose = rho4 * E_transpose;
		// printf("Longkang test 111.\n");
		// esq = rho4_E_transpose * (*E_ptr);
		// mat_mul_vec(rho3_C_transpose, *d_ptr, temp_vec_for_mat_mul);
	}

// printf("Longkang test 11.\n");

	// free matrix expression. 
	if(matrix_expressions.size()){
		std::vector<std::vector<const SparseMatrix*>>::iterator iter = matrix_expressions.begin();
		for(; iter!= matrix_expressions.end();){
			// printf("Test in update expression: size matrix_expression: %ld\n",matrix_expressions.size());
			iter = matrix_expressions.erase(iter);
		}
	}
	// printf("Test in update expression: size matrix_expression: %ld\n",matrix_expressions.size());

	
// printf("Longkang test 12.\n");

		/* Storing the matrix 2 * A + (rho1 + rho2) * I to save calculation time */
	// _2A_plus_rho1_rho2 = 2 * (*A_ptr);
	// _2A_plus_rho1_rho2.diagonal().array() += rho1 + rho2;

	// DenseMatrix tmp(n,n);
	// tmp.diagonal().array() = rho1 + rho2;
	// _2A_plus_rho1_rho2 = tmp.sparseView();

	// _2A_plus_rho1_rho2 = SparseMatrix(n,n);
	// _2A_plus_rho1_rho2.setZero();
	// _2A_plus_rho1_rho2.diagonal().array() += rho1 + rho2;




	// SparseMatrix preconditioner_diag_mat(n, n); /* The diagonal matrix used by the preconditioner */
	// preconditioner_diag_mat.reserve(n);
	std::vector<Triplet> preconditioner_diag_mat_triplets;
	preconditioner_diag_mat_triplets.reserve(n);
	/* The diagonal elements in the original expression evaluated by the preconditioner is given by the
	 * diagonal elements of 2 * _A + (rho1 + rho2) * I + rho3 * _C^T * _C
	 */
	for (int i = 0; i < n; i++) {
		preconditioner_diag_mat_triplets.push_back(Triplet(i, i, 0));
	}
	_2A_plus_rho1_rho2.setZero();
	_2A_plus_rho1_rho2 = SparseMatrix(n,n);
	_2A_plus_rho1_rho2.setFromTriplets(preconditioner_diag_mat_triplets.begin(), preconditioner_diag_mat_triplets.end());
	_2A_plus_rho1_rho2.diagonal().array() += rho1 + rho2;
	_2A_plus_rho1_rho2.makeCompressed();
	// printf("test longkang: sum of rho_1_rho2: %f\n",_2A_plus_rho1_rho2.sum());

	preconditioner_diag_mat.setZero();
	preconditioner_diag_mat = SparseMatrix(n, n); 
	// preconditioner_diag_mat.reserve(n);
	// preconditioner_diag_mat.setFromTriplets(preconditioner_diag_mat_triplets.begin(), preconditioner_diag_mat_triplets.end());
	// preconditioner_diag_mat.diagonal().array() = _2A_plus_rho1_rho2.diagonal().array();
	// preconditioner_diag_mat.makeCompressed();
	preconditioner_diag_mat = _2A_plus_rho1_rho2;
	// printf("test longkang: sum of preconditioner_: %f\n",preconditioner_diag_mat.sum());

	//free preconditioner_diag_mat_triplets
	if(preconditioner_diag_mat_triplets.size()){
		std::vector<Triplet>::iterator iter_t = preconditioner_diag_mat_triplets.begin();
		for(; iter_t!= preconditioner_diag_mat_triplets.end();){
			// printf("Test in update expression: size matrix_expression: %ld\n",matrix_expressions.size());
			iter_t = preconditioner_diag_mat_triplets.erase(iter_t);
		}
	}

// printf("Longkang test 13.\n");

	/* The matrix expression for the unconstraint case is 2 * _A + (rho1 + rho2) * I */
	// std::vector<std::vector<const SparseMatrix*>> matrix_expressions;
	matrix_expressions.emplace_back();
	matrix_expressions.back().push_back(&_2A_plus_rho1_rho2);


	/* If the problem has equality constraint, add rho4 * E^T * E to the matrix expression */
	if (instruction.problem_type == inequality) {
		matrix_expressions.emplace_back();
		matrix_expressions.back().push_back(E_ptr);
		matrix_expressions.back().push_back(&rho4_E_transpose);

		/* Calculating the diagonal elements of Esq for preconditioner */
		Esq_diag = DenseVector(n);
		Esq_diag.setZero();
		// printf("this is in the update expresison: E_transpose.outerSize(): %d\n",(*E_ptr).outerSize());
		for(int j = 0; j < (*E_ptr).outerSize(); ++j) {
			typename SparseMatrix::InnerIterator it((*E_ptr), j);
			while (it) {
				if(it.value() != 0.0) {
					Esq_diag[j] += it.value() * it.value();
				}
				++it;
			}
		}
		preconditioner_diag_mat.diagonal().array() += rho4 * Esq_diag.array();
	}

	// this->preconditioner_diag_mat = preconditioner_diag_mat; 
	// this->preconditioner_diag_mat_triplets = preconditioner_diag_mat_triplets;
	// this->matrix_expressions = matrix_expressions;
	
	// if(iter > 6000){
	// 	esq = E_transpose * (*E_ptr);
	// 	esq += _2A_plus_rho1_rho2;
	// 	printf("Longkang test: esq is calculated\n");
	// }
	
}


void readDenseVec(FILE *fp, DenseVector &vec, int vecLen) {
	for (int i = 0; i < vecLen; i++) {
		if (fscanf(fp, "%lf\n", &(vec.data()[i]))== 0) {
			printf("error when reading dense vector\n");
			exit(-1);
		}
	}
}

void readSparseMat(FILE *fp, SparseMatrix &mat, int k) {
	std::vector<Triplet> triplets;
	int row, col;
	double val;
	int max_row = 0;
	int max_col = 0;
	// printf("longkang test 3\n");
	while(true) {
		// std::cout << fscanf(fp, "%d,%d,%lf\n", &row, &col, &val) << std::endl;
		if (fscanf(fp, "%d,%d,%lf\n", &row, &col, &val) != 3) {
			break;
		}
		// printf("longkang test 4\n");
		if (row > max_row) {
			max_row = row;
		}

		if (col > max_col) {
			max_col = col;
		}
		if(k==2)
			triplets.push_back(Triplet(row - 1, col - 1, -1.0*val));
		else
			triplets.push_back(Triplet(row - 1, col - 1, val));
	}
	mat = SparseMatrix(max_row, max_col);
    // printf("max_col: %d; max_row: %d\n", max_col, max_row);
	mat.setFromTriplets(triplets.begin(), triplets.end());
}

void LPboxADMMsolver::readFile(int i, int k, int j){ //i-instance; k-item_num; j-bid_num. 
	char path_C[100];
	char path_b[100];
	char path_xiter_[100];
	char log[100];
	std::string root = "../cython_solver/data";
	std::string subroot;
	// std::string psize;

	// if(k==1) subroot = "cauctions/";
	// if(k==2) subroot = "setcover/";
	// if(k==3) subroot = "indset/";
	// if(k==4) subroot = "facility/";
	// root = root + subroot;

	// if(j==11) psize='s';      // 100-500
	// else if(j==12) psize='m'; // 200-1000
	// else if(j==13) psize='l'; // 300-1500
	// else if(j==14) psize="xl"; // 800-4000 

	// else if(j==1) psize="100_10000";
	// else if(j==2) psize="500_50000";
	// else if(j==3) psize="1000_100000";
	// else if(j==4) psize="2000_200000";
	// else if(j==5) psize="2000_500000";
	// else if(j==6) psize="2000_1000000";
	// else if(j==7) psize="500_10000";
	// else if(j==8) psize="1000_10000";


	// else if(j==17) psize="100_20000";
	// else if(j==18) psize="100_50000";
	// else if(j==19) psize="100_100000";
	// else if(j==20) psize="100_200000";
	// else if(j==21) psize="100_500000";
	// else if(j==22) psize="100_1000000";

	// else if(j==25) psize="1000_200000";
	// else if(j==26) psize="1000_500000";
	// else if(j==27) psize="1000_1000000";


	std::cout << root << std::endl;
	// printf("test 101\n");
	std::string csv = root + "/xiter/allres.csv";
	// printf("test 101: %s\n",csv.c_str());
	sprintf(path_C, "%s/instance/%d_%d/instance_%d_C.txt",root.c_str(),k,j,i); //E=C
	// printf("test 102: %s\n",path_C);
	sprintf(path_b, "%s/instance/%d_%d/instance_%d_b.txt",root.c_str(),k,j,i); 
	// printf("test 103 %s\n",path_b);
	sprintf(log, "%s/log/%d_%d_log_%d.txt",root.c_str(),k,j,i);
	// printf("test 104 %s\n",log);
	sprintf(path_xiter_, "%s/xiter/%d_%d_xiters_%d.csv",root.c_str(),k,j,i);
	// std::string tmp = root + "xiter/" + psize + "_xiters.csv";
	// printf("test 105 %s\n",path_xiter_);
	printf("C: %s\nlog: %s\npathX: %s\n",path_C,log,path_xiter_);

	SparseMatrix E; //E=C
    const char* inputFile_c = path_C;
	// printf("longkang test 0\n"); 
    FILE *input_mat_fp = fopen(inputFile_c, "r");
	// printf("longkang test 1\n"); 
    readSparseMat(input_mat_fp, E, k);
	// printf("longkang test 2\n"); 
    int row = E.rows();
    int col = E.cols();
    std::cout << col << ", " << row << ", " << E.nonZeros() << std::endl; 
	std::cout << "Density:" << (1.0*E.nonZeros()/col)/row  << std::endl;
	DenseVector b(col);
    const char* inputFile = path_b; //"test_data/test_100_500/instance_1_b.txt";
    FILE *input_vec_fp = fopen(inputFile, "r");
    readDenseVec(input_vec_fp, b, col);
	
	// if(k==1 || k==3)
	b.array() =  - 1.0  * b.array();

	DenseVector f = DenseVector::Ones(row);
	// if(k==2) f.array() = - 1.0 * f.array();
	
	E_ptr = new SparseMatrix(row, col);
	org_E_ptr = new SparseMatrix(row, col);
	*E_ptr = E;
	*org_E_ptr = E; 

	f_ptr = new DenseVector(row);
	*f_ptr = f;

	b_ptr = new DenseVector(col);
	*b_ptr = b;

	// std::string outputFile = log; 
	// std::string x_path_ = path_xiter_;
	// set_log_file(outputFile);
	this->log_file_path = log;
	// printf("Writing log output to %s\n", log);
	this->path_xiter = path_xiter_;
	// printf("Writing x_iters to %s\n", path_xiter_);
	file_idx = i;
	this->csv = csv;
}

