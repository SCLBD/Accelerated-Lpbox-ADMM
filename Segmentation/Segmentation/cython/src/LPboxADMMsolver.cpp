//
// Created by squintingsmile on 2020/9/22.
//

#define OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT


#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <Eigen4/Dense>
#include <Eigen4/Sparse>

#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
// #include "image_segmentation_utils.h"
#include <getopt.h>
#include <unistd.h>
#include <fstream>

// #include "image_segmentation_utils.h"
#include <limits>
#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include "LPboxADMMsolver.h"


using namespace Eigen::internal;
using std::abs;
using std::sqrt;
Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols,
		", ", ", ", "", "", " << ", ";");






// From: image_segmentation_utils



/* input = [a_1 a_2 ... a_n], output = [a_1; a_2; ... a_n] where a_i are column vectors */
void vectorize(const DenseMatrix &input, DenseVector &output) {
	int numRows = input.rows();
	int numCols = input.cols();
	output = DenseVector(numRows * numCols);
	for (int i = 0; i < numCols; i++) {
		output.block(i * numRows, 0, numRows, 1).array() = input.block(0, i, numRows, 1).array();
	}
}

void get_unary_cost(const DenseMatrix &_nodes, _double_t sigma, _double_t b, _double_t f1, _double_t f2, DenseMatrix &unary_cost) {
	DenseVector nodes;
	vectorize(_nodes, nodes);
	const _double_t c = std::log(2.0 * M_PI) / 2.0 + std::log(sigma);
	DenseVector alpha_b = (nodes.array() - b).pow(2.0) / (2 * sigma * sigma) + c;
	DenseVector aa = (-(nodes.array() - f1).pow(2.0) / (2 * sigma * sigma)).exp() 
		+ (-(nodes.array() - f2).pow(2) / (2 * sigma * sigma)).exp();
	DenseVector alpha_f = - (aa.array() + std::numeric_limits<_double_t>::epsilon()).log() + c + std::log(2.0);

	// printf("this const is %lf\n", c);
	// printf("this is b\n");
	// for(int i = 0; i < alpha_b.rows(); i++){
	// 	std::cout << alpha_b[i] << ", ";
	// }
	// std::cout << std::endl;

	// printf("this is f\n");
	// for(int i = 0; i < alpha_f.rows(); i++){
	// 	std::cout << alpha_f[i] << ", ";
	// }
	// std::cout << std::endl;

	
	unary_cost = DenseMatrix(2, alpha_b.rows());
	unary_cost.block(0, 0, 1, alpha_b.rows()) = alpha_b.transpose();
	unary_cost.block(1, 0, 1, alpha_f.rows()) = alpha_f.transpose();
}

DenseMatrix get_row(const DenseMatrix &mat, int row_index) {
	int num_cols = mat.cols();
	DenseMatrix res = mat.block(row_index, 0, 1, num_cols);
	return res;
}

void repmat(const DenseMatrix &mat, int numRows, int numCols, DenseMatrix &result) {

	result = DenseMatrix(numRows * mat.rows(), numCols * mat.cols());
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			result.block(i * mat.rows(), j * mat.cols(), mat.rows(), mat.cols()) = mat;
		}
	}
}

/* the output is [A; B] */
void verticle_concat(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &output) {
	output = DenseMatrix(A.rows() + B.rows(), A.cols());
	output.block(0, 0, A.rows(), A.cols()) = A;
	output.block(A.rows(), 0, B.rows(), B.cols()) = B;
}

void verticle_concat(const DenseVector &A, const DenseVector &B, DenseVector &output) {
	output = DenseVector(A.rows() + B.rows());
	output.block(0, 0, A.rows(), 1) = A;
	output.block(A.rows(), 0, B.rows(), 1) = B;
}
	
void get_offsets(int k, DenseMatrix &offsets) {
	int edgeLen = 2 * k + 1;
	offsets = DenseMatrix(edgeLen * edgeLen, 2);
	DenseMatrix row_offsets = DenseMatrix(edgeLen, edgeLen);
	for (int i = 0; i < edgeLen; i++) {
		row_offsets.block(i, 0, 1, edgeLen).array() = i - k;
	}
	DenseMatrix col_offsets = row_offsets.transpose();

	for (int i = 0; i < edgeLen; i++) {
		offsets.block(i * edgeLen, 0, edgeLen, 1) = row_offsets.block(0, i, edgeLen, 1);
		offsets.block(i * edgeLen, 1, edgeLen, 1) = col_offsets.block(0, i, edgeLen, 1);
	}
}

void meshgrid(int x, int y, DenseMatrix &X, DenseMatrix &Y) {
	X = DenseMatrix(y, x);
	Y = DenseMatrix(y, x);
	for (int i = 0; i < x; i++) {
		X.block(0, i, y, 1).array() = i + 1;
	}

	for (int i = 0; i < y; i++) {
		Y.block(i, 0, 1, x).array() = i + 1;
	}
}

struct pair {
	int idx1;
	int idx2;
};

void generate_pixel_pairs(int nrows, int ncols, int k, DenseIntMatrix &res) {

	//printf("matrix rows: %d", ((2 * k + 1) * (2 * k + 1) - 1) * nrows * ncols);
	DenseIntMatrix tmp(((2 * k + 1) * (2 * k + 1) - 1) * nrows * ncols, 2);
	//std::vector<pair> vec;
	//vec.reserve();
	int counter = 0;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			for (int a = -k; a < k + 1; a++) {
				for (int b = -k; b < k + 1; b++) {
					if (a != b && i + a >= 0 && i + a < nrows && j + b >= 0 && j + b < ncols) {
						//vec.push_back({i * ncols + j + 1 , (i + a) * ncols + (j + b + 1)});
						tmp(counter, 0) = i * ncols + j + 1;
						tmp(counter, 1) = (i + a) * ncols + (j + b + 1);
						counter += 1;
					}
				}
			}
		}
	}
	res = tmp.block(0, 0, counter, 2).matrix();
	
	//for (int i = 0; i < vec.size(); i++) {
		//res(i, 0) = vec[i].idx1;
		//res(i, 1) = vec[i].idx2;
	//}
}

void get_binary_cost(const DenseMatrix &image, SparseMatrix &binary_cost_matrix) {

	int nrows = image.rows();
	int ncols = image.cols();
	int num_pixels = nrows * ncols;

	DenseVector image_vec;
	vectorize(image, image_vec);
	_double_t sigma = std::sqrt((image_vec.array() - image_vec.mean()).square().sum()/(image_vec.size()-1));;

    uint8_t *indicator = new uint8_t[num_pixels];
    memset(indicator, 1, num_pixels);

	DenseIntMatrix pairs;
	generate_pixel_pairs(nrows, ncols, 1, pairs);
	DenseVector I1(pairs.rows());
	DenseVector I2(pairs.rows());

	for (int i = 0; i < pairs.rows(); i++) {
		I1(i) = image((pairs(i, 0) - 1) % nrows, (pairs(i, 0) - 1) / nrows);
		I2(i) = image((pairs(i, 1) - 1) % nrows, (pairs(i, 1) - 1) / nrows);
	}
	
	DenseVector diff_vec = (I1.array() - I2.array()).pow(2.0) / sigma;
	
	diff_vec.array() = (-diff_vec.array()).exp();

	binary_cost_matrix = SparseMatrix(num_pixels, num_pixels);
	
	std::vector<Triplet> triplets;
	for (int i = 0; i < pairs.rows(); i++) {
		int row = pairs(i, 0);
		int col = pairs(i, 1);
		triplets.push_back({row - 1, col - 1, std::round(3 * diff_vec(i))});
		if (row == col) {
			//printf("row: %d\n", row);
			indicator[row - 1] = 0;
		}
	}

	for (int i = 0; i < num_pixels; i++) {
		if (indicator[i] == 1) {
			//printf("row: %d\n", i);
			//printf("Encountering zero diagonal element\n");
			triplets.push_back({i, i, 0.0});
		}
	}

	binary_cost_matrix.setFromTriplets(triplets.begin(), triplets.end());
	delete[] indicator;

}

void get_A_b_from_cost(const DenseMatrix &unary_cost, const SparseMatrix &binary_cost, SparseMatrix &A, DenseVector &b, _double_t &c) {
	int n = binary_cost.rows();
	
	DenseMatrix U1 = get_row(unary_cost, 0);
	DenseMatrix U2 = get_row(unary_cost, 1);

	b = (U2 - U1).eval().transpose();
	
	A = - binary_cost;
	DenseVector ones = DenseVector::Ones(binary_cost.cols());
	DenseVector We = - A * ones;
	A.diagonal().array() += We.array();
	A = 2 * A;

	// _double_t tmp = 0;
	// for(int i = 0; i < U1.cols(); i++){
	// 	tmp += U1[i]; 
	// }
	c = U1.array().sum();
	// printf("U1.shape: %ld, %ld\n",U1.rows(), U1.cols());
	// printf("this is C: %lf\n",c);
	// return c;
}






















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
 * allocating vectors
 */

void _conjugate_gradient(const std::vector<std::vector<const SparseMatrix*>> &mat_expressions, const DenseVector& rhs,
		DenseVector& x, const Eigen::DiagonalPreconditioner<_double_t>& precond, int& iters,
		typename DenseVector::RealScalar& tol_error, DenseVector &temp_vec_for_cg, DenseVector &temp_vec_for_mat_mul) {

	typedef typename DenseVector::RealScalar RealScalar;
	typedef typename DenseVector::Scalar Scalar;
	typedef Eigen::Matrix<Scalar,Dynamic,1> VectorType;

	RealScalar tol = tol_error;
	int maxIters = iters;

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


void LPboxADMMsolver::ADMM_bqp_linear_eq_init() {
	stop_threshold = 1e-4;
	std_threshold = 1e-6;
	gamma_val = 1.6;
	gamma_factor = 0.95;
	rho_change_step = 5;
	max_iters = 5e3;
	initial_rho = 1;
	history_size = 3;
	learning_fact = 1 + 5.0 / 100;
	pcg_tol = 1e-4;
	pcg_maxiters = 1e3;
	rel_tol = 5e-5;
	projection_lp=2;
}

void LPboxADMMsolver::ADMM_bqp_linear_ineq_init() {
	stop_threshold = 1e-4;
	std_threshold = 1e-6;
	gamma_val = 1.6;
	gamma_factor = 0.95;
	rho_change_step = 5;
	max_iters = 1e4;
	initial_rho = 25;
	history_size = 3;
	learning_fact = 1 + 1.0 / 100;
	pcg_tol = 1e-4;
	pcg_maxiters = 1e3;
	rel_tol = 5e-5;
	projection_lp = 2;

}

void LPboxADMMsolver::ADMM_bqp_linear_eq_and_uneq_init() {
	stop_threshold = 1e-4;
	std_threshold = 1e-6;
	gamma_val = 1.6;
	gamma_factor = 0.95;
	rho_change_step = 5;
	max_iters = 1e4;
	initial_rho = 25;
	history_size = 3;
	learning_fact = 1 + 1.0 / 100;
	pcg_tol = 1e-4;
	pcg_maxiters = 1e3;
	rel_tol = 5e-5;
	projection_lp = 2;
}

LPboxADMMsolver::LPboxADMMsolver(){};

LPboxADMMsolver::LPboxADMMsolver(int node){
	printf("Object with node is created!\n");
	this->node = node;
}

LPboxADMMsolver::LPboxADMMsolver(int node, int problem){
	printf("Object with node is created!\n");
	this->node = node;
	this->problem = problem; 
	this->print_info = 0;
}

LPboxADMMsolver::LPboxADMMsolver(int print_info, int node, int problem){
	printf("Object with node is created with three inputs!\n");
	this->node = node;
	this->problem = problem; 
	this->print_info = print_info;
}


void LPboxADMMsolver::ADMM_bqp_unconstrained_init() {
	std_threshold = 1e-6;
	gamma_val = 1.0;
	gamma_factor = 0.99;
	initial_rho = 5;
	learning_fact = 1 + 3.0 / 100;
	rho_upper_limit = 1000;
	history_size = 5;
	rho_change_step = 5;
	rel_tol = 1e-5;
	stop_threshold = 1e-3;
	max_iters = 1e4;
	projection_lp = 2; // p norm 
	pcg_tol = 1e-3;
	pcg_maxiters = 1e3;

	// printf("Finished generating matrices, starting algorithm\n");

	int numNodes=this->node;
	// const char* imagePath = "data/standard_test_images/house.tif"; 
	// const char* imagePath = "data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"; 

	// const char* imagePath = "data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg";
	// const char* output = "result/airplane.out"; 
	// std::string log = "result/airplane.log";
	
	int i = problem;

	char imagePath[100];
	char output[100];
	char log[100];
	char xiter[100];
	char xiter_all[100]="../result/xiter_all.csv";
	sprintf(imagePath, "../data/%d.jpg", i);
	sprintf(output, "../result/output_%d.png", i);
	sprintf(log, "../result/log_%d.txt", i);
	this->set_log_file(log);
	sprintf(xiter, "../xiter/%d.csv", i);
	

	this->output = output;
	this->xiter = xiter;
	this->xiter_all = xiter_all;
	// this->print_info = 0;

	printf("The input file is: %s, and the size is: %d\n",imagePath,numNodes);

	cv::Mat image = cv::imread(imagePath, 0);  // read in the images in grey. 
	double lambda = 3.0;
	/* Scaling the picture */
	double scale = std::sqrt(numNodes / (double) (image.rows * image.cols));

	printf("Origin image size: %d X %d = %d\n",image.rows,image.cols,image.rows*image.cols);

	cv::Mat scaled_image;
	cv::Size scaled_size = cv::Size(std::round(scale * image.cols), std::round(scale * image.rows));
	resize(image, scaled_image, cv::Size(), scale, scale);

	scaled_row = scaled_image.rows;
	scaled_col = scaled_image.cols;

	int numPixels = scaled_image.rows * scaled_image.cols;
	printf("Reshaped image size: %d X %d = %d\n",scaled_image.rows,scaled_image.cols,numPixels);

	DenseMatrix I; // from Mat to Eigen::DenseMatrix.
	cv2eigen(scaled_image, I);
	
	// my_I = I; 

	I.array() = I.array() / 263.0;

	DenseMatrix unary_cost;
	SparseMatrix binary_cost;

	// printf("I  %ld\n",I.rows()*I.cols());

	double sigma = 0.1;
	double b = 0.6;
	double f1 = 0.2;
	double f2 = 0.2;

	get_unary_cost(I, sigma, b, f1, f2, unary_cost);
	get_binary_cost(I, binary_cost); /* Actually calculating round(lambda * W) in this so we can get the 
									  * binary cost directly */

	unary_cost.array() = unary_cost.array().round();

	SparseMatrix _A;
	DenseVector _b;
	get_A_b_from_cost(unary_cost, binary_cost, _A, _b, _c);
	printf("Finished generating matrices, starting algorithm\n");

	org_A = new SparseMatrix(_A.rows(), _A.cols());
	*org_A = _A/2;
	org_b = new DenseVector(_b.rows());
	*org_b = _b;

	A_ptr = new SparseMatrix(_A.rows(), _A.cols());
	*A_ptr = _A/2;
	b_ptr = new DenseVector(_b.rows());
	*b_ptr = _b;
	n = _A.cols(); 
	printf("The x size: %d\n",n);
	x0 = DenseVector::Zero(n);
	x_sol = DenseVector::Zero(n);
	// x_sol = x0;
	y1       = DenseVector(n);
	y2       = DenseVector(n);
	z1       = DenseVector(n);
	z2       = DenseVector(n);
	prev_idx = DenseVector(n);
	best_sol = DenseVector(n);
	temp_vec = DenseVector(n);
	cur_idx  = DenseVector(n);
	temp_mat = SparseMatrix(n, n);

	z1.array() = 0;
	z2.array() = 0;
	cur_idx.array() = 0;

	rho1 = initial_rho;
	rho2 = initial_rho;
	prev_rho1 = rho1;
	prev_rho2 = rho2;

	/* temp_mat stores the matrix that is used in the conjugate gradient step */
	temp_mat = 2 * (*A_ptr);
	temp_mat.diagonal().array() += rho1 + rho2;
	temp_mat.makeCompressed();

	y1 = x_sol;
	y2 = x_sol;
	prev_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
	best_sol = x_sol;
	best_bin_obj = compute_cost(x_sol, (*A_ptr), (*b_ptr));

	org_n = n;
	fix_n = 0;
	fix_sum = 0;
	ret_idx = -1 * DenseVector::Ones(n);
	ret_val = -1 * DenseVector::Ones(n);
	fix_obj = 0;
	sum_fix_obj = 0;

	left_idx = DenseVector::Zero(n);
	for(int i = 0 ; i < n; i++){
		left_idx.array()[i] = i;
		// x_sol.array()[i] = 1;
	}
	printf("this is the end of init.\n");


}

void LPboxADMMsolver::save_img(){
	double* x = new double[org_n];
	x = get_x_sol();
	// double* x = get_x_sol();
	DenseVector xx = DenseVector::Zero(org_n);
	for(int i = 0 ; i < org_n; i++){
		xx.array()[i] = x[i];
	}

	/* Reshape the solution to fit the size of the image */
	DenseMatrix reshaped_mat = Eigen::Map<DenseMatrix>(xx.data(), scaled_row, scaled_col);

	// Output the saved images.
	/* Setting the location with value 1 to be white */
	DenseIntMatrix r_mat = (reshaped_mat.array() >= 0.5).matrix().cast<int>() * 255;

	// for(int i = 0 ; i < org_n; i++){
	// 	if(xx.array()[i] >= 0.5){
	// 		r_mat.data()[i] = int(I.data()[i]*263); //255; 
	// 	}
	// }
	cv::Mat res_mat(r_mat.rows(), r_mat.cols(), CV_8UC1);//CV_32FC3
	eigen2cv(r_mat, res_mat);
	imwrite(output, res_mat);
	printf("sucessful write image: %s\n", output.c_str());
}


// get x_iterations for Learning to Early Fix. 
double* LPboxADMMsolver::get_x_iters_d(int ws){ 
	int rows = x_iters.rows(); // n_num
	int cols = ws; // 500/  100
	printf("In get_x_iters_d: rows:%d, cols:%d, n: %d.\n",rows, cols, n);
	double* ret = new double[rows*cols]; 
	for(int i = 0; i < rows; i++){
		for(int j = 0 ; j < cols; j++){
			ret[i*cols + j] = x_iters(i,j);
		}
	}
	return ret;
}

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

double LPboxADMMsolver::get_final_obj(){
	// int len = x_sol.rows();
	// if(n!=0){
	// 	// printf("Calculate objective:\n");
	// 	// printf("fixed sum: %d; x_sol size: %d; n=%d; \n",fix_sum, len, n);
	// 	// printf("Final Objective: %f\n",sum_fix_obj+cur_obj);
	// 	return sum_fix_obj+cur_obj;
	// }
	// else{
	// 	printf("Final Objective: %f\n",sum_fix_obj);
	// 	return sum_fix_obj;
	// }

	double* x = new double[org_n];
	x = get_x_sol();
	// double* x = get_x_sol();
	DenseVector xx = DenseVector::Zero(org_n);
	for(int i = 0 ; i < org_n; i++){
		xx.array()[i] = x[i];
	}
	double obj = compute_cost(xx, (*org_A), (*org_b));

	printf("The obj is %f. The C is %f, \n", obj, _c);

	return obj+_c; 
}

double* LPboxADMMsolver::get_x_sol(){
	if(n!=0){
		printf("fixed index shape: %ld\n", ret_idx_prev.rows());
		printf("current left solution shape: %ld\n", x_sol.rows());
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


int LPboxADMMsolver::ADMM_bqp_unconstrained_l2f(int iter_start, int iter_end, double* vec, int fix_num) {
	int ret = 0; 
	FILE *fp;
	if (does_log) {
		fp = fopen(log_file_path.c_str(), "w+");
	}

	x_iters = DenseMatrix::Zero(n-fix_num,10);
	int cc = 0; 
// printf("test 101\n");
	if(fix_num!=0){
		fix_sum += fix_num;
		Eigen::ArrayXi det_idx(n); // all reverse index  
		Eigen::ArrayXi det_idx_fixed(fix_num);
		Eigen::ArrayXi det_idx_unfixed(n-fix_num);

		Eigen::ArrayXi fix_idx(fix_num); // fixed index
		Eigen::ArrayXi non_fix_idx(n-fix_num); //left index
		DenseVector org_fix_val(fix_num); // fixed value
		
		// std::vector<Triplet> mab,mcd;
		std::vector<Triplet> ma,mb; 
		std::vector<double> mab_vec; // tmp  
		int j=0, k=0;

		// printf("Test 101. n=%d, fixed=%d, left=%d, \n non_zeros=%ld, col=%ld, row=%ld\n", n, fix_num, n-fix_num, (*A_ptr).nonZeros(), (*A_ptr).cols(), (*A_ptr).rows() ); 

		for(int i = 0 ; i < n; i++){
			if(vec[i] == 1){  // 1 - fix to 1
				fix_idx[j] = i;
				org_fix_val[j] = 1;
				// det_idx_fixed[i] = j;
				det_idx[i] = j;
				j++;
			}					
			else if(vec[i] == 0){  // 0 - fix to 0
				fix_idx[j] = i;
				org_fix_val[j] = 0;
				// det_idx_fixed[i] = j;
				det_idx[i] = j;
				j++;
			}
			else{  // -1 - not fix
				non_fix_idx[k] = i;
				// det_idx_unfixed[i] = k;
				det_idx[i] = k;
				
				for(Eigen::SparseMatrix<_double_t, Eigen::RowMajor>::InnerIterator it(*A_ptr,i); it; ++it){
					// mab.push_back(Triplet(k-1, it.col(), it.value()));
					mab_vec.push_back(k);
					mab_vec.push_back(it.col());
					mab_vec.push_back(double(it.value()));
				}

				k++;
			}
		}

		for(int i=0; i < mab_vec.size()/3; i++){
			int row = int(mab_vec[i*3]);
			int col = int(mab_vec[i*3+1]);
			double val = mab_vec[i*3+2];
			if(vec[col]==-1){
				ma.push_back(Triplet(row, int(det_idx[col]), val));
				// if(row<0 || int(det_idx[col])<0){
				// 	printf("there is error1: row=%d, col=%d\n", row, int(det_idx[col]));
				// }
				// if(row >= n-fix_num || int(det_idx[col]) >= n-fix_num ){
				// 	printf("there is error2: row=%d, col=%d\n", row, int(det_idx[col]));
				// }
			} 
			else{	
				mb.push_back(Triplet(row, int(det_idx[col]), val));
				// if(row<0 || int(det_idx[col])<0){
				// 	printf("there is error3: row=%d, col=%d\n", row, int(det_idx[col]));
				// }
				// if(row >= n-fix_num || int(det_idx[col]) >= fix_num ){
				// 	printf("there is error4: row=%d, col=%d\n", row, int(det_idx[col]));
				// }
			}
		}

		// for(int i=0 ; i < n-fix_num; i++){
		// 	if(non_fix_idx[i] >= n){
		// 		printf("error: non_fix_idx[i] >= n. \n");
		// 	}
		// }

		// Ma.setZero();
		Ma = SparseMatrix(n-fix_num, n-fix_num);  
		Ma.setFromTriplets(ma.begin(), ma.end()); 
		Ma.makeCompressed();
		// printf("Test 103. Ma, non_zeros=%ld, col=%ld, row=%ld\n", (Ma).nonZeros(), (Ma).cols(), (Ma).rows() );

 		// Mb.setZero();
		Mb = SparseMatrix(n-fix_num, fix_num);  
		Mb.setFromTriplets(mb.begin(), mb.end());
		Mb.makeCompressed();
		// printf("Test 104. Mb, non_zeros=%ld, col=%ld, row=%ld\n", (Mb).nonZeros(), (Mb).cols(), (Mb).rows() );

		// printf("test 103!!\n");

		org_fix_idx = left_idx(fix_idx);
		org_non_fix_idx = left_idx(non_fix_idx);
		left_idx = org_non_fix_idx;

		ret_idx.resize(ret_idx_prev.rows()+org_fix_idx.rows(), 1);
		ret_idx << ret_idx_prev, org_fix_idx;
		ret_val.resize(ret_val_prev.rows()+org_fix_val.rows(), 1);
		ret_val << ret_val_prev, org_fix_val;
		ret_idx_prev = ret_idx;
		ret_val_prev = ret_val;

		if(n-fix_num==0){
			ret = 1;
			n = 0;
			iter_end = iter_start;
		}
		else{
			temp = x_sol(non_fix_idx);
			x_sol = temp;
			x_prev = x_sol;
			prev_idx = (x_prev.array() >= 0.5).matrix().cast<_double_t>();

			temp = y1(non_fix_idx); y1 = temp; temp = y2(non_fix_idx); y2 = temp; 
			temp = z1(non_fix_idx); z1 = temp; temp = z2(non_fix_idx); z2 = temp; 
			
			b1 = (*b_ptr)(non_fix_idx); b2 = (*b_ptr)(fix_idx);
			x2 = org_fix_val;
			
			n = n - fix_num;
			A_ptr = new SparseMatrix(n, n);
			*A_ptr = Ma;
			b_ptr = new DenseVector(n);
			mat_mul_vec(Mb, x2, temp_vec_for_mat_mul2);
			*b_ptr = 2 * temp_vec_for_mat_mul2 + b1;

			temp_mat = SparseMatrix(n, n);
			temp_mat = 2 * (*A_ptr);
			temp_mat.diagonal().array() += rho1 + rho2;
			temp_mat.makeCompressed();
		}

		printf("Iter: %d; Fixed %ld Elements; Totally Fixed %d Elements; Left %ld Elements.\n",
					iter_start,fix_idx.rows(),fix_sum,non_fix_idx.rows());
	}
	
	std::chrono::steady_clock::time_point start, end;
	start = std::chrono::steady_clock::now();
	long time_elapsed = 0;
	for (int iter = iter_start; iter < iter_end; iter++) {
		temp_vec = x_sol + z1 / rho1;

		if (does_log) {
			fprintf(fp, "Iteration: %d\n", iter);
		}
		/* Project vector on [0, 1] box to calculate y1 */
		project_box(n, temp_vec, y1);

		temp_vec = x_sol + z2 / rho2;

		/* Project vector on shifted lp box to calculate y2 */
		project_shifted_Lp_ball(n, temp_vec, projection_lp, y2);

		/* If the iteration is nonzero and it divides rho_change_step, it means
		 * that the rho updated in the last iteration and the matrix used by conjugate
		 * gradient should be updated
		 */
		if (iter != 0 && rhoUpdated) {
			temp_mat.diagonal().array() += (prev_rho1 + prev_rho2) * rho_change_ratio;
			temp_mat.makeCompressed();
		}
// printf("test 102\n");	
		/* Calculate the vector b in the conjugate gradient algorithm */
		temp_vec = rho1 * y1 + rho2 * y2 - ((*b_ptr) + z1 + z2);
		
		/* Explicit version of conjugate gradient used for profiling */

		/* Since the matrix used by the conjugate gradient changes only when after rho is updated
		 * we only need to recalculate the preconditioner if rho is updated in the last iteration
		 */
		if (rhoUpdated) {
			diagonalPreconditioner.compute(temp_mat);
			rhoUpdated = false;
		}
		// printf("Test 222. x_sol shape:%d, temp_mat.shape:%d,%d , temp_vec.shape:%d  \n", x_sol.rows(), temp_mat.rows(), temp_mat.cols() ,temp_vec.rows()  );

		x_sol = y1;
		_double_t tol = pcg_tol;
		int maxiter = pcg_maxiters;
		_conjugate_gradient(temp_mat, temp_vec, x_sol, diagonalPreconditioner, maxiter, tol);
		if (does_log) {
			fprintf(fp, "Conjugate gradient stops after %d iterations\n", maxiter);
			fprintf(fp, "Conjugate gradient stops with residual %lf\n", tol);
		}

		if(1){ 
			x_iters.block(0,cc,n,1) = x_sol;
			cc++;
		}

// printf("test 103\n");
		z1 = z1 + gamma_val * rho1 * (x_sol - y1);
		z2 = z2 + gamma_val * rho2 * (x_sol - y2);


		/* Testing the conditions to see if the algorithm converges */
		_double_t temp0 = std::max(x_sol.norm(), _double_t(2.2204e-16));
		cvg_test1 = (x_sol-y1).norm() / temp0;
		cvg_test2 = (x_sol-y2).norm() / temp0;
		if (cvg_test1 <= stop_threshold && cvg_test2 <= stop_threshold) {
			ret = 1;
			printf("Terminate by condition xyy. iter: %d, stop_threshold: %.6f\n", iter, std::max(cvg_test1, cvg_test2));
			if (does_log) {
				fprintf(fp, "iter: %d, stop_threshold: %.6f\n", iter, std::max(cvg_test1, cvg_test2));
			}
			break;
		}

		/* Update the rho value every rho_change_step */
		if ((iter+1) % rho_change_step == 0) {
			prev_rho1 = rho1;
			prev_rho2 = rho2;
			rho1 = learning_fact * rho1;
			rho2 = learning_fact * rho2;
			gamma_val = std::max(gamma_val * gamma_factor, _double_t(1.0));
			rhoUpdated = true;
			rho_change_ratio = learning_fact - 1.0;
		}

		/* Computer the relaxed cost function (x is not binary)*/
		_double_t obj_val = compute_cost(x_sol,(*A_ptr),(*b_ptr));
		obj_list.push_back(obj_val);
		if (obj_list.size() >= history_size) {
			std_obj = compute_std_obj(obj_list, history_size);
		}
		if (std_obj <= std_threshold) {
			ret = 1;
			printf("Terminate by condition obj_std. iter: %d, std_threshold: %.6f\n", iter, std_obj);
			if (does_log) {
				fprintf(fp, "iter: %d, std_threshold: %.6f\n", iter, std_obj);
			}
			break;
		}
		

		/* Calculating the actual cost */
		cur_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>(); /* The value type of the vector should be double
																	  therefore needs casting the boolean vector to double */
		prev_idx = cur_idx;
		cur_obj = compute_cost(prev_idx, (*A_ptr), (*b_ptr));

		/* Setting the best binary solution */
		if (best_bin_obj >= cur_obj) { 	
			best_bin_obj = cur_obj;
			best_sol = x_sol;
		}


	}
	printf("Finish from iter %d to iter %d\n", iter_start, iter_end);


	// end = std::chrono::steady_clock::now();
	// time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	// std::cout << "Time elapsed for all: " << time_elapsed*1.0/1000 << "s" << std::endl;

	if (does_log) {
		// fprintf(fp, "Time elapsed: %lfs\n", time_elapsed*1.0/1000);
		fclose(fp);
	}

	// cur_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
	// cur_obj = compute_cost(cur_idx, (*A_ptr), (*b_ptr));

	// printf("obj: %lf; c: %lf; Energy: %lf\n", cur_obj, _c, cur_obj+_c);
	// return cur_obj + _c;
	return ret;
}




int LPboxADMMsolver::ADMM_bqp_unconstrained_legacy() {
	
	printf("this is the start of lpbox.\n");
	// printf("test 101!!\n");
	FILE *fp;
	if (does_log) {
		fp = fopen(log_file_path.c_str(), "w+");
	}
	
	FILE *xi;
	if(print_info==1){
		xi = fopen(xiter.c_str(), "w+");
		// fprintf(xi, "This is the start!!!\n");
	}

	FILE *al;
	al = fopen(xiter_all.c_str(), "a+");

	std::chrono::steady_clock::time_point start, end;
	start = std::chrono::steady_clock::now();
	long time_elapsed = 0;
	int iter=0;
	for (iter = 0; iter < max_iters; iter++) {
		temp_vec = x_sol + z1 / rho1;

		if (does_log) {
			fprintf(fp, "Iteration: %d\n", iter);
		}
		/* Project vector on [0, 1] box to calculate y1 */
		project_box(n, temp_vec, y1);

		temp_vec = x_sol + z2 / rho2;

		/* Project vector on shifted lp box to calculate y2 */
		project_shifted_Lp_ball(n, temp_vec, projection_lp, y2);

		/* If the iteration is nonzero and it divides rho_change_step, it means
		 * that the rho updated in the last iteration and the matrix used by conjugate
		 * gradient should be updated
		 */
		if (iter != 0 && rhoUpdated) {
			temp_mat.diagonal().array() += (prev_rho1 + prev_rho2) * rho_change_ratio;
			temp_mat.makeCompressed();
		}

		/* Calculate the vector b in the conjugate gradient algorithm */
		temp_vec = rho1 * y1 + rho2 * y2 - ((*b_ptr) + z1 + z2);

		/* Explicit version of conjugate gradient used for profiling */

		/* Since the matrix used by the conjugate gradient changes only when after rho is updated
		 * we only need to recalculate the preconditioner if rho is updated in the last iteration
		 */
		if (rhoUpdated) {
			diagonalPreconditioner.compute(temp_mat);
			rhoUpdated = false;
		}

		x_sol = y1;
		_double_t tol = pcg_tol;
		int maxiter = pcg_maxiters;
		_conjugate_gradient(temp_mat, temp_vec, x_sol, diagonalPreconditioner, maxiter, tol);
		if (does_log) {
			fprintf(fp, "Conjugate gradient stops after %d iterations\n", maxiter);
			fprintf(fp, "Conjugate gradient stops with residual %lf\n", tol);
		}

		// printf("test 102!!\n");

		
		if(print_info==1){
			fprintf(xi, "Iter%d,", iter+1);
			int n = x_sol.rows();
			for(int i = 0 ; i < n-1; i++){
				fprintf(xi, "%lf,",x_sol.array()[i]);
			}
			fprintf(xi, "%lf\n",x_sol.array()[n-1]);
		}


		z1 = z1 + gamma_val * rho1 * (x_sol - y1);
		z2 = z2 + gamma_val * rho2 * (x_sol - y2);


		/* Testing the conditions to see if the algorithm converges */
		_double_t temp0 = std::max(x_sol.norm(), _double_t(2.2204e-16));
		cvg_test1 = (x_sol-y1).norm() / temp0;
		cvg_test2 = (x_sol-y2).norm() / temp0;
		if (cvg_test1 <= stop_threshold && cvg_test2 <= stop_threshold) {
			printf("Terminate by condition xyy. iter: %d, stop_threshold: %.6f\n", iter, std::max(cvg_test1, cvg_test2));
			if (does_log) {
				fprintf(fp, "iter: %d, stop_threshold: %.6f\n", iter, std::max(cvg_test1, cvg_test2));
			}
			break;
		}

		/* Update the rho value every rho_change_step */
		if ((iter+1) % rho_change_step == 0) {
			prev_rho1 = rho1;
			prev_rho2 = rho2;
			rho1 = learning_fact * rho1;
			rho2 = learning_fact * rho2;
			gamma_val = std::max(gamma_val * gamma_factor, _double_t(1.0));
			rhoUpdated = true;
			rho_change_ratio = learning_fact - 1.0;
		}

		/* Computer the relaxed cost function (x is not binary)*/
		_double_t obj_val = compute_cost(x_sol,(*A_ptr),(*b_ptr));
		obj_list.push_back(obj_val);
		if (obj_list.size() >= history_size) {
			std_obj = compute_std_obj(obj_list, history_size);
		}
		if (std_obj <= std_threshold) {
			printf("Terminate by condition obj_std. iter: %d, std_threshold: %.6f\n", iter, std_obj);
			if (does_log) {
				fprintf(fp, "iter: %d, std_threshold: %.6f\n", iter, std_obj);
			}
			break;
		}
		

		/* Calculating the actual cost */
		cur_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>(); /* The value type of the vector should be double
																	  therefore needs casting the boolean vector to double */
		prev_idx = cur_idx;
		cur_obj = compute_cost(prev_idx, (*A_ptr), (*b_ptr));

		



		/* Setting the best binary solution */
		if (best_bin_obj >= cur_obj) { 	
			best_bin_obj = cur_obj;
			best_sol = x_sol;
		}

		if (does_log) { 
			fprintf(fp, "current objective: %lf\n", obj_val);
			fprintf(fp, "current binary objective: %lf\n", cur_obj);
			fprintf(fp, "norm of x_sol: %lf\n", x_sol.norm());
			fprintf(fp, "norm of binary x_sol: %lf\n", cur_idx.norm());
			fprintf(fp, "norm of y1: %lf\n", y1.norm());
			fprintf(fp, "norm of y2: %lf\n", y2.norm());

			fprintf(fp, "norm of z1: %lf\n", z1.norm());
			fprintf(fp, "norm of z2: %lf\n", z2.norm());

			fprintf(fp, "rho1: %lf\n", rho1);
			fprintf(fp, "rho2: %lf\n", rho2);

			fprintf(fp, "time elapsed: %lf\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()*1.0/1000);
			fprintf(fp, "-------------------------------------------------\n");
		}


	}


	end = std::chrono::steady_clock::now();
	time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Time elapsed for all: " << time_elapsed*1.0/1000 << "s" << std::endl;

	if (does_log) {
		fprintf(fp, "Time elapsed: %lfs\n", time_elapsed*1.0/1000);
		fclose(fp);
	}

	if(print_info==1) fclose(xi);

	cur_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
	cur_obj = compute_cost(cur_idx, (*A_ptr), (*b_ptr));

	printf("obj: %lf; c: %lf; Energy: %lf\n", cur_obj, _c, cur_obj+_c);

	fprintf(al, "%d,%f,%f,%d,%f\n", problem, cur_obj, cur_obj+_c, iter+1, time_elapsed*1.0/1000);
	fclose(al);

	return int(cur_obj + _c);
}



int LPboxADMMsolver::ADMM_bqp(MatrixInfo matrix_info, SolverInstruction instruction, 
		Solution& sol) {


	const SparseMatrix *A_ptr, *C_ptr, *E_ptr;
	const DenseVector *b_ptr, *d_ptr, *f_ptr;

	A_ptr = matrix_info.A;
	b_ptr = matrix_info.b;

	SparseMatrix C_transpose, E_transpose, _2A_plus_rho1_rho2, rho3_C_transpose, rho4_E_transpose;

	DenseVector y3, z3, z4, Csq_diag, Esq_diag;

	int l = matrix_info.l;
	int m = matrix_info.m;
	int n = matrix_info.n;

	FILE *fp;
	if (does_log) {
		fp = fopen(log_file_path.c_str(), "w+");
	}

	DenseVector x_sol    = DenseVector(n);
	DenseVector y1       = DenseVector(n);
	DenseVector y2       = DenseVector(n);
	DenseVector z1       = DenseVector(n);
	DenseVector z2       = DenseVector(n);
	DenseVector prev_idx = DenseVector(n);
	DenseVector best_sol = DenseVector(n);
	DenseVector temp_vec = DenseVector(n);
	DenseVector cur_idx  = DenseVector(n);
	SparseMatrix temp_mat = SparseMatrix(n, n);

	/* The temp_vec_for_cg and temp_vec_for_mat_mat are vector that used to store the temporary vectors
	 * in the algorithm to prevent allocating new vectos. Adding this shall has around 5% of performance
	 * improvement (although might reduce the performance under some tasks by 10% since it increases the cost 
	 * of other operations besides matrix multiplication, such as vector addition and vector multiplication
	 * for reasons unknown)
	 */
	DenseVector temp_vec_for_cg = DenseVector(n);
	DenseVector temp_vec_for_mat_mul = DenseVector(n);
	_double_t cur_obj;
	bool rhoUpdated = true;

	x_sol = *(matrix_info.x0);

	z1.array() = 0;
	z2.array() = 0;
	cur_idx.array() = 0;


	Eigen::DiagonalPreconditioner<_double_t> diagonalPreconditioner;

	_double_t rho1 = initial_rho;
	_double_t rho2 = initial_rho;
	_double_t rho3 = initial_rho;
	_double_t rho4 = initial_rho;
	_double_t prev_rho1 = rho1;
	_double_t prev_rho2 = rho2;
	_double_t prev_rho3 = rho3;
	_double_t prev_rho4 = rho4;

	std::vector<_double_t> obj_list; /* Stores the objective value calculated during each iteration */
	_double_t std_obj = 1;

	_double_t cvg_test1;
	_double_t cvg_test2;
	_double_t rho_change_ratio;

	if (instruction.update_y3) {
		y3 = DenseVector(l);
	}

	if (instruction.update_z3) {
		z3 = DenseVector(m);
		z3.array() = 0;
	}

	if (instruction.update_z4) {
		z4 = DenseVector(l);
		z4.array() = 0;
	}


	/* If this task contains equality constraints, initialize C and d and relevant matrices */
	if (instruction.problem_type & equality) {
		C_ptr = matrix_info.C;
		C_transpose = (*C_ptr).transpose();
		rho3_C_transpose = rho3 * C_transpose;
		d_ptr = matrix_info.d;
	}

	/* If this task contains inequality constraints, initialize E and f and relevant matrices */
	if (instruction.problem_type & inequality) {
		E_ptr = matrix_info.E;
		E_transpose = (*E_ptr).transpose();
		rho4_E_transpose = rho4 * E_transpose;
		f_ptr = matrix_info.f;
	}

	/* Storing the matrix 2 * A + (rho1 + rho2) * I to save calculation time */
	_2A_plus_rho1_rho2 = 2 * (*A_ptr);
	_2A_plus_rho1_rho2.diagonal().array() += rho1 + rho2;

	SparseMatrix preconditioner_diag_mat(n, n); /* The diagonal matrix used by the preconditioner */
	preconditioner_diag_mat.reserve(n);
	std::vector<Triplet> preconditioner_diag_mat_triplets;
	preconditioner_diag_mat_triplets.reserve(n);
	/* The diagonal elements in the original expression evaluated by the preconditioner is given by the
	 * diagonal elements of 2 * _A + (rho1 + rho2) * I + rho3 * _C^T * _C
	 */
	for (int i = 0; i < n; i++) {
		preconditioner_diag_mat_triplets.push_back(Triplet(i, i, 0));
	}
	preconditioner_diag_mat.setFromTriplets(preconditioner_diag_mat_triplets.begin(), preconditioner_diag_mat_triplets.end());
	preconditioner_diag_mat.diagonal().array() = _2A_plus_rho1_rho2.diagonal().array();
	preconditioner_diag_mat.makeCompressed();


	/* The matrix expression for the unconstraint case is 2 * _A + (rho1 + rho2) * I */
	std::vector<std::vector<const SparseMatrix*>> matrix_expressions;
	matrix_expressions.emplace_back();
	matrix_expressions.back().push_back(&_2A_plus_rho1_rho2);

	/* If the problem has equality constraint, add rho3 * C^T * C to the matrix expression */
	if (instruction.problem_type & equality) {
		matrix_expressions.emplace_back();
		matrix_expressions.back().push_back(C_ptr);
		matrix_expressions.back().push_back(&rho3_C_transpose);

		/* Calculating the diagonal elements of Csq for preconditioner */
		Csq_diag = DenseVector(n); 		
		Csq_diag.setZero();
		for(int j = 0; j < C_transpose.outerSize(); ++j) {
			typename SparseMatrix::InnerIterator it(C_transpose, j);
			while (it) {
				if(it.value() != 0.0) {
					Csq_diag[j] += it.value() * it.value();
				}
				++it;
			}
		}
		preconditioner_diag_mat.diagonal().array() += rho3 * Csq_diag.array();
	}

	/* If the problem has equality constraint, add rho4 * E^T * E to the matrix expression */
	if (instruction.problem_type &inequality) {
		matrix_expressions.emplace_back();
		matrix_expressions.back().push_back(E_ptr);
		matrix_expressions.back().push_back(&rho4_E_transpose);

		/* Calculating the diagonal elements of Esq for preconditioner */
		Esq_diag = DenseVector(n);
		Esq_diag.setZero();
		for(int j = 0; j < E_transpose.outerSize(); ++j) {
			typename SparseMatrix::InnerIterator it(E_transpose, j);
			while (it) {
				if(it.value() != 0.0) {
					Esq_diag[j] += it.value() * it.value();
				}
				++it;
			}
		}
		preconditioner_diag_mat.diagonal().array() += rho4 * Esq_diag.array();
	}

	y1 = x_sol;
	y2 = x_sol;
	if (instruction.update_y3) {
		y3 = *f_ptr - *E_ptr * x_sol;
	}

	prev_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
	best_sol = x_sol;

	_double_t best_bin_obj = compute_cost(x_sol, *A_ptr, *b_ptr, temp_vec_for_mat_mul);

	if (does_log) {
		fprintf(fp, "Initial state\n");
        fprintf(fp, "norm of x_sol: %lf\n", x_sol.norm());
        fprintf(fp, "norm of b: %lf\n", (*b_ptr).norm());
        fprintf(fp, "norm of y1: %lf\n", y1.norm());
        fprintf(fp, "norm of y2: %lf\n", y2.norm());
        if (instruction.update_y3) {
            fprintf(fp, "norm of y3: %lf\n", y3.norm());
        }

        fprintf(fp, "norm of z1: %lf\n", z1.norm());
        fprintf(fp, "norm of z2: %lf\n", z2.norm());

        if (instruction.update_z3) {
            fprintf(fp, "norm of z3: %lf\n", z3.norm());
        }

        if (instruction.update_z4) {
            fprintf(fp, "norm of z4: %lf\n", z4.norm());
        }

        fprintf(fp, "norm of cur_idx: %lf\n", cur_idx.norm());
        fprintf(fp, "rho1: %lf\n", rho1);
        fprintf(fp, "rho2: %lf\n", rho2);
        fprintf(fp, "rho3: %lf\n", rho3);
        fprintf(fp, "rho4: %lf\n", rho4);
        fprintf(fp, "-------------------------------------------------\n");
	}

	std::chrono::steady_clock::time_point start, end;
	start = std::chrono::steady_clock::now();


	long time_elapsed = 0;
	for (int iter = 0; iter < max_iters; iter++) {
		if (does_log) {
			fprintf(fp, "Iteration: %d\n", iter);
		}
		temp_vec = x_sol + z1 / rho1;

		/* Project vector on [0, 1] box */
		project_box(n, temp_vec, y1);

		temp_vec = x_sol + z2 / rho2;

		/* Project vector on shifted lp box */
		project_shifted_Lp_ball(n, temp_vec, projection_lp, y2);

		if (instruction.update_y3) {
			mat_mul_vec(*E_ptr, x_sol, temp_vec_for_mat_mul);
			y3 = *f_ptr - temp_vec_for_mat_mul - z4 / rho4;
			project_vec_less_than(y3, y3, 0, 0); 
		}

		/* If the iteration is nonzero and it divides rho_change_step, it means
		 * that the rho updated in the last iteration
		 */
		if (iter != 0 && rhoUpdated) {
			/* Note that we need the previous rho to update in order to get the difference between
			 * the updated matrix and the not updated matrix. Another possible calculation is by
			 * calculating rho * rho_change_ration / learning_fact
			 */
			_2A_plus_rho1_rho2.diagonal().array() += rho_change_ratio * (prev_rho1 + prev_rho2);
			if (instruction.problem_type != unconstrained) {
				preconditioner_diag_mat.diagonal().array() += rho_change_ratio * (prev_rho1 + prev_rho2);
			}

			if (instruction.update_rho3) {
				preconditioner_diag_mat.diagonal().array() += rho_change_ratio * prev_rho3 * Csq_diag.array();
				rho3_C_transpose = learning_fact * rho3_C_transpose;	
			}

			if (instruction.update_rho4) {
				preconditioner_diag_mat.diagonal().array() += rho_change_ratio * prev_rho4 * Esq_diag.array();
				rho4_E_transpose = learning_fact * rho4_E_transpose;
			}
		}



		/* If the problem in unconstrained, the rhs vector is 
		 * rho1 * y1 + rho2 * y2 - (b + z1 + z2)
		 */
		if (instruction.problem_type == unconstrained) {
			temp_vec = rho1 * y1 + rho2 * y2 - (*b_ptr + z1 + z2);
		}

		/* If the problem in equality, the rhs vector is 
		 * rho1 * y1 + rho2 * y2 + rho3 * C^T * d - (b + z1 + z2 + C^T * z3)
		 */
		if (instruction.problem_type == equality) {
			temp_vec = rho1 * y1 + rho2 * y2 - (*b_ptr + z1 + z2);
			mat_mul_vec(rho3_C_transpose, *d_ptr, temp_vec_for_mat_mul);
			temp_vec += temp_vec_for_mat_mul;
			mat_mul_vec(C_transpose, z3, temp_vec_for_mat_mul);
			temp_vec -= temp_vec_for_mat_mul;
		}

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

		/* If the problem in equality, the rhs vector is 
		 * rho1 * y1 + rho2 * y2 + rho3 * C^T * d + rho4 * E^T * (f - y3) - (b + z1 + z2 + C^T * z3 + E^T * z4)
		 */
		if (instruction.problem_type == equality_and_inequality) {
			temp_vec = rho1 * y1 + rho2 * y2 - (*b_ptr + z1 + z2);
			mat_mul_vec(rho3_C_transpose, *d_ptr, temp_vec_for_mat_mul);
			temp_vec += temp_vec_for_mat_mul;
			mat_mul_vec(C_transpose, z3, temp_vec_for_mat_mul);
			temp_vec -= temp_vec_for_mat_mul;
			mat_mul_vec(rho4_E_transpose, *f_ptr - y3, temp_vec_for_mat_mul);
			temp_vec += temp_vec_for_mat_mul;
			mat_mul_vec(E_transpose, z4, temp_vec_for_mat_mul);
			temp_vec -= temp_vec_for_mat_mul;
		}


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
		_conjugate_gradient(matrix_expressions, temp_vec, x_sol, diagonalPreconditioner, 
				maxiter, tol, temp_vec_for_cg, temp_vec_for_mat_mul);

		if (does_log) {
			fprintf(fp, "Conjugate gradient stops after %d iterations\n", maxiter);
			fprintf(fp, "Conjugate gradient stops with residual %lf\n", tol);
		}

		z1 = z1 + gamma_val * rho1 * (x_sol - y1);
		z2 = z2 + gamma_val * rho2 * (x_sol - y2);
		if (instruction.update_z3) {
			z3 = z3 + gamma_val * rho3 * (*C_ptr * x_sol - *d_ptr);
		}

		if (instruction.update_z4) {
			z4 = z4 + gamma_val * rho4 * ((*E_ptr) * x_sol + y3 - (*f_ptr));
		}


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

		_double_t obj_val = compute_cost(x_sol, *A_ptr, *b_ptr);
		obj_list.push_back(obj_val);
		if (obj_list.size() >= history_size) {
			std_obj = compute_std_obj(obj_list, history_size);
		}
		if (std_obj <= std_threshold) {
			if (does_log) {
				fprintf(fp, "iter: %d, std_threshold: %.6f\n", iter, std_obj);
			}
			printf("iter: %d, std_threshold: %.6f\n", iter, std_obj);
			break;
		}

		cur_idx = (x_sol.array() >= 0.5).matrix().cast<_double_t>();
		prev_idx = cur_idx;
		cur_obj = compute_cost(prev_idx, *A_ptr, *b_ptr);

		if (best_bin_obj >= cur_obj) {
			best_bin_obj = cur_obj;
			best_sol = x_sol;
		}

		if (does_log) {
			fprintf(fp, "current objective: %lf\n", obj_val);
			fprintf(fp, "current binary objective: %lf\n", cur_obj);

			if (instruction.problem_type == equality || instruction.problem_type == equality_and_inequality) {
				fprintf(fp, "equality constraint violation: %lf\n", (*matrix_info.C * cur_idx - *matrix_info.d).norm() / x_sol.rows());
			}

			if (instruction.problem_type == inequality || instruction.problem_type == equality_and_inequality) {
				DenseVector diff = *matrix_info.E * cur_idx - *matrix_info.f;
				project_vec_less_than(diff, diff, 0, 0);
				fprintf(fp, "inequality constraint violation: %lf\n", diff.norm() / x_sol.rows());
			}

			fprintf(fp, "norm of x_sol: %lf\n", x_sol.norm());
			fprintf(fp, "norm of y1: %lf\n", y1.norm());
			fprintf(fp, "norm of y2: %lf\n", y2.norm());
			if (instruction.update_y3) {
				fprintf(fp, "norm of y3: %lf\n", y3.norm());
			}

			fprintf(fp, "norm of z1: %lf\n", z1.norm());
			fprintf(fp, "norm of z2: %lf\n", z2.norm());

			if (instruction.update_z3) {
				fprintf(fp, "norm of z3: %lf\n", z3.norm());
			}

			if (instruction.update_z4) {
				fprintf(fp, "norm of z4: %lf\n", z4.norm());
			}

			fprintf(fp, "norm of cur_idx: %lf\n", cur_idx.norm());
			fprintf(fp, "rho1: %lf\n", rho1);
			fprintf(fp, "rho2: %lf\n", rho2);
			if (instruction.update_rho3) {
				fprintf(fp, "rho3: %lf\n", rho3);
			}
			if (instruction.update_rho4) {
				fprintf(fp, "rho4: %lf\n", rho4);
			}
			fprintf(fp, "-------------------------------------------------\n");
		}
	}

	sol.x_sol = new DenseVector(x_sol);
	sol.y1 = new DenseVector(y1);
	sol.y2 = new DenseVector(y2);
	sol.best_sol = new DenseVector(best_sol);

	end = std::chrono::steady_clock::now();
	time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //micro
	std::cout << "Time elapsed: " << time_elapsed << "us//" << 1.0*time_elapsed/1000 << "s;" << std::endl;
	if (does_log) {
		fprintf(fp, "Time elapsed: %ldus\n", time_elapsed);
		fclose(fp);
	}
	sol.time_elapsed = time_elapsed;

	return 1;
}

int LPboxADMMsolver::ADMM_bqp_unconstrained(int n, const SparseMatrix &_A, const DenseVector &_b, 
		const DenseVector &x0, Solution& sol) {

	SolverInstruction solver_instruction;
	MatrixInfo matrix_info;

	matrix_info.x0 = &x0;
	matrix_info.A = &_A;
	matrix_info.b = &_b;
	matrix_info.n = n;

	solver_instruction.problem_type = unconstrained;
	solver_instruction.update_y3 = 0;
	solver_instruction.update_z3 = 0;
	solver_instruction.update_z4 = 0;
	solver_instruction.update_rho3 = 0;
	solver_instruction.update_rho4 = 0;

	ADMM_bqp(matrix_info, solver_instruction, sol);
	return 1;
}


int LPboxADMMsolver::ADMM_bqp_unconstrained(int n, _double_t *A, _double_t *b, _double_t *x0, 
		Solution& sol) {
	/* Initialize the parameters */
	auto _A = SparseMatrix(n, n);
	auto _b = DenseVector(n);
	auto _x0 = DenseVector(n);

	/* Generating the sparse matrix for A */
	std::vector<Triplet> triplet_list;
	for (int i = 0; i < n * n; i++) {
		int row = i % n;
		int col = i / n;

		if (A[i] == 0.0 && !(row == col)) {
			continue;
		}
		triplet_list.push_back(Triplet(row, col, A[i]));
	}
	_A.setFromTriplets(triplet_list.begin(), triplet_list.end());
	memcpy(_b.data(), b, n * sizeof(_double_t));
	memcpy(_x0.data(), x0, n * sizeof(_double_t));

	int ret = ADMM_bqp_unconstrained(n, _A, _b, _x0, sol);
	return ret;
}


int LPboxADMMsolver::ADMM_bqp_linear_eq(int n, const SparseMatrix &_A, const DenseVector &_b, 
		const DenseVector &x0, int m, const SparseMatrix &_C, const DenseVector &_d, Solution& sol) {

	SolverInstruction solver_instruction;
	MatrixInfo matrix_info;

	matrix_info.x0 = &x0;
	matrix_info.A = &_A;
	matrix_info.b = &_b;
	matrix_info.n = n;
	matrix_info.C = &_C;
	matrix_info.d = &_d; 
	matrix_info.m = m;

	solver_instruction.problem_type = equality;
	solver_instruction.update_y3 = 0;
	solver_instruction.update_z3 = 1;
	solver_instruction.update_z4 = 0;

	/* If this value is set to be true, then the admm update will update
     * rho3 at each iteration with rate learning_fact. The default setting
     * is false, following the strategy in the clustering task */
	solver_instruction.update_rho3 = 0;
	solver_instruction.update_rho4 = 0;

	ADMM_bqp(matrix_info, solver_instruction, sol);
	return 1;
}


int LPboxADMMsolver::ADMM_bqp_linear_eq(int n, _double_t *A, _double_t *b, _double_t *x0, int m, 
		_double_t *C, _double_t *d, Solution& sol) {
	/* Initialize the parameters */
	auto _A = SparseMatrix(n, n);
	auto _b = DenseVector(n);
	auto _x0 = DenseVector(n);
	auto _C = SparseMatrix(m, n);
	auto _d = DenseVector(m);

	/* Generating the sparse matrix for A */
	std::vector<Triplet> triplet_list; /* Maybe one can reserve some space here */
	for (int i = 0; i < n * n; i++) {
		if (A[i] == 0.0) {
			continue;
		}
		int row = i % n;
		int col = i / n;
		triplet_list.push_back(Triplet(row, col, A[i]));
	}
	_A.setFromTriplets(triplet_list.begin(), triplet_list.end());

	/* Generating the sparse matrix for C */
	triplet_list.clear();
	for (int i = 0; i < m * n; i++) {
		if (C[i] == 0.0) {
			continue;
		}
		int row = i % n;
		int col = i / n;
		triplet_list.push_back(Triplet(row, col, C[i]));
	}
	_C.setFromTriplets(triplet_list.begin(), triplet_list.end());

	memcpy(_x0.data(), x0, n * sizeof(_double_t));
	memcpy(_b.data(), b, n * sizeof(_double_t));
	memcpy(_d.data(), d, m * sizeof(_double_t));

	int ret = ADMM_bqp_linear_eq(n, _A, _b, _x0, m, _C, _d, sol);
	return ret;
}

int LPboxADMMsolver::ADMM_bqp_linear_ineq(int n, const SparseMatrix &_A, const DenseVector &_b,
		const DenseVector &x0, int l, const SparseMatrix &_E, const DenseVector &_f, Solution& sol) {

	SolverInstruction solver_instruction;
	MatrixInfo matrix_info;

	matrix_info.x0 = &x0;
	matrix_info.A = &_A;
	matrix_info.b = &_b;
	matrix_info.n = n;
	matrix_info.E = &_E;
	matrix_info.f = &_f; 
	matrix_info.l = l;

	solver_instruction.problem_type = inequality;
	solver_instruction.update_y3 = 1;
	solver_instruction.update_z3 = 0;
	solver_instruction.update_z4 = 1;
	solver_instruction.update_rho3 = 0;
	solver_instruction.update_rho4 = 1;

	ADMM_bqp(matrix_info, solver_instruction, sol);
	return 1;
}

int LPboxADMMsolver::ADMM_bqp_linear_ineq(int n, _double_t *A, _double_t *b, _double_t *x0, 
		int l, _double_t *E, _double_t *f, Solution& sol) {
	/* Initialize the parameters */
	auto _A       = SparseMatrix(n, n);
	auto _b       = DenseVector(n);
	auto _x0	  = DenseVector(n);
	auto _E       = SparseMatrix(l, n);
	auto _f       = DenseVector(l);

	/* Generating the sparse matrix for A */
	std::vector<Triplet> triplet_list; /* Maybe one can reserve some space here */
	for (int i = 0; i < n * n; i++) {
		if (A[i] == 0.0) {
			continue;
		}
		int row = i % n;
		int col = i / n;
		triplet_list.push_back(Triplet(row, col, A[i]));
	}
	_A.setFromTriplets(triplet_list.begin(), triplet_list.end());

	/* Generating the sparse matrix for C */
	triplet_list.clear();
	for (int i = 0; i < l * n; i++) {
		if (E[i] == 0.0) {
			continue;
		}
		int row = i % n;
		int col = i / n;
		triplet_list.push_back(Triplet(row, col, E[i]));
	}
	_E.setFromTriplets(triplet_list.begin(), triplet_list.end());

	memcpy(_x0.data(), x0, n * sizeof(_double_t));
	memcpy(_b.data(), b, n * sizeof(_double_t));
	memcpy(_f.data(), f, l * sizeof(_double_t));

	int ret = ADMM_bqp_linear_ineq(n, _A, _b, _x0, l, _E, _f, sol);
	return ret;
}



int LPboxADMMsolver::ADMM_bqp_linear_eq_and_uneq(int n, const SparseMatrix &_A, 
		const DenseVector &_b, const DenseVector &x0, int m, const SparseMatrix &_C, const DenseVector &_d, 
		int l, const SparseMatrix &_E, const DenseVector &_f, Solution& sol) {

	SolverInstruction solver_instruction;
	MatrixInfo matrix_info;

	matrix_info.x0 = &x0;
	matrix_info.A = &_A;
	matrix_info.b = &_b;
	matrix_info.n = n;
	matrix_info.C = &_C;
	matrix_info.d = &_d;
	matrix_info.m = m;
	matrix_info.E = &_E;
	matrix_info.f = &_f; 
	matrix_info.l = l;

	solver_instruction.problem_type = equality_and_inequality;
	solver_instruction.update_y3 = 1;
	solver_instruction.update_z3 = 1;
	solver_instruction.update_z4 = 1;
	solver_instruction.update_rho3 = 1;
	solver_instruction.update_rho4 = 1;

	ADMM_bqp(matrix_info, solver_instruction, sol);
	return 1;

}


/* Currently assuming that the input matrix are column majored */
int LPboxADMMsolver::ADMM_bqp_linear_eq_and_uneq(int n, _double_t *A, _double_t *b, 
		_double_t *x0, int m, _double_t *C, _double_t *d, int l, _double_t *E, _double_t *f, Solution& sol) {
	/* Initialize the parameters */
	auto _A       = SparseMatrix(n, n);
	auto _b       = DenseVector(n);
	auto _x0      = DenseVector(n);
	auto _C       = SparseMatrix(m, n);
	auto _d       = DenseVector(m);
	auto _E       = SparseMatrix(l, n);
	auto _f       = DenseVector(l);

	/* Generating the sparse matrix for A */

	std::vector<Triplet> triplet_list; /* Maybe one can reserve some space here */
	for (int i = 0; i < n * n; i++) {
		if (A[i] == 0.0) {
			continue;
		}
		int row = i % n;
		int col = i / n;
		triplet_list.push_back(Triplet(row, col, A[i]));
	}
	_A.setFromTriplets(triplet_list.begin(), triplet_list.end());

	/* Generating the sparse matrix for C */
	triplet_list.clear();
	for (int i = 0; i < m * n; i++) {
		if (E[i] == 0.0) {
			continue;
		}
		int row = i % n;
		int col = i / n;
		triplet_list.push_back(Triplet(row, col, E[i]));
	}
	_E.setFromTriplets(triplet_list.begin(), triplet_list.end());

	/* Generating the sparse matrix for C */
	triplet_list.clear();
	for (int i = 0; i < l * n; i++) {
		if (C[i] == 0.0) {
			continue;
		}
		int row = i % n;
		int col = i / n;
		triplet_list.push_back(Triplet(row, col, C[i]));
	}
	_C.setFromTriplets(triplet_list.begin(), triplet_list.end());

	memcpy(_x0.data(), x0, n * sizeof(_double_t));
	memcpy(_b.data(), b, n * sizeof(_double_t));
	memcpy(_d.data(), d, m * sizeof(_double_t));
	memcpy(_f.data(), f, l * sizeof(_double_t));

	int ret = ADMM_bqp_linear_eq_and_uneq(n, _A, _b, _x0, m, _C, _d, l, _E, _f, sol);
	return ret;
}