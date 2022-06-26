#define OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT

#include <unistd.h>
#include <Eigen4/Dense>
#include <Eigen4/Sparse>
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
// #include "image_segmentation_utils.h"
#include <getopt.h>

#include "LPboxADMMsolver.h"

using namespace cv;

int main(int argc, char *argv[]) {
	int o;
	char* imagePath;
	int numNodes;
	char* outputPath;
    std::string logFile;
    bool log_file_is_set = false;
	int option_idx;

	for(int i = 0 ; i < 100; i++){
		LPboxADMMsolver solver(1, 1e4, i);  // 0-donot print fix info; 1-get xiters
		solver.ADMM_bqp_unconstrained_init();
		int obj = solver.ADMM_bqp_unconstrained_legacy();
		// solver.save_img(); 
	}
	return 1;
}


