//Udacity HW1 Solution

#include "timer.h"
#include "utils.h"
#include "reference_calc.h"
#include "compare.h"

// include the definitions of the above functions for this homework
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <unistd.h>
#include <string>
#include <cstdio>
#include <cmath>
#include <iostream>


//   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~

/**
 * The CUDA wrapper, defined in student_func.cu
 */
void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, 
		uchar4 * const d_rgbaImage,
		unsigned char* const d_greyImage,
		size_t numRows, size_t numCols);

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        * d_rgbaImage__;
unsigned char * d_greyImage__;


//   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~

//   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~
size_t numRows() { return imageRGBA.rows; }

size_t numCols() { return imageRGBA.cols; }


auto is_pow_2 = [] (int n, const std::string & L)
{
	const bool ISP2 = (n and (n & (n - 1)) == 0);
	if (ISP2 == false)
	{
		std::cerr << "\tWarning: " << L << " are not a power of 2, bad behaviour (black belt)" << std::endl;
	}
};


/**
 * return types are void since any internal error will be handled by quitting
 * no point in returning error codes...
 * returns a pointer to an RGBA version of the input image
 * and a pointer to the single channel grey-scale output
 * on both the host and device
*/
void preProcess(uchar4 ** inputImage, unsigned char ** greyImage,
		uchar4 ** d_rgbaImage, unsigned char ** d_greyImage,
		const std::string & filename)
{
	// make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	cv::Mat image;
	image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
	if (image.empty())
	{
		const size_t MAXPATHLEN = 2048;
		char temp[MAXPATHLEN];
		const std::string CWD = (getcwd(temp, MAXPATHLEN) ? std::string(temp) : std::string(""));
		std::cerr << "Couldn't open file: " << filename << ", cwd: " << CWD << std::endl;
		exit(1);
	}
	else
	{
		std::cout << "Loaded file " << filename << ", Pixels: " << image.rows << " r x " << image.cols << " c" << std::endl;
		if (image.rows != image.cols)
		{
			std::cerr << "\tWarning: image is not a square, bad behaviour (distortion)";
		}
		is_pow_2(image.rows, "rows");
		is_pow_2(image.cols, "cols");
	}

	cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);

	// allocate memory for the output
	imageGrey.create(image.rows, image.cols, CV_8UC1);

	// This shouldn't ever happen given the way the images are created
	// at least based upon my limited understanding of OpenCV, but better to check
	if ((imageRGBA.isContinuous() and imageGrey.isContinuous()) == false)
	{
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}

	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	// same stuff, more warnings
	// *inputImage = imageRGBA(0, 0);
	*greyImage  = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();

	// allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
	// make sure no memory is left laying around
	checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)));

	// copy input array to the GPU
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}


/**
 * Outputs image on disk
 */
void postProcess(const std::string & output_file, unsigned char * data_ptr)
{
	cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);

	// output the image
	cv::imwrite(output_file.c_str(), output);
}


/**
 * Free GPU allocations
 */
void cleanup()
{
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
}


/**
 * Outputs reference image on disk
 */
void generateReferenceImage(const std::string & input_filename, const std::string & output_filename)
{
	cv::Mat reference = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);

	cv::imwrite(output_filename, reference);
}


/**
 * MAIN
 */
int main(int argc, char **argv)
{
	std::cout << "Welcome to HW1 built at " << __DATE__ << ", " << __TIME__ << std::endl;

	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;

	std::string input_file;
	std::string output_file;
	std::string reference_file;
	double perPixelError = 0.0;
	double globalError   = 0.0;
	bool useEpsCheck = false;

	switch (argc)
	{
	case 2:
		input_file = std::string(argv[1]);
		output_file = "HW1_output.png";
		reference_file = "HW1_reference.png";
		break;
	case 3:
		input_file  = std::string(argv[1]);
		output_file = std::string(argv[2]);
		reference_file = "HW1_reference.png";
		break;
	case 4:
		input_file  = std::string(argv[1]);
		output_file = std::string(argv[2]);
		reference_file = std::string(argv[3]);
		break;
	case 6:
		useEpsCheck=true;
		input_file  = std::string(argv[1]);
		output_file = std::string(argv[2]);
		reference_file = std::string(argv[3]);
		perPixelError = atof(argv[4]);
		globalError   = atof(argv[5]);
		break;
	default:
		std::cerr << "Usage: ./HW1 input_file [output_filename [reference_filename [perPixelError globalError]]]" << std::endl;
		exit(1);
	}

	// load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

	GpuTimer timer;
	timer.Start();

	// -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
	// call the students' code
	your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
	// -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

	timer.Stop();
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	std::cout <<"Your code ran in: " << timer.Elapsed() << " msecs" << std::endl;

	const size_t numPixels = numRows() * numCols();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

	// check results and output the grey image
	postProcess(output_file, h_greyImage);

	referenceCalculation(h_rgbaImage, h_greyImage, numRows(), numCols());

	postProcess(reference_file, h_greyImage);

	generateReferenceImage(input_file, reference_file);
	compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

	cleanup();

	return 0;
}
