//Udacity HW2 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

#include "reference_calc.h"
#include "compare.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 * d_inputImageRGBA__;
uchar4 * d_outputImageRGBA__;

float * h_filter__;


size_t numRows()
{
	return imageInputRGBA.rows;
}


size_t numCols()
{
	return imageInputRGBA.cols;
}

/**
 * Return types are void since any internal error will be handled by quitting. no point in returning error codes.
 * returns a pointer to an RGBA version of the input image, and a pointer to the single channel grey-scale output,
 * on both the host and device
 */
void preProcess(uchar4 ** h_inputImageRGBA, uchar4 ** h_outputImageRGBA,
				uchar4 ** d_inputImageRGBA, uchar4 ** d_outputImageRGBA,
				unsigned char ** d_redBlurred,
				unsigned char ** d_greenBlurred,
				unsigned char ** d_blueBlurred,
				float ** h_filter, int * filterWidth,
				const std::string & filename)
{
	//make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	cv::Mat image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
	if (image.empty())
	{
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageInputRGBA, cv::COLOR_BGR2RGBA);

	//allocate memory for the output
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageInputRGBA.isContinuous() or !imageOutputRGBA.isContinuous())
	{
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}

	*h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
	*h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();

	// allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around

	// copy input array to the GPU
	checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_inputImageRGBA__  = *d_inputImageRGBA;
	d_outputImageRGBA__ = *d_outputImageRGBA;

	// now create the filter that they will use
	const int blurKernelWidth = 9;
	const float blurKernelSigma = 2.;

	*filterWidth = blurKernelWidth;

	// create and fill the filter we will convolve with
	*h_filter = new float[blurKernelWidth * blurKernelWidth];
	h_filter__ = *h_filter;

	float filterSum = 0.f; //for normalization

	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r)
	{
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c)
		{
			const float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
			const size_t filterIdx = (r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2;
			(*h_filter)[filterIdx] = filterValue;
			filterSum += filterValue;
		}
	}

	const float normalizationFactor = 1.f / filterSum;

	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r)
	{
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c)
		{
			const size_t filterIdx = (r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2;
			(*h_filter)[filterIdx] *= normalizationFactor;
		}
	}

	//blurred
	checkCudaErrors(cudaMalloc(d_redBlurred,    sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels));
}


void postProcess(const std::string & output_file, uchar4 * data_ptr)
{
	cv::Mat output(numRows(), numCols(), CV_8UC4, (void*)data_ptr);

	cv::Mat imageOutputBGR;
	cv::cvtColor(output, imageOutputBGR, cv::COLOR_RGBA2BGR);
	//output the image
	cv::imwrite(output_file.c_str(), imageOutputBGR);
}


void cleanUp(void)
{
	cudaFree(d_inputImageRGBA__);
	cudaFree(d_outputImageRGBA__);
	delete[] h_filter__;
}


/**
 * An unused bit of code showing how to accomplish this assignment using OpenCV.  It is much faster
 * than the naive implementation in reference_calc.cpp.
 */
void generateReferenceImage(std::string input_file, std::string reference_file, int kernel_size)
{
	cv::Mat input = cv::imread(input_file);
	// Create an identical image for the output as a placeholder
	cv::Mat reference = cv::imread(input_file);
	cv::GaussianBlur(input, reference, cv::Size2i(kernel_size, kernel_size),0);
	cv::imwrite(reference_file, reference);
}


/**
 * DEFINED IN student_func.cu
 */
void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
		uchar4* const d_outputImageRGBA,
		const size_t numRows, const size_t numCols,
		unsigned char *d_redBlurred,
		unsigned char *d_greenBlurred,
		unsigned char *d_blueBlurred,
		const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
		const float* const h_filter, const size_t filterWidth);


/*******  Begin main *********/

int main(int argc, char **argv)
{
	uchar4 * h_inputImageRGBA,  * d_inputImageRGBA;
	uchar4 * h_outputImageRGBA, * d_outputImageRGBA;
	unsigned char * d_redBlurred, * d_greenBlurred, * d_blueBlurred;

	float * h_filter;
	int filterWidth;

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
		output_file = "HW2_output.png";
		reference_file = "HW2_reference.png";
		break;
	case 3:
		input_file  = std::string(argv[1]);
		output_file = std::string(argv[2]);
		reference_file = "HW2_reference.png";
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
		std::cerr << "Usage: ./HW2 input_file [output_filename] [reference_filename] [perPixelError] [globalError]" << std::endl;
		exit(1);
	}

	// load the image and give us our input and output pointers
	preProcess(	&h_inputImageRGBA, &h_outputImageRGBA,
				&d_inputImageRGBA, &d_outputImageRGBA,
				&d_redBlurred, &d_greenBlurred, &d_blueBlurred,
				&h_filter, &filterWidth, input_file);

	allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);

	GpuTimer timer;
	timer.Start();

	//call the students' code
	your_gaussian_blur(	h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA,
						numRows(), numCols(),
						d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
	timer.Stop();

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

	if (err < 0)
	{
		//Couldn't print! Probably the student closed stdout - bad news
		std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
		exit(1);
	}

	//check results and output the blurred image

	const size_t numPixels = numRows() * numCols();
	//copy the output back to the host
	checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

	postProcess(output_file, h_outputImageRGBA);

	referenceCalculation(h_inputImageRGBA, h_outputImageRGBA, numRows(), numCols(), h_filter, filterWidth);

	postProcess(reference_file, h_outputImageRGBA);

	// Cheater easy way with OpenCV
	//generateReferenceImage(input_file, reference_file, filterWidth);

	compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

	checkCudaErrors(cudaFree(d_redBlurred));
	checkCudaErrors(cudaFree(d_greenBlurred));
	checkCudaErrors(cudaFree(d_blueBlurred));

	cleanUp();

	return 0;
}
