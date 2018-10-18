// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "utils.h"
#include <iostream>



/**
 * Fill in the kernel to convert from color to greyscale
 * the mapping from components of a uchar4 to RGBA is:
 *  .x -> R ; .y -> G ; .z -> B ; .w -> A
 *
 * The output (greyImage) at each pixel should be the result of
 * applying the formula: output = .299f * R + .587f * G + .114f * B;
 * Note: We will be ignoring the alpha channel for this conversion
 *
 * First create a mapping from the 2D block and grid locations
 * to an absolute 2D location in the image, then use that to
 * calculate a 1D offset
 */
__global__
void rgba_to_greyscale(
		const uchar4* const rgbaImage,
		unsigned char* const greyImage,
		int numRows, int numCols)
{
	const int GRID_Y = (blockDim.y * blockIdx.y) + threadIdx.y;
	const int GRID_X = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int OFFSET_X = GRID_Y * gridDim.x * blockDim.x;
	const int i = OFFSET_X + GRID_X;

	greyImage[i] = .299f * rgbaImage[i].x
	             + .587f * rgbaImage[i].y
	             + .114f * rgbaImage[i].z;

	// I cannot. std::cout << "Kernel: pixel " << i << " = " << greyImage[i] << std::endl;
}


/**
 * https://stackoverflow.com/questions/16619274/cuda-griddim-and-blockdim
 *
 * blockDim.x,y,z	 		number of threads in a block		in the particular direction
 *
 * gridDim.x,y,z			number of blocks in a grid			in the particular direction
 *
 * blockDim.x * gridDim.x	number of threads in a grid			in the x direction
 *
 */
void your_rgba_to_greyscale(const uchar4* const h_rgbaImage, uchar4* const d_rgbaImage,
		unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
/*
			const int threadsPerBlock = 256;
			// = (50000 + 256 - 1) / 256 = 196
			const int blocksPerGrid = (m_numElements + threadsPerBlock - 1) / threadsPerBlock;
			printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
			d_vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m_numElements);
*/

	const int N_BLOCKS = 1024;
	const int N_BLOCKS_PER_DIM = sqrt(N_BLOCKS);

	// number of threads per block (up to 512/1024 based on GPU model)
	const dim3 blockSize(numCols/N_BLOCKS_PER_DIM, numRows/N_BLOCKS_PER_DIM, 1);
	const int nThreadsPerBlocks = (blockSize.x * blockSize.y);

	// number of blocks
	const dim3 gridSize(N_BLOCKS_PER_DIM, N_BLOCKS_PER_DIM, 1);
	const int nBlocks = (gridSize.x * gridSize.y);
	const int nThreads = nBlocks * nThreadsPerBlocks;

	std::cout << "\nBlocks per dimension: " << N_BLOCKS_PER_DIM
				<< "\nTotal Threads: " << nThreads << "\n"
				<< "\nType      \tC(x)\tR(x)\tTot\tLabel"
				<< "\nElements  \t" << numCols << "\t" << numRows << "\t" << (numRows*numCols) << "\tPixels"
				<< "\nGrid Size \t" << gridSize.x << "\t" << gridSize.y << "\t" << nBlocks << "\tBlocks"
				<< "\nBlock Size\t" << blockSize.x << "\t" << blockSize.y << "\t" << nThreadsPerBlocks << "\tThreads/Block"
				<< "\n" << std::endl;

	rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}
