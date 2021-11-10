#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include "CUDA_color_conversion.h"

#include <npp.h>
#include <nppi.h>
#include <npps.h>

void YUV420pToRGB_NPP(const unsigned char *input, unsigned char *output, int width,
	int height, int device_id) {

	int curDev = -1;
	cudaGetDevice(&curDev);
	if (curDev != device_id) {
		cudaSetDevice(device_id);
	}

	Npp8u *pNppInput;
	int nppInputStep;
	pNppInput = nppiMalloc_8u_C1(width, height / 2 * 3, &nppInputStep);
	cudaMemcpy(pNppInput, input, sizeof(Npp8u) * width * height / 2 * 3,
		cudaMemcpyHostToDevice);

	Npp8u *pNppInput_V_half = pNppInput + width * height;

	Npp8u *pNppInput_U_half = pNppInput_V_half + width * height / 4;

	Npp8u *pNppInputArray[3] = { pNppInput, pNppInput_U_half, pNppInput_V_half };
	int nppInputSteps[3] = { width, width / 2, width / 2 };

	Npp8u *pNppOutput;
	int nppOutputStep;
	pNppOutput = nppiMalloc_8u_C3(width, height, &nppOutputStep);

	NppiSize nppSize;
	nppSize.width = width;
	nppSize.height = height;
	nppiYUV420ToBGR_8u_P3C3R(pNppInputArray, nppInputSteps, pNppOutput, width * 3,
		nppSize);

	cudaMemcpy(output, pNppOutput, sizeof(unsigned char) * width * height * 3,
		cudaMemcpyDeviceToHost);

	cudaFree(pNppInput);
	cudaFree(pNppOutput);

	if (curDev != device_id) {
		cudaSetDevice(curDev);
	}
}

__global__ void cvtNV12_BGR(unsigned char* A,unsigned char* B,
	const int height,const int width,const int linesize)
{
	int IDX = blockDim.x * blockIdx.x + threadIdx.x;
	long len = width * height;
	if (IDX < len)
	{
		int j = IDX % width;
		int i = (IDX - j) / width;

		int bgr[3];
		int yIdx, uvIdx, idx;
		int y, u, v;

		yIdx = i * linesize + j;
		uvIdx = linesize * height + (i / 2)*linesize + j - j % 2;

		y = A[yIdx];
		u = A[uvIdx];
		v = A[uvIdx + 1];

		bgr[0] = y + 1.772 * (u - 128);
		bgr[1] = y - 0.34414 * (u - 128) - 0.71414 * (v - 128);
		bgr[2] = y + 1.402 * (v - 128);

		for (int k = 0; k < 3; k++) {
			idx = (i * width + j) * 3 + k;
			if (bgr[k] >= 0 && bgr[k] < 255) {
				B[idx] = bgr[k];
			}
			else {
				B[idx] = bgr[k] < 0 ? 0 : 255;
			}
		}
	}
}

int cvtColor(unsigned char *d_req,
	unsigned char *d_res,
	int resolution,
	int height,
	int width,
	int linesize)
{
	int threadPerBlock = 256;
	int blockPerGrid = (resolution + threadPerBlock - 1) / threadPerBlock;
	//params_type-->switch?
	cvtNV12_BGR << <blockPerGrid, threadPerBlock >> > (d_req, d_res, height, width, line);

	return 0;
}