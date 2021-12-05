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

/*****************************************************************
 *
 *  cutColor yuv2bgr i420
 */
#define DESCALE(x, n)    (((x) + (1 << ((n)-1)))>>(n))
#define COEFFS_0 		(22987)
#define COEFFS_1 		(-11698)
#define COEFFS_2 		(-5636)
#define COEFFS_3 		(29049)
#define clip(minv, maxv, value)  ((value)<minv) ? minv : (((value)>maxv) ? maxv : (value))
__global__ void gpuConvertI420toBGR_kernel(
	unsigned char *src_y, unsigned char *src_u, unsigned char *src_v, unsigned char *dst,
	unsigned int width, unsigned int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx * 2 >= width) {
		return;
	}
	int height_half = (height >> 1);
	int width_half = (width >> 1);

	for (int i = 0; i < height_half; ++i) {
		int cb = src_u[i*width_half + idx];
		int cr = src_v[i*width_half + idx];
		int b = DESCALE((cb - 128)*COEFFS_3, 14);
		int g = DESCALE((cb - 128)*COEFFS_2 + (cr - 128)*COEFFS_1, 14);
		int r = DESCALE((cr - 128)*COEFFS_0, 14);
		int y00 = src_y[(i * 2 + 0)*width + idx * 2 + 0];
		int y01 = src_y[(i * 2 + 0)*width + idx * 2 + 1];
		int y10 = src_y[(i * 2 + 1)*width + idx * 2 + 0];
		int y11 = src_y[(i * 2 + 1)*width + idx * 2 + 1];

		dst[(i * 2 + 0)*width * 3 + idx * 6 + 0] = clip(0, 255, y00 + b);//B
		dst[(i * 2 + 0)*width * 3 + idx * 6 + 1] = clip(0, 255, y00 + g);//G
		dst[(i * 2 + 0)*width * 3 + idx * 6 + 2] = clip(0, 255, y00 + r);//R
		dst[(i * 2 + 0)*width * 3 + idx * 6 + 3] = clip(0, 255, y01 + b);//B
		dst[(i * 2 + 0)*width * 3 + idx * 6 + 4] = clip(0, 255, y01 + g);//G
		dst[(i * 2 + 0)*width * 3 + idx * 6 + 5] = clip(0, 255, y01 + r);//R
		dst[(i * 2 + 1)*width * 3 + idx * 6 + 0] = clip(0, 255, y10 + b);//B
		dst[(i * 2 + 1)*width * 3 + idx * 6 + 1] = clip(0, 255, y10 + g);//G
		dst[(i * 2 + 1)*width * 3 + idx * 6 + 2] = clip(0, 255, y10 + r);//R
		dst[(i * 2 + 1)*width * 3 + idx * 6 + 3] = clip(0, 255, y11 + b);//B
		dst[(i * 2 + 1)*width * 3 + idx * 6 + 4] = clip(0, 255, y11 + g);//G
		dst[(i * 2 + 1)*width * 3 + idx * 6 + 5] = clip(0, 255, y11 + r);//R
	}
}
static cudaError_t cuConvert_yuv2bgr_i420(PCUOBJ pObj, Mat &dst)
{
	cudaError_t ret = cudaSuccess;
	int width = dst.cols, height = dst.rows;

	unsigned char *d_src = pObj->d_mem[pObj->imem];
	unsigned char *d_dst = dst.data;

	unsigned int blockSize = 1024;
	unsigned int numBlocks = (width / 2 + blockSize - 1) / blockSize;

	gpuConvertI420toBGR_kernel << <numBlocks, blockSize >> > (
		d_src, d_src + width * height, d_src + width * height + ((width >> 1)*(height >> 1)), d_dst,
		width, height);
	cudaStreamSynchronize(NULL);

	return ret;
}

cudaError_t cuConvertConn_yuv2bgr_i420(int chId, Mat &dst, int flag)
{
	//LOCK;
	cudaError_t ret = cudaSuccess;
	ret = cuConvert_yuv2bgr_i420(&gObjs[chId], dst);
	if (ret != cudaSuccess) {
		printf("%s(%i)  : cudaGetLastError() CUDA error: %d\n", __FILE__, __LINE__, (int)cudaGetLastError());
	}
	ret = cudaDeviceSynchronize();
	//UNLOCK;
	return ret;
}

//    Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//    Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//    Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//    Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
//    U U U U U U      V V V V V V      U V U V U V      V U V U V U
//    V V V V V V      U U U U U U      U V U V U V      V U V U V U
//    - I420 -          - YV12 -         - NV12 -         - NV21 -

#define GPU_BLOCK_THREADS  512

static dim3 gridDims(int numJobs) {
	int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
	return dim3(ceil(numJobs / (float)numBlockThreads));
}

static dim3 blockDims(int numJobs) {
	return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

__global__ void convert_nv12_to_bgr_float_kernel(const uint8_t* y, const uint8_t* uv, int width, int height, int linesize,
	H264Codec::CudaNorm norm, float* dst_b, float* dst_g, float* dst_r, int edge) {

	int position = blockDim.x * blockIdx.x + threadIdx.x;
	if (position >= edge) return;

	int ox = position % width;
	int oy = position / width;

	const uint8_t& yvalue = y[oy * linesize + ox];
	int offset_uv = oy / 2 * linesize + (ox / 2) * 2;
	const uint8_t& u = uv[offset_uv + 0];
	const uint8_t& v = uv[offset_uv + 1];

	float b = 1.164f * (yvalue - 16.0f) + 2.018f * (u - 128.0f);
	float g = 1.164f * (yvalue - 16.0f) - 0.813f * (v - 128.0f) - 0.391 * (u - 128.0f);
	float r = 1.164f * (yvalue - 16.0f) + 1.596f * (v - 128.0f);

	if (norm.type == H264Codec::CudaNormType::MeanSTD) {
		dst_b[position] = (b * norm.scale - norm.mean[0]) / norm.std[0];
		dst_g[position] = (g * norm.scale - norm.mean[1]) / norm.std[1];
		dst_r[position] = (r * norm.scale - norm.mean[2]) / norm.std[2];
	}
	else if (norm.type == H264Codec::CudaNormType::ScaleAdd) {
		dst_b[position] = b * norm.scale + norm.add;
		dst_g[position] = g * norm.scale + norm.add;
		dst_r[position] = r * norm.scale + norm.add;
	}
	else {
		dst_b[position] = b;
		dst_g[position] = g;
		dst_r[position] = r;
	}
}

namespace H264Codec {

	void convert_nv12_to_bgr_float(const uint8_t* y, const uint8_t* uv, int width, int height, int linesize, const CudaNorm& norm, float* dst, cudaStream_t stream) {

		int total = width * height;
		auto grid = gridDims(total);
		auto block = blockDims(total);

		convert_nv12_to_bgr_float_kernel << <grid, block, 0, stream >> > (
			y, uv, width, height, linesize, norm,
			dst + width * height * 0,
			dst + width * height * 1,
			dst + width * height * 2,
			total
			);
	}
};