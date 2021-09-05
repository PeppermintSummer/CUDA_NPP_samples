#pragma once
#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nppi.h>

#include "CUDA_npp_resize.h"

static void copyYUVCPU2GPU(Npp8u*pDst, uint8_t*pSrcY, uint8_t*pSrcU, uint8_t*pSrcV, int width, int height)
{
	if (pDst == nullptr || pSrcY == nullptr || pSrcU == nullptr || pSrcV == nullptr) {
		return;
	}

	uint8_t*pTemp = new uint8_t[width*height * 3];
	memcpy(pTemp, pSrcY, width*height);

	uint8_t *pTempDst = pTemp + width * height;
	uint8_t *pTempSrc = pSrcU;
	for (int i = 0; i < height / 2; i++) {
		memcpy(pTempDst, pTempSrc, width / 2);
		pTempDst += width;
		pTempSrc += width / 2;
	}

	pTempDst = pTemp + width * height * 2;
	pTempSrc = pSrcV;
	for (int i = 0; i < height / 2; i++) {
		memcpy(pTempDst, pTempSrc, width / 2);
		pTempDst += width;
		pTempSrc += width / 2;
	}

	cudaMemcpy(pDst, pTemp, width*height * 3, cudaMemcpyHostToDevice);

	delete[] pTemp;
}


int main()
{
	const char* file_yuv = "out240x128.yuv";
	int width = 240;
	int height = 128;

	size_t srcSize = width * height * 3 / 2;
	uint8_t* pInData = new uint8_t[srcSize];
	Npp8u* pYUV_dev; //uchar
	Npp8u* pRGB_dev;
	cudaMalloc((void**)&pYUV_dev, width * height * 3 * sizeof(Npp8u));
	cudaMalloc((void**)&pRGB_dev, width * height * 3 * sizeof(Npp8u));

	FILE* fp = fopen(file_yuv, "rb");
	if (!fp)
	{
		printf("open %s error!!\n", file_yuv);
		return 0;
	}

	int i = 0;
	while (fread(pInData,1,srcSize,fp) == srcSize) //YUV 422
	{
		uint8_t* pY = pInData;
		uint8_t* pU = pY + width * height;
		uint8_t* pV = pU + width * height / 4;

		copyYUVCPU2GPU(pYUV_dev, pY, pU, pV, width, height);
		NppiSize nppSize = { width,height };
		printf("[%s:%d],nppSize(%d,%d)\n", __FILE__, __LINE__, nppSize.width, nppSize.height);

		/*YUV格式有两大类：planar和packed。
		对于planar的YUV格式，先连续存储所有像素点的Y，紧接着存储所有像素点的U，随后是所有像素点的V。
		对于packed的YUV格式，每个像素点的Y, U, V是连续交*存储的。
		 YUV420P，Y，U，V三个分量都是平面格式，分为I420和YV12。I420格式和YV12格式的不同处在U平面和V平面的位置不同。
		 在I420格式中，U平面紧跟在Y平面之后，然后才是V平面（即：YUV）；但YV12则是相反（即：YVU）
		 YUV420SP, Y分量平面格式，UV打包格式, 即NV12。 NV12与NV21类似，U 和 V 交错排列,不同在于UV顺序。
		 */

		//auto ret = nppiYUVToRGB_8u_P3R((const Npp8u * const*)pYUV_dev, width * 3, &pRGB_dev,width*3, nppSize);
		auto ret = nppiYUVToRGB_8u_C3R(pYUV_dev, width * 3, pRGB_dev, width*3, nppSize);
		if (ret != 0)
		{
			printf("nppiYUVToRGB_8u_C3R error:%d\n", ret);
			return 0;
		}

		cv::Mat img(height, width, CV_8UC3);
		cudaMemcpy(img.data, pRGB_dev, width*height * 3, cudaMemcpyDeviceToHost);
		std::string str1 = std::to_string(i + 1) + ".jpg";
		cv::imwrite(str1.c_str(), img);
		//cv::waitKey(1);
		i++;
		if (i>5)
		{
			break;
		}
	}
	delete[] pInData;
	cudaFree(pYUV_dev);
	cudaFree(pRGB_dev);
	fclose(fp);

	return 0;
}

