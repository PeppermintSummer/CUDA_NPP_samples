#pragma once
#include <nppi.h>
#include <opencv2/opencv.hpp>

int cvtColor(unsigned char *d_req,
	unsigned char *d_res,
	int resolution,
	int height,
	int width,
	int linesize);

int cvtcolor_by_npp(Npp8u** pYUV_dev,Npp8u* pRGB_dev,int linesize[], void** data,int width,int height)
{
	//cudaMalloc

	//memcpy
	cudaMemcpy(pYUV_dev[0], data[0], width*height * sizeof(Npp8u), cudaMemcpyHostToDevice);
	cudaMemcpy(pYUV_dev[1], data[1], width*height * sizeof(Npp8u) / 4, cudaMemcpyHostToDevice);
	cudaMemcpy(pYUV_dev[2], data[2], width*height * sizeof(Npp8u) / 4, cudaMemcpyHostToDevice);
	NppiSize nppSize = { width,height };
	auto ret = nppiYUV420ToRGB_8u_P3C3R(pYUV_dev, linesize, pRGB_dev, linesize[0] * 3, nppSize); //channel=3
	if (ret != 0)
	{
		printf("nppiYUVToRGB_8u_C3R error:%d\n", ret);
		//return;
	}
	cv::Mat tmpMat(height, width, CV_8UC3);
	cudaMemcpy(tmpMat.data, pRGB_dev, width*height * 3, cudaMemcpyDeviceToHost);
	cv::imshow("img", tmpMat);

	//cudafree
}