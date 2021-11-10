#pragma once
#include <nppi.h>
#include <opencv2/opencv.hpp>

void YUV420pToRGB_NPP(const unsigned char *input, unsigned char *output, int width,
	int height, int device_id);

int cvtColor(unsigned char *d_req,
	unsigned char *d_res,
	int resolution,
	int height,
	int width,
	int linesize);
