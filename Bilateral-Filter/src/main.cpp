/*
The MIT License (MIT)

Copyright (c) 2015 Nrupatunga

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>

#pragma comment(lib, "opencv_core300d.lib") // core functionalities
#pragma comment(lib, "opencv_highgui300d.lib") //GUI
#pragma comment(lib, "opencv_imgcodecs300d.lib")
#pragma comment(lib, "opencv_imgproc300d.lib") // Histograms, Edge detection

using namespace cv;
using namespace std;

#define PI (3.1412)

//Generate 2D Gaussian kernel of defined size
void GenerateGauss2D(Mat &sMatGaussGray, int s32Size, float fSigma)
{
	double dVal, dKerVal;
	int s32Shift = s32Size / 2;
	Mat sMatGauss;

	if (!(s32Size && 0x01))
		s32Size = s32Size + 1;
	sMatGauss     = Mat(Size(s32Size, s32Size), CV_32FC1);
	sMatGaussGray = Mat(Size(s32Size, s32Size), CV_32FC1);

	const float kfScale = 1 / (sqrt(2 * PI) * fSigma );
	for (int i = -s32Shift; i < s32Size - s32Shift; i++){
		float *ptr = sMatGauss.ptr<float>(i+s32Shift);
		for (int j = -s32Shift; j < s32Size - s32Shift; j++){
			if ((i*i + j*j) > (s32Size*s32Size))
				continue;
			dVal = ((i * i ) + (j * j ))/ (2 * fSigma * fSigma);
			dKerVal = exp(-dVal);
			ptr[j+s32Shift] = dKerVal*kfScale;
		}
	}
	normalize(sMatGauss, sMatGaussGray, 0, 1, NORM_MINMAX);
}

//Bilateral filter
void bilateral2D(Mat &sMatInput, Mat &sMatOutput, int s32KernelSize, float fSigmaD, float fSigmaR)
{
	Mat sMatInputPad;
	int s32NumPad = s32KernelSize >> 1;
	if ( CV_8UC1 != sMatInput.depth() ) {
		cout << "Input should be gray image" << endl;
		return;
	}

	int s32Width  = sMatInput.cols;
	int s32Height = sMatInput.rows;
	sMatOutput = Mat(Size(s32Width, s32Height), CV_32FC1);
	//Space Kernel
	Mat sMatSpaceKernel =  Mat(Size(s32KernelSize, s32KernelSize), CV_32FC1);
	GenerateGauss2D(sMatSpaceKernel, s32KernelSize, fSigmaD);

	//Range Kernel
	Mat sMatRangeKernel = Mat(Size(256, 1), CV_32FC1);
	double dScale = 1 / (sqrt(2 * PI) * fSigmaR);
	for (int i = 0; i < sMatRangeKernel.cols; i++) {
		double dVal = ((i * i)) / (2 * fSigmaR * fSigmaR);
		double dKernelVal = exp(-dVal);
		sMatRangeKernel.at<float>(i) = (float)dKernelVal;
	}
	normalize(sMatRangeKernel, sMatRangeKernel, 0, 1, NORM_MINMAX);
	copyMakeBorder(sMatInput, sMatInputPad, s32NumPad, s32NumPad, s32NumPad, s32NumPad, BORDER_REPLICATE);

	s32Width  = sMatInputPad.cols;
	s32Height = sMatInputPad.rows;
	for (int i = s32NumPad; i < s32Height - s32NumPad; i++){
		for (int j = s32NumPad; j < s32Width - s32NumPad; j++){
			int s32TargetPixel = sMatInputPad.at<unsigned char>(i, j);
			float fResult = 0; float fSumCoeff = 0;
			for (int m = -s32NumPad; m < s32NumPad; m++){
				for (int n = -s32NumPad; n < s32NumPad; n++){
					int s32CurrentPixel = sMatInputPad.at<unsigned char>(i + m, j + n);
					float fCoeff = sMatSpaceKernel.at<float>(m + s32NumPad, n + s32NumPad) * sMatRangeKernel.at<float>(abs(s32CurrentPixel - s32TargetPixel));
					fResult += s32CurrentPixel * fCoeff;
					fSumCoeff += fCoeff;
				}
			}
			sMatOutput.at<float>(i - s32NumPad, j - s32NumPad) = fResult/fSumCoeff;
		}
	}
	sMatOutput.convertTo(sMatOutput, CV_8UC1);
}

int main(int argc, char *argv[])
{
	 Mat sMatInput = imread("..\\statue.jpg", IMREAD_GRAYSCALE);
	 Mat sMatOutputBF, sMatOutputGauss;
	 float fSigmaD, fSigmaR;
	 int s32FilterSize = 7;
	 fSigmaD = fSigmaR = 50;
	 bilateral2D(sMatInput, sMatOutputBF, s32FilterSize, fSigmaD, fSigmaR);
	 GaussianBlur(sMatInput, sMatOutputGauss, Size(s32FilterSize, s32FilterSize), fSigmaD, fSigmaR);

	 String strInput = "Input"; String strOutputBF = "OutputBF"; String strOutputGauss = "OutputGauss";
	 namedWindow(strInput, WINDOW_KEEPRATIO);
	 namedWindow(strOutputBF, WINDOW_KEEPRATIO);
	 namedWindow(strOutputGauss, WINDOW_KEEPRATIO);

	 imshow(strInput, sMatInput);
	 imshow(strOutputBF, sMatOutputBF);
	 imshow(strOutputGauss, sMatOutputGauss);

#if 1
	 imwrite("Input.jpg", sMatInput);
	 imwrite("BilateralOutput.jpg", sMatOutputBF);
	 imwrite("GaussOutput.jpg", sMatOutputGauss);
#endif
	 waitKey(0);


}
