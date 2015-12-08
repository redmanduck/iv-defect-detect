#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <ctime>

using namespace cv;

const int CAM_NUMBER = 0;

int main(int argc, char** argv)
{
	VideoCapture cap;
	if (!cap.open(CAM_NUMBER)) {
		std::cout << "Unable to open cam stream" << std::endl;
		return 0;
	}
	

	std::clock_t start;
	double duration;
	
	start = std::clock();
	int g = 0;

	for (;;)
	{
		Mat frame;

		

		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		cap >> frame;

		if ((int)duration == 1) {
			std::cout << "Capturing.." << std::endl;
			imshow("Frame" + std::to_string(g++), frame);

			start = std::clock();
		}

		imshow("FrameX", frame);

		if (waitKey(1) == 27) {
			break;
		}
	}


}