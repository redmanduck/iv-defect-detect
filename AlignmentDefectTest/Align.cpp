#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <ctime>

using namespace cv;

const int CAM_NUMBER = 1;

int main(int argc, char** argv)
{
	Mat im1_gray, im2_gray;
	Mat im1_kp, im2_kp;
	std::vector<KeyPoint> SIFT_keypoints;
	VideoCapture cap;
	if (!cap.open(CAM_NUMBER)) {
		std::cout << "Unable to open cam stream" << std::endl;
		return 0;
	}


	Mat im1 = imread("C:\\Users\\ecegr\\Dropbox\\CV_Exp\\images\\BAR1.jpg");
	Mat im2 = imread("C:\\Users\\ecegr\\Dropbox\\CV_Exp\\images\\BAR2.jpg");

	std::clock_t start;
	double duration;
	start = std::clock();

	int j = 0, q = 0;
	for (;;)
	{
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

		std::cout << duration << " elapsed" << std::endl;
		Mat frame;
		cap >> frame;
		if (frame.empty()) break;

		imshow("frame", frame);

		if (waitKey(1) == 27) {
			if (j == 0) {
				std::cout << "IM1 Cap" << std::endl;
				frame.copyTo(im1);
				start = std::clock();

			}
			else {
				break;
			}
			j++;
		}

		if (j > 0) {

			frame.copyTo(im2);

			Mat blended;
			double alpha = 0.5; double beta;
			beta = (1.0 - alpha);
			addWeighted(im1, alpha, im2, beta, 0.0, blended);


			if ((int)duration == 2) {
				destroyWindow("calculation");

				cvtColor(im1, im1_gray, CV_BGR2GRAY);
				cvtColor(im2, im2_gray, CV_BGR2GRAY);
				double termination_eps = 1e-4;
				const int warp_mode = MOTION_TRANSLATION;
				int number_of_iterations = 6000;
				TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations, termination_eps);
				Mat warp_matrix = Mat::eye(2, 3, CV_32F);
				findTransformECC(
					im1_gray,
					im2_gray,
					warp_matrix,
					warp_mode,
					criteria
					);

				float dst = warp_matrix.at<float>(0, 2);
				float dst2 = warp_matrix.at<float>(1, 2);
				
				float displacement = sqrt(pow(dst, 2) + pow(dst2, 2));

				putText(blended, "X: " + std::to_string(dst) + ", Y: " + std::to_string(dst2), Point(10, 50), FONT_HERSHEY_PLAIN, 1.8, Scalar(0, 0, 255), 3, 8);
				putText(blended, "|R| = " + std::to_string(displacement) , Point(10, 100), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 0), 2, 8);

				//putText(blended, std::to_string(duration) + " SEC", Point(10, 100), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 0), 2, 8);
				start = std::clock();

				imshow("calculation", blended);
				
			}
			
			imshow("blended", blended);

			
		}

	}

	destroyAllWindows();
	
	


	waitKey(0);
}
