#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;

const int CAM_NUMBER = 1;
void showSideBySide(String name, Mat &im1, Mat &im2);

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

	int j = 0;
	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()) break;

		imshow("frame", frame);

		if (waitKey(1) == 27) {
			if (j == 0) {
				std::cout << "IM1 Cap" << std::endl;
				frame.copyTo(im1);
			}
			else {
				std::cout << "IM2 Cap" << std::endl;
				frame.copyTo(im2);
				break;
			}
			j++;
		}
	}

	destroyAllWindows();
	
	cvtColor(im1, im1_gray, CV_BGR2GRAY);
	cvtColor(im2, im2_gray, CV_BGR2GRAY);

	// Specify the threshold of the increment
	// in the correlation coefficient between two iterations
	double termination_eps = 1e-6;
	const int warp_mode = MOTION_TRANSLATION;
	int number_of_iterations = 15000;

	// Define termination criteria
	TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations, termination_eps);
	Mat warp_matrix = Mat::eye(2, 3, CV_32F);

	std::cout << "Finding ECC" << std::endl;
	// Run the ECC algorithm. The results are stored in warp_matrix.
	findTransformECC(
		im1_gray,
		im2_gray,
		warp_matrix,
		warp_mode,
		criteria
		);

	std::cout << warp_matrix << std::endl;

	Mat blended;
	double alpha = 0.5; double beta;
	beta = (1.0 - alpha);
	addWeighted(im1, alpha, im2, beta, 0.0, blended);
	float dst = warp_matrix.at<float>(0,2);
	float dst2 = warp_matrix.at<float>(1, 2);

	putText(blended, "X: " + std::to_string(dst) + ", Y: " + std::to_string(dst2), Point(10, 50), FONT_HERSHEY_PLAIN, 1.8, Scalar(0,0, 255), 3, 8);
	imshow("blended", blended);

	waitKey(0);
}

void showSideBySide(String name, Mat &im1, Mat &im2) {
	Size sz1 = im1.size();
	Size sz2 = im2.size();
	Mat im3(sz1.height, sz1.width + sz2.width, CV_8UC3);
	Mat left(im3, Rect(0, 0, sz1.width, sz1.height));
	im1.copyTo(left);
	Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
	im2.copyTo(right);
	imshow(name, im3);
}