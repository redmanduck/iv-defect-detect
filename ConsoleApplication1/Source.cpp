#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	Mat im1 = imread("C:\\Users\\ecegr\\Dropbox\\CV_Exp\\images\\hello.jpg");
	Mat im2 = imread("C:\\Users\\ecegr\\Dropbox\\CV_Exp\\images\\hello_defect.jpg");
	Mat im1_gray, im2_gray;

	cvtColor(im1, im1_gray, CV_BGR2GRAY);
	cvtColor(im2, im2_gray, CV_BGR2GRAY);

	const int warp_mode = MOTION_EUCLIDEAN;

	// Set a 2x3 or 3x3 warp matrix depending on the motion model.
	Mat warp_matrix;

	warp_matrix = Mat::eye(2, 3, CV_32F);

	int number_of_iterations = 10;

	// Specify the threshold of the increment
	// in the correlation coefficient between two iterations
	double termination_eps = 1e-5;


	// Define termination criteria
	TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations, termination_eps);
	cout << "Finding ECC" << endl;
	// Run the ECC algorithm. The results are stored in warp_matrix.
	findTransformECC(
		im1_gray,
		im2_gray,
		warp_matrix,
		warp_mode,
		criteria
	);

	// Storage for warped image.
	Mat im2_aligned;
	Mat im2_aligned_gray;

	cout << "Warping" << endl;
	warpAffine(im2, im2_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);

	cvtColor(im2_aligned, im2_aligned_gray, CV_BGR2GRAY);

	cout << im2_aligned.size().width  << "x" << im2_aligned.size().height << endl;

	// Find difference in probe image with sample
	Mat diff = im1_gray - im2_aligned_gray;

	//Calculate thresholded contour for the differences

	Mat threshold_frame;
	vector<vector<Point>> contours;

	threshold(diff.clone() , threshold_frame, 100, 255, 3);

	findContours(threshold_frame, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	size_t count = contours.size();
	vector<Point2i> center;
	vector<int> radius;
	Scalar red(0, 0, 255);

	for (int i = 0; i<count; i++)
	{
		cv::Point2f c;
		float r;
		cv::minEnclosingCircle(contours[i], c, r);

		center.push_back(c);
		radius.push_back(r);
	}

	for (int i = 0; i < count; i++)
	{
		cv::circle(im1, center[i], radius[i], red, 3);
	}


	imshow("Defect", im1);
	imshow("Defect GS", diff);

	waitKey(0);

	return 0;
}