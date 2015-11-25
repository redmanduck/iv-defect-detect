#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;

/*
	Bug: Algorithm assumes that
	im2 has more content than im1,
	so when thresholded the subtraction of the opposite
	it will not pick up the threshold

	Bug2: Thresholding too strong! it destroy some diff
*/
int main(int argc, char** argv)
{
	float thres = 30;
	VideoCapture cap;

	if (!cap.open(0)) {
		cout << "Unable to open cam stream" << endl;
		return 0;
	}
	Mat im1, im2;
	int j = 0;
	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()) break; // end of video stream
		imshow("Streaming..", frame);
		if (waitKey(1) == 27) {
			if (j == 0) {
				cout << "IM1 Cap" << endl;
				frame.copyTo(im1);
			}
			else {
				cout << "IM2 Cap" << endl;
				frame.copyTo(im2);
				break;
			}
			j++;
		}
	}

	destroyAllWindows();
	/*
	Mat im1 = imread("C:\\Users\\ecegr\\Dropbox\\CV_Exp\\images\\starbucks_small.jpg");
	Mat im2 = imread("C:\\Users\\ecegr\\Dropbox\\CV_Exp\\images\\starbucks_small_defect2.jpg");
	*/
	Mat im1_gray, im2_gray;	

	cvtColor(im1, im1_gray, CV_BGR2GRAY);
	cvtColor(im2, im2_gray, CV_BGR2GRAY);

	// Alignment correction

	const int warp_mode = MOTION_EUCLIDEAN;

	// Set a 2x3 or 3x3 warp matrix depending on the motion model.
	Mat warp_matrix;

	warp_matrix = Mat::eye(2, 3, CV_32F);

	int number_of_iterations = 250;

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

	cout << "Warping.." << endl;
	warpAffine(im2, im2_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);

	cvtColor(im2_aligned, im2_aligned_gray, CV_BGR2GRAY);

	imshow("Defect Test A - Im2 Warped Grey", im2_aligned_gray);


	// Find difference in probe image with sample

	Mat diff;
	absdiff(im1_gray, im2_aligned_gray, diff);

	//Calculate thresholded contour for the differences
	//Mat diff_thrs;
	vector<vector<Point>> contours;
	
	//threshold(diff, diff_thrs, 100, 255, 0);
	Mat fmask = Mat::zeros(diff.rows, diff.cols, CV_8UC1);
	for (int j = 0; j<diff.rows; ++j)
		for (int i = 0; i<diff.cols; ++i)
		{
			Scalar pix = diff.at<uchar>(j, i);
			
			if (pix.val[0] > thres)
			{
				fmask.at<uchar>(j, i) = 255;
			}
			
		}

	imshow("FMASK", fmask);
	//Highlighting Diff area
	cout << "Marking.." << endl;
	findContours(fmask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	size_t count = contours.size();
	vector<Point2i> center;
	vector<int> radius;
	Scalar red(0, 0, 255);

	for (int i = 0; i<count; i++)
	{
		cv::Point2f c;
		float r = 0.00f;
		cv::minEnclosingCircle(contours[i], c, r);

		cout << r << endl;

		if (r > 5.0f) {
			center.push_back(c);
			radius.push_back(r);
		}
	}

	for (int i = 0; i < center.size(); i++)
	{
		cv::circle(im2_aligned, center[i], radius[i] + 1, red, 2);
	}

	imshow("Defect Test A - Original", im1);
	imshow("Defect Test A - Im B with marking", im2_aligned);
	imshow("Defect Test A - Diff Before Thres", diff);
	

	waitKey(0);

	return 0;
}