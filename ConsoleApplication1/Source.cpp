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
void showSideBySide(Mat &im1, Mat &im2);

/*
	Bug: Algorithm assumes that
	im2 has more content than im1,
	so when thresholded the subtraction of the opposite
	it will not pick up the threshold

	Bug2: Thresholding too strong! it destroy some diff
*/
int main(int argc, char** argv)
{

	const float thres = 33.0f;
	const float sensitivity = 4.0f;
	const int number_of_iterations = 255;

	const bool use_feed = false;

	Mat im1, im2;
	std::cout << "Opening capture stream.." << std::endl;
	if(use_feed){
		VideoCapture cap;

		if (!cap.open(0)) {
			std::cout << "Unable to open cam stream" << std::endl;
			return 0;
		}
		int j = 0;
		for (;;)
		{
			Mat frame;
			cap >> frame;
			if (frame.empty()) break; // end of video stream
			imshow("Streaming..", frame);
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
	}
	else {
		 im1 = imread("C:\\Users\\ecegr\\Dropbox\\CV_Exp\\images\\hello.jpg");
		 im2 = imread("C:\\Users\\ecegr\\Dropbox\\CV_Exp\\images\\hello_defect.jpg");
		 //im1 = imread("C:\\Users\\ecegr\\Desktop\\PGarten\\A.png");
	//	 im2 = imread("C:\\Users\\ecegr\\Desktop\\PGarten\\B.png");
	}
	
	if (im1.empty()) {
		std::cout << "File not found" << std::endl;
		return 0;
	}
	
	Mat im1_gray, im2_gray;	

	cvtColor(im1, im1_gray, CV_BGR2GRAY);
	cvtColor(im2, im2_gray, CV_BGR2GRAY);

	// Alignment correction

	const int warp_mode = MOTION_EUCLIDEAN;

	// Set a 2x3 or 3x3 warp matrix depending on the motion model.
	Mat warp_matrix;

	warp_matrix = Mat::eye(2, 3, CV_32F);

	// Specify the threshold of the increment
	// in the correlation coefficient between two iterations
	double termination_eps = 1e-5;

	// Define termination criteria
	TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, number_of_iterations, termination_eps);

	std::cout << "Finding ECC" << std::endl;
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

	std::cout << "Warping.." << std::endl;
	warpAffine(im2, im2_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);

	cvtColor(im2_aligned, im2_aligned_gray, CV_BGR2GRAY);


	// Find difference in probe image with sample

	Mat diff;
	absdiff(im1_gray, im2_aligned_gray, diff);

	//Calculate thresholded contour for the differences
	//Mat diff_thrs;
	std::vector<std::vector<Point>> contours;
	
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
	std::cout << "Marking.." << std::endl;
	findContours(fmask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	size_t count = contours.size();
	std::vector<Point2i> center;
	std::vector<int> radius;
	Scalar red(0, 0, 255);

	for (int i = 0; i<count; i++)
	{
		cv::Point2f c;
		float r = 0.00f;
		cv::minEnclosingCircle(contours[i], c, r);

		if (r > sensitivity) {
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
	//showSideBySide(im1, im2_aligned);
	imshow("Defect Test A - Diff Before Thres", diff);
	

	waitKey(0);

	return 0;
}

void showSideBySide(Mat &im1, Mat &im2) {
	Size sz1 = im1.size();
	Size sz2 = im2.size();
	Mat im3(sz1.height, sz1.width + sz2.width, CV_8UC3);
	Mat left(im3, Rect(0, 0, sz1.width, sz1.height));
	im1.copyTo(left);
	Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
	im2.copyTo(right);
	imshow("Defect", im3);
}