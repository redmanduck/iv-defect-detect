#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace cv;
void showSideBySide(Mat &im1, Mat &im2);
void showSideBySide(String name, Mat &im1, Mat &im2);
int alignmentScore(Mat &ab);

//Mask treshold
const float MASK_THRES = 10.0f;
//Error elimination (eliminate small)
const float MIN_ERR_RADIUS = 0.0f;
//Alignment accuracy
const int number_of_iterations = 3555;
const double AAE_DIM_PERCENTILE = 0.05; //Alignment Artifact Elimination Dimension Percentile (how far from the corner do we start disallowing center of marker)
//Other control variables
const unsigned int CAM_NUMBER = 1;
const bool use_feed = true;


int main(int argc, char** argv)
{
	Mat im1, im2;
	std::cout << "Opening capture stream.." << std::endl;
	if(use_feed){
		VideoCapture cap;

		if (!cap.open(CAM_NUMBER)) {
			std::cout << "Unable to open cam stream" << std::endl;
			return 0;
		}
		int j = 0;
		for (;;)
		{
			Mat frame, shframe;
			cap >> frame;
			if (frame.empty()) break; // end of video stream
			
			if (j == 0) {
				shframe = frame;
				imshow("Streaming..", shframe);
			}
			else {
				absdiff(frame, im1, shframe);
				int score = alignmentScore(shframe);
				std::cout << score << " White Count (Lower is better, minimize this)" << std::endl;

				showSideBySide("Streaming..", shframe, frame);
			}

			

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
		 //im2 = imread("C:\\Users\\ecegr\\Desktop\\PGarten\\B.png");
	}
	
	if (im1.empty() || im2.empty()) {
		std::cout << "File not found" << std::endl;
		return 0;
	}

	Mat im1_gray, im2_gray;
	cvtColor(im1, im1_gray, CV_BGR2GRAY);
	cvtColor(im2, im2_gray, CV_BGR2GRAY);

	//std::cout << "Denoising.." << std::endl;
	//fastNlMeansDenoising(im1_gray, im1_gray, 3.0, 7, 21);
	//fastNlMeansDenoising(im2_gray, im2_gray, 3.0, 7, 21);

	// Alignment correction
	const int warp_mode = MOTION_AFFINE;

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
			
			if (pix.val[0] > MASK_THRES)
			{
				fmask.at<uchar>(j, i) = 255;
			}
			
		}

	//Highlighting Diff area
	

	//Mat fmask_canny;
	std::cout << "Morphing.." << std::endl;
	//Canny(fmask, fmask_canny, 20, 20 * 3, 3);
	erode(fmask, fmask, Mat(), Point(-1,1),1,1, 1);

	std::cout << "Finding Contours.." << std::endl;
	findContours(fmask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	imshow("FMASK", fmask);


	size_t count = contours.size();
	std::vector<Point2i> center;
	std::vector<int> radius;
	Scalar red(0, 0, 255);

	for (int i = 0; i<count; i++)
	{
		cv::Point2f c;
		float r = 0.00f;
		cv::minEnclosingCircle(contours[i], c, r);
		
		//Discard alignment artifacts, which is circles that are as large as the image and has radius in the corner
		bool IN_CORNER = (c.x < AAE_DIM_PERCENTILE*im1.size().width ||
			c.x > (1- AAE_DIM_PERCENTILE)*im1.size().width || c.y < im1.size().height*AAE_DIM_PERCENTILE || 
			c.y > im1.size().height*(1- AAE_DIM_PERCENTILE));
		bool ALM_ARTIFACT = (r >= im1.size().height / 2 || r >= im1.size().width / 2 || IN_CORNER );
		//Don't include useless markers (false alarm)
		if (r > MIN_ERR_RADIUS && !ALM_ARTIFACT) {
			center.push_back(c);
			radius.push_back(r);
		}
	}

	for (int i = 0; i < center.size(); i++)
	{
		cv::circle(im2_aligned, center[i], radius[i] + 1, red, 2);
	}

	showSideBySide(im2_aligned, im1);
	imshow("Defect Test A - Diff Before Thres", diff);

	waitKey(0);

	return 0;
}

void showSideBySide(Mat &im1, Mat &im2) {
	showSideBySide("Defect", im1, im2);
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

int alignmentScore(Mat &ab) {
	//cvtColor(ab.clone(), ab, CV_BGR2GRAY);
	int whcount = countNonZero(ab > 127);
	return whcount;
}