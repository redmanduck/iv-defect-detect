#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <ctime>


const int CAM_NUMBER = 0;

using namespace cv;

void diag(String c) {
	std::cout << c << std::endl;
}

int main(int argc, char** argv)
{

	std::clock_t start;
	double duration;
	start = std::clock();
	float fps = 0.00;
	int frames = 0;

	VideoCapture cap;
	if (!cap.open(CAM_NUMBER)) {
		std::cout << "Unable to open cam stream" << std::endl;
		return 0;
	}

	Point roiLoc(0,0);
	int rect_incr = 10;
	Rect rect_pt(100, 100, 100, 100);

	bool DRAW_MODE = false;
	bool ROI_PROC = false;
	bool TEMPL_PROC = false;

	int mode_scale_idx = 0;

	Mat templ;

	for (;;)
	{
		
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	
		if (duration >= 1) {
			fps = frames / 1.0;
			start = std::clock();
			frames = 0;
		}

		Mat frame;
		cap >> frame;

		frames++;

		if (frame.empty()) break;

		//Flag triggering
		int b = waitKey(1);
		if (b == 27) {
			if (!TEMPL_PROC) {
				break;
			}
			TEMPL_PROC = false;
		}
		else if (b == 13) {
			if (DRAW_MODE) {
				ROI_PROC = true;
				DRAW_MODE = false;
			}
			else {
				DRAW_MODE = true;
			}
		}
		else if (b == 2621440) {
			//DOWN
			rect_pt.y = rect_pt.y + rect_incr;
		}
		else if (b == 2424832) {
			//LEFT
			rect_pt.x = rect_pt.x - rect_incr;
		}
		else if (b == 2555904) {
			//RIGHT
			rect_pt.x = rect_pt.x + rect_incr;
		}
		else if (b == 2490368) {
			//UP
			rect_pt.y = rect_pt.y - rect_incr;
		}
		else if (b == 105) {
			//i
			if (mode_scale_idx == 0 || mode_scale_idx  == 1) rect_pt.width -= rect_incr;
			if (mode_scale_idx == 0 || mode_scale_idx == 2)  rect_pt.height -= rect_incr;
		}
		else if (b == 111) {
			//o
			if (mode_scale_idx == 0 || mode_scale_idx == 1)rect_pt.width += rect_incr;
			if (mode_scale_idx == 0 || mode_scale_idx == 2) rect_pt.height += rect_incr;
		}
		else if (b == 45) {
			//minus
			rect_incr -= 1;
		}
		else if (b == 61) {
			//plus
			rect_incr += 1;
		}
		else if (b == 109) {
			//m
			if (mode_scale_idx == 2) {
				mode_scale_idx = 0;
			}
			else {
				mode_scale_idx++;
			}
		}

		//Flag Processing

		if (ROI_PROC) {
			//capture area inside rect
			Mat cropped = frame(rect_pt);
			//cvtColor(cropped, cropped, CV_BGR2GRAY);
			roiLoc = Point(rect_pt.x, rect_pt.y);

			imshow("ROI", cropped);

			cropped.copyTo(templ);

			TEMPL_PROC = true;
			ROI_PROC = false;
			cv::destroyAllWindows();
		}

		Point minLoc; Point maxLoc; double maxVal; double minVal;
		Scalar green(0, 255, 0);
		Scalar orange(30, 255, 255);

		if (TEMPL_PROC) {
			imshow("roi", templ);
			Mat tmp, frame_grey;
			//cvtColor(frame, frame_grey, CV_BGR2GRAY);

			matchTemplate(frame, templ, tmp, CV_TM_CCOEFF_NORMED);
			
			threshold(tmp, tmp, 0.7, 1., CV_THRESH_TOZERO);
			

			minMaxLoc(tmp, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

			if (maxLoc.x != 0 && maxLoc.y != 0) {
				Scalar match_color = green;

				if (maxVal < 0.9) {
					match_color = orange;
				}

				rectangle(frame, maxLoc, Point(maxLoc.x + templ.cols, maxLoc.y + templ.rows), match_color, 1, 8, 0);
				putText(frame, "OK", Point(10, frame.size().height - 50), FONT_HERSHEY_PLAIN, 3, green, 2, 8);
				
				putText(frame, std::to_string(maxVal),
					Point(maxLoc.x, maxLoc.y + templ.rows + 15), FONT_HERSHEY_PLAIN, 0.75, match_color, 1, 8);

			}
			else {
				putText(frame, "NG", Point(10, frame.size().height - 50), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 2 , 8);
			}
			
			
		}


		if (DRAW_MODE) {
			rectangle(frame, rect_pt, Scalar(0, 0, 255), 1);
			putText(frame, mode_scale_idx == 0 ? "UNIFORM" : (mode_scale_idx == 1 ? "WIDTH ADJUST" : "HEIGHT ADJUST"),
				Point(rect_pt.x, rect_pt.y + rect_pt.height + 15), FONT_HERSHEY_PLAIN, 0.7, Scalar(0, 0, 255), 1, 8);
		}

		putText(frame, "ADJ " + std::to_string(rect_incr),
			Point(10, 50), FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0), 1, 8);
		

		putText(frame, "fps " + std::to_string((int)fps), Point(10, frame.size().height - 10), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 1, 8);

		putText(frame, "Match " + std::to_string(maxLoc.x) + "," + std::to_string(maxLoc.y), Point(10, 100), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 1, 8);
		putText(frame, "ROI " + std::to_string(roiLoc.x) + "," + std::to_string(roiLoc.y), Point(10, 120), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0), 1, 8);
		double roiDiff = sqrt(pow(roiLoc.x - maxLoc.x, 2) + pow(roiLoc.y - maxLoc.y, 2));
		Scalar color(0, 0, 255);

		if (roiDiff < 100) {
			color = Scalar(0, 255, 0);
		}

		putText(frame, "Diff " + std::to_string(roiDiff), Point(10, 140), FONT_HERSHEY_PLAIN, 1, color, 1, 8);

		imshow("frame", frame);

	}

	destroyAllWindows();




	waitKey(0);
}
