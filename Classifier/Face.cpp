#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <ctime>

using namespace cv;

const int CAM_NUMBER = 0;
void detectAndDisplay(Mat frame);
CascadeClassifier cascade;


#ifndef ALPHA
int main(int argc, char** argv)
{
	String cascade_name = "C:\\opencv\\opencv\\sources\\samples\\winrt\\FaceDetection\\FaceDetection\\Assets\\haarcascade_frontalface_alt.xml";

	if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };


	VideoCapture cap;
	if (!cap.open(CAM_NUMBER)) {
		std::cout << "Unable to open cam stream" << std::endl;
		return 0;
	}

	for (;;)
	{
		Mat frame;
		cap >> frame;

		detectAndDisplay(frame);

		if (waitKey(1) == 27) {
			break;
		}
	}


}

void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	std::vector<Rect> eyes;

	cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{

		Mat faceROI = frame_gray(faces[i]);

		//-- In each face, detect eyes
		cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			rectangle(frame, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), Scalar(0, 255, 0));
		}
	}
	//-- Show what you got
	imshow("window_name", frame);
}
#endif