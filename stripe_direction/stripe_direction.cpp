#include  <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv){

    VideoCapture capture("../../video/1-1080p.mp4");
    Mat prevFrame, prevGray;

    if(!capture.read(prevFrame)){
        cout << "Input Error!\n"<< endl;
        return -1;
    }

    cvtColor(prevFrame, prevGray, COLOR_RGB2GRAY);

    while(true){
        Mat nextFrame, nextGray;

        if(!capture.read(nextFrame)){
            break;
        }
        imshow("VideoInput", nextFrame);

        cvtColor(nextFrame, nextGray, COLOR_BGR2GRAY);
        Mat_<Point2f>flow;
        calcOpticalFlowFarneback(prevFrame, nextFrame, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        Mat xV = Mat::zeros(prevFrame.size(), CV_32FC1);
        Mat yV = Mat::zeros(prevFrame.size(), CV_32FC1);

        for(int row = 0; row < flow.rows; row++){
            for(int col = 0; col < flow.cols; col++){
                 const Point2f& flow_xy = flow.at<Point2f>(row, col);
                 xV.at<float>(row, col) = flow_xy.x;
                 yV.at<float>(row, col) = flow_xy.y;
            }
        }

        Mat magnitude, angle;
        cartToPolar(xV, yV, magnitude, angle);
        angle = angle * 180.0 / CV_PI / 2.0;
        normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);
        convertScaleAbs(magnitude, magnitude);
        convertScaleAbs(angle, angle);

        Mat HSV = Mat::zeros(prevFrame.size(), prevFrame.type());
        vector<Mat> result;
        split(HSV, result);
        result[0] = angle;
        result[1] = Scalar(255);
        result[2] = magnitude;

        merge(result, HSV);

        Mat rgbImg;
        cvtColor(HSV, rgbImg, COLOR_HSV2BGR);

        imshow("VideoOutput", rgbImg);
        int ch = waitKey(5);
        if(ch == 27){
            break;
        }
    }
    waitKey(0);
    return 0;
}