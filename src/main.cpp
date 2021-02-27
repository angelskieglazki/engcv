#include <opencv2/core/mat.hpp>
#include <string>
#include <iostream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

std::vector<cv::Rect> detectLetters(cv::Mat img)
{
    std::vector<cv::Rect> boundRect;
    cv::Mat img_gray, img_sobel, img_threshold, element;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    cv::Sobel(img_gray, img_sobel, CV_8U, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT);
    cv::threshold(img_sobel, img_threshold, 0, 255, THRESH_OTSU+THRESH_BINARY);
    element = getStructuringElement(cv::MORPH_RECT, cv::Size(15, 3) );
    cv::morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element); //Does the trick
    std::vector< std::vector< cv::Point> > contours;
    cv::findContours(img_threshold, contours, 0, 1); 
    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
        if (contours[i].size()>100)
        { 
            cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
            cv::Rect appRect( boundingRect( cv::Mat(contours_poly[i]) ));
            if (appRect.width>appRect.height) 
                boundRect.push_back(appRect);
        }
    return boundRect;
}

int main(int argc, char** argv){
  std::cout<<"Hello cv" <<"\n";
  auto image = cv::imread("test.png");
  tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
  ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
  ocr->SetPageSegMode(tesseract::PSM_AUTO);
  ocr->SetImage(image.data, image.cols, image.rows, 3, image.step);
  auto outText = string(ocr->GetUTF8Text());
  std::cout<<outText<<"\n";
  ocr->End();
  return 0;
  std::vector<cv::Rect> letterBBoxes=detectLetters(image);
//Display
  for(int i=0; i< letterBBoxes.size(); i++)
    cv::rectangle(image,letterBBoxes[i],cv::Scalar(0,255,0),1,8,0);
    
  cv::imwrite( "test-out.jpg", image);

  return 0;
}
