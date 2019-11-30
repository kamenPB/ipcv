// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <cmath>
#include<bits/stdc++.h>
#include <vector>
using namespace cv;
using namespace std;

#define pi 3.14159265

//void normolization(Mat x,Mat y,double min_x,double max_x);
//void getMag(Mat &Gx,Mat &Gy,Mat &dst,Mat &img);
//void getDirection(Mat &Gx,Mat &Gy,Mat &dst);
void getThresholdedMag(Mat &input, Mat &output,int magThrFun);

int main( int argc, char** argv )
{

 // LOADING THE IMAGE
 Mat image;
 image = imread( "originals/dart8.jpg", 1 );
 image.convertTo(image, CV_32F);
 Mat image2 = image.clone();
 if( argc != 1 || !image.data)
 {
   printf( " No image data \n " );
   return -1;
 }


 //GaussianBlur( image, image, Size(3,3), 0, 0, BORDER_DEFAULT );


 // CONVERT COLOUR, BLUR AND SAVE
 Mat gray_image;
 gray_image.create(image.size(), CV_32F);
 cvtColor( image, gray_image, CV_BGR2GRAY );

 GaussianBlur( gray_image, gray_image, Size(9, 9), 2, 2 );


 // instilize two filtires dx and dy
 Mat dx,dy,dxKernel,dyKernel;
 dx.create(image.size(), CV_32F);
 dy.create(image.size(), CV_32F);

 dxKernel = (Mat_<float>(3,3) << -1, 0, 1,
                                     -2, 0, 2,
                                     -1, 0, 1);

 dyKernel = (Mat_<float>(3,3) << -1,-2,-1,
                                      0, 0, 0,
                                      1, 2, 1);

 Mat Gx,Gy;
 Gx.create(image.size(), CV_32F);
 Gy.create(image.size(), CV_32F);
 Point anchor = Point( -1, -1 );
 double delta = 0;
 filter2D(gray_image, Gx, 5 , dxKernel, anchor, delta, BORDER_DEFAULT );
 filter2D(gray_image, Gy, 5 , dyKernel, anchor, delta, BORDER_DEFAULT );

 Mat mag,theta,houghspaces;

 int rho_width = gray_image.rows;
 int rho_height = gray_image.cols;
 houghspaces.create(2*(rho_width+rho_height),360,CV_32F);
 houghspaces = Scalar(0,0,0);


Mat Mag, Angle;
//Mag.create(gray_image.size(),CV_64F);
Angle.create(gray_image.size(),CV_32F);
cartToPolar(Gx, Gy, Mag, Angle, 1);
//imwrite( "Angle.jpg", Angle );
// display images

float angle = 0.0;
float radian = 0.0;
float radians = 0.0;
float rho_value = 0.0;
float theta_value= 0.0;
Mat MagThresh;
MagThresh.create(Mag.size(),CV_32F);


// threshholds
int magThr = 120;
int houghSpaceThr = 120;

getThresholdedMag(Mag, MagThresh, magThr);


int rMin = 150;
int rMax = 200;

int sizes[] = {2*(rho_width+rho_height), 360, rMax}; // rho, theta, radius
Mat houghspaceCircle(3, sizes, CV_32F, Scalar(0));


// create houghspace for lines from threshholded magnitude image
 for ( int i = 0; i < MagThresh.rows; i++ ){
		for( int j = 0; j < MagThresh.cols; j++ ){
        if (MagThresh.at<float>(i, j) > 0) { // value is either 0 or 255
          for(int theta = 0; theta < 360; theta++){
            for(int r = rMin; r < rMax; r++){
              radians = theta * (pi / 180);
              int x0 = round(i - r * cos(radians));
              int y0 = round(j - r * sin(radians));
              //cout<<"x0: "<<x0<<" y0: "<<y0<<endl;
              if(x0 > 0 && x0 < MagThresh.cols && y0 > 0 && y0 << MagThresh.cols){
                houghspaceCircle.at<float>(x0, y0, r) += 1;
                //cout<<"i: "<<i<<"j: "<<j<<"id: "<<id<<endl;
              }
            }
          }

		 }
	}
}

normalize(houghspaces, houghspaces, 0, 255, NORM_MINMAX);
imwrite( "5houghspaceLines.jpg", houghspaces );

int th = 300;

int count=0;
for (int i = 0; i < houghspaceCircle.rows; i++) {
  for (int j = 0; j < houghspaceCircle.cols; j++) {
    for(int r = rMin; r < rMax; r++){
      float val = 0.0;
      val = houghspaces.at<float>(i, j, r);
      if (val > th){
        houghspaces.at<float>(i, j, r) = 255;
        count++;
      }

      else houghspaces.at<float>(i, j, r) = 0;
    }
  }
 }

 vector<float>  rhoValues;
 vector<float>  thetaValues;
 vector<float>  radiusValues;
 rhoValues.resize(count);
 thetaValues.resize(count);
 radiusValues.resize(count);

 int index=0;
 for (int i = 0; i < houghspaceCircle.rows; i++) {
   for (int j = 0; j < houghspaceCircle.cols; j++) {
     for(int r = rMin; r < rMax; r++){
       float val = 0.0;
       val = houghspaces.at<float>(i, j, r);
       if (val == 255){
         rhoValues[index]=i;
         thetaValues[index]=j;
         radiusValues[index]=r;
         index++;
       }
     }
   }
 }

// circle
// for( int i = 0; i < count; i++ )
// {
//     float rhos = rhoValues[i]-rho_width - rho_height, thetas = thetaValues[i];
//     float radians = thetas *pi/ 180;
//     //cout<<"rhos: "<<rhos<<", thetas: "<<radians<<endl;
//     Point pt1;
//     double a = cos(radians), b = sin(radians);
//     double x0 = a*rhos, y0 = b*rhos;
//     pt1.x = cvRound(x0 + 1000*(-b));
//     pt1.y = cvRound(y0 + 1000*(a));
//     pt2.x = cvRound(x0 - 1000*(-b));
//     pt2.y = cvRound(y0 - 1000*(a));
//     //cout<<"pt1.x: "<<pt1.x<<", pt1.y: "<<pt1.y<<", pt2.x: "<<pt2.x<<", pt2.y: "<<pt2.y<<endl;
//     line( image2, pt1, pt2, Scalar(0,0,255), 2, CV_AA);
// }



// storing images
 imwrite( "1edgeGx.jpg", Gx );
 imwrite( "2edgeGY.jpg", Gy );
 imwrite( "3lines.jpg", image2 );
 imwrite( "4image.jpg", image);
 imwrite( "6Mag.jpg", Mag);
 imwrite( "7MagThresh.jpg", MagThresh);
 imwrite( "8Angle.jpg", Angle);
 return 0;
}


// apply threshhold to magnitude
void getThresholdedMag(Mat &input, Mat &output, int magThr) {
	Mat img;
	img.create(input.size(), CV_32F);

	normalize(input, img, 0, 255, NORM_MINMAX);

	for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {

      double val = 0;
      val = img.at<float>(y, x);

      if (val > magThr) output.at<float>(y, x) = 255.0;
      else output.at<float>(y, x) = 0.0;
    }
  }
}







void getMag(Mat &Gx,Mat &Gy,Mat &dst,Mat &img){
  //Mat mm;
  //mm.create(dst.size(),CV_64F);
  for ( int i = 0; i < dst.rows; i++ ){
 		for( int j = 0; j < dst.cols; j++ ){
      dst.at<float>(i,j) = sqrt(pow(Gx.at<float>(i,j),2)+pow(Gy.at<float>(i,j),2));
    }
  }
  normalize(dst, dst, 0, 255, NORM_MINMAX);
}

void getDirection(Mat &Gx,Mat &Gy,Mat &dst){
  float angle=0.0;
  for ( int i = 0; i < dst.rows; i++ ){
 		for( int j = 0; j < dst.cols; j++ ){
      if((Gx.at<float>(i,j) !=0) && (Gy.at<float>(i,j) != 0)){
        angle =atan2(Gy.at<float>(i,j),Gx.at<float>(i,j));
      }else{
        angle = (float)atan(0.0);
      }
      dst.at<float>(i,j) = angle;
    }
  }
  Mat img;
	img.create(Gx.size(), CV_32F);


  normalize(dst, dst, 0, 255, NORM_MINMAX);
}
