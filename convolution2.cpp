// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <cmath>
#include<bits/stdc++.h>
#include <vector>
#include <list>
using namespace cv;
using namespace std;

#define pi 3.14159265

struct Lines {
    Point p1;
    Point p2;
    list <Point> intersectionPoint;
    int intersectionCount;
    //int intersectionPointSize;
    //vector<Point> intersectionPoint;
    //intersectionPoint.resize(100);
};

//  Declare an instance of Book and a vector of books
  //Lines book;

//void normolization(Mat x,Mat y,double min_x,double max_x);
//void getMag(Mat &Gx,Mat &Gy,Mat &dst,Mat &img);
//void getDirection(Mat &Gx,Mat &Gy,Mat &dst);
void getThresholdedMag(Mat &input, Mat &output,int magThrFun);
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r);
void showListPoint(list<Point> l);
void showListPointonImage(Mat &img,list<Point> l);
int main( int argc, char** argv )
{

 // LOADING THE IMAGE
 Mat image;
 image = imread( "dart1.jpg", 1 );
 image.convertTo(image, CV_32F);
 Mat image2 = image.clone();
 if( argc != 1 || !image.data)
 {
   printf( " No image data \n " );
   return -1;
 }
// GaussianBlur( image, image, Size(3,3), 0, 0, BORDER_DEFAULT );
 // CONVERT COLOUR, BLUR AND SAVE
 Mat gray_image;
 gray_image.create(image.size(), CV_32F);
 cvtColor( image, gray_image, CV_BGR2GRAY );
GaussianBlur( gray_image, gray_image, Size(9,9), 2, 2, BORDER_DEFAULT );
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
 Point anchor=Point( -1, -1 );
 double delta=0;
 filter2D(gray_image, Gx, 5 , dxKernel, anchor, delta, BORDER_DEFAULT );
 filter2D(gray_image, Gy, 5 , dyKernel, anchor, delta, BORDER_DEFAULT );

 cv::Mat mag,theta,houghspaces;

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

float angle =0.0;
float radian =0.0;
float radians =0.0;
float rho_value=0.0;
float theta_value=0.0;
Mat MagThresh;
MagThresh.create(Mag.size(),CV_32F);
//vector<Vec2f> lines;

// Read from matrix.

// Save to matrix.
int magThr =254;
int magThrFun =200;
int houghSpaceThr =200;
getThresholdedMag(Mag,MagThresh,magThrFun);
 for ( int i = 0; i < Mag.rows; i++ ){
		for( int j = 0; j < Mag.cols; j++ ){
      if (MagThresh.at<float>(i, j) > magThr) { //250 is threshhold
        angle = Angle.at<float>(i, j);
        if (angle > 0) radian = (angle * (180/pi));
        else radian = 360 + (angle * (180/pi));

        radian = round(radian);

        for (int theta_value = 0; theta_value <180; theta_value ++) {
          radians = theta_value * (pi/ 180);
          rho_value = (j * cos(radians)) + (i * sin(radians)) + rho_width + rho_height;
          houghspaces.at<float>( rho_value , theta_value ) += 1;
        }
		}
	}
}

normalize(houghspaces, houghspaces, 0, 255, NORM_MINMAX);
imwrite( "5houghspaces.jpg", houghspaces );

int count=0;
for (int i = 0; i < houghspaces.rows; i++) {
  for (int j = 0; j < houghspaces.cols; j++) {
    float val = 0.0;
    val = houghspaces.at<float>(i, j);
    if (val > houghSpaceThr){ //150 is threshhold
      houghspaces.at<float>(i, j) = 255;
      count++;
    }

    else houghspaces.at<float>(i, j) = 0.0;
  }
 }
 vector<float> rhoValues;
 vector<float> thetaValues;
 rhoValues.resize(count);
 thetaValues.resize(count);

 int index=0;
 for (int i = 0; i < houghspaces.rows; i++) {
   for (int j = 0; j < houghspaces.cols; j++) {
     float val = 0.0;
     val = houghspaces.at<float>(i, j);
     if (val == 255){
       rhoValues[index]=i-rho_width - rho_height;
       thetaValues[index]=j;
       index++;
     }
   }
 }


 /*int no_of_cols = count;
 int no_of_rows = count;
 int initial_value = 0;

 vector<vector<float> > lines;
 lines.resize(no_of_rows, vector<float>(no_of_cols, initial_value));*/
vector<Lines> allLines;
allLines.resize(count);
for( int i = 0; i < count; i++ )
{
    float rhos = rhoValues[i], thetas = thetaValues[i];
    float radians = thetas *pi/ 180;
    cout<<"rhos: "<<rhos<<", thetas: "<<radians<<endl;
    Point pt1, pt2;
    Point2f ptIntersect;
    double a = cos(radians), b = sin(radians);
    double x0 = a*rhos, y0 = b*rhos;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
    //intersection(pt1,pt2,pt1,pt2,ptIntersect);

    //cout<<"ptIntersect.x: "<<ptIntersect.x<<", ptIntersect.y: "<<ptIntersect.y<<endl;
    //cout<<"pt1.x: "<<pt1.x<<", pt1.y: "<<pt1.y<<", pt2.x: "<<pt2.x<<", pt2.y: "<<pt2.y<<endl;
    //std::vector<Lines> line;
    //line[0].p1 = pt1;
    //line[0].p2 = pt2;

    allLines[i].p1 = pt1;
    allLines[i].p2 = pt2;
    allLines[i].intersectionCount=0;
    //showListPoint(allLines[i].intersectionPoint);
    //allLines[0].intersectionPoint[0] = pt2;
    //cout<<"i: "<<i<<", p1: "<<allLines[i].p1<<", p2: "<<allLines[i].p2<<endl;
    line( image2, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
}


 for(int i=0; i< allLines.size(); i++){
   for(int j=0; j< allLines.size(); j++){
     if(i != j){
       Point2f pp;
       intersection(allLines[i].p1,allLines[i].p2,allLines[j].p1,allLines[j].p2,pp);
       if((pp.x !=0) && (pp.y !=0)){
          allLines[i].intersectionPoint.push_back(pp);
          allLines[i].intersectionCount ++;
       }
     }
   }
 }
 //showListPointonImage(image2,allLines[0].intersectionPoint);
 //circle( image2,allLines[0].intersectionPoint,5,Scalar( 255, 0, 0 ),3,8 );
 imwrite( "1edgeGx.jpg", Gx );
 imwrite( "2edgeGY.jpg", Gy );
 imwrite( "3image2.jpg", image2 );
 imwrite( "4image.jpg", image);
 imwrite( "6MagThresh.jpg", MagThresh);
 imwrite( "7Mag.jpg", Mag);
 imwrite( "8Angle.jpg", Angle);
 return 0;
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

void getThresholdedMag(Mat &input, Mat &output, int magThrFun) {
	Mat img;
	img.create(input.size(), CV_32F);

	normalize(input, img, 0, 255, NORM_MINMAX);

	for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {

      double val = 0;
      val = img.at<float>(y, x);

      if (val > magThrFun) output.at<float>(y, x) = 255.0;
      else output.at<float>(y, x) = 0.0;
    }
  }

  imwrite("MagThresholded.jpg", output);
}

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r){
    Point2f x = o2 - o1;
    Point2f d1 = p1 - o1;
    Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < 1e-8){
      r.x=0;
      r.y=0;
      return false;
    }

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    //cout<<"r: "<<r<<endl;
    //count ++;
    return true;
}
void showListPoint(list<Point> l){
  list<Point>::iterator it;
  for(it=l.begin();it!=l.end();it++){
    cout <<  *it << " ";
  }
  cout << "\n";
}
void showListPointonImage(Mat &img,list<Point> l){
  list<Point>::iterator it;
  for(it=l.begin();it!=l.end();it++){
    circle( img,*it,5,Scalar( 255, 0, 0 ),3,8 );
    //cout <<  *it << " ";
  }
}
/*
double find_min(Mat x){
  double min=x.at<double>(0,0);
  for ( int i = 0; i < x.rows; i++ ){
 		for( int j = 0; j < x.cols; j++ ){
      if(x.at<double>(i,j) < min){
        min = x.at<double>(i,j);
      }
    }
  }
  return min;
}

double find_max(Mat x){
  double max=x.at<double>(0,0);
  for ( int i = 0; i < x.rows; i++ ){
 		for( int j = 0; j < x.cols; j++ ){
      if(x.at<double>(i,j) > max){
        max = x.at<double>(i,j);
      }
    }
  }
  return max;
}

void normolization(Mat x,Mat y,double min_x,double max_x){

  for ( int i = 0; i < y.rows; i++ ){
 		for( int j = 0; j < y.cols; j++ ){
      y.at<double>(i,j) = (double)((x.at<double>(i,j)-min_x)/(max_x-min_x));
      //cout<<"i: "<<i<<", j: "<<j<<", x.at<double>(i,j)"<<x.at<double>(i,j)<<", y.at<double>(i,j)"<<y.at<double>(i,j)<<endl;
    }
  }
  //cout<<"y.rows: "<<y.rows<<endl;
  //cout<<"y.cols: "<<y.cols<<endl;
}*/
/*Mat convolve(cv::Mat &input, cv::Mat &kernel) {
	cv::Mat blurredOutput;
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );
		cout << "helo ";

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );
					cout << imageval;

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
	cout << "world";
	return blurredOutput;
}*/
