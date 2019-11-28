// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <cmath>
#include<bits/stdc++.h>

using namespace cv;
using namespace std;

#define pi 3.14159265

double find_min(Mat x);
double find_max(Mat x);
void normolization(Mat x,Mat y,double min_x,double max_x);
int main( int argc, char** argv )
{
 // LOADING THE IMAGE
 Mat image;
 image = imread( "detected.jpg", 1 );

 if( argc != 1 || !image.data)
 {
   printf( " No image data \n " );
   return -1;
 }

 // CONVERT COLOUR, BLUR AND SAVE
 Mat gray_image;
 cvtColor( image, gray_image, CV_BGR2GRAY );
 Mat gray_image2;
 cvtColor( image, gray_image2, CV_BGR2GRAY );

 // instilize two filtires dx and dy
 int d1[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
 int d2[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
 cv::Mat dx= cv::Mat(3, 3, CV_32S, d1);
 cv::Mat dy= cv::Mat(3, 3, CV_32S, d2);

 // Apply the dx and dy filtiers on the gray image
 //Gx= dx * gray_image
 //Gy= dy * gray_image
 Mat Gx,Gy;
 Point anchor=Point( -1, -1 );
 double delta=0;
 filter2D(gray_image, Gx, 5 , dx, anchor, delta, BORDER_DEFAULT );
 filter2D(gray_image, Gy, 5 , dy, anchor, delta, BORDER_DEFAULT );

 //instilize  Matrices
 cv::Mat mag(Gx.size(), Gx.type());
 cv::Mat theta(mag.size(), mag.type());

 cv::Mat rho(Gx.size(), Gx.type());
 cv::Mat houghspaces;

 //int rho_length = 2 * (rho_width+rho_height);
 //int theta_length = 360;
 //calculating the Magnitude which is: mag=sqrt(Gx^2 + Gy^2)
 cv::magnitude(Gx, Gy, mag);


 int rho_width = rho.rows;
 int rho_height = rho.cols;
 houghspaces.create(2*(rho_width+rho_height),360,CV_64F);
 houghspaces = Scalar(0,0,0);

 Mat mag3, angle3;
 cartToPolar(Gx, Gy, mag3, angle3, 1);

/*double gx = 0.0;
double gy = 0.0;
double k=0.0;
 for ( int i = 0; i < theta.rows; i++ ){
    for( int j = 0; j < theta.cols; j++ ){
      gx = Gx.at<double>(i,j);
      gy = Gy.at<double>(i,j);
      if(gx !=0 && gy != 0){
        k=atan2(gy,gx);
      }else{
        k=(double) atan(0);
      }
      theta.at<double>(i,j) =k;
    }
  }
*/
 cv::Mat diff = angle3 != theta;
// Equal if no elements disagree
//bool eq = cv::countNonZero(diff) == 0;
//cout<<"Eq: "<<eq<<endl;

 for ( int i = 0; i < mag.rows; i++ ){
		for( int j = 0; j < mag.cols; j++ ){
      double angle =atan2(Gy.at<double>(i,j),Gx.at<double>(i,j));
      //angle = round(angle3.at<double>(i,j));
      //angle = 0.4;
      //cout<<"angle: "<<angle<<endl;
      double radian;
      if(angle<0){
        radian = 360 + (angle*180/pi);
      }else{
        radian =  (angle*180/pi);
      }

      //double degree= radian*180/pi;
      double rho_value  = (i*cos(radian)) + (j*sin(radian)) + rho_width + rho_height;

      //theta.at<double>(i, j)=angle;
      rho.at<double>(i,j)=rho_value;

      houghspaces.at<double>(round(rho_value),round(angle)) += 1;
      //houghspaces.at<double>(rho,theta) += 1;
		}
	}

  for( int i = 0; i < houghspaces.rows; i++ ){
    for( int j = 0; j < houghspaces.cols; j++ ){
      if(houghspaces.at<double>(i,j)>90){
         double rho2 = i-rho_width - rho_height;
         double theta2 = j;

         Point pt1, pt2;
         double a = cos(theta2), b = sin(theta2);
         double x0 = a*rho2, y0 = b*rho2;

         pt1.x = cvRound(x0 + 1000*(-b));
         pt1.y = cvRound(y0 + 1000*(a));
         pt2.x = cvRound(x0 - 1000*(-b));
         pt2.y = cvRound(y0 - 1000*(a));

         line( image, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
      }
     }
   }

double min_theta= find_min(theta);
double max_theta= find_max(theta);
double min_rho= find_min(rho);
double max_rho= find_max(rho);
/*cout<<"min_theta: "<<min_theta<<endl;
cout<<"max_theta: "<<max_theta<<endl;
cout<<"min_rho: "<<min_rho<<endl;
cout<<"max_rho: "<<max_rho<<endl;*/

 imwrite( "edgeGx.jpg", Gx );
 imwrite( "edgeGY.jpg", Gy );
 imwrite( "mag.jpg", mag);
 imwrite( "mag3.jpg", mag3);
 imwrite( "angle3.jpg", angle3);
 imwrite( "houghspace.jpg", houghspaces);
 imwrite( "gray_image2.jpg", image);
 //imwrite( "gradient_direction.jpg", theta);

 return 0;
}
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
}
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
