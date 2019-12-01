// header inclusion

#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core_c.h>
#include <cmath>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

using namespace cv;

using namespace std;



#define pi 3.14159265



//void normolization(Mat x,Mat y,double min_x,double max_x);

//void getMag(Mat &Gx,Mat &Gy,Mat &dst,Mat &img);

//void getDirection(Mat &Gx,Mat &Gy,Mat &dst);

void getThresholdedMag(Mat& input, Mat& output, int magThrFun);



int main(int argc, char** argv)

{



	// LOADING THE IMAGE

	Mat image;

	image = imread("dart1.jpg", 1);

	image.convertTo(image, CV_32F);

	Mat image2 = image.clone();

	// add the following lines
	if (image.empty())
		std::cout << "failed to open img.jpg" << std::endl;
	else
		std::cout << "img.jpg loaded OK" << std::endl;





	//GaussianBlur( image, image, Size(3,3), 0, 0, BORDER_DEFAULT );





	// CONVERT COLOUR, BLUR AND SAVE

	Mat gray_image;

	gray_image.create(image.size(), CV_32F);

	cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);



	GaussianBlur(gray_image, gray_image, Size(9, 9), 2, 2);





	// instilize two filtires dx and dy

	Mat dx, dy, dxKernel, dyKernel;

	dx.create(image.size(), CV_32F);

	dy.create(image.size(), CV_32F);



	dxKernel = (Mat_<float>(3, 3) << -1, 0, 1,

		-2, 0, 2,

		-1, 0, 1);



	dyKernel = (Mat_<float>(3, 3) << -1, -2, -1,

		0, 0, 0,

		1, 2, 1);



	Mat Gx, Gy;

	Gx.create(image.size(), CV_32F);

	Gy.create(image.size(), CV_32F);

	Point anchor = Point(-1, -1);

	double delta = 0;

	filter2D(gray_image, Gx, 5, dxKernel, anchor, delta, BORDER_DEFAULT);

	filter2D(gray_image, Gy, 5, dyKernel, anchor, delta, BORDER_DEFAULT);



	Mat mag, theta, houghspaces;



	int rho_width = gray_image.rows;

	int rho_height = gray_image.cols;

	houghspaces.create(2 * (rho_width + rho_height), 360, CV_32F);

	houghspaces = Scalar(0, 0, 0);





	Mat Mag, Angle;

	//Mag.create(gray_image.size(),CV_64F);

	Angle.create(gray_image.size(), CV_32F);

	cartToPolar(Gx, Gy, Mag, Angle, 1);

	//imwrite( "Angle.jpg", Angle );

	// display images



	float angle = 0.0;

	float radian = 0.0;

	float radians = 0.0;

	float rho_value = 0.0;

	float theta_value = 0.0;

	Mat MagThresh;

	MagThresh.create(Mag.size(), CV_32F);





	// threshholds

	int magThr = 160;

	int houghSpaceThr = 200;



	getThresholdedMag(Mag, MagThresh, magThr);


	int rMin = 40;

	int rMax = 60;

	int th = 180;

	int sizes[] = { 2 * (rho_width + rho_height), 2 * 360 + 1, rMax - rMin };

	Mat houghspaceCircle(3, sizes, CV_32F, Scalar(0));


	vector<Vec3f> circles;

	
	// create houghspace for circles from threshholded magnitude image

	for (int i = 0; i < MagThresh.rows; i++) {

		for (int j = 0; j < MagThresh.cols; j++) {

			if (MagThresh.at<float>(i, j) > 0) { // value is either 0 or 255


				for (int r = 0; r < rMax - rMin; r++) {
					
					for (int theta = 0; theta < 360; theta++) {

						radians = theta * (pi / 180);

						int x0 = i - (r + rMin) * cos(radians);

						int y0 = j - (r + rMin) * sin(radians);
						
						//int x1 = i + (r + rMin) * cos(radians);

						//int y1 = j + (r + rMin) * sin(radians);

						//circles[i][j] = 

						//cout <<"x0: "<<x0<<" y0: "<<y0<<endl;

						if (x0 > 0 && x0 < MagThresh.cols && y0 > 0 && y0 < MagThresh.cols) {

							//cout << "done" << endl;
							//cout << houghspaceCircle.at<float>(x0, y0, r) << endl;

							//cout <<"x0: "<<x0<<" y0: "<<y0<<" r: "<< r <<endl;

							houghspaceCircle.at<float>(x0, y0, r) += 1;

							

						}


					}

				}



			}

		}

	}


	normalize(houghspaceCircle, houghspaceCircle, 0, 255, NORM_MINMAX);

	int count = 0;

	for (int i = 0; i < sizes[0]; i++) {

		for (int j = 0; j < sizes[1]; j++) {

			for (int r = 0; r < rMax - rMin; r++) {

				float val = 0.0;

				val = houghspaceCircle.at<float>(i, j, r);

				

				if (val > th) {

					houghspaceCircle.at<float>(i, j, r) = 255;

					count++;

					//cout << "store r = " << r << endl;

					//cout << "count " << count << endl;
					//cout << "val " << val << endl;
					
					

				}



				else {
					houghspaceCircle.at<float>(i, j, r) = 0;
					//cout << "val" << val << endl;
				}

			}

		}

	}

	cout << "count " << count << endl;

	vector<float>  rhoValues;

	vector<float>  thetaValues;

	vector<float>  radiusValues;

	rhoValues.resize(count);

	thetaValues.resize(count);

	radiusValues.resize(count);

	float rho, theta2;

	

	int index = 0;

	for (int i = 0; i < sizes[0]; i++) {

		for (int j = 0; j < sizes[1]; j++) {

			for (int r = 0; r < rMax - rMin; r++) {

				float val = 0.0;

				val = houghspaceCircle.at<float>(i, j, r);

				


				if (val == 255) {
					//cout << "taken r = " << r << endl;

					radiusValues[index] = r + rMin;

					rhoValues[index] = i;
					//rho = i - 2 * radiusValues[index];
					
					thetaValues[index] = j;

					//theta2 = j ;
					
					//cout << "actual r" << radiusValues[index] <<endl;

					index++;

				}

			}

		}

	} 



	//float maxX = 0;
	//float maxY = 0;
	//float maxR = 0;
	//cout << image.cols << "   " << image.rows;
	//float minX = image.cols;
	//float minY = image.rows;

	//float sumX = 0;
	//float sumY = 0;
	//float sumR = 0;

	for (int i = 0; i < count; i++) {
		//cout << "rho: " << rhoValues[i];
		//cout << "theta: " << thetaValues[i];
		float r = radiusValues[i];

		//float radians = (thetaValues[i]) * pi / 180;

		float x0 = rhoValues[i];
		float y0 = thetaValues[i];

		//cout << thetaValues[i] << endl;
		

		//sumX += x0;
		//sumY += y0;
		//sumR += r;

		//cout << "x0; " << x0 << endl;
		//cout << "y0; " << y0 << endl;
		
		//if (x0 > maxX) maxX = x0;
		//if (y0 > maxY) maxY = y0;
		//if (r > maxR) maxR = r;
		//if (x0 < minX) minX = x0;
		//if (y0 < minY) minY = y0;

		Point center;
		center.x = y0;
		center.y = x0;
		//cout << "theta: " << thetaValues[i] << endl;

		//cout << "r: " << r;

		//cout << "x0 " << center.x << " y0 " << center.y << endl;
		circle(image2, center, r, Scalar(0, 0, 255), 2); 
		
	}

	//circle(image2, Point((minX + maxX) / 2, (minY + maxY) / 2), (maxX - minX) / 2, Scalar(0, 255, 0), 2);
	//rectangle(image2, Point(minY, minX), Point(maxY, maxX), Scalar(0, 255, 0), 2);


	// storing images

	imwrite("1edgeGx.jpg", Gx);

	imwrite("2edgeGY.jpg", Gy);

	imwrite("3circles.jpg", image2);

	imwrite("4image.jpg", image);

	imwrite("6Mag.jpg", Mag);

	imwrite("7MagThresh.jpg", MagThresh);

	imwrite("8Angle.jpg", Angle);


	cout << "images written";

	return 0;

}





// apply threshhold to magnitude

void getThresholdedMag(Mat& input, Mat& output, int magThr) {

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















void getMag(Mat& Gx, Mat& Gy, Mat& dst, Mat& img) {

	//Mat mm;

	//mm.create(dst.size(),CV_64F);

	for (int i = 0; i < dst.rows; i++) {

		for (int j = 0; j < dst.cols; j++) {

			dst.at<float>(i, j) = sqrt(pow(Gx.at<float>(i, j), 2) + pow(Gy.at<float>(i, j), 2));

		}

	}

	normalize(dst, dst, 0, 255, NORM_MINMAX);

}



void getDirection(Mat& Gx, Mat& Gy, Mat& dst) {

	float angle = 0.0;

	for (int i = 0; i < dst.rows; i++) {

		for (int j = 0; j < dst.cols; j++) {

			if ((Gx.at<float>(i, j) != 0) && (Gy.at<float>(i, j) != 0)) {

				angle = atan2(Gy.at<float>(i, j), Gx.at<float>(i, j));

			}
			else {

				angle = (float)atan(0.0);

			}

			dst.at<float>(i, j) = angle;

		}

	}

	Mat img;

	img.create(Gx.size(), CV_32F);





	normalize(dst, dst, 0, 255, NORM_MINMAX);

}
