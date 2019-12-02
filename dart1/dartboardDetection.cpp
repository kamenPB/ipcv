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
};

struct intersectionLines {
    Point intersectionPoint;
    int intersectionCount;
};

/** Function Headers */
void getThresholdedMag(Mat &input, Mat &output,int magThrFun);
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r);
void showListPoint(list<Point> l);
void showListPointonImage(Mat &img,list<Point> l);
void showList3DPoint(list<Point3d> l);
void detectAndDisplay( Mat frame ,int &numberOfBoxes,vector<Point> &corner1,vector<Point> &corner2);
int ***malloc3dArray(int dim1, int dim2, int dim3);

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

int main( int argc, char** argv )
{
  Mat frame;
  Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  //image = imread( "originals/dart15.jpg", 1 );
  frame = image.clone();
  image.convertTo(image, CV_32F);

  // 2. Load the Strong Classifier in a structure called `Cascade'
  if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  // 3. Detect Faces and Display Result
  int numberOfBoxes = 0;
  vector<Point> corner1;
  vector<Point> corner2;
  detectAndDisplay( frame,numberOfBoxes,corner1,corner2);

  // 4. Save Result Imagedart1
  //imwrite( "Viola-Jones_detector.jpg", frame );

 Mat image2 = image.clone();
 if( argc != 2 || !image.data)
 {
   printf( " No image data \n " );
   return -1;
 }
 //list <int,int,int> ddd;

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

// Read from matrix.

// Save to matrix.
//int magThr =120;
int magThrFun =210;
int houghSpaceThr =175;
int pointIntersectionThr = 0;
getThresholdedMag(Mag,MagThresh,magThrFun);
 for ( int i = 0; i < Mag.rows; i++ ){
		for( int j = 0; j < Mag.cols; j++ ){
      if (MagThresh.at<float>(i, j) > 0) { //250 is threshhold
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
imwrite( "out/1houghspaces.jpg", houghspaces );

int count=0;
for (int i = 0; i < houghspaces.rows; i++) {
  for (int j = 0; j < houghspaces.cols; j++) {
    float val = 0.0;
    val = houghspaces.at<float>(i, j);
    if (val > houghSpaceThr){
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

vector<Lines> allLines;
allLines.resize(count);
for( int i = 0; i < count; i++ )
{
    float rhos = rhoValues[i], thetas = thetaValues[i];
    float radians = thetas *pi/ 180;

    Point pt1, pt2;
    Point2f ptIntersect;
    double a = cos(radians), b = sin(radians);
    double x0 = a*rhos, y0 = b*rhos;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));

    allLines[i].p1 = pt1;
    allLines[i].p2 = pt2;
    line( image2, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
}
imwrite( "out/7MagThresh.jpg", MagThresh);
imwrite( "out/8Mag.jpg", Mag);
MagThresh.release();
Mag.release();
// threshholds
//magThr = 100;
int circlehoughSpaceThr = 121;

//getThresholdedMag(Mag, MagThresh, magThr);

int rMin = 80;
int rMax = 160;
int diff= rMax-rMin;
//int th = 180;
int rr=0;
int ***houghspaceCircle;
//int dim1 =rho_width , dim2 = rho_height, dim3 = rMax - rMin;
//  int i, j, k;
houghspaceCircle = malloc3dArray(Mag.rows,Mag.cols, diff);
/*for (int i = 0; i < Mag.rows; i++){
  for (int j = 0; j < Mag.cols; j++){
    for (int k = 0; k < diff; k++){
      houghspaceCircle[i][j][k]=0;
    }
  }
}*/

//HoughSpace Circles
//VOTING ARRAY
/*for (int i = 0; i < Mag.rows; i++) {
  for (int j = 0; j < Mag.cols; j++) {
    if (Mag.at<float>(i, j) > 130) {
      for (int theta = 0; theta < 360; theta++) {
        for (int r = 0; r < diff; r++) {
          radians = theta * (pi / 180);
          rr = r + rMin;
          int x0 = j - (rr) * cos(radians);
          int y0 = i - (rr) * sin(radians);

          if (x0 > 0 && x0 < Mag.rows && y0 > 0 && y0 < Mag.cols) {
            houghspaceCircle[x0][y0][r] += 1;
          }
        }
      }
    }
  }
}*/
/*for (int x = 0; x < Mag.rows; x++) {
  for (int y = 0; y < Mag.cols; y++) {
    for (int r = 0; r < diff; r++) {
      if(houghspaceCircle[x][y][r] > circlehoughSpaceThr){
        //circle( frame,Point(x,y),5,Scalar( 0, 255, 255 ),3,8 );
        //cout<<houghspaceCircle[x][y][r]<<endl;
      }
    }
  }
}*/
// calculate the All IntersectionLines and return all points that intersect with how many number has been occure
int num=0;
for(int i=0; i< allLines.size(); i++){
 for(int j=0; j< allLines.size(); j++){
   if(i != j && j>i){
     Point2f pp;
     intersection(allLines[i].p1,allLines[i].p2,allLines[j].p1,allLines[j].p2,pp);
     if((pp.x !=0) && (pp.y !=0)){
       num++;
      }
    }
  }
}

vector<intersectionLines> allIntersectionLines;
allIntersectionLines.resize(num);
vector<intersectionLines> dectingDarts;
int dectingDartsIndex =0;
dectingDarts.resize(3*num);
for(int i=0; i<num;i++){
  allIntersectionLines[i].intersectionCount =0;
  allIntersectionLines[i].intersectionPoint.x =0;
  allIntersectionLines[i].intersectionPoint.y =0;
  dectingDarts[i].intersectionCount =0;
  dectingDarts[i].intersectionPoint.x =0;
  dectingDarts[i].intersectionPoint.y =0;
  //cout<<"X: "<< allIntersectionLines[i].intersectionPoint.x<<", Y: "<<allIntersectionLines[i].intersectionPoint.y<<", Count: "<<allIntersectionLines[i].intersectionCount<<endl;
}

bool flag2=false;
 for(int i=0; i< allLines.size(); i++){
   for(int j=0; j< allLines.size(); j++){
     if(i != j && j>i){
       Point2f pp;
       intersection(allLines[i].p1,allLines[i].p2,allLines[j].p1,allLines[j].p2,pp);
       pp.x = cvRound(pp.x);
       pp.y = cvRound(pp.y);
       if((pp.x !=0) && (pp.y !=0)){
         flag2 =true;
         int l=0;
         for(int k=0; k<count;k++){
           if((allIntersectionLines[k].intersectionPoint.x ==pp.x) && (allIntersectionLines[k].intersectionPoint.y ==pp.y)){
             allIntersectionLines[k].intersectionCount ++;
           }else{

             while(l <num){
               if((allIntersectionLines[l].intersectionPoint.x == 0) && (allIntersectionLines[l].intersectionPoint.y == 0)){
                 allIntersectionLines[l].intersectionPoint.x = pp.x;
                 allIntersectionLines[l].intersectionPoint.y = pp.y;
                 allIntersectionLines[l].intersectionCount ++;
                 l = num;
               }
               l++;
              }
            }
          }
        }
      }
    }
  }

if(num>0 && flag2){
  //int maxIntersectionPoint = allIntersectionLines[0].intersectionCount;
  //int maxIntersectionIndex = 0;

  for(int i=0; i<num; i++){


     //cout<<", X: "<<allIntersectionLines[i].intersectionPoint.x;
     //cout<<", Y: "<<allIntersectionLines[i].intersectionPoint.y;
     //cout<<"i: "<<i<<", Count: "<<allIntersectionLines[i].intersectionCount<<endl;
     for(int j=0; j<numberOfBoxes; j++){
       Point p    = allIntersectionLines[i].intersectionPoint;
       int Pcount =  allIntersectionLines[i].intersectionCount;
       if((p.x>corner1[j].x) && (p.x<corner2[j].x) && (p.y > corner1[j].y) && (p.y < corner2[j].y)){
         int maxPoint = Pcount;
         int maxIndex = i;
         for(int k=0; k<num; k++){
           Point pp    = allIntersectionLines[k].intersectionPoint;
           int PPcount =  allIntersectionLines[k].intersectionCount;

           if((pp.x>corner1[j].x) && (pp.x<corner2[j].x) && (pp.y > corner1[j].y) && (pp.y < corner2[j].y)){
             if(PPcount>maxPoint){
               maxPoint = PPcount;
               maxIndex = k;
             }
           }
         }

         if(allIntersectionLines[maxIndex].intersectionCount> pointIntersectionThr){
           dectingDarts[dectingDartsIndex].intersectionPoint =allIntersectionLines[maxIndex].intersectionPoint;
           dectingDartsIndex++;
           //cout<<allIntersectionLines[maxIndex].intersectionCount<<" "<<dectingDartsIndex<<endl;
           //circle( frame,allIntersectionLines[maxIndex].intersectionPoint,1,Scalar( 255, 255, 0 ),3,8 );
           rectangle(frame, corner1[j], corner2[j], Scalar( 0, 255, 255 ), 2);
         }
         //circle( frame,Point(allIntersectionLines[i].intersectionPoint.x,allIntersectionLines[i].intersectionPoint.y),2,Scalar( 255, 0, 0 ),3,8 );
       }
     }
     /*if(allIntersectionLines[i].intersectionCount>maxIntersectionPoint){
       maxIntersectionPoint = allIntersectionLines[i].intersectionCount;
       maxIntersectionIndex = i;
     }*/
 }
  //cout<<"index: "<<maxIntersectionIndex<<", Count: "<<maxIntersectionPoint<<endl;
}
//circle( frame,corner2[0],2,Scalar( 255, 255, 0 ),3,8 );
//for(int j=0; j<numberOfBoxes; j++){
  //if( (allIntersectionLines[i].intersectionPoint.x>corner1[j].x) && (allIntersectionLines[i].intersectionPoint.x<corner2[j].x) && (allIntersectionLines[i].intersectionPoint.y > corner1[j]) && (allIntersectionLines[i].intersectionPoint.y < corner2[j])){
//numberOfBoxes;
//vector<Point> corner1;
//vector<Point> corner2;

//circle( image2,Point(294,225),2,Scalar( 0, 255, 0 ),3,8 );

imwrite( "out/2Lines.jpg", image2);
imwrite( "detected.jpg", frame );
 imwrite( "out/4edgeGx.jpg", Gx );
 imwrite( "out/5edgeGY.jpg", Gy );
 imwrite( "out/6orginal.jpg", image);

 imwrite( "out/9Angle.jpg", Angle);
 return 0;
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
void showList3DPoint(list<Point3d> l){
  list<Point3d>::iterator it;
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

void detectAndDisplay( Mat frame, int &numberOfBoxes,vector<Point> &corner1,vector<Point> &corner2)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // 3. Print number of Faces found
	std::cout <<"#Boxes found by Viola-Jones algorithm: "<<faces.size() << std::endl;
  numberOfBoxes = faces.size();
  corner1.resize(numberOfBoxes);
  corner2.resize(numberOfBoxes);

  // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		//rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
    corner1[i] = Point(faces[i].x, faces[i].y);
    corner2[i] = Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
	}
}

int ***malloc3dArray(int dim1, int dim2, int dim3){
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	for (j = 0; j < dim2; j++) {
  	    array[i][j] = (int *) malloc(dim3 * sizeof(int));
	}
    }
    return array;
}
