#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
 
using namespace cv;
using namespace std;

cv::Mat fastGuidedFilter(cv::Mat I_org, cv::Mat p_org, int r, double eps, int s)
{
	cv::Mat I,_I;
	I_org.convertTo(_I, CV_64FC1, 1.0 / 255);
 
	resize(_I,I,Size(),1.0/s,1.0/s,INTER_CUBIC);
	//size = (int(round(w*s)), int(round(h*s)));
 
	cv::Mat p,_p;
	p_org.convertTo(_p, CV_64FC1, 1.0 / 255);
	//p = _p;
	resize(_p, p, Size(),1.0/s,1.0/s,INTER_CUBIC);
 
	//[hei, wid] = size(I);    
	int hei = I.rows;
	int wid = I.cols;
	
	r = round(r/s) ;
	//r = (2 * r + 1)/s;//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4   
	//r = (2 * r + 1)/s+1
	//mean_I = boxfilter(I, r) ./ N;    
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));
 
	//mean_p = boxfilter(p, r) ./ N;    
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));
 
	//mean_Ip = boxfilter(I.*p, r) ./ N;    
	cv::Mat mean_Ip;
	cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));
 
	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.    
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
 
	//mean_II = boxfilter(I.*I, r) ./ N;    
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));
 
	//var_I = mean_II - mean_I .* mean_I;    
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);
 
	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;       
	cv::Mat a = cov_Ip / (var_I + eps);
 
	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;    
	cv::Mat b = mean_p - a.mul(mean_I);
 
	//mean_a = boxfilter(a, r) ./ N;    
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
	Mat rmean_a;
	resize(mean_a, rmean_a, Size(I_org.cols, I_org.rows),INTER_CUBIC);
 
	//mean_b = boxfilter(b, r) ./ N;    
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
	Mat rmean_b;
	resize(mean_b, rmean_b, Size(I_org.cols, I_org.rows),INTER_CUBIC);
	
	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;    
	cv::Mat q = rmean_a.mul(_I) + rmean_b;
 
	return q;
}


int main(){
	cv::Mat img = cv::imread("F:/JZYY/code/ImageEnhancement/2018Endoscope_image_enhancement/result/myresult/2.png");
	if (img.empty()){
		return -1;
	}
	cv::Mat ycrcb_I, ycrcb_P, dst, Re1, Re2; 

	if(img.channels() > 1){
		cv::cvtColor(img, ycrcb_I, COLOR_BGR2YCrCb);
		ycrcb_I.copyTo(ycrcb_P);
	}
	else{
		return -1;
	}	
	
	std::vector<cv::Mat> I, P, q;
	cv::split(ycrcb_I, I);
	cv::split(ycrcb_P, P);

	int r = 50;
	double eps = 0.12;
	int s = 2;
	double w = 0.5;
	dst = fastGuidedFilter(I[0], P[0], r, eps, s);
	dst.convertTo(dst, CV_8U, 255);

	addWeighted(I[0],1+w,dst,-w,0,dst);
	
	q.push_back(dst);

	q.push_back(P[1]);
	q.push_back(P[2]);
	cv::merge(q, Re1);

	cv::cvtColor(Re1, Re2, COLOR_YCrCb2BGR);
	
	cv::imshow("GuidedFilter_box", img);
	cv::imwrite("./result/2gf505.png", Re2);
	cv::waitKey(0);


}
