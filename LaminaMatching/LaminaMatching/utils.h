#pragma once
#include <iostream>
#include<opencv.hpp>
#include <numeric>
#include <algorithm>
#include<cmath>
#include<inference_engine.hpp>

using namespace std;
using namespace cv;

void PAF_finetune(vector<vector<double>>& PAF_x, vector<vector<double>>& PAF_y, Mat& PAF);
void Get_all_points(Mat PCM_LEFT, Mat PCM_RIGHT, vector<Point>& points_left, vector<Point>& points_right, int ori_width, int ori_height);
vector<Point> Get_pairs_idx(vector<Point>& points_left, vector<Point>& points_right, vector<vector<double>>& PAF_MAP_X, vector<vector<double>>& PAF_MAP_Y);

int readModel(std::string binPath, std::string xmlPath, InferenceEngine::CNNNetwork& model);
    
int initiateModel(std::string binPath,
    std::string xmlPath,
    InferenceEngine::CNNNetwork& model,
    InferenceEngine::InferRequest& infer_request);

vector<vector<Point>> predict(InferenceEngine::CNNNetwork& model,
    InferenceEngine::InferRequest& infer_request,
    cv::Mat& image);
    