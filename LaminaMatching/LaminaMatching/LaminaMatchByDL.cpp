
#include "LaminaMatchByDL.h"

LaminaMatchByDL::LaminaMatchByDL(std::string IR_xml, std::string IR_bin) {
	this->initiateModel(IR_xml, IR_bin, this->model, this->infer_request);
}

void LaminaMatchByDL::initiateModel(std::string xmlPath, std::string binPath, InferenceEngine::CNNNetwork& model, InferenceEngine::InferRequest& infer_request){

    model = this->ie.ReadNetwork(xmlPath, binPath);
    //prepare input blobs
    InferenceEngine::InputInfo::Ptr input_info = model.getInputsInfo().begin()->second;
    std::string input_name = (model.getInputsInfo().begin())->first;
    input_info->setLayout(InferenceEngine::Layout::NCHW);
    input_info->setPrecision(InferenceEngine::Precision::FP32);

    //prepare output blobs
    InferenceEngine::DataPtr output_PAF = model.getOutputsInfo().begin()->second;
    std::string output_name_PAF = model.getOutputsInfo().begin()->first;
    output_PAF->setPrecision(InferenceEngine::Precision::FP32);

    InferenceEngine::DataPtr output_PCM = (++model.getOutputsInfo().begin())->second;
    std::string output_name_PCM = (++model.getOutputsInfo().begin())->first;
    output_PCM->setPrecision(InferenceEngine::Precision::FP32);

    InferenceEngine::DataPtr output_bbox = (++(++model.getOutputsInfo().begin()))->second;
    std::string output_name_bbox = (++(++model.getOutputsInfo().begin()))->first;
    output_bbox->setPrecision(InferenceEngine::Precision::FP32);

    //load model to device
    this->executable_network = this->ie.LoadNetwork(model, "CPU");

    //create infer request
    infer_request = this->executable_network.CreateInferRequest();

}

vector<vector<Point>> LaminaMatchByDL::predict(cv::Mat& image) {
    int ori_height = image.rows;
    int ori_width = image.cols;
    resize(image, image, Size(128, 512));
    //blur(image, image, Size(3, 3));

    std::string input_name = this->model.getInputsInfo().begin()->first;
    std::string output_paf = this->model.getOutputsInfo().begin()->first;
    std::string output_pcm = (++this->model.getOutputsInfo().begin())->first;
    std::string output_bbox = (++(++this->model.getOutputsInfo().begin()))->first;

    InferenceEngine::Blob::Ptr input = this->infer_request.GetBlob(input_name);

    InferenceEngine::Blob::Ptr output_PAF = this->infer_request.GetBlob(output_paf);
    InferenceEngine::Blob::Ptr output_PCM = this->infer_request.GetBlob(output_pcm);
    InferenceEngine::Blob::Ptr output_Bbox = this->infer_request.GetBlob(output_bbox);


    size_t channels_number = input->getTensorDesc().getDims()[1];
    size_t image_size = input->getTensorDesc().getDims()[3] * input->getTensorDesc().getDims()[2];

    auto input_data1 = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    size_t num = 0;

    for (size_t pid = 0; pid < image_size; ++pid)
    {
        for (size_t ch = 0; ch < channels_number; ++ch)
        {
            input_data1[pid] = (float)image.at<uchar>(pid) / 255.0f;
        }
    }

    this->infer_request.Infer();

    auto output_data_PAF = output_PAF->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    auto output_data_PCM = output_PCM->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    auto output_data_Bbox = output_Bbox->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

    Mat PCM_LEFT = Mat(512, 128, CV_8U, Scalar::all(0));
    Mat PCM_RIGHT = Mat(512, 128, CV_8U, Scalar::all(0));
    for (int j = 0; j < 512; j++) {
        for (int z = 0; z < 128; z++) {
            if (output_data_PCM[0] >= 0.5) {
                PCM_LEFT.at<uchar>(j, z) = 255;
            }
            else {
                PCM_LEFT.at<uchar>(j, z) = 0;
            }
            ++output_data_PCM;
        }
    }

    for (int j = 0; j < 512; j++) {
        for (int z = 0; z < 128; z++) {
            if (output_data_PCM[0] >= 0.5) {
                PCM_RIGHT.at<uchar>(j, z) = 255;
            }
            else {
                PCM_RIGHT.at<uchar>(j, z) = 0;
            }
            ++output_data_PCM;
        }
    }

    vector<Point> points_left, points_right;
    this->Get_all_points(PCM_LEFT, PCM_RIGHT, points_left, points_right, ori_width, ori_height);

    Mat PAF_X = Mat(512, 128, CV_32F, Scalar::all(0));
    Mat PAF_Y = Mat(512, 128, CV_32F, Scalar::all(0));
    Mat PAF_vector_size = Mat(512, 128, CV_32F, Scalar::all(0));

    // ---------------------------------------
    for (int y = 0; y < 512; y++) {
        for (int x = 0; x < 128; x++) {
            PAF_X.at<float>(y, x) = output_data_PAF[0];
            ++output_data_PAF;
        }
    }

    for (int y = 0; y < 512; y++) {
        for (int x = 0; x < 128; x++) {
            PAF_Y.at<float>(y, x) = output_data_PAF[0];
            ++output_data_PAF;
        }
    }

    for (int y = 0; y < 512; y++) {
        for (int x = 0; x < 128; x++) {
            if (sqrt(pow(PAF_X.at<float>(y, x), 2) + pow(PAF_Y.at<float>(y, x), 2)) >= 0.5) {
                PAF_vector_size.at<float>(y, x) = 255;
            }
            else {
                PAF_vector_size.at<float>(y, x) = 0;
            }
            PAF_Y.at<float>(y, x) = output_data_PAF[0];
        }
    }

    PAF_vector_size.convertTo(PAF_vector_size, CV_8U);

    vector<double> w(128, 0);
    vector<vector<double>> PAF_MAP_X(512, w), PAF_MAP_Y(512, w);

    this->PAF_finetune(PAF_MAP_X, PAF_MAP_Y, PAF_vector_size);
    this->PAF_global = PAF_vector_size;

    vector<Point> pairs_point_idx;
    vector<vector<Point>> pair;
    pairs_point_idx = this->Get_pairs_idx(points_left, points_right, PAF_MAP_X, PAF_MAP_Y);

    for (size_t i = 0; i < pairs_point_idx.size(); i++) {
        Point current_left_point, current_right_point;
        vector<Point> current_pair;
        current_left_point = points_left[pairs_point_idx[i].x];
        current_right_point = points_right[pairs_point_idx[i].y];
        current_pair.push_back(current_left_point);
        current_pair.push_back(current_right_point);
        pair.push_back(current_pair);
    }

    cout << "×óºáÍ»×ø±ê(x, y)" << "              " << "ÓÒºáÍ»×ø±ê(x, y)" << endl;
    for (size_t i = 0; i < pair.size(); i++) {
        cout << pair[i][0].x << " " << pair[i][0].y << "                 " << pair[i][1].x << " " << pair[i][1].y << " " << endl;
        if (abs(pair[i][1].y - pair[i][0].y) / (pair[i][1].x - pair[i][0].x) > 0.7) {
            continue;
        }

        line(image, pair[i][0], pair[i][1], Scalar(255));
    }

    imshow("PAF_global",this->PAF_global);
    waitKey(0);

    imshow("image", image);
    waitKey(0);

    return pair;
}

void LaminaMatchByDL::PAF_finetune(vector<vector<double>>& PAF_x, vector<vector<double>>& PAF_y, Mat& PAF) {

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(PAF, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

    for (auto contour = contours.cbegin(); contour != contours.cend(); contour++) {
        int xmin = 0, xmax = 0;
        for (auto c_iter = (*contour).cbegin(); c_iter != (*contour).cend(); c_iter++) {
            if (c_iter == (*contour).cbegin()) {
                xmin = (*c_iter).x;
                xmax = (*c_iter).x;
                continue;
            }
            if ((*c_iter).x < xmin) {
                xmin = (*c_iter).x;
            }
            if ((*c_iter).x > xmax) {
                xmax = (*c_iter).x;
            }
        }
        int y_LeftPoint = 0, y_RightPoint = 0, y_LeftPointNum = 0, y_RightPointNum = 0;

        for (auto c_iter = (*contour).cbegin(); c_iter != (*contour).cend(); c_iter++) {
            if ((*c_iter).x == xmin) {
                y_LeftPoint += (*c_iter).y;
                y_LeftPointNum += 1;
            }
            if ((*c_iter).x == xmax) {
                y_RightPoint += (*c_iter).y;
                y_RightPointNum += 1;
            }
        }
        y_LeftPoint = y_LeftPoint / y_LeftPointNum;
        y_RightPoint = y_RightPoint / y_RightPointNum;
        int x_LeftPoint = xmin, x_RightPoint = xmax;
        //cout << "x_LeftPoint: " << xmin <<  " y_LeftPoint: " << y_LeftPoint << " x_RightPoint:" << xmax << " y_RightPoint: " << y_RightPoint << endl;

        cv::Mat blobMask(512, 128, CV_8U, cv::Scalar(0));
        fillConvexPoly(blobMask, *contour, Scalar(255));

        vector<float> leftRegion_x, leftRegion_y, rightRegion_x, rightRegion_y;
        for (rsize_t y = 0; y < 512; y++) {
            for (rsize_t x = 0; x < 128; x++) {
                if (blobMask.at<uchar>(y, x) == 255) {
                    if (sqrt(pow((float(y) - float(y_LeftPoint)), 2) + pow((float(x) - float(x_LeftPoint)), 2)) < sqrt(pow((float(y) - float(y_RightPoint)), 2) + pow((float(x) - float(x_RightPoint)), 2)))
                    {
                        leftRegion_x.push_back(x);
                        leftRegion_y.push_back(y);
                    }
                    else {
                        rightRegion_x.push_back(x);
                        rightRegion_y.push_back(y);
                    }

                }
            }
        }

        int finalContourLeftWeight_x, finalContourLeftWeight_y, finalContourRightWeight_x, finalContourRightWeight_y;

        double finalContourLeftWeight_xSumValue = accumulate(begin(leftRegion_x), end(leftRegion_x), 0.0);
        finalContourLeftWeight_x = finalContourLeftWeight_xSumValue / leftRegion_x.size();

        double finalContourLeftWeight_ySumValue = accumulate(begin(leftRegion_y), end(leftRegion_y), 0.0);
        finalContourLeftWeight_y = finalContourLeftWeight_ySumValue / leftRegion_y.size();

        double finalContourRightWeight_xSumValue = accumulate(begin(rightRegion_x), end(rightRegion_x), 0.0);
        finalContourRightWeight_x = finalContourRightWeight_xSumValue / rightRegion_x.size();

        double finalContourRightWeight_ySumValue = accumulate(begin(rightRegion_y), end(rightRegion_y), 0.0);
        finalContourRightWeight_y = finalContourRightWeight_ySumValue / rightRegion_y.size();

        for (rsize_t y = 0; y < 512; y++) {
            for (rsize_t x = 0; x < 128; x++) {
                if (blobMask.at<uchar>(y, x) == 255) {

                    PAF_x[y][x] = float(finalContourRightWeight_x - finalContourLeftWeight_x) / sqrt(pow(float(finalContourRightWeight_x) - float(finalContourLeftWeight_x), 2) + pow(float(finalContourRightWeight_y) - float(finalContourLeftWeight_y), 2));
                    PAF_y[y][x] = -float(finalContourRightWeight_y - finalContourLeftWeight_y) / sqrt(pow(float(finalContourRightWeight_x) - float(finalContourLeftWeight_x), 2) + pow(float(finalContourRightWeight_y) - float(finalContourLeftWeight_y), 2));

                }
                //cout << setw(2) << setprecision(2) <<  PAF_y[y][x] << " ";
            }
            //cout << endl;
        }

    }

}

void LaminaMatchByDL::Get_all_points(Mat PCM_LEFT, Mat PCM_RIGHT, vector<Point>& points_left, vector<Point>& points_right, int ori_width, int ori_height) {
    vector<vector<Point>> LEFT_contours, RIGHT_contours;
    vector<Vec4i> LEFT_hierarchy, RIGHT_hierarchy;
    findContours(PCM_LEFT, LEFT_contours, LEFT_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
    findContours(PCM_RIGHT, RIGHT_contours, RIGHT_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

    for (auto l_contour = LEFT_contours.cbegin(); l_contour != LEFT_contours.cend(); l_contour++) {
        float p_x_sum = 0, p_y_sum = 0, p_x_mean = 0, p_y_mean = 0;
        Point current_point;
        for (auto l_c_iter = (*l_contour).cbegin(); l_c_iter != (*l_contour).cend(); l_c_iter++) {
            p_x_sum += float((*l_c_iter).x);
            p_y_sum += float((*l_c_iter).y);
        }
        p_x_mean = (p_x_sum / (*l_contour).size());
        p_y_mean = (p_y_sum / (*l_contour).size());
        /*       current_point.x = p_x_mean * (ori_width / 128);
               current_point.y = p_y_mean * (ori_height / 512);*/
        current_point.x = p_x_mean;
        current_point.y = p_y_mean;
        //cout << current_point.x << " " << current_point.y << endl;
        points_left.push_back(current_point);
    }

    for (auto r_contour = RIGHT_contours.cbegin(); r_contour != RIGHT_contours.cend(); r_contour++) {
        float p_x_sum = 0, p_y_sum = 0, p_x_mean = 0, p_y_mean = 0;
        Point current_point;
        for (auto r_c_iter = (*r_contour).cbegin(); r_c_iter != (*r_contour).cend(); r_c_iter++) {
            p_x_sum += float((*r_c_iter).x);
            p_y_sum += float((*r_c_iter).y);
        }
        p_x_mean = (p_x_sum / (*r_contour).size());
        p_y_mean = (p_y_sum / (*r_contour).size());
        //current_point.x = p_x_mean * (ori_width / 128);
        //current_point.y = p_y_mean * (ori_height / 512);
        current_point.x = p_x_mean;
        current_point.y = p_y_mean;
        //cout << current_point.x << " " << current_point.y << endl;
        points_right.push_back(current_point);
    }
}

vector<Point> LaminaMatchByDL::Get_pairs_idx(vector<Point>& points_left,
    vector<Point>& points_right,
    vector<vector<double>>& PAF_MAP_X,
    vector<vector<double>>& PAF_MAP_Y) {

    vector<double> w(points_right.size(), 0);
    vector<vector<double>> confidence(points_left.size(), w);

    int integral_num = 100;

    int idx_leftPoint = 0;
    for (auto point_left = points_left.cbegin(); point_left != points_left.cend(); point_left++) {
        int idx_rightPoint = 0;
        for (auto point_right = points_right.cbegin(); point_right != points_right.cend(); point_right++) {
            double target_vector_x = 0.0, target_vector_y = 0.0;
            target_vector_x = (double((*point_right).x - (*point_left).x)) / (sqrt(pow(double((*point_right).x) - (*point_left).x, 2) + pow(double((*point_right).y) - (*point_left).y, 2)));
            target_vector_y = -(double((*point_right).y - (*point_left).y)) / (sqrt(pow(double((*point_right).x) - (*point_left).x, 2) + pow(double((*point_right).y) - (*point_left).y, 2)));

            double integral_interval_x = (double((*point_right).x - (*point_left).x)) / integral_num;
            double integral_interval_y = (double((*point_right).y - (*point_left).y)) / integral_num;
            double integral_pair = 0.0, integral_pair_temp = 0.0;
            for (rsize_t n = 1; n < integral_num; n++) {
                int integral_x = 0, integral_y = 0;
                integral_x = int((*point_left).x + integral_interval_x * n);
                integral_y = int((*point_left).y + integral_interval_y * n);
                integral_pair_temp = (PAF_MAP_X[integral_y][integral_x] * target_vector_x) + (PAF_MAP_Y[integral_y][integral_x] * target_vector_y);
                integral_pair += integral_pair_temp;
            }
            confidence[idx_leftPoint][idx_rightPoint] = integral_pair;
            idx_rightPoint += 1;
        }
        idx_leftPoint += 1;
    }
    //for (size_t nrow = 0; nrow < points_left.size(); nrow++)
    //{
    //    cout << endl;
    //    for (size_t ncol = 0; ncol < points_right.size(); ncol++)
    //    {
    //        cout << setprecision(5) << confidence[nrow][ncol] << "    ";
    //    }
    //    cout << endl;
    //}

    vector<double> pair_value;
    vector<Point> pair_idx;
    for (size_t left_idx = 0; left_idx < points_left.size(); left_idx++) {
        for (size_t right_idx = 0; right_idx < points_right.size(); right_idx++) {
            pair_value.push_back(confidence[left_idx][right_idx]);
            Point current_pair_idx;
            current_pair_idx.x = left_idx;
            current_pair_idx.y = right_idx;
            pair_idx.push_back(current_pair_idx);
        }
    }

    for (size_t i = 0; i < pair_value.size(); i++) {
        for (size_t j = i + 1; j < pair_value.size(); j++) {
            if (pair_value[j] > pair_value[i]) {
                double temp;
                temp = pair_value[j];
                pair_value[j] = pair_value[i];
                pair_value[i] = temp;

                Point temp_point;
                temp_point = pair_idx[j];
                pair_idx[j] = pair_idx[i];
                pair_idx[i] = temp_point;
            }

        }
    }


    vector<int> confidence_max_left_already_in, confidence_max_right_already_in;
    int min_detected_points_number = min(points_left.size(), points_right.size());
    for (size_t i = 0; i < pair_value.size(); i++) {
        vector<int>::iterator find_left, find_right;
        find_left = find(confidence_max_left_already_in.begin(), confidence_max_left_already_in.end(), pair_idx[i].x);
        find_right = find(confidence_max_right_already_in.begin(), confidence_max_right_already_in.end(), pair_idx[i].y);
        if (find_left == confidence_max_left_already_in.end() && find_right == confidence_max_right_already_in.end()) {
            confidence_max_left_already_in.push_back(pair_idx[i].x);
            confidence_max_right_already_in.push_back(pair_idx[i].y);
        }
    }

    vector<Point> pair_point_idx;
    for (size_t i = 0; i < confidence_max_right_already_in.size(); i++) {
        Point pair;
        pair.x = confidence_max_left_already_in[i];
        pair.y = confidence_max_right_already_in[i];
        pair_point_idx.push_back(pair);
    }

    return pair_point_idx;
}