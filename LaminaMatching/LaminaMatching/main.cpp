#include<iostream>
#include<opencv.hpp>
#include<stdlib.h>
#include <typeinfo>
#include <vector>
#include <stdio.h>
#include <iomanip>

#include "LaminaMatchByDL.h"

using namespace cv;
using namespace std;


int main() {
    std::string IR_xml = "C:\\Users\\11443\\Desktop\\Projects\\LaminaMatching\\deployment\\Unet2D_test.xml";
    std::string IR_bin = "C:\\Users\\11443\\Desktop\\Projects\\LaminaMatching\\deployment\\Unet2D_test.bin";
    std::string image_root = "C:\\Users\\11443\\Desktop\\dataset\\jizhui_point\\all image\\≤‚ ‘Õº∆¨170\\HKUSZH00100015220190222002P4.bmp";
    
    LaminaMatchByDL my_model(IR_xml, IR_bin);
    
    Mat image = imread(image_root);
    cvtColor(image, image, COLOR_BGR2GRAY);
     
    vector<vector<Point>> pair;
    pair = my_model.predict(image);

    return 0;
}