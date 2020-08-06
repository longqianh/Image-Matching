#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <string.h>
#include <typeinfo>
#include <vector>
// #include <memory>
using namespace std;
int main(int argc, char *argv[])
{
	hello();
    string sp_path="../assets/superpoint.pt";
    torch::jit::script::Module* sp_module =new torch::jit::script::Module(torch::jit::load(sp_path));

    string imgpath="../assets/anchor.jpg";
    auto img=cv::imread(imgpath, cv::IMREAD_GRAYSCALE);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(cv2tensor(img));
    auto res = sp_module->forward(inputs).toTuple();
    auto scores=res->elements()[0];
    auto keypoints=res->elements()[1];
    auto descriptors=res->elements()[2];


   

    return 0;

}