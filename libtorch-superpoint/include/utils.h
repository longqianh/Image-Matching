#pragma once
#include <vector>
#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include <opencv2/opencv.hpp>
void hello();
torch::Tensor cv2tensor(cv::Mat img);