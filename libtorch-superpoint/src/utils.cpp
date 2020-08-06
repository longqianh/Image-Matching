#include "utils.h"
void hello(){
    std::cout << "Hello world!" << std::endl;
}
torch::Tensor cv2tensor(cv::Mat img){
    cv::Mat img_f;
    cv::resize(img,img_f,cv::Size(640,480));
    cv::imwrite("imgf.jpg",img_f);
    img_f.convertTo(img_f, CV_32F, 1.0 / 255);
    torch::Tensor tensor_image = torch::from_blob(img_f.data,{1,1,img_f.rows,img_f.cols} ,torch::kFloat32);
    return std::move(tensor_image);

}
    