// #include "superpoint.h"


// torch::nn::Conv2dOptions conv_options(int64_t in_channels, int64_t out_channels, int64_t kernel_size=3,
//                           int64_t stride=1, int64_t padding=1){
//     torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size);
//     conv_options.stride(stride);
//     conv_options.padding(padding);
//     return conv_options;
// }

// torch::nn::BatchNormOptions bn_options(int64_t features){
//     torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(features);
//     // bn_options.affine(true);
//     return bn_options;
// }

// torch::nn::MaxPool2dOptions mxp_options(int64_t kernel_size,int64_t stride) {
// 	torch::nn::MaxPool2dOptions mxp_options = torch::nn::MaxPool2dOptions(kernel_size);
// 	mxp_options.stride(stride);
// 	return mxp_options;
// }

// struct EmptyLayer : torch::nn::Module
// {
//     EmptyLayer(){
        
//     }

//     torch::Tensor forward(torch::Tensor x) {
//         return x; 
//     }
// };

// struct UpsampleLayer : torch::nn::Module
// {
// 	int _stride;
//     UpsampleLayer(int stride){
//         _stride = stride;
//     }

//     torch::Tensor forward(torch::Tensor x) {

//     	torch::IntList sizes = x.sizes();

//     	int64_t w, h;

//     	if (sizes.size() == 4)
//     	{
//     		w = sizes[2] * _stride;
//     		h = sizes[3] * _stride;

// 			x = torch::upsample_nearest2d(x, {w, h});
//     	}
//     	else if (sizes.size() == 3)
//     	{
// 			w = sizes[2] * _stride;
// 			x = torch::upsample_nearest1d(x, {w});
//     	}   	
//         return x; 
//     }
// };

// struct MaxPoolLayer2D : torch::nn::Module
// {
// 	int _kernel_size;
// 	int _stride;
//     MaxPoolLayer2D(int kernel_size, int stride){
//         _kernel_size = kernel_size;
//         _stride = stride;
//     }

//     torch::Tensor forward(torch::Tensor x) {	
//     	if (_stride != 1)
//     	{
//     		x = torch::max_pool2d(x, {_kernel_size, _kernel_size}, {_stride, _stride});
//     	}
//     	else
//     	{
//     		int pad = _kernel_size - 1;

//        		torch::Tensor padded_x = torch::replication_pad2d(x, {0, pad, 0, pad});
//     		x = torch::max_pool2d(padded_x, {_kernel_size, _kernel_size}, {_stride, _stride});
//     	}       

//         return x;
//     }
// };

// struct DetectionLayer : torch::nn::Module
// {
// 	vector<float> _anchors;

//     DetectionLayer(vector<float> anchors)
//     {
//         _anchors = anchors;
//     }
    
//     torch::Tensor forward(torch::Tensor prediction, int inp_dim, int num_classes, torch::Device device)
//     {
//     	return predict_transform(prediction, inp_dim, _anchors, num_classes, device);
//     }

//     torch::Tensor predict_transform(torch::Tensor prediction, int inp_dim, vector<float> anchors, int num_classes, torch::Device device)
//     {
//     	int batch_size = prediction.size(0);
//     	int stride = floor(inp_dim / prediction.size(2));
//     	int grid_size = floor(inp_dim / stride);
//     	int bbox_attrs = 5 + num_classes;
//     	int num_anchors = anchors.size()/2;

//     	for (int i = 0; i < anchors.size(); i++)
//     	{
//     		anchors[i] = anchors[i]/stride;
//     	}
//     	torch::Tensor result = prediction.view({batch_size, bbox_attrs * num_anchors, grid_size * grid_size});
//     	result = result.transpose(1,2).contiguous();
//     	result = result.view({batch_size, grid_size*grid_size*num_anchors, bbox_attrs});
    	
//     	result.select(2, 0).sigmoid_();
//         result.select(2, 1).sigmoid_();
//         result.select(2, 4).sigmoid_();

//         auto grid_len = torch::arange(grid_size);

//         std::vector<torch::Tensor> args = torch::meshgrid({grid_len, grid_len});

//         torch::Tensor x_offset = args[1].contiguous().view({-1, 1});
//         torch::Tensor y_offset = args[0].contiguous().view({-1, 1});

//         // std::cout << "x_offset:" << x_offset << endl;
//         // std::cout << "y_offset:" << y_offset << endl;

//         x_offset = x_offset.to(device);
//         y_offset = y_offset.to(device);

//         auto x_y_offset = torch::cat({x_offset, y_offset}, 1).repeat({1, num_anchors}).view({-1, 2}).unsqueeze(0);
//         result.slice(2, 0, 2).add_(x_y_offset);

//         torch::Tensor anchors_tensor = torch::from_blob(anchors.data(), {num_anchors, 2});
//         //if (device != nullptr)
//         	anchors_tensor = anchors_tensor.to(device);
//         anchors_tensor = anchors_tensor.repeat({grid_size*grid_size, 1}).unsqueeze(0);

//         result.slice(2, 2, 4).exp_().mul_(anchors_tensor);
//         result.slice(2, 5, 5 + num_classes).sigmoid_();
//    		result.slice(2, 0, 4).mul_(stride);

//     	return result;
//     }
// };


// //---------------------------------------------------------------------------
// // SuperPoint
// //---------------------------------------------------------------------------

// SuperPoint::SuperPoint(const char *cfg_file, torch::Device *device="cpu") {

// 	load_cfg(cfg_file);
//     _device = device;
// 	// create_modules();
// 	relu = register_module("relu", torch::nn::ReLU())
// 	pool = torch::nn::MaxPool2d(mxp_options(2, 2));
// 	conv1a =register_module("conv1a", torch::nn::Conv2d(conv_options(1, 64)));
// 	conv1b =register_module("conv1b", torch::nn::Conv2d(conv_options(64, 64)));
// 	conv2a =register_module("conv2a", torch::nn::Conv2d(conv_options(64, 64)));
// 	conv2b =register_module("conv2b", torch::nn::Conv2d(conv_options(64, 64)));
// 	conv3a =register_module("conv3a", torch::nn::Conv2d(conv_options(64, 128)));
// 	conv3b =register_module("conv3b", torch::nn::Conv2d(conv_options(128, 128)));
// 	conv4a =register_module("conv4a", torch::nn::Conv2d(conv_options(128, 128)));
// 	conv4b =register_module("conv4b", torch::nn::Conv2d(conv_options(128, 128)));
// 	convPa =register_module("convPa", torch::nn::Conv2d(conv_options(128, 256)));
// 	convDa =register_module("convDa", torch::nn::Conv2d(conv_options(128, 256)));
// 	convPb =register_module("convPb", torch::nn::Conv2d(conv_options(256, 65, 1, 1, 0)));
// 	load_weights();
// }

// void SuperPoint::load_cfg(const char *cfg_file)
// {
// 	ifstream fs(cfg_file);
// 	string line;
 
// 	if(!fs) 
// 	{
// 		std::cout << "Fail to load cfg file:" << cfg_file << endl;
// 		return;
// 	}

// 	while (getline (fs, line))
// 	{ 
// 		trim(line);

// 		if (line.empty())
// 		{
// 			continue;
// 		}		

// 		if ( line.substr (0,1)  == "[")
// 		{
// 			map<string, string> block;			

// 			string key = line.substr(1, line.length() -2);
// 			block["type"] = key;  

// 			blocks.push_back(block);
// 		}
// 		else
// 		{
// 			map<string, string> *block = &blocks[blocks.size() -1];

// 			vector<string> op_info;

// 			split(line, op_info, "=");

// 			if (op_info.size() == 2)
// 			{
// 				string p_key = op_info[0];
// 				string p_value = op_info[1];
// 				block->operator[](p_key) = p_value;
// 			}			
// 		}				
// 	}
// 	fs.close();
// }

// void SuperPoint::create_modules()
// {
// 	torch::nn::Sequential module;

// 	// relu = register_module("relu",torch::nn::ReLU())
// 	// pool = torch::nn::MaxPool2d(mxp_options(2,2));
// 	// conv1a =register_module("conv1a", torch::nn::Conv2d(conv_options(1, 64)));
// 	// conv1b =register_module("conv1b", torch::nn::Conv2d(conv_options(64, 64)));
// 	// conv2a =register_module("conv2a",torch::nn::Conv2d(conv_options(64,64))); 
// 	// conv2b =register_module("conv2b",torch::nn::Conv2d(conv_options(64,64))); 
// 	// conv3a =register_module("conv3a",torch::nn::Conv2d(conv_options(64,128))); 
// 	// conv3b =register_module("conv3b",torch::nn::Conv2d(conv_options(128,128))); 
// 	// conv4a =register_module("conv4a",torch::nn::Conv2d(conv_options(128,128))); 
// 	// conv4b =register_module("conv4b",torch::nn::Conv2d(conv_options(128,128))); 
// 	// convPa =register_module("convPa",torch::nn::Conv2d(conv_options(128, 256))); 
// 	// convDa =register_module("convDa",torch::nn::Conv2d(conv_options(128, 256))); 
// 	// convPb =register_module("convPb",torch::nn::Conv2d(conv_options(256, 65, 1, 1, 0))); 
// 	// bn=register_module("bn", torch::nn::BatchNorm2d(bn_options()));

// }

// map<string, string>* SuperPoint::get_net_info()
// {
// 	if (blocks.size() > 0)
// 	{
// 		return &blocks[0];
// 	}
// }

// void SuperPoint::load_weights(const char *weight_file)
// {
// 	ifstream fs(weight_file, ios::binary);

// 	// header info: 5 * int32_t
// 	int32_t header_size = sizeof(int32_t)*5;

// 	int64_t index_weight = 0;

// 	fs.seekg (0, fs.end);
//     int64_t length = fs.tellg();
//     // skip header
//     length = length - header_size;

//     fs.seekg (header_size, fs.beg);
//     float *weights_src = (float *)malloc(length);
//     fs.read(reinterpret_cast<char*>(weights_src), length);

//     fs.close();

//     at::TensorOptions options= torch::TensorOptions()
//         .dtype(torch::kFloat32)
//         .is_variable(true);
//     at::Tensor weights = torch::from_blob(weights_src, {length/4});

// 	for (int i = 0; i < module_list.size(); i++)
// 	{
// 		map<string, string> module_info = blocks[i + 1];

// 		string module_type = module_info["type"];

// 		// only conv layer need to load weight
// 		if (module_type != "convolutional")	continue;
		
// 		torch::nn::Sequential seq_module = module_list[i];

// 		auto conv_module = seq_module.ptr()->ptr(0);
// 		torch::nn::Conv2dImpl *conv_imp = dynamic_cast<torch::nn::Conv2dImpl *>(conv_module.get());

// 		int batch_normalize = get_int_from_cfg(module_info, "batch_normalize", 0);

// 		if (batch_normalize > 0)
// 		{
// 			// second module
// 			auto bn_module = seq_module.ptr()->ptr(1);

// 			torch::nn::BatchNormImpl *bn_imp = dynamic_cast<torch::nn::BatchNormImpl *>(bn_module.get());

// 			int num_bn_biases = bn_imp->bias.numel();

// 			at::Tensor bn_bias = weights.slice(0, index_weight, index_weight + num_bn_biases);
// 			index_weight += num_bn_biases;
	
// 			at::Tensor bn_weights = weights.slice(0, index_weight, index_weight + num_bn_biases);
// 			index_weight += num_bn_biases;

// 			at::Tensor bn_running_mean = weights.slice(0, index_weight, index_weight + num_bn_biases);
// 			index_weight += num_bn_biases;

// 			at::Tensor bn_running_var = weights.slice(0, index_weight, index_weight + num_bn_biases);
// 			index_weight += num_bn_biases;

// 			bn_bias = bn_bias.view_as(bn_imp->bias);
// 			bn_weights = bn_weights.view_as(bn_imp->weight);
// 			bn_running_mean = bn_running_mean.view_as(bn_imp->running_mean);
//             bn_running_var = bn_running_var.view_as(bn_imp->running_var);

// 			bn_imp->bias.set_data(bn_bias);
// 			bn_imp->weight.set_data(bn_weights);
// 			bn_imp->running_mean.set_data(bn_running_mean);
//             bn_imp->running_var.set_data(bn_running_var);
// 		}
// 		else
// 		{
// 			int num_conv_biases = conv_imp->bias.numel();

// 			at::Tensor conv_bias = weights.slice(0, index_weight, index_weight + num_conv_biases);
// 			index_weight += num_conv_biases;

// 			conv_bias = conv_bias.view_as(conv_imp->bias);
// 			conv_imp->bias.set_data(conv_bias);
// 		}		

// 		int num_weights = conv_imp->weight.numel();
	
// 		at::Tensor conv_weights = weights.slice(0, index_weight, index_weight + num_weights);
// 		index_weight += num_weights;	

// 		conv_weights = conv_weights.view_as(conv_imp->weight);
// 		conv_imp->weight.set_data(conv_weights);
// 	}
// }

// torch::Tensor SuperPoint::forward(torch::Tensor x) 
// {
// 	int module_count = module_list.size();

// 	std::vector<torch::Tensor> outputs(module_count);

// 	torch::Tensor result;
// 	int write = 0;

// 	for (int i = 0; i < module_count; i++)
// 	{
// 		map<string, string> block = blocks[i+1];

// 		string layer_type = block["type"];

// 		if (layer_type == "net")
// 			continue;

// 		if (layer_type == "convolutional" || layer_type == "upsample" || layer_type == "maxpool")
// 		{
// 			torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *>(module_list[i].ptr().get());
			
// 			x = seq_imp->forward(x);
// 			outputs[i] = x;
// 		}
// 		else if (layer_type == "route")
// 		{
// 			int start = std::stoi(block["start"]);
// 			int end = std::stoi(block["end"]);

// 			if (start > 0) start = start - i;

// 			if (end == 0)
// 			{
// 				x = outputs[i + start];
// 			}
// 			else
// 			{
// 				if (end > 0) end = end - i;

// 				torch::Tensor map_1 = outputs[i + start];
// 				torch::Tensor map_2 = outputs[i + end];

// 				x = torch::cat({map_1, map_2}, 1);
// 			}

// 			outputs[i] = x;
// 		}
// 		else if (layer_type == "shortcut")
// 		{
// 			int from = std::stoi(block["from"]);
// 			x = outputs[i-1] + outputs[i+from];
//             outputs[i] = x;
// 		}
// 		else if (layer_type == "yolo")
// 		{
// 			torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *>(module_list[i].ptr().get());

// 			map<string, string> net_info = blocks[0];
// 			int inp_dim = get_int_from_cfg(net_info, "height", 0);
// 			int num_classes = get_int_from_cfg(block, "classes", 0);

// 			x = seq_imp->forward(x, inp_dim, num_classes, *_device);

// 			if (write == 0)
// 			{
// 				result = x;
// 				write = 1;
// 			}
// 			else
// 			{
// 				result = torch::cat({result,x}, 1);
// 			}

// 			outputs[i] = outputs[i-1];
// 		}
// 	}
// 	return result;
// }

// torch::Tensor SuperPoint::write_results(torch::Tensor prediction, int num_classes, float confidence, float nms_conf)
// {
// 	// get result which object confidence > threshold
// 	auto conf_mask = (prediction.select(2,4) > confidence).to(torch::kFloat32).unsqueeze(2);
	
// 	prediction.mul_(conf_mask);
// 	auto ind_nz = torch::nonzero(prediction.select(2, 4)).transpose(0, 1).contiguous();	

// 	if (ind_nz.size(0) == 0) 
// 	{
//         return torch::zeros({0});
//     }

// 	torch::Tensor box_a = torch::ones(prediction.sizes(), prediction.options());
// 	// top left x = centerX - w/2
// 	box_a.select(2, 0) = prediction.select(2, 0) - prediction.select(2, 2).div(2);
// 	box_a.select(2, 1) = prediction.select(2, 1) - prediction.select(2, 3).div(2);
// 	box_a.select(2, 2) = prediction.select(2, 0) + prediction.select(2, 2).div(2);
// 	box_a.select(2, 3) = prediction.select(2, 1) + prediction.select(2, 3).div(2);

//     prediction.slice(2, 0, 4) = box_a.slice(2, 0, 4);

//     int batch_size = prediction.size(0);
//     int item_attr_size = 5;

//     torch::Tensor output = torch::ones({1, prediction.size(2) + 1});
//     bool write = false;

//     int num = 0;

//     for (int i = 0; i < batch_size; i++)
//     {
//     	auto image_prediction = prediction[i];

//     	// get the max classes score at each result
//     	std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(image_prediction.slice(1, item_attr_size, item_attr_size + num_classes), 1);

//     	// class score
//     	auto max_conf = std::get<0>(max_classes);
//     	// index
//     	auto max_conf_score = std::get<1>(max_classes);
//     	max_conf = max_conf.to(torch::kFloat32).unsqueeze(1);
//     	max_conf_score = max_conf_score.to(torch::kFloat32).unsqueeze(1);

//     	// shape: n * 7, left x, left y, right x, right y, object confidence, class_score, class_id
//     	image_prediction = torch::cat({image_prediction.slice(1, 0, 5), max_conf, max_conf_score}, 1);
    	
//     	// remove item which object confidence == 0
//         auto non_zero_index =  torch::nonzero(image_prediction.select(1,4));
//         auto image_prediction_data = image_prediction.index_select(0, non_zero_index.squeeze()).view({-1, 7});

//         // get unique classes 
//         std::vector<torch::Tensor> img_classes;

// 	    for (int m = 0, len = image_prediction_data.size(0); m < len; m++) 
// 	    {
// 	    	bool found = false;	        
// 	        for (int n = 0; n < img_classes.size(); n++)
// 	        {
// 	        	auto ret = (image_prediction_data[m][6] == img_classes[n]);
// 	        	if (torch::nonzero(ret).size(0) > 0)
// 	        	{
// 	        		found = true;
// 	        		break;
// 	        	}
// 	        }
// 	        if (!found) img_classes.push_back(image_prediction_data[m][6]);
// 	    }

//         for (int k = 0; k < img_classes.size(); k++)
//         {
//         	auto cls = img_classes[k];

//         	auto cls_mask = image_prediction_data * (image_prediction_data.select(1, 6) == cls).to(torch::kFloat32).unsqueeze(1);
//         	auto class_mask_index =  torch::nonzero(cls_mask.select(1, 5)).squeeze();

//         	auto image_pred_class = image_prediction_data.index_select(0, class_mask_index).view({-1,7});
//         	// ascend by confidence
//         	// seems that inverse method not work
//         	std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(image_pred_class.select(1,4));

//         	auto conf_sort_index = std::get<1>(sort_ret);
        	
//         	// seems that there is something wrong with inverse method
//         	// conf_sort_index = conf_sort_index.inverse();

//         	image_pred_class = image_pred_class.index_select(0, conf_sort_index.squeeze()).cpu();

//            	for(int w = 0; w < image_pred_class.size(0)-1; w++)
//         	{
//         		int mi = image_pred_class.size(0) - 1 - w;

//         		if (mi <= 0)
//         		{
//         			break;
//         		}

//         		auto ious = get_bbox_iou(image_pred_class[mi].unsqueeze(0), image_pred_class.slice(0, 0, mi));

//         		auto iou_mask = (ious < nms_conf).to(torch::kFloat32).unsqueeze(1);
//         		image_pred_class.slice(0, 0, mi) = image_pred_class.slice(0, 0, mi) * iou_mask;

//         		// remove from list
//         		auto non_zero_index = torch::nonzero(image_pred_class.select(1,4)).squeeze();
//         		image_pred_class = image_pred_class.index_select(0, non_zero_index).view({-1,7});
//         	}
        	
//         	torch::Tensor batch_index = torch::ones({image_pred_class.size(0), 1}).fill_(i);

//         	if (!write)
//         	{
//         		output = torch::cat({batch_index, image_pred_class}, 1);
//         		write = true;
//         	}
//         	else
//         	{
//         		auto out = torch::cat({batch_index, image_pred_class}, 1);
//         		output = torch::cat({output,out}, 0);
//         	}

//         	num += 1;
//         }
//     }

//     if (num == 0)
//     {
//     	return torch::zeros({0});
//     }

//     return output;
// }