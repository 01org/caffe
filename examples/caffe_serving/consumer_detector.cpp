#include <caffe/caffe.hpp>
#include "caffe/data_transformer.hpp"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <thread>

#include "common.h"

using namespace caffe; 

struct detect_result {
	int imgid;
	string classlabel;
	float confidence;
	float left;
	float right;
	float top;
	float bottom;
};

enum ColorFormat {
	VA_RGB = 0,
	VA_BGR = 1
};

template <typename Dtype>
class Detector {
public:
	Detector(const string& model_file,
		const string& weights_file,
		const string& label_file,
		int gpu,
		int batch_size);
	void Preprocess(const vector<cv::Mat> &imgs);

	void Detect(vector<vector<detect_result>> &result,		
		int wait_key,
		bool visualize = false);

	void ShowResult(const vector<cv::Mat> &imgs,
		const vector<vector<detect_result>> objects,
		int wait_key);

	~Detector() {
		//delete input_blobs_;
		delete data_transformer_;
	}

private:
	shared_ptr<Net<Dtype> > net_;
	cv::Size input_blob_size_;
	cv::Size image_size_;
	int num_channels_;
	int batch_size_;
	bool use_yolo_format_;
	Blob<Dtype>* input_blobs_;
	const vector<cv::Mat> *origin_imgs_;
	DataTransformer<Dtype> *data_transformer_;
	ColorFormat input_color_format_;
	vector<string> labels_;
	Blob<Dtype> *input_blob_;
};

Detector<half> *detectorHalfPtr;
Detector<float> *detectorFloatPtr;

template <typename Dtype>
Detector<Dtype>::Detector(const string& model_file,
	const string& weights_file,
	const string& label_file,
	int gpu,
	int batch_size) {
	// Set device id and mode
	if (gpu != -1) {
		std::cout << "Use GPU with device ID " << gpu << std::endl;
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(gpu);
	}
	else {
		std::cout << "Use CPU" << std::endl;
		Caffe::set_mode(Caffe::CPU);
	}
	batch_size_ = batch_size;
	/* Load the network. */
	net_.reset(new Net<Dtype>(model_file, TEST, Caffe::GetDefaultDevice()));
	net_->CopyTrainedLayersFrom(weights_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<Dtype>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_blob_size_ = cv::Size(input_layer->width(), input_layer->height());

	// Check whether the model we will using.
	// Different models need different preprocessing parameters.
	const shared_ptr<Layer<Dtype>> output_layer = net_->layers().back();
	TransformationParameter transform_param;
	caffe::ResizeParameter *resize_param = transform_param.mutable_resize_param();
	if (output_layer->layer_param().type() == "YoloDetectionOutput") {
		use_yolo_format_ = !output_layer->layer_param().
			yolo_detection_output_param().ssd_format();
		resize_param->set_resize_mode(caffe::ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD);
		resize_param->add_pad_value(127.5);
		transform_param.set_scale(1. / 255.);
		transform_param.set_force_color(true);
		std::cout << "Using Yolo: " << net_->name() << std::endl;
		input_color_format_ = VA_RGB;
	}
	else if (output_layer->layer_param().type() == "DetectionOutput") {
		use_yolo_format_ = false;
		resize_param->set_resize_mode(caffe::ResizeParameter_Resize_mode_WARP);
		if (net_->name().find("MobileNet") != std::string::npos) {
			transform_param.add_mean_value(127.5);
			transform_param.add_mean_value(127.5);
			transform_param.add_mean_value(127.5);
			transform_param.set_scale(1. / 127.5);
			std::cout << "Using SSD(MobileNet)." << std::endl;
		}
		else {
			// For standard SSD VGG or DSOD
			transform_param.add_mean_value(104);
			transform_param.add_mean_value(117);
			transform_param.add_mean_value(123);
			std::cout << "Using SSD : " << net_->name() << std::endl;
		}
		input_color_format_ = VA_BGR;
	}
	else {
		std::cerr << "The model is not a valid object detection model."
			<< std::endl;
		exit(-1);
	}
	resize_param->set_width(input_blob_size_.width);
	resize_param->set_height(input_blob_size_.height);
	resize_param->set_prob(1.0);
	resize_param->add_interp_mode(caffe::ResizeParameter_Interp_mode_LINEAR);
	data_transformer_ = new DataTransformer<Dtype>(transform_param,
		TEST,
		Caffe::GetDefaultDevice());

	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (getline(labels, line))
		labels_.push_back(string(line));
}

template <typename Dtype>
void Detector<Dtype>::Preprocess(const vector<cv::Mat> &imgs) {
	/*if (imgs.size() == 0)
	return;*/
	origin_imgs_ = &imgs;
	int batch_id = 0;

	image_size_.width = imgs[0].cols;
	image_size_.height = imgs[0].rows;

	vector<cv::Mat> batch_imgs(imgs);
	FixupChannels(batch_imgs, num_channels_, input_color_format_);
	Blob<Dtype> * blob = new Blob<Dtype>;
	int batch_size = batch_imgs.size();
	blob->Reshape(batch_size,
		num_channels_,
		input_blob_size_.height,
		input_blob_size_.width);
	data_transformer_->Transform(batch_imgs, blob);
	input_blobs_ = blob;
}

static void
FixupChannels(vector <cv::Mat> &imgs, int num_channels, enum ColorFormat color_format) {
	for (int i = 0; i < imgs.size(); i++) {
		/* Convert the input image to the input image format of the network. */
		cv::Mat img = imgs[i];
		if (img.channels() != num_channels) {
			cv::Mat sample;
			if (img.channels() == 3 && num_channels == 1)
				cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
			else if (img.channels() == 4 && num_channels == 1)
				cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
			else if (img.channels() == 4 && num_channels == 3)
				cv::cvtColor(img, sample,
					color_format == VA_BGR ?
					cv::COLOR_BGRA2BGR : cv::COLOR_BGRA2RGB);
			else if (img.channels() == 1 && num_channels == 3)
				cv::cvtColor(img, sample,
					color_format == VA_BGR ?
					cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2RGB);
			else {
				// Should not enter here, just in case.
				if (color_format == VA_BGR)
					sample = img;
				else
					cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
			}
			imgs[i] = sample;
		}
	}
}

// Normalized coordinate fixup for yolo
float fixup_norm_coord(float coord, float ratio) {
	if (ratio >= 1)
		return coord;
	else
		return (coord - (1. - ratio) / 2) / ratio;
}

template <typename Dtype>
void Detector<Dtype>::Detect(vector<vector<detect_result>> &all_objects,
	int wait_key,
	bool visualize) {
	int w = image_size_.width;
	int h = image_size_.height;

	Blob<Dtype>* input_layer = net_->input_blobs()[0];
	input_layer->ReshapeLike(*input_blobs_);

	net_->Reshape();

	//std::cout << batch_to_detect << std::endl;
	int batch_size = input_blobs_->num();
	input_layer->ReshapeLike(*input_blobs_);
	input_layer->ShareData(*input_blobs_);
	net_->Forward();
	Caffe::Synchronize(Caffe::GetDefaultDevice()->id());
	Blob<Dtype>* result_blob = net_->output_blobs()[0];
	const Dtype* result = result_blob->cpu_data();
	const int num_det = result_blob->height();
	for (int k = 0; k < num_det * 7; k += 7) {
		detect_result object;
		object.classlabel = labels_[(int)result[k + 1]]; //SSD
		//object.classlabel = labels_[(int)result[k + 1] + 1]; //YOLO
		object.confidence = result[k + 2];
		if (object.confidence > 0.3)
		{
			if (use_yolo_format_) {
				object.left = (int)(fixup_norm_coord((result[k + 3] -
					result[k + 5] / 2.0), float(w) / h) * w);
				object.right = (int)(fixup_norm_coord((result[k + 3] +
					result[k + 5] / 2.0), float(w) / h) * w);
				object.top = (int)(fixup_norm_coord((result[k + 4] -
					result[k + 6] / 2.0), float(h) / w) * h);
				object.bottom = (int)(fixup_norm_coord((result[k + 4] +
					result[k + 6] / 2.0), float(h) / w) * h);
			}
			else {
				object.left = (int)(result[k + 3] * w);
				object.top = (int)(result[k + 4] * h);
				object.right = (int)(result[k + 5] * w);
				object.bottom = (int)(result[k + 6] * h);
			}
			if (object.left < 0) object.left = 0;
			if (object.top < 0) object.top = 0;
			if (object.right >= w) object.right = w - 1;
			if (object.bottom >= h) object.bottom = h - 1;
			all_objects[result[k]].push_back(object);
		}
	}

	delete input_blobs_;
}

void Processor(vector<int> serverIds, int batch_size, int wait_key, bool visualize, bool halfType = false)
{
	namespace bip = boost::interprocess;

	// open Shared Memory Manager
	bip::managed_shared_memory msm(bip::open_or_create, MEMORY_NAME, bip::read_write);

	// open Shared Mat Header
	SharedImageHeader* shared_image_header = msm.find<SharedImageHeader>("MatHeader").first;

	// get header information from Shared Memory 
	cv::Mat shared;
	shared = cv::Mat(
		shared_image_header->size,
		shared_image_header->type,
		msm.get_address_from_handle(shared_image_header->handle));

	int totalimg = shared_image_header->total;
	unsigned long nrto = 0; // number of timeouts.
	int i = totalimg;
	int serverId;
	serverId = serverIds[0];
	shared_image_header->serverId = serverId;

	std::vector<cv::Mat> imgs;
	std::vector<std::string> labels;
	while (true) {
		int timeout = 10000;
		while (shared_image_header->isActive == 0) {
			timeout--;
			if (timeout == 0) {
				nrto++;
				break;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		if (timeout > 0) {
			imgs.push_back(shared.clone());
			labels.push_back(std::to_string(shared_image_header->serverId) + ":" + shared_image_header->tag);
			for (size_t j = 0; j < serverIds.size(); j++)
			{
				if (serverId == serverIds[j])
				{
					if (j + 1 > serverIds.size() - 1)
						serverId = serverIds[0];
					else
						serverId = serverIds[j + 1];
					break;
				}
			}
			shared_image_header->isActive = 0;
			shared_image_header->serverId = serverId;
		}

		if (imgs.size() == batch_size || (i < batch_size && i == imgs.size()) || batch_size == 1)
		{
			i -= imgs.size();
			if (halfType)
				detectorHalfPtr->Preprocess(imgs);
			else
				detectorFloatPtr->Preprocess(imgs);
			vector<vector<detect_result>> objects(imgs.size());
			if (halfType)
				detectorHalfPtr->Detect(objects, wait_key, visualize);
			else
				detectorFloatPtr->Detect(objects, wait_key, visualize);
			for (size_t k = 0; k < objects.size(); ++k) {
				std::cout << "---------- Detection for "
					<< labels[k] << " ----------" << std::endl;
				if (objects[k].size() > 0)
					for (int j = 0; j < objects[k].size(); j++) {

						detect_result obj = objects[k][j];
						std::stringstream ss;
						ss << std::fixed << std::setprecision(4) << obj.confidence
							<< " - \""
							<< obj.classlabel
							<< ":"
							<< std::fixed << std::setprecision(0)
							<< obj.left << "," << obj.top << "," << obj.right << "," << obj.bottom
							<< "\"";
						std::cout << ss.str() << std::endl;
					}
				else
					std::cout << "no detection" << std::endl;
			}
			imgs.clear();
			labels.clear();

			std::cout << i << std::endl;
			if (i <= 0)
				break;
		}
	}
}

int main(int argc, char** argv) 
{
	const char* keys =
		"{ server_ids       | <none> | server IDs }"
		"{ prototxt_file    | <none> | prototxt file }"
		"{ caffemodel_file  | <none> | caffemodel file }"
		"{ label_file       | <none> | label file }"
		"{ batch_size       | 1      | batch size }"
		"{ gpu              | 0      | gpu device }"
		"{ cpu              | false  | use cpu device }"
		"{ use_fp16         | false  | use fp16 forward engine }"
		;

	cv::CommandLineParser parser(argc, argv, keys);

	string serverList = parser.get<string>("server_ids");
	string model_file = parser.get<string>("prototxt_file");
	string trained_file = parser.get<string>("caffemodel_file");
	string label_file = parser.get<string>("label_file");
	int batch_size = parser.get<int>("batch_size");
	bool visualize = parser.get<bool>("visualize");
	int wait_key = parser.get<int>("wait_key");
	int gpu = parser.get<int>("gpu");
	bool cpu = parser.get<bool>("cpu");
	bool use_fp16 = parser.get<bool>("use_fp16");

	vector<int> serverIds;
	if (serverList == "")
		return -1;
	else if (serverList.size() == 1)
		serverIds.push_back(stoi(serverList));
	else
	{
		size_t pos = 0;
		string token;
		while ((pos = serverList.find(",")) != string::npos) {
			token = serverList.substr(0, pos);
			serverIds.push_back(stoi(token));
			serverList.erase(0, pos + 1);
		}
		serverIds.push_back(std::stoi(serverList));
	}	

	if (cpu)
		gpu = -1;

	bool halfType = false;
	if (use_fp16)
	{
#ifdef HAS_HALF_SUPPORT
		Detector<half> detector(model_file, trained_file, label_file, gpu, batch_size);
		detectorHalfPtr = &detector;
		halfType = true;
		Processor(serverIds, batch_size, wait_key, false, halfType);
#else
		std::cout << "fp16 is not supported." << std::endl;
#endif
	}
	if (!halfType) {
		Detector<float> detector(model_file, trained_file, label_file, gpu, batch_size);
		detectorFloatPtr = &detector;		
		Processor(serverIds, batch_size, wait_key, false);
	}

	return 0;
}