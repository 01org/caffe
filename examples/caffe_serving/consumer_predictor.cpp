#include <caffe/caffe.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <thread>

#include "common.h"

using namespace caffe; 

template <typename Dtype>
class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	void Classify(const std::vector<cv::Mat>& imgs, int_tp N = 1);
	std::vector<std::vector<std::pair<string, Dtype>>> predictions;

private:
	void SetMean(const string& mean_file);

	std::vector<std::vector<Dtype> > Predict(const std::vector<cv::Mat>& imgs);

	void WrapInputLayer(std::vector<Dtype *> &input_channels, int n);

	void Preprocess(const cv::Mat& img, std::vector<Dtype *> input_channels);

private:
	shared_ptr<Net<Dtype> > net_;

	cv::Size input_geometry_;
	cv::Size input_newwh_;
	int_tp num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
};

Classifier<half> *classifierHalfPtr;
Classifier<float> *classifierFloatPtr;

// Get all available GPU devices
static void get_gpus(vector<int>* gpus) {
	int count = 0;
	count = Caffe::EnumerateDevices(true);
	for (int i = 0; i < count; ++i) {
		gpus->push_back(i);
	}
}

template <typename Dtype>
Classifier<Dtype>::Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file) {
	// Set device id and mode
	vector<int> gpus;
#ifndef CPU_ONLY
	get_gpus(&gpus);
#endif
	if (gpus.size() != 0) {
		std::cout << "Use GPU with device ID " << gpus[0] << std::endl;
		Caffe::SetDevices(gpus);
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(gpus[0]);
	}
	else {
		std::cout << "Use CPU" << std::endl;
		Caffe::set_mode(Caffe::CPU);
	}

	/* Load the network. */
	net_.reset(new Net<Dtype>(model_file, TEST, Caffe::GetDefaultDevice()));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<Dtype>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->shape(1);
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<Dtype>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->shape(1))
		<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int_tp>& lhs,
	const std::pair<float, int_tp>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
template <typename Dtype>
static std::vector<int_tp> Argmax(const std::vector<Dtype>& v, int_tp N) {
	std::vector<std::pair<Dtype, int_tp> > pairs;
	for (size_t i = 0; i < v.size(); ++i) {
		pairs.push_back(std::make_pair(v[i], static_cast<int_tp>(i)));
	}
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int_tp> result;
	for (int_tp i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

template <typename Dtype>
void Classifier<Dtype>::Classify(const std::vector<cv::Mat>& imgs, int_tp N) {
	std::vector<std::vector<Dtype>> outputs = Predict(imgs);

	for (int j = 0; j < outputs.size(); ++j) {
		std::vector<Dtype> output = outputs[j];

		N = std::min<int>(labels_.size(), N);
		std::vector<int> maxN = Argmax(output, N);
		std::vector<std::pair<string, Dtype>> prediction;
		for (int i = 0; i < N; ++i) {
			int idx = maxN[i];
			prediction.push_back(std::make_pair(labels_[idx], output[idx]));
		}
		predictions.push_back(prediction);
	}
}

/* Load the mean file in binaryproto format. */
template <typename Dtype>
void Classifier<Dtype>::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.shape(1), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int_tp i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

template <typename Dtype>
std::vector<std::vector<Dtype> > Classifier<Dtype>::Predict(const std::vector<cv::Mat>& imgs) {
	Blob<Dtype>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(imgs.size(), num_channels_, input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	for (int i = 0; i < imgs.size(); ++i)
	{
		std::vector<Dtype *> input_channels;
		WrapInputLayer(input_channels, i);
		Preprocess(imgs[i], input_channels);
	}

	net_->Forward();
	Caffe::Synchronize(Caffe::GetDefaultDevice()->id());

	std::vector<std::vector<Dtype> > outputs;

	Blob<Dtype>* output_layer = net_->output_blobs()[0];
	for (int i = 0; i < output_layer->num(); ++i) {
		const Dtype* begin = output_layer->cpu_data() + i * output_layer->channels();
		const Dtype* end = begin + output_layer->channels();
		/* Copy the output layer to a std::vector */
		outputs.push_back(std::vector<Dtype>(begin, end));
	}
	return outputs;
}

template <typename Dtype>
void Classifier<Dtype>::WrapInputLayer(std::vector<Dtype *> &input_channels, int n) {
	Blob<Dtype>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	int channels = input_layer->channels();
	Dtype* input_data = input_layer->mutable_cpu_data() + n * width * height * channels;
	for (int i = 0; i < channels; ++i) {
		input_channels.push_back(input_data);
		input_data += width * height;
	}
}

template <typename Dtype>
void Classifier<Dtype>::Preprocess(const cv::Mat& img, std::vector<Dtype *> input_channels) {
	/* Convert the input image to the input image format of the network. */

	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2RGB);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2RGB);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	for (int i = 0; i < sample_normalized.rows; i++) {
		for (int j = 0; j < sample_normalized.cols; j++) {
			int pos = i * input_geometry_.width + j;
			if (num_channels_ == 3) {
				cv::Vec3f pixel = sample_normalized.at<cv::Vec3f>(i, j);
				input_channels[0][pos] = pixel.val[0];
				input_channels[1][pos] = pixel.val[1];
				input_channels[2][pos] = pixel.val[2];
			}
			else {
				cv::Scalar pixel = sample_normalized.at<float>(i, j);
				input_channels[0][pos] = pixel.val[0];
			}
		}
	}

}

void Processor(vector<int> serverIds, int batch_size, bool halfType = false)
{
	namespace bip = boost::interprocess;

	std::vector<cv::Mat> imgs;
	std::vector<std::string> labels;

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
	unsigned long nrto = 0; // number of timeouts
	int i = totalimg;
	int serverId;
	serverId = serverIds[0];
	shared_image_header->serverId = serverId;

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
			if (halfType) {
				classifierHalfPtr->predictions.clear();
				classifierHalfPtr->Classify(imgs);
				/* Print the top N predictions. */
				for (size_t k = 0; k < classifierHalfPtr->predictions.size(); ++k) {
					std::cout << "---------- Prediction for "
						<< labels[k] << " ----------" << std::endl;

					std::vector<std::pair<string, half>>& predictions = classifierHalfPtr->predictions[k];
					for (size_t j = 0; j < predictions.size(); ++j) {
						std::pair<string, half> p = predictions[j];
						std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
							<< p.first << "\"" << std::endl;
					}
				}
			}
			else {
				classifierFloatPtr->predictions.clear();
				classifierFloatPtr->Classify(imgs);
				/* Print the top N predictions. */
				for (size_t k = 0; k < classifierFloatPtr->predictions.size(); ++k) {
					std::cout << "---------- Prediction for "
						<< labels[k] << " ----------" << std::endl;

					std::vector<std::pair<string, float>>& predictions = classifierFloatPtr->predictions[k];
					for (size_t j = 0; j < predictions.size(); ++j) {
						std::pair<string, float> p = predictions[j];
						std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
							<< p.first << "\"" << std::endl;
					}
				}
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
		"{ binaryproto_file | <none> | binaryproto file }"
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
	string mean_file = parser.get<string>("binaryproto_file");
	string label_file = parser.get<string>("label_file");
	int batch_size = parser.get<int>("batch_size");	
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
		Classifier<half> classifier(model_file, trained_file, mean_file, label_file);
		classifierHalfPtr = &classifier;
		halfType = true;
		Processor(serverIds, batch_size, halfType);
#else
		std::cout << "fp16 is not supported." << std::endl;
#endif
	}
	if (!halfType) {
		Classifier<float> classifier(model_file, trained_file, mean_file, label_file);
		classifierFloatPtr = &classifier;
		Processor(serverIds, batch_size);
	}

	return 0;
}