#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "caffe/data_transformer.hpp"
#include <caffe/caffe.hpp>

using namespace caffe;

int navSliderIdx, navSliderMax, confSliderIdx, nmsSliderIdx;

struct label_map {
	map<string, int> groupToIndex;
	map<int, string> indexToLabel;
	map<string, int> labelToIndex;
	string activeGroup;
};

vector<cv::Rect> prevBBoxes;
vector<float> prevScores;

int wait_key, restore_key;

bool halfType;

struct class_rect {
	int index;
	map<string, cv::Rect> groupToRect;
	map<string, cv::Rect> labelToRect;
	bool timeout;
	cv::Mat image;
};

struct detect_result {
	int classid;
	string classlabel;
	float confidence;
	float left;
	float right;
	float top;
	float bottom;
	cv::Rect labelrect;
};

enum DrawComponent {
	DrwLine = 0,
	DrwMask = 1,
	DrwCaption = 2,
	DrwBBox = 3,
	DrwCapBBox = 4,
	DrwHelp = 5
};

enum SaveComponent {
	SvImage = 0,
	SvCrop = 1,
	SvSSD = 2,
	SvYOLO = 3
};

enum SourceType {
	SrcImage = 0,
	SrcVideo = 1,
	SrcStream = 2
};

enum OperationMode {
	OpNorm = 0,
	OpCtrl = 1,
	OpNew = 2,
	OpEdit = 3
};

struct processor_state {
	SourceType sourceType;
	string sourceDir;
	vector<string> sourceName;
	string sourceExt;
	string sourceURI;
	cv::VideoCapture cap;
	int skipCount;
	int imageIndex;
	double videoIndex;
	bool supportAutoDetect;
	bool autoDetect;
	map<string, string> autoDetectMap;
	bool priorFilter;
	float confThresh;
	float nmsThresh;
	string handleName;
	cv::Mat image;
	cv::Mat overlay;
	string maskFileName;
	cv::Mat mask;
	vector<cv::Point> maskPoints;
	vector<vector<cv::Point>> maskPointsList;
	vector<detect_result> objects;
	int objIndexToEdit;
	string exportPrefix;
	bool showHelp;
	bool preProcess;
	string exportDir;
	bool exportImage;
	bool exportCrop;
	bool exportSSD;
	bool exportYOLO;
	label_map labelMap;
	bool committed;
	OperationMode mode;	
};

processor_state *processorStatePtr;

void ShowResult(bool redraw);
void Processor();
void Preprocessor();

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
		bool deep_detect);
	void Preprocess(const cv::Mat &imgs);

	void Detect(bool priorFilter, cv::Mat &mask,
		vector<detect_result> &result,
		float confTresh,
		float nmsTresh);

	~Detector() {
		for (size_t i = 0; i < input_blobs_.size(); ++i)
			delete input_blobs_[i];
		delete data_transformer_;
	}

private:
	shared_ptr<Net<Dtype> > net_;
	cv::Size input_blob_size_;
	cv::Size image_size_;
	int num_channels_;
	int batch_size_;
	bool use_yolo_format_;
	vector<Blob<Dtype>*> input_blobs_;
	const cv::Mat *origin_imgs_;
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
	bool auto_detect) {

	if (auto_detect) {
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
		batch_size_ = 1;
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
}

template <typename Dtype>
void Detector<Dtype>::Preprocess(const cv::Mat &imgs) {
	origin_imgs_ = &imgs;
	int batch_id = 0;

	image_size_.width = imgs.cols;
	image_size_.height = imgs.rows;
	int batch_count = batch_size_;
	input_blob_ = new Blob<Dtype>;
	input_blob_->Reshape(batch_size_,
		num_channels_,
		input_blob_size_.height,
		input_blob_size_.width);
	int pos = batch_id * batch_size_;
	cv::Mat img(imgs);
	FixupChannels(img, num_channels_, input_color_format_);
	data_transformer_->Transform(img, input_blob_);
}

static void
FixupChannels(cv::Mat &imgs, int num_channels,
	enum ColorFormat color_format) {
	cv::Mat img = imgs;
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
		imgs = sample;
	}
}

template <typename Dtype>
void Detector<Dtype>::Detect(bool priorFilter, cv::Mat &mask,
	vector<detect_result> &all_objects,
	float conf_thresh, float nms_thresh)
{
	Blob<Dtype>* input_layer = net_->input_blobs()[0];
	input_layer->ReshapeLike(*input_blob_);
	net_->Reshape();

	int batch_to_detect = batch_size_;
	int w = image_size_.width;
	int h = image_size_.height;

	bool maskFilter;
	maskFilter = cv::countNonZero(mask) > 0 ? true : false;
	vector<cv::Rect> bboxes;
	vector<float> scores;
	const float score_thresh = conf_thresh;
	vector<int> indices;
	for (int batch_id = 0; batch_id < batch_to_detect; batch_id++) {
		int real_batch_id = batch_id % 1;
		int batch_size = input_blob_->num();
		input_layer->ReshapeLike(*input_blob_);
		input_layer->ShareData(*input_blob_);
		net_->Forward();
		Blob<Dtype>* result_blob = net_->output_blobs()[0];
		const Dtype* result = result_blob->cpu_data();
		const int num_det = result_blob->height();
		vector<detect_result> objects;
		for (size_t i = 0; i < all_objects.size(); ++i) {
			detect_result object = all_objects[i];
			objects.push_back(object);
			cv::Rect roi(cv::Point(object.left, object.top), cv::Point(object.right, object.bottom));
			bboxes.push_back(roi);
			scores.push_back(object.confidence);
		}
		all_objects.clear();

		int count = 0;
		for (int k = 0; k < num_det * 7; k += 7) {
			detect_result object;
			object.confidence = result[k + 2];
			string label = labels_[(int)result[k + 1]];
			std::transform(label.begin(), label.end(), label.begin(), ::toupper);
			int idx;
			bool valid = false;
			auto li = processorStatePtr->labelMap.labelToIndex.find(label);
			if (li != processorStatePtr->labelMap.labelToIndex.end()) {
				idx = li->second;
				valid = true;
			}
			if (object.confidence >= conf_thresh && valid)
			{
				object.classlabel = label;
				object.classid = idx;

				detect_result paddedObject;
				if (use_yolo_format_) {
					object.left = (int)((result[k + 3] - result[k + 5] / 2.0) * w);
					object.right = (int)((result[k + 3] + result[k + 5] / 2.0) * w);
					object.top = (int)((result[k + 4] - result[k + 6] / 2.0) * h);
					object.bottom = (int)((result[k + 4] + result[k + 6] / 2.0) * h);
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

				if (maskFilter) {
					cv::Mat blob = cv::Mat::zeros(mask.size(), CV_8UC1);
					cv::rectangle(blob, cv::Point(object.left, object.top), cv::Point(object.right, object.bottom), cv::Scalar(255), -1);
					int blobUnmasked = cv::countNonZero(blob);
					int blobMasked = cv::countNonZero(blob & mask);
					if (blobMasked != blobUnmasked)
						continue;
				}

				objects.push_back(object);

				cv::Rect roi(cv::Point(object.left, object.top), cv::Point(object.right, object.bottom));
				bboxes.push_back(roi);
				scores.push_back(object.confidence);
				count++;
			}
		}

		if (!bboxes.empty()) {
			//compare IOU of curr 
			cv::dnn::NMSBoxes(bboxes, scores, score_thresh, nms_thresh, indices);
			vector<int> currIndices;
			if (bboxes.size() > indices.size())
				for (size_t i = 0; i < indices.size(); ++i)
					currIndices.push_back(indices[i]);
			else
				currIndices = indices;

			vector<int> finalIndices;

			//compare IOU of curr vs. prev
			if (priorFilter && !prevBBoxes.empty())
				for (size_t i = 0; i < currIndices.size(); ++i) {
					auto tempBBoxes = prevBBoxes;
					auto tempScores = prevScores;
					tempBBoxes.push_back(bboxes[currIndices[i]]);
					tempScores.push_back(scores[currIndices[i]]);
					indices.clear();
					cv::dnn::NMSBoxes(tempBBoxes, tempScores, score_thresh, nms_thresh, indices);
					if (tempBBoxes.size() == indices.size())
						finalIndices.push_back(currIndices[i]);
				}

			if (finalIndices.empty() && prevBBoxes.empty())
				finalIndices = currIndices;
			//filter curr bboxes
			prevBBoxes.clear();
			prevScores.clear();
			for (size_t i = 0; i < finalIndices.size(); ++i) {
				all_objects.push_back(objects[finalIndices[i]]);
				if (priorFilter) {
					prevBBoxes.push_back(bboxes[finalIndices[i]]);
					prevScores.push_back(scores[finalIndices[i]]);
				}
			}
		}
	}
	delete input_blob_;
}

void DrawOverlay(cv::Mat &img, DrawComponent draw, int idx = -1)
{
	if (draw == DrwLine || draw == DrwMask)
		if (draw == DrwLine)
			for (size_t i = 0; i < processorStatePtr->maskPoints.size(); ++i)
				cv::polylines(img, processorStatePtr->maskPoints, false, cv::Scalar(0, 0, 255), 2);
		else {
			cv::Mat parentMask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC1);
			for (size_t i = 0; i < processorStatePtr->maskPointsList.size(); ++i) {
				cv::Mat childMask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC1);
				cv::fillConvexPoly(childMask, processorStatePtr->maskPointsList[i], cv::Scalar(255));
				parentMask = parentMask | childMask;
			}
			if (cv::countNonZero(parentMask) > 0) {
				processorStatePtr->mask = parentMask;
				cv::Mat mask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC3);
				cv::cvtColor(parentMask, mask, cv::COLOR_GRAY2BGR);
				cv::addWeighted(img, 0.7, mask, 0.3, 0.0, img);
			}
		}
	else if (draw == DrwCaption || draw == DrwBBox || draw == DrwCapBBox) {
		detect_result obj = processorStatePtr->objects[idx];
		int baseLine = 0;
		string label = obj.classlabel;
		std::transform(label.begin(), label.end(), label.begin(), ::toupper);
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		cv::Rect labelRect = cv::Rect(cv::Point(obj.left, obj.top - labelSize.height - baseLine),
			cv::Size(labelSize.width + baseLine, labelSize.height + baseLine));
		bool offset = false;
		if (labelRect.y < 0) {
			labelRect = cv::Rect(cv::Point(obj.left, obj.bottom),
				cv::Size(labelSize.width + baseLine, labelSize.height + 2 + baseLine));
			offset = true;
		}
		processorStatePtr->objects[idx].labelrect = labelRect;
		if (draw == DrwCaption || draw == DrwCapBBox) {
			if (label != "UNKNOWN")
				cv::rectangle(img, labelRect, cv::Scalar(0, 0, 0), cv::FILLED);
			else
				cv::rectangle(img, labelRect, cv::Scalar(0, 0, 255), cv::FILLED);
			if (offset)
				cv::putText(img, label, cv::Point(obj.left, obj.bottom + labelSize.height - 1 + baseLine),
					cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
			else
				cv::putText(img, label, cv::Point(obj.left, obj.top - baseLine),
					cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		}
		if (draw == DrwBBox || draw == DrwCapBBox)
			cv::rectangle(img, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(255, 242, 35), 2);
		if (processorStatePtr->committed)
			cv::rectangle(img, cvRect(0, 0, img.cols - 1, img.rows - 1), cv::Scalar(0, 255, 0), 5);
	}
	else {
		vector<string> currentImage;
		currentImage.push_back("CURRENTPROCESS:");
		if (processorStatePtr->sourceType == SrcImage)
			currentImage.push_back(processorStatePtr->sourceName[processorStatePtr->imageIndex] +
				"." + processorStatePtr->sourceExt);
		else {
			if (processorStatePtr->sourceType == SrcVideo)
				currentImage.push_back(processorStatePtr->sourceName[0] +
					"." + processorStatePtr->sourceExt);
			else
				currentImage.push_back(processorStatePtr->sourceName[0]);
			currentImage.push_back(processorStatePtr->sourceName[1]);
			currentImage.push_back(processorStatePtr->sourceName[2]);
		}
		vector<string> shortKeys;
		shortKeys.push_back("SHORTKEYS:");
		if (processorStatePtr->preProcess) {
			shortKeys.push_back("(h)elp");
			shortKeys.push_back("(w)rite");
			shortKeys.push_back("(q)uit");
		}
		else {
			if (wait_key == 0) {
				shortKeys.push_back("f(i)lter");
				if (processorStatePtr->supportAutoDetect) {
					if (processorStatePtr->mode == OpCtrl)
						shortKeys.push_back("con(t)rolOff");
					else
						shortKeys.push_back("con(t)rolOn");
					if (processorStatePtr->autoDetect)
						shortKeys.push_back("(d)etectOff");
					else
						shortKeys.push_back("(d)etectOn");
				}
				if (processorStatePtr->mode == OpNew)
					shortKeys.push_back("(space)/(c)ancel");
				else
					shortKeys.push_back("(n)ew");
				if (!processorStatePtr->objects.empty())
					shortKeys.push_back("(w)rite");
				if (processorStatePtr->sourceType != SrcStream)
					shortKeys.push_back("(b)ack");
				shortKeys.push_back("(s)kip");
				shortKeys.push_back("re(f)resh");
				shortKeys.push_back("(h)elpOn/Off");
				shortKeys.push_back("(r)esume");
				shortKeys.push_back("(q)uit");
			}
			else {
				shortKeys.push_back("(h)elpOn/Off");
				shortKeys.push_back("(p)pause");
				shortKeys.push_back("(q)uit");
			}
		}
		vector<string> mouseClicks;
		mouseClicks.push_back("MOUSECLICKS:");
		if (processorStatePtr->preProcess) {
			mouseClicks.push_back("(LB)line");
			mouseClicks.push_back("(DLB)mask");
			mouseClicks.push_back("(RB)remove");
		}
		else {
			if (wait_key == 0) {
				mouseClicks.push_back("(DLB)edit");
				mouseClicks.push_back("(RB)remove");
			}
		}

		int baseLine = 0, baseLine_ = 0;
		cv::Size textSize = cv::getTextSize("(space)/(c)ancel", cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		cv::Size textSize_ = cv::getTextSize(currentImage[0].size() > currentImage[1].size() ?
			currentImage[0] : currentImage[1], cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine_);
		int baseWidth;
		if (processorStatePtr->preProcess || wait_key == 0)
			baseWidth = (textSize.width + baseLine) * 2;
		else
			baseWidth = textSize.width + baseLine;
		cv::Rect textRect = cv::Rect(cv::Point(0, 0), cv::Size(baseWidth + textSize_.width + baseLine_,
			(textSize.height + baseLine + 2) * (shortKeys.size() + 1) + shortKeys.size()));

		cv::Mat help = img(textRect);
		cv::Mat rect(help.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		int indent = baseLine;
		for (size_t i = 0; i < shortKeys.size(); ++i)
			cv::putText(rect, shortKeys[i], cv::Point(indent, (textSize.height + baseLine + 2) * (i + 1) + i),
				cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 255));
		if (processorStatePtr->preProcess || wait_key == 0) {
			indent += baseLine + textSize.width;
			for (size_t i = 0; i < mouseClicks.size(); ++i)
				cv::putText(rect, mouseClicks[i], cv::Point(indent, (textSize.height + baseLine + 2) * (i + 1) + i),
					cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 255));
		}
		indent += baseLine + textSize.width;
		for (size_t i = 0; i < currentImage.size(); ++i)
			cv::putText(rect, currentImage[i], cv::Point(indent, (textSize.height + baseLine + 2) * (i + 1) + i),
				cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 255));

		cv::addWeighted(rect, 0.8, help, 0.2, 0.0, help);
	}
}

cv::Mat DrawMenu(class_rect &classRect)
{
	int count;
	int index = 0;
	bool group = false;
	string baseLabel;
	string rootLabel = "";
	if (!processorStatePtr->labelMap.groupToIndex.empty()) {
		for (auto gi : processorStatePtr->labelMap.groupToIndex) {
			if (gi.first.size() > rootLabel.size())
				rootLabel = gi.first;
			if (gi.first == processorStatePtr->labelMap.activeGroup) {
				count = gi.second;
				group = true;
			}
			if (!group)
				index += gi.second;
		}
		for (size_t i = 0; i < processorStatePtr->labelMap.groupToIndex.size(); ++i)
			baseLabel += "X" + rootLabel;
	}
	else {
		baseLabel = "XXXXXXXXXXXXXXXX";
		count = processorStatePtr->labelMap.indexToLabel.size() - 1;
	}
	int baseLine = 0;
	cv::Size labelSize = cv::getTextSize(baseLabel, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	cv::Mat menu = cv::Mat::zeros(cv::Size(labelSize.width, (labelSize.height + baseLine) *	(count + 1)), CV_8UC1);
	map<string, cv::Rect> groupRects;
	map<string, cv::Rect> labelRects;
	int rect, text;
	if (group) {
		int i = 0;
		int j = 1;
		for (auto il : processorStatePtr->labelMap.indexToLabel) {
			if (i >= index && i < index + count) {
				if (group) {
					group = false;
					int k = 1;
					for (auto gi : processorStatePtr->labelMap.groupToIndex) {
						if (gi.first == processorStatePtr->labelMap.activeGroup) {
							rect = 125;
							text = 255;
						}
						else {
							rect = 200;
							text = 0;
						}
						float menuCount = processorStatePtr->labelMap.groupToIndex.size();
						cv::Rect labelRect = cv::Rect(cv::Point((labelSize.width / menuCount) * (k - 1), 0),
							cv::Size(labelSize.width / menuCount + 1, labelSize.height + baseLine));
						groupRects[gi.first] = labelRect;
						cv::rectangle(menu, labelRect, cv::Scalar(rect), cv::FILLED);
						cv::putText(menu, gi.first, cv::Point((labelSize.width / menuCount) * (k - 1), labelSize.height),
							cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(text));
						k++;
					}
				}
				if (j % 2 == 0) {
					rect = 255;
					text = 0;
				}
				else {
					rect = 0;
					text = 255;
				}
				cv::Rect labelRect = cv::Rect(cv::Point(0, (labelSize.height + baseLine) * (j - 1)
					+ (labelSize.height + baseLine)), cv::Size(labelSize.width, labelSize.height + baseLine));
				labelRects[il.second] = labelRect;
				cv::rectangle(menu, labelRect, cv::Scalar(rect), cv::FILLED);
				cv::putText(menu, il.second, cv::Point(0, (labelSize.height + baseLine) +
					(labelSize.height + baseLine) * j - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(text));
				j++;
			}
			i++;
		}
	}
	else {
		int i = 1;
		for (auto il : processorStatePtr->labelMap.indexToLabel) {
			if (i % 2 == 0) {
				rect = 255;
				text = 0;
			}
			else {
				rect = 0;
				text = 255;
			}
			cv::Rect labelRect = cv::Rect(cv::Point(0, (labelSize.height + baseLine) * (i - 1)),
				cv::Size(labelSize.width, labelSize.height + baseLine));
			labelRects[il.second] = labelRect;
			cv::rectangle(menu, labelRect, cv::Scalar(rect), cv::FILLED);
			cv::putText(menu, il.second, cv::Point(0, (labelSize.height + baseLine) * i - 2),
				cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(text));
			i++;
		}
	}

	classRect.groupToRect = groupRects;
	classRect.labelToRect = labelRects;

	return menu;
}

//Mouse event for PopupMenu
void MouseCallbackMenu(int event, int x, int y, int flag, void *param)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		auto menus = (class_rect*)param;
		//Validate tab selection
		bool validTab = false;
		for (auto gr : menus->groupToRect) {
			if (gr.second.contains(cv::Point(x, y)) && gr.first != processorStatePtr->labelMap.activeGroup) {
				processorStatePtr->labelMap.activeGroup = gr.first;
				validTab = true;
				break;
			}
		}
		if (validTab) {
			cv::Mat menu = DrawMenu(*menus);
			cv::imshow("Classes", menu);
		}
		//Validate menu selection
		if (!validTab) {
			bool validMenu = false;
			for (auto lr : menus->labelToRect) {
				if (lr.second.contains(cv::Point(x, y))) {
					menus->timeout = false;
					auto li = processorStatePtr->labelMap.labelToIndex.find(lr.first);
					if (li != processorStatePtr->labelMap.labelToIndex.end())
						processorStatePtr->objects[menus->index].classid = li->second;
					processorStatePtr->objects[menus->index].classlabel = lr.first;
					if (lr.first == "UNKNOWN")
						processorStatePtr->objects[menus->index].confidence = processorStatePtr->confThresh;
					else
						processorStatePtr->objects[menus->index].confidence = 1;
					validMenu = true;
					break;
				}
			}
			if (validMenu) {
				cv::destroyWindow("Classes");
				if (processorStatePtr->mode == OpEdit) {
					processorStatePtr->mode = OpNorm;
					DrawOverlay(menus->image, DrwCapBBox, menus->index);
					if (processorStatePtr->showHelp)
						DrawOverlay(menus->image, DrwHelp);
					cv::imshow(processorStatePtr->handleName, menus->image);
				}
				else
					ShowResult(true);
			}
		}
	}
}

void PopupMenu(int idx, cv::Mat &img, int x, int y, bool refresh = false)
{
	class_rect classRect;
	classRect.index = idx;
	classRect.timeout = true;
	classRect.image = img;
	cv::Mat menu = DrawMenu(classRect);

	if (!refresh) {
		cv::namedWindow("Classes", cv::WINDOW_AUTOSIZE);
		cv::setMouseCallback("Classes", MouseCallbackMenu, (void*)&classRect);
		cv::moveWindow("Classes", x, y);
	}
	cv::imshow("Classes", menu);
	cv::waitKey(3000);

	if (classRect.timeout)
		PopupMenu(idx, img, x, y, true);
}

//Mouse event for Processor
void MouseCallbackProcessor(int event, int x, int y, int flag, void *param)
{
	//Update caption or bounding box
	if (event == cv::EVENT_LBUTTONDBLCLK)
	{
		int idx;
		bool updateCap = false;
		bool updateBBox = false;
		for (size_t i = 0; i < processorStatePtr->objects.size(); ++i) {
			if (processorStatePtr->objects[i].labelrect.contains(cv::Point(x, y))) {
				idx = i;
				updateCap = true;
				break;
			}
			cv::Rect objRect = cv::Rect(cv::Point(processorStatePtr->objects[i].left, processorStatePtr->objects[i].top),
				cv::Point(processorStatePtr->objects[i].right, processorStatePtr->objects[i].bottom));
			if (objRect.contains(cv::Point(x, y))) {
				idx = i;
				updateBBox = true;
				break;
			}
		}
		if (updateCap || updateBBox) {
			cv::Mat img = processorStatePtr->image.clone();
			if (!processorStatePtr->mask.empty()) {
				cv::Mat mask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC3);
				cv::cvtColor(processorStatePtr->mask, mask, cv::COLOR_GRAY2BGR);
				cv::addWeighted(img, 0.7, mask, 0.3, 0.0, img);
			}
			for (size_t i = 0; i < processorStatePtr->objects.size(); ++i) {
				if (i == idx && updateBBox)
					DrawOverlay(img, DrwCaption, i);
				if (i != idx)
					DrawOverlay(img, DrwCapBBox, i);
			}
			if (updateCap) {
				processorStatePtr->mode = OpEdit;
				PopupMenu(idx, img, x, y);
			}
			if (updateBBox) {
				processorStatePtr->objIndexToEdit = idx;
				if (processorStatePtr->showHelp)
					DrawOverlay(img, DrwHelp);
				processorStatePtr->overlay = img;
				cv::imshow(processorStatePtr->handleName, img);
			}
		}
	}
	//Remove object
	else if (event == cv::EVENT_RBUTTONDOWN) {
		int idx;
		bool removeObj = false;
		for (size_t i = 0; i < processorStatePtr->objects.size(); ++i) {
			cv::Rect objRect = cv::Rect(cv::Point(processorStatePtr->objects[i].left, processorStatePtr->objects[i].top),
				cv::Point(processorStatePtr->objects[i].right, processorStatePtr->objects[i].bottom));
			if (objRect.contains(cv::Point(x, y))) {
				idx = i;
				removeObj = true;
				break;
			}
		}
		if (removeObj) {
			cv::Mat img = processorStatePtr->image.clone();
			if (!processorStatePtr->mask.empty()) {
				cv::Mat mask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC3);
				cv::cvtColor(processorStatePtr->mask, mask, cv::COLOR_GRAY2BGR);
				cv::addWeighted(img, 0.7, mask, 0.3, 0.0, img);
			}
			processorStatePtr->objects.erase(processorStatePtr->objects.begin() + idx);
			for (size_t i = 0; i < processorStatePtr->objects.size(); ++i)
				DrawOverlay(img, DrwCapBBox, i);
			if (processorStatePtr->showHelp)
				DrawOverlay(img, DrwHelp);
			processorStatePtr->overlay = img;
			cv::imshow(processorStatePtr->handleName, img);
		}
	}
}

//Mouse event for Preprocessor
void MouseCallbackPreprocessor(int event, int x, int y, int flag, void *param)
{
	//Accumulate ROI points
	if (event == cv::EVENT_LBUTTONDOWN) {
		processorStatePtr->maskPoints.push_back(cv::Point(x, y));
		if (processorStatePtr->maskPoints.size() > 1) {
			cv::Mat img = processorStatePtr->image.clone();
			DrawOverlay(img, DrwLine);
			if (!processorStatePtr->mask.empty()) {
				cv::Mat mask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC3);
				cv::cvtColor(processorStatePtr->mask, mask, cv::COLOR_GRAY2BGR);
				cv::addWeighted(img, 0.7, mask, 0.3, 0.0, img);
			}
			if (processorStatePtr->showHelp)
				DrawOverlay(img, DrwHelp);
			cv::imshow("Filter", img);
		}
	}
	//Commit ROI mask
	else if (event == cv::EVENT_LBUTTONDBLCLK) {
		if (processorStatePtr->maskPoints.size() > 2) {
			processorStatePtr->maskPoints.push_back(processorStatePtr->maskPoints[0]);
			cv::Mat mask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC1);
			cv::fillConvexPoly(mask, processorStatePtr->maskPoints, cv::Scalar(255));
			if (!processorStatePtr->mask.empty())
				processorStatePtr->mask = processorStatePtr->mask | mask;
			else
				processorStatePtr->mask = mask;
			processorStatePtr->maskPointsList.push_back(processorStatePtr->maskPoints);
			processorStatePtr->maskPoints.clear();
			cv::Mat img = processorStatePtr->image.clone();
			DrawOverlay(img, DrwMask);
			if (processorStatePtr->showHelp)
				DrawOverlay(img, DrwHelp);
			cv::imshow("Filter", img);
		}
	}
	//Delete ROI mask
	else if (event == cv::EVENT_RBUTTONDOWN) {
		bool valid = false;
		int idx;
		for (size_t i = 0; i < processorStatePtr->maskPointsList.size(); ++i) {
			cv::Mat mask = cv::Mat::ones(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC1) * 255;
			cv::fillConvexPoly(mask, processorStatePtr->maskPointsList[i], cv::Scalar(0));
			if (cv::pointPolygonTest(processorStatePtr->maskPointsList[i], cv::Point(x, y), false) != -1) {
				idx = i;
				valid = true;
				break;
			}
		}
		if (valid) {
			cv::Mat img = processorStatePtr->image.clone();
			processorStatePtr->maskPointsList.erase(processorStatePtr->maskPointsList.begin() + idx);
			cv::Mat parentMask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC1);
			for (size_t i = 0; i < processorStatePtr->maskPointsList.size(); ++i) {
				cv::Mat childMask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC1);
				cv::fillConvexPoly(childMask, processorStatePtr->maskPointsList[i], cv::Scalar(255));
				parentMask = parentMask | childMask;
			}
			processorStatePtr->mask = parentMask;
			DrawOverlay(img, DrwMask);
			if (processorStatePtr->showHelp)
				DrawOverlay(img, DrwHelp);
			cv::imshow("Filter", img);
		}
	}
}

void SaveResult(SaveComponent save)
{
	if (save == SvImage) {
		if (processorStatePtr->sourceType == SrcImage) {
			string src = processorStatePtr->sourceDir + "image/" + processorStatePtr->sourceName[processorStatePtr->imageIndex] + "." + processorStatePtr->sourceExt;
			string dst = processorStatePtr->exportDir + "image/" + processorStatePtr->sourceName[processorStatePtr->imageIndex] + "." + processorStatePtr->sourceExt;
			std::ifstream ifs(src, std::ios::binary);
			std::ofstream ofs(dst, std::ios::binary);
			ofs << ifs.rdbuf();
		}
		else {
			string datestamp = processorStatePtr->sourceName[1];
			datestamp.erase(std::remove(datestamp.begin(), datestamp.end(), '-'), datestamp.end());
			string timestamp = processorStatePtr->sourceName[2];
			timestamp.erase(std::remove(timestamp.begin(), timestamp.end(), ':'), timestamp.end());
			string filename = processorStatePtr->exportDir
				+ "image/"
				+ processorStatePtr->sourceName[0]
				+ "_"
				+ datestamp
				+ "-"
				+ timestamp
				+ ".jpg";
			cv::imwrite(filename, processorStatePtr->image, vector<int>({ cv::IMWRITE_JPEG_QUALITY, 100 }));
		}
	}

	if (save == SvCrop) {
		for (size_t i = 0; i < processorStatePtr->objects.size(); ++i) {
			detect_result obj = processorStatePtr->objects[i];
			cv::Rect roi(cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom));
			ostringstream filename;
			string prefix;
			if (processorStatePtr->sourceType == SrcImage)
				prefix = processorStatePtr->sourceName[processorStatePtr->imageIndex];
			else {
				string datestamp = processorStatePtr->sourceName[1];
				datestamp.erase(std::remove(datestamp.begin(), datestamp.end(), '-'), datestamp.end());
				string timestamp = processorStatePtr->sourceName[2];
				timestamp.erase(std::remove(timestamp.begin(), timestamp.end(), ':'), timestamp.end());
				prefix = processorStatePtr->sourceName[0] + "_" + datestamp + "-" + timestamp;
			}
			filename << processorStatePtr->exportDir
				<< "crop/"
				<< obj.classlabel
				<< "/"
				<< prefix
				<< "_"
				<< i
				<< ".jpg";
			cv::imwrite(filename.str(), processorStatePtr->image(roi), vector<int>({ cv::IMWRITE_JPEG_QUALITY, 100 }));
		}
	}

	if (save == SvSSD || save == SvYOLO) {
		string subdir = save == SvSSD ? "ssd/" : "yolo/";
		string filename;
		string fileext = wait_key > 0 ? (save == SvSSD ? ".ssd" : ".yolo") : ".txt";
		if (processorStatePtr->sourceType == SrcImage)
			filename = processorStatePtr->exportDir
			+ "label/"
			+ subdir
			+ processorStatePtr->sourceName[processorStatePtr->imageIndex]
			+ fileext;
		else {
			string datestamp = processorStatePtr->sourceName[1];
			datestamp.erase(std::remove(datestamp.begin(), datestamp.end(), '-'), datestamp.end());
			string timestamp = processorStatePtr->sourceName[2];
			timestamp.erase(std::remove(timestamp.begin(), timestamp.end(), ':'), timestamp.end());
			filename = processorStatePtr->exportDir
				+ "label/"
				+ subdir
				+ processorStatePtr->sourceName[0]
				+ "_"
				+ datestamp
				+ "-"
				+ timestamp
				+ fileext;
		}

		ostringstream ss;
		std::ofstream ofs(filename);
		if (ofs.is_open()) {
			for (size_t i = 0; i < processorStatePtr->objects.size(); ++i) {
				detect_result obj = processorStatePtr->objects[i];
				if (wait_key > 0)
					ss << obj.classlabel;
				else
					ss << obj.classid;
				if (save == SvYOLO) {
					float w = (obj.right - obj.left);
					float h = (obj.bottom - obj.top);
					float x = (w / 2 + obj.left) / processorStatePtr->image.cols;
					float y = (h / 2 + obj.top) / processorStatePtr->image.rows;
					w /= processorStatePtr->image.cols;
					h /= processorStatePtr->image.rows;
					ss << " "
						<< std::fixed
						<< std::setprecision(6)
						<< x
						<< " "
						<< y
						<< " "
						<< w
						<< " "
						<< h;
				}
				else
					ss << " "
					<< (int)obj.left
					<< " "
					<< (int)obj.top
					<< " "
					<< (int)obj.right
					<< " "
					<< (int)obj.bottom;

				ofs << ss.str() << "\n";
				ss.clear();
				ss.str("");
			}
			ofs.close();
		}
	}
}

void ExportResult()
{
	bool committedSSD = false, committedYOLO = false;
	if (wait_key > 0) {
		if (processorStatePtr->exportImage)
			SaveResult(SvImage);
		if (processorStatePtr->exportSSD)
			SaveResult(SvSSD);
		if (processorStatePtr->exportYOLO)
			SaveResult(SvYOLO);
	}
	else {
		if (processorStatePtr->exportImage)
			SaveResult(SvImage);
		if (processorStatePtr->exportCrop)
			SaveResult(SvCrop);
		if (processorStatePtr->exportSSD) {
			SaveResult(SvSSD);
			committedSSD = true;
		}
		if (processorStatePtr->exportYOLO) {
			SaveResult(SvYOLO);
			committedYOLO = true;
		}
	}
	processorStatePtr->committed = committedSSD | committedYOLO;
}

static void TrackbarCallbackController(int, void*)
{
	processorStatePtr->confThresh = confSliderIdx / 10.0;
	processorStatePtr->nmsThresh = nmsSliderIdx / 10.0;
}

static void TrackbarCallbackNavigator(int, void*)
{
	if (processorStatePtr->sourceType == SrcImage)
		processorStatePtr->imageIndex = navSliderIdx * (processorStatePtr->skipCount + 1);
	else {
		double frame_rate = processorStatePtr->cap.get(cv::CAP_PROP_FPS);
		double frame_msec = 1000 / frame_rate;
		processorStatePtr->videoIndex = navSliderIdx * (processorStatePtr->skipCount + 1) * frame_msec;
	}
}

void ShowResult(bool redraw = false)
{
	if (redraw) {
		cv::Mat img = processorStatePtr->image.clone();

		//Draw mask
		if (!processorStatePtr->mask.empty()) {
			cv::Mat mask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC3);
			cv::cvtColor(processorStatePtr->mask, mask, cv::COLOR_GRAY2BGR);
			cv::addWeighted(img, 0.7, mask, 0.3, 0.0, img);
		}

		//Draw overlay
		for (size_t i = 0; i < processorStatePtr->objects.size(); ++i)
			DrawOverlay(img, DrwCapBBox, i);

		if (processorStatePtr->showHelp)
			DrawOverlay(img, DrwHelp);

		processorStatePtr->overlay = img;
	}

	cv::setMouseCallback(processorStatePtr->handleName, MouseCallbackProcessor, (void*)processorStatePtr);
	cv::imshow(processorStatePtr->handleName, processorStatePtr->overlay);

	int key = cv::waitKey(static_cast<char>(wait_key));
	if (wait_key == 0) {
		if (key == 'b') {
			bool valid = false;
			if (processorStatePtr->sourceType != SrcStream) {
				if (navSliderIdx > 0) {
					if (processorStatePtr->sourceType == SrcImage)
						processorStatePtr->imageIndex = navSliderIdx * (processorStatePtr->skipCount + 1);
					else {
						double frame_rate = processorStatePtr->cap.get(cv::CAP_PROP_FPS);
						double frame_msec = 1000 / frame_rate;
						processorStatePtr->cap.set(cv::CAP_PROP_POS_MSEC, navSliderIdx * (processorStatePtr->skipCount + 1) * frame_msec - frame_msec);
					}
					cv::setTrackbarPos("Index", processorStatePtr->handleName, --navSliderIdx);
					valid = true;
				}
			}
			valid ? Processor() : ShowResult();
		}
		else if (key == 'd') {
			if (processorStatePtr->supportAutoDetect) {
				processorStatePtr->autoDetect = processorStatePtr->autoDetect ? false : true;
				if (processorStatePtr->autoDetect) {
					cv::Mat img = processorStatePtr->image.clone();
					if (halfType) {
						detectorHalfPtr->Preprocess(img);
						detectorHalfPtr->Detect(processorStatePtr->priorFilter, processorStatePtr->mask, processorStatePtr->objects,
							processorStatePtr->confThresh, processorStatePtr->nmsThresh);
					}
					else {
						detectorFloatPtr->Preprocess(img);
						detectorFloatPtr->Detect(processorStatePtr->priorFilter, processorStatePtr->mask, processorStatePtr->objects,
							processorStatePtr->confThresh, processorStatePtr->nmsThresh);
					}
				}
				ShowResult(true);
			}
			ShowResult();
		}
		else if (key == 'f') {
			if (processorStatePtr->sourceType != SrcStream) {
				if (processorStatePtr->sourceType == SrcImage)
					processorStatePtr->imageIndex = navSliderIdx * (processorStatePtr->skipCount + 1);
				else
					processorStatePtr->cap.set(cv::CAP_PROP_POS_MSEC, processorStatePtr->videoIndex);
				Processor();
			}
			else if (processorStatePtr->autoDetect) {
				cv::Mat img = processorStatePtr->image.clone();
				if (halfType) {
					detectorHalfPtr->Preprocess(img);
					detectorHalfPtr->Detect(processorStatePtr->priorFilter, processorStatePtr->mask, processorStatePtr->objects,
						processorStatePtr->confThresh, processorStatePtr->nmsThresh);
				}
				else {
					detectorFloatPtr->Preprocess(img);
					detectorFloatPtr->Detect(processorStatePtr->priorFilter, processorStatePtr->mask, processorStatePtr->objects,
						processorStatePtr->confThresh, processorStatePtr->nmsThresh);
				}
				ShowResult(true);
			}
			else
				ShowResult();
		}
		else if (key == 'h') {
			processorStatePtr->showHelp = processorStatePtr->showHelp ? false : true;
			ShowResult(true);
		}
		else if (key == 'i') {
			processorStatePtr->preProcess = true;
			cv::namedWindow("Filter", cv::WINDOW_NORMAL);
			cv::moveWindow("Filter", 0, 0);
			cv::setMouseCallback("Filter", MouseCallbackPreprocessor);
			Preprocessor();
			processorStatePtr->preProcess = false;
			ShowResult(true);
		}
		else if (key == 'n') {
			processorStatePtr->mode = OpNew;
			if (processorStatePtr->showHelp) {
				cv::Mat img = processorStatePtr->image.clone();
				if (!processorStatePtr->mask.empty()) {
					cv::Mat mask = cv::Mat::zeros(cv::Size(processorStatePtr->image.cols, processorStatePtr->image.rows), CV_8UC3);
					cv::cvtColor(processorStatePtr->mask, mask, cv::COLOR_GRAY2BGR);
					cv::addWeighted(img, 0.7, mask, 0.3, 0.0, img);
				}
				for (size_t i = 0; i < processorStatePtr->objects.size(); ++i) {
					if (i == processorStatePtr->objIndexToEdit)
						DrawOverlay(img, DrwCaption, processorStatePtr->objIndexToEdit);
					else
						DrawOverlay(img, DrwCapBBox, i);
				}
				DrawOverlay(img, DrwHelp);
				processorStatePtr->overlay = img;
			}
			cv::Rect rect = cv::selectROI(processorStatePtr->handleName, processorStatePtr->overlay, false);
			processorStatePtr->mode = OpNorm;
			if (!rect.empty()) {
				if (processorStatePtr->objIndexToEdit != -1) {
					processorStatePtr->objects[processorStatePtr->objIndexToEdit].left = rect.x;
					processorStatePtr->objects[processorStatePtr->objIndexToEdit].top = rect.y;
					processorStatePtr->objects[processorStatePtr->objIndexToEdit].right = rect.x + rect.width;
					processorStatePtr->objects[processorStatePtr->objIndexToEdit].bottom = rect.y + rect.height;
					processorStatePtr->objIndexToEdit = -1;
					ShowResult(true);
				}
				else {
					detect_result obj;
					obj.left = rect.x;
					obj.top = rect.y;
					obj.right = rect.x + rect.width;
					obj.bottom = rect.y + rect.height;
					processorStatePtr->objects.push_back(obj);
					int idx = processorStatePtr->objects.size() - 1;
					PopupMenu(idx, processorStatePtr->overlay, rect.x, rect.y);
				}
			}
			else
				ShowResult();
		}
		else if (key == 'q')
			exit(1);
		else if (key == 'r')
			wait_key = restore_key;
		else if (key == 's') {
			if (processorStatePtr->sourceType != SrcStream)
				cv::setTrackbarPos("Index", processorStatePtr->handleName, ++navSliderIdx);
		}
		else if (key == 't') {
			if (processorStatePtr->autoDetect)
				if (processorStatePtr->mode == OpCtrl) {
					processorStatePtr->mode = OpNorm;
					cv::destroyWindow(processorStatePtr->handleName);
					cv::namedWindow(processorStatePtr->handleName, cv::WINDOW_NORMAL);
					cv::moveWindow(processorStatePtr->handleName, 0, 0);
					cv::setMouseCallback(processorStatePtr->handleName, MouseCallbackProcessor, (void*)processorStatePtr);
					if (processorStatePtr->sourceType != SrcStream)
						cv::createTrackbar("Index", processorStatePtr->handleName, &navSliderIdx, navSliderMax, TrackbarCallbackNavigator);
				}
				else {
					confSliderIdx = processorStatePtr->confThresh * 10;
					nmsSliderIdx = processorStatePtr->nmsThresh * 10;
					cv::createTrackbar("Conf", processorStatePtr->handleName, &confSliderIdx, 10, TrackbarCallbackController);
					cv::createTrackbar("NMS", processorStatePtr->handleName, &nmsSliderIdx, 10, TrackbarCallbackController);
					processorStatePtr->mode = OpCtrl;
				}
			else
				ShowResult();
		}
		else if (key == 'w') {
			if (!processorStatePtr->objects.empty()) {
				ExportResult();
				if (processorStatePtr->sourceType != SrcStream)
					cv::setTrackbarPos("Index", processorStatePtr->handleName, ++navSliderIdx);
			}
			else
				ShowResult();
		}
		else
			ShowResult();
	}
	else {//if (wait_key > 0) {
		if (key == 'h') {
			processorStatePtr->showHelp = processorStatePtr->showHelp ? false : true;
			ShowResult(true);
		}
		else if (key == 'p') {
			if (restore_key == NULL)
				restore_key = wait_key;
			wait_key = 0;
		}
		else if (key == 'q')
			exit(1);
		else {
			if (!processorStatePtr->objects.empty())
				ExportResult();
			if (processorStatePtr->sourceType != SrcStream)
				cv::setTrackbarPos("Index", processorStatePtr->handleName, ++navSliderIdx);
		}
	}
}

//Filter for ROI mask
void Preprocessor()
{
	cv::Mat img = processorStatePtr->image.clone();

	if (!processorStatePtr->maskPointsList.empty())
		DrawOverlay(img, DrwMask);
	if (processorStatePtr->showHelp)
		DrawOverlay(img, DrwHelp);
	cv::setMouseCallback("Filter", MouseCallbackPreprocessor);
	cv::imshow("Filter", img);
	int key = cv::waitKey(static_cast<char>(0));
	if (key == 'h') {
		processorStatePtr->showHelp = processorStatePtr->showHelp ? false : true;
		Preprocessor();
	}
	else if (key == 'w') {
		if (!processorStatePtr->maskPointsList.empty()) {
			std::ofstream maskFile;
			if (processorStatePtr->maskFileName.empty())
				processorStatePtr->maskFileName = "newMask.txt";
			maskFile.open(processorStatePtr->maskFileName);
			for (size_t i = 0; i < processorStatePtr->maskPointsList.size(); ++i) {
				for (size_t j = 0; j < processorStatePtr->maskPointsList[i].size() - 1; ++j)
					maskFile << processorStatePtr->maskPointsList[i][j].x << "," << processorStatePtr->maskPointsList[i][j].y << ";";
				maskFile << processorStatePtr->maskPointsList[i][processorStatePtr->maskPointsList[i].size() - 1].x << "," <<
					processorStatePtr->maskPointsList[i][processorStatePtr->maskPointsList[i].size() - 1].y << "\n";
			}
			maskFile.close();
		}
		cv::destroyWindow("Filter");
	}
	else if (key == 'q')
		exit(1);
	else
		Preprocessor();
}

//Load annotation from file
bool LoadLabel(const string &filename, SaveComponent format, cv::Mat &mask)
{
	bool maskFilter;
	maskFilter = cv::countNonZero(mask) > 0 ? true : false;
	bool success = false;
	std::ifstream ifs(filename);
	if (ifs.is_open()) {
		for (string line; getline(ifs, line);) {
			std::stringstream ss(line);
			detect_result object;
			bool found = false;
			int idx;
			string label;
			float conf = 1;
			ss >> label;
			ss >> object.left;
			ss >> object.top;
			ss >> object.right;
			ss >> object.bottom;
			if (format == SvYOLO) {
				float w = round(object.right * processorStatePtr->image.cols);
				float h = round(object.bottom * processorStatePtr->image.rows);
				object.left = round(object.left * processorStatePtr->image.cols - w / 2);
				object.top = round(object.top * processorStatePtr->image.rows - h / 2);
				object.right = object.left + w;
				object.bottom = object.top + h;
			}
			if (maskFilter) {
				cv::Mat blob = cv::Mat::zeros(mask.size(), CV_8UC1);
				cv::rectangle(blob, cv::Point(object.left, object.top), cv::Point(object.right, object.bottom), cv::Scalar(255), -1);
				int blobUnmasked = cv::countNonZero(blob);
				int blobMasked = cv::countNonZero(blob & mask);
				if (blobMasked != blobUnmasked)
					continue;
			}
			bool index = (label.find_first_not_of("0123456789") == string::npos);
			if (index) {
				auto il = processorStatePtr->labelMap.indexToLabel.find(stoi(label));
				if (il != processorStatePtr->labelMap.indexToLabel.end()) {
					idx = il->first;
					label = il->second;
					found = true;
				}
			}
			else {
				std::transform(label.begin(), label.end(), label.begin(), ::toupper);
				auto li = processorStatePtr->labelMap.labelToIndex.find(label);
				if (li != processorStatePtr->labelMap.labelToIndex.end()) {
					idx = li->second;
					found = true;
				}
			}
			if (!found) {
				label = "UNKNOWN";
				idx = processorStatePtr->labelMap.labelToIndex[label];
				conf = processorStatePtr->confThresh;
			}
			object.classid = idx;
			object.classlabel = label;
			object.confidence = conf;
			processorStatePtr->objects.push_back(object);
		}
		ifs.close();
		success = true;
	}
	return success;
}

string PositionToTimestamp(double totalInMilliseconds)
{
	int64 msec = std::abs(int64(std::round(totalInMilliseconds)));

	auto hours = msec / (1000 * 60 * 60);
	auto minutes = msec / (1000 * 60) % 60;
	auto seconds = msec / 1000 % 60;
	auto milliseconds = msec % 1000;

	char timestamp[15] = "";
	sprintf(timestamp, "%02d:%02d:%02d.%03d", hours, minutes, seconds, milliseconds);

	return timestamp;
}

void Processor()
{
	int frameLostCount = 0;
	int count = processorStatePtr->skipCount;
	if (processorStatePtr->sourceType == SrcImage)
		for (size_t i = processorStatePtr->imageIndex; i < processorStatePtr->sourceName.size(); ++i) {
			if ((processorStatePtr->skipCount == 0) || (count > 0 && count % processorStatePtr->skipCount == 0)) {
				cv::Mat img;
				img = cv::imread(processorStatePtr->sourceDir + "image/" + processorStatePtr->sourceName[i] + "." + processorStatePtr->sourceExt, -1);
				if (!img.empty()) {
					processorStatePtr->image = img;
					processorStatePtr->imageIndex = i;

					if (processorStatePtr->preProcess) {
						cv::namedWindow("Filter", cv::WINDOW_NORMAL);
						cv::moveWindow("Filter", 0, 0);
						Preprocessor();
						processorStatePtr->preProcess = false;
					}

					processorStatePtr->objects.clear();
					string labelFile;
					SaveComponent labelFormat;
					if (processorStatePtr->exportSSD) {
						labelFile = processorStatePtr->exportDir + "label/ssd/" + processorStatePtr->sourceName[processorStatePtr->imageIndex] + ".txt";
						labelFormat = SvSSD;
					}
					else if (processorStatePtr->exportYOLO) {
						labelFile = processorStatePtr->exportDir + "label/yolo/" + processorStatePtr->sourceName[processorStatePtr->imageIndex] + ".txt";
						labelFormat = SvYOLO;
					}
					bool committed = LoadLabel(labelFile, labelFormat, processorStatePtr->mask);
					if (!committed) {
						labelFile = processorStatePtr->sourceDir + "label/ssd/" + processorStatePtr->sourceName[processorStatePtr->imageIndex] + ".ssd";
						labelFormat = SvSSD;
						if (!LoadLabel(labelFile, labelFormat, processorStatePtr->mask)) {
							labelFile = processorStatePtr->sourceDir + "label/yolo/" + processorStatePtr->sourceName[processorStatePtr->imageIndex] + ".yolo";
							labelFormat = SvYOLO;
							LoadLabel(labelFile, labelFormat, processorStatePtr->mask);
						}
					}

					if ((processorStatePtr->mode == OpCtrl) || (processorStatePtr->autoDetect && !committed)) {
						if (halfType) {
							detectorHalfPtr->Preprocess(img);
							detectorHalfPtr->Detect(processorStatePtr->priorFilter, processorStatePtr->mask, processorStatePtr->objects,
								processorStatePtr->confThresh, processorStatePtr->nmsThresh);
						}
						else {
							detectorFloatPtr->Preprocess(img);
							detectorFloatPtr->Detect(processorStatePtr->priorFilter, processorStatePtr->mask, processorStatePtr->objects,
								processorStatePtr->confThresh, processorStatePtr->nmsThresh);
						}
					}

					processorStatePtr->committed = committed;

					ShowResult(true);
				}
				count = 0;
			}
			else
				count++;
		}
	else {
		for (;;) {
			if ((processorStatePtr->skipCount == 0) || (count > 0 && count % processorStatePtr->skipCount == 0)) {
				cv::Mat img;
				processorStatePtr->cap >> img;
				if (!img.empty()) {
					if (processorStatePtr->sourceType == SrcStream) {					
						boost::posix_time::ptime dateTime = boost::posix_time::microsec_clock::local_time();
						string datestamp = boost::posix_time::to_iso_extended_string(dateTime).substr(0, 10);
						string timestamp = boost::posix_time::to_iso_extended_string(dateTime).substr(11, 12);
						processorStatePtr->sourceName[1] = datestamp;
						processorStatePtr->sourceName[2] = timestamp;
					}
					else {
						processorStatePtr->videoIndex = processorStatePtr->cap.get(cv::CAP_PROP_POS_MSEC);
						processorStatePtr->sourceName[2] = PositionToTimestamp(processorStatePtr->videoIndex);
					}

					processorStatePtr->image = img;

					if (processorStatePtr->preProcess) {
						cv::namedWindow("Filter", cv::WINDOW_NORMAL);
						cv::moveWindow("Filter", 0, 0);
						Preprocessor();
						processorStatePtr->preProcess = false;
					}

					processorStatePtr->objects.clear();
					bool committed = false;
					if (processorStatePtr->sourceType == SrcVideo) {
						string datestamp = processorStatePtr->sourceName[1];
						datestamp.erase(std::remove(datestamp.begin(), datestamp.end(), '-'), datestamp.end());
						string timestamp = processorStatePtr->sourceName[2];
						timestamp.erase(std::remove(timestamp.begin(), timestamp.end(), ':'), timestamp.end());
						string labelFile;
						string rootdir = processorStatePtr->exportDir;
						SaveComponent labelFormat;
						if (processorStatePtr->exportSSD) {
							rootdir += "label/ssd/";
							labelFormat = SvSSD;
						}
						else if (processorStatePtr->exportYOLO) {
							rootdir += "label/yolo/";
							labelFormat = SvYOLO;
						}
						labelFile = processorStatePtr->sourceName[0]
							+ "_"
							+ datestamp
							+ "-"
							+ timestamp;
						committed = LoadLabel(rootdir + labelFile + ".txt", labelFormat, processorStatePtr->mask);
						if (!committed) {
							rootdir = processorStatePtr->sourceDir + "label/";
							labelFormat = SvSSD;
							if (!LoadLabel(rootdir + "ssd/" + labelFile + ".ssd", labelFormat, processorStatePtr->mask)) {
								labelFormat = SvYOLO;
								LoadLabel(rootdir + "yolo/" + labelFile + ".yolo", labelFormat, processorStatePtr->mask);
							}
						}
					}

					if ((processorStatePtr->mode == OpCtrl) || (processorStatePtr->autoDetect && !committed)) {
						if (halfType) {
							detectorHalfPtr->Preprocess(img);
							detectorHalfPtr->Detect(processorStatePtr->priorFilter, processorStatePtr->mask, processorStatePtr->objects,
								processorStatePtr->confThresh, processorStatePtr->nmsThresh);
						}
						else {
							detectorFloatPtr->Preprocess(img);
							detectorFloatPtr->Detect(processorStatePtr->priorFilter, processorStatePtr->mask, processorStatePtr->objects,
								processorStatePtr->confThresh, processorStatePtr->nmsThresh);
						}
					}

					processorStatePtr->committed = committed;
				}
				else {
					frameLostCount++;
					if (frameLostCount > 100) {
						processorStatePtr->cap.release();
						processorStatePtr->cap.open(processorStatePtr->sourceURI, cv::CAP_FFMPEG);
						frameLostCount = 0;
					}
				}
				count = 0;
			}
			else
				count++;

			ShowResult(true);
		}
	}
}

vector<string> FilenameParser(string folder, string ext)
{
	vector<string> names;
	if (!boost::filesystem::exists(folder) || !boost::filesystem::is_directory(folder)) return names;

	boost::filesystem::directory_iterator it(folder);
	boost::filesystem::directory_iterator endit;

	while (it != endit)
	{
		if (boost::filesystem::is_regular_file(*it) && it->path().extension() == "." + ext) {
			string filename = it->path().filename().string();
			filename.replace(filename.end() - 4, filename.end(), "");
			names.push_back(filename);
		}
		++it;
	}

	return names;
}

int main(int argc, char** argv) {

	const char* keys =
		"{ model model_file        | <none> | model file }"
		"{ weights weights_file    | <none> | weights file }"
		"{ label label_file        | <none> | label file }"
		"{ export_label            | <none> | output label }"
		"{ export_prefix           | <none> | output prefix }"
		"{ export_ssd              | true   | export label to SSD format }"
		"{ export_yolo             | false  | export label to YOLO format }"
		"{ show_help               | true   | display short-cut keys }"
		"{ image_dir               | <none> | path to image }"
		"{ image_ext               | <none> | image extension }"
		"{ video                   | <none> | path to video }"
		"{ export_dir              | <none> | path to export object }"
		"{ export_image            | <none> | path to export image }"
		"{ export_crop             | false  | export cropped object }"
		"{ edit_mask               | false  | foreground masking }"
		"{ mask_file             | <none> | roi points file }"
		"{ auto_detect           | false  | persistent object detection }"
		"{ conf_threshold        | 0.3    | detection confidence threshold }"
		"{ nms_threshold         | 0.5    | non-maximum suppression threshold }"
		"{ label_filter          | <none> | auto-detect label filter file }"
		"{ prior_filter          | false  | filter overlaps from previous frame }"
		"{ i iter                | 1      | iterations to be run }"
		"{ w wait_key            | 0      | display interval }"
		"{ g gpu                 | 0      | gpu device }"
		"{ c cpu                 | false  | use cpu device }"
		"{ fp16 use_fp16         | false  | use fp16 forward engine. }"
		"{ frame_skip            | 0      | number of frame to skip}"
		"{ help                  | false  | display this help and exit  }"
		;


	cv::CommandLineParser parser(argc, argv, keys);
	const string model_file = parser.get<std::string>("model_file");
	const string weights_file = parser.get<std::string>("weights_file");
	const string video_source = parser.get<std::string>("video");
	const string image_dir = parser.get<std::string>("image_dir");
	const string image_ext = parser.get<std::string>("image_ext");
	const string exportPrefix = parser.get<std::string>("export_prefix");
	const string label_file = parser.get<std::string>("label");

	string exportDir = parser.get<string>("export_dir");
	bool exportImage = parser.get<bool>("export_image");
	bool exportCrop = parser.get<bool>("export_crop");
	bool exportSSD = parser.get<bool>("export_ssd");
	bool exportYOLO = parser.get<bool>("export_yolo");
	bool showHelp = parser.get<bool>("show_help");
	bool editMask = parser.get<bool>("edit_mask");
	const string exportLabel = parser.get<std::string>("export_label");
	const float conf_thresh = parser.get<float>("conf_threshold");
	const float nms_thresh = parser.get<float>("nms_threshold");
	bool priorFilter = parser.get<bool>("prior_filter");
	bool autoDetect = parser.get<bool>("auto_detect");
	string labelFilter = parser.get<string>("label_filter");
	int iter = parser.get<int>("iter");
	string mask_file = parser.get<string>("mask_file");
	wait_key = parser.get<int>("wait_key");
	int gpu = parser.get<int>("gpu");
	bool cpu = parser.get<bool>("cpu");
	bool use_fp16 = parser.get<bool>("use_fp16");
	int frameSkipCount = parser.get<int>("frame_skip");

	if (!exportDir.empty()) {
		if (!boost::filesystem::exists(exportDir.c_str()))
			boost::filesystem::create_directory(exportDir.c_str());
		if (exportImage)
			boost::filesystem::create_directory((exportDir + "image/").c_str());
		if (exportCrop)
			boost::filesystem::create_directory((exportDir + "crop/").c_str());
		if (exportSSD || exportYOLO) {
			boost::filesystem::create_directory((exportDir + "label/").c_str());
			if (exportSSD)
				boost::filesystem::create_directory((exportDir + "label/ssd").c_str());
			if (exportYOLO)
				boost::filesystem::create_directory((exportDir + "label/yolo").c_str());
		}
	}

	label_map labelMap;
	std::ifstream labelFile(exportLabel);
	string line;
	int idx = 0;
	string firstGroup, currGroup;
	while (getline(labelFile, line)) {
		string label = line;
		std::transform(label.begin(), label.end(), label.begin(), ::toupper);
		if (label[0] == '#') {
			currGroup = label.substr(1, label.size());
			if (firstGroup.empty())
				firstGroup = currGroup;
			labelMap.groupToIndex[currGroup] = 0;
		}
		else {
			if (exportCrop) {
				string label_ = label;
				std::transform(label_.begin(), label_.end(), label_.begin(), ::tolower);
				boost::filesystem::create_directory((exportDir + "crop/" + label_).c_str());
			}
			if (!currGroup.empty())
				labelMap.groupToIndex[currGroup] += 1;
			labelMap.indexToLabel[idx] = label;
			labelMap.labelToIndex.insert(std::make_pair(label, idx++));
		}
	}
	if (!firstGroup.empty())
		labelMap.activeGroup = firstGroup;

	map<string, string> autoDetectMap;
	if (!labelFilter.empty()) {
		std::ifstream labelFile(labelFilter);
		while (getline(labelFile, line)) {
			string label;
			size_t pos = line.find(",");
			string src = line.substr(0, pos);
			string dst = line.substr(pos + 1, line.size());
			autoDetectMap.insert(std::make_pair(src, dst));
		}
	}

	if (cpu)
		gpu = -1;

	int totalFrameCount = 0;
	string videotype;
	cv::VideoCapture cap_;
	vector<string> imgfiles;
	cv::Mat img;
	int image_width;
	int image_height;
	if (image_dir != "") {
		imgfiles = FilenameParser(image_dir + "image/", image_ext);
		img = cv::imread(image_dir + "image/" + imgfiles[0] + "." + image_ext, -1);
		image_width = img.cols;
		image_height = img.rows;
	}
	else {
		if (video_source.find("http") != std::string::npos)
			videotype = "http";
		else if (video_source.find("rtsp") != std::string::npos)
			videotype = "rtsp";
		else
			videotype = "vid";
		if (!cap_.open(video_source, cv::CAP_FFMPEG))
			LOG(FATAL) << "Failed to video source: " << video_source;
		image_width = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
		image_height = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
	}

	vector<vector<cv::Point>> maskPoints;
	cv::Mat mask;
	if (!mask_file.empty()) {
		mask = cv::Mat::zeros(cv::Size(image_width, image_height), CV_8UC1);
		int cnt = 0;
		std::ifstream maskFile(mask_file.c_str());
		while (getline(maskFile, line)) {
			size_t pos = 0, pos2;
			string token, pointX, pointY;
			vector<cv::Point> pts;
			while ((pos = line.find(";")) != string::npos) {
				token = line.substr(0, pos);
				pos2 = token.find(",");
				pointX = token.substr(0, pos2);
				pointY = token.substr(pos2 + 1, token.size());
				pts.push_back(cv::Point(std::stoi(pointX), std::stoi(pointY)));
				line.erase(0, pos + 1);
			}
			pos2 = line.find(",");
			pointX = line.substr(0, pos2);
			pointY = line.substr(pos2 + 1, line.size());
			int x = std::stoi(pointX), y = std::stoi(pointY);
			x = x > image_width - 1 ? image_width - 1 : x;
			y = y > image_height - 1 ? image_height - 1 : y;
			pts.push_back(cv::Point(x, y));
			cv::fillConvexPoly(mask, pts, cv::Scalar(255));
			maskPoints.push_back(pts);
		}
	}

	processor_state processorState;
	vector<detect_result> objects;
	processorState.supportAutoDetect = autoDetect;
	processorState.autoDetect = autoDetect;
	processorState.autoDetectMap = autoDetectMap;
	processorState.priorFilter = priorFilter;
	processorState.confThresh = conf_thresh;
	processorState.maskPointsList = maskPoints;
	processorState.mask = mask;
	processorState.confThresh = conf_thresh;
	processorState.nmsThresh = nms_thresh;
	processorState.exportPrefix = exportPrefix;
	processorState.preProcess = editMask;
	processorState.maskFileName = mask_file;
	processorState.showHelp = showHelp;
	processorState.exportDir = exportDir;
	processorState.exportImage = exportImage;
	processorState.exportCrop = exportCrop;
	processorState.exportSSD = exportSSD;
	processorState.exportYOLO = exportYOLO;
	processorState.labelMap = labelMap;
	processorState.mode = OpNorm;
	processorState.skipCount = frameSkipCount;
	processorState.committed = false;
	processorState.objIndexToEdit = -1;

	if (!imgfiles.empty()) {
		processorState.sourceDir = image_dir;
		processorState.sourceName = imgfiles;
		processorState.sourceExt = image_ext;
		processorState.imageIndex = 0;
		processorState.sourceType = SrcImage;
		processorState.handleName = "Image";
	}
	else {
		processorState.sourceURI = video_source;
		if (videotype == "vid") {
			processorState.sourceType = SrcVideo;
			processorState.handleName = "Video";
			processorState.videoIndex = cap_.get(cv::CAP_PROP_POS_MSEC);
			size_t pos = video_source.find_last_of("/\\") + 1;
			string rootname = video_source.substr(pos);
			processorState.sourceName.push_back(rootname.substr(0, rootname.size() - 4));
			processorState.sourceName.push_back("0000-00-00");
			processorState.sourceName.push_back("00:00:00.000");
			processorState.sourceDir = video_source.substr(0, pos);
			processorState.sourceExt = rootname.substr(rootname.size() - 3);
		}
		else {
			processorState.sourceType = SrcStream;
			processorState.handleName = "Stream";
			if (!exportPrefix.empty())
				processorState.sourceName.push_back(exportPrefix);
			else
				processorState.sourceName.push_back("LiveStream");
			boost::posix_time::ptime timeInfo = boost::posix_time::second_clock::local_time();
			char datestamp[10] = "";
			sprintf(datestamp, "%04d-%02d-%02d", timeInfo.date().year(), timeInfo.date().month(),
				timeInfo.date().day());
			char timestamp[15] = "";
			sprintf(timestamp, "%02d:%02d:%02d.%03d", timeInfo.time_of_day().hours(),
				timeInfo.time_of_day().minutes(), timeInfo.time_of_day().seconds(),
				timeInfo.time_of_day().fractional_seconds() / 1000);
			processorState.sourceName.push_back(datestamp);
			processorState.sourceName.push_back(timestamp);
		}
		processorState.cap = cap_;

	}

	cv::namedWindow(processorState.handleName, cv::WINDOW_NORMAL);
	cv::moveWindow(processorState.handleName, 0, 0);
	if (processorState.sourceType != SrcStream) {
		navSliderIdx = 0;
		if (processorState.sourceType == SrcImage)
			navSliderMax = (processorState.sourceName.size() - 1) / (frameSkipCount + 1);
		else
			navSliderMax = cap_.get(cv::CAP_PROP_FRAME_COUNT) / (frameSkipCount + 1);
		cv::createTrackbar("Index", processorState.handleName, &navSliderIdx, navSliderMax, TrackbarCallbackNavigator);
	}

	if (use_fp16) {
#ifdef HAS_HALF_SUPPORT
		Detector<half> detector(model_file, weights_file, label_file, gpu, autoDetect);	
		detectorHalfPtr = &detector;
		processorStatePtr = &processorState;
		cv::setMouseCallback(processorState.handleName, MouseCallbackProcessor, (void*)processorStatePtr);
		halfType = true;
		Processor();		
#else
		std::cout << "fp16 is not supported." << std::endl;		
#endif
	}
	if (!halfType) {
		Detector<float> detector(model_file, weights_file, label_file, gpu, autoDetect);
		detectorFloatPtr = &detector;
		processorStatePtr = &processorState;
		cv::setMouseCallback(processorState.handleName, MouseCallbackProcessor, (void*)processorStatePtr);
		halfType = false;
		Processor();
	}				
	return 0;
}
