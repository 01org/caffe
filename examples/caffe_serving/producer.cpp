#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/filesystem.hpp>
#include <chrono>
#include <thread>

#include "common.h"

using namespace cv;
using namespace std;


vector<string> FilenameParser(string folder, string ext)
{
	namespace bfs = boost::filesystem;
	vector<string> names;
	if (!bfs::exists(folder) || !bfs::is_directory(folder)) return names;

	bfs::directory_iterator it(folder);
	bfs::directory_iterator endit;

	while (it != endit)
	{
		if (it->path().extension() == "." + ext) {
			string filename = it->path().filename().string();
			names.push_back(filename);
		}
		++it;
	}

	return names;
}

int main(int argc, char** argv)
{
	const char* keys =
		"{ server_id  | <none> | server ID }"
		"{ image_path | <none> | image path }"
		"{ image_ext  | jpg    | image extension }"
		"{ image_size | 0      | image size }"
		"{ reset_mem  | 1      | reset shared memory }"
		;

	cv::CommandLineParser parser(argc, argv, keys);
	int serverId = parser.get<int>("server_id");
	string imagePath = parser.get<string>("image_path");
	string imageExt = parser.get<string>("image_ext");
	int imageSize = parser.get<int>("image_size");
	int resetMem = parser.get<int>("reset_mem");

	namespace bip = boost::interprocess;


	// remove existing shared memory object
	if (resetMem == 1)
		bip::shared_memory_object::remove(MEMORY_NAME);

	vector<string> imgFiles = FilenameParser(imagePath, imageExt);

	string filename = imagePath + imgFiles[0];
	cv::Mat captured_image;
	captured_image = cv::imread(filename, -1);
	if (captured_image.empty())
		cout << "Unable to decode image " << imgFiles[0] << endl;
	if (imageSize != 0)
		cv::resize(captured_image, captured_image, cv::Size(imageSize, imageSize));

	// calculate total data size
	const int data_size	= captured_image.total() * captured_image.elemSize();

	// reserve shared memory
	bip::managed_shared_memory msm(bip::open_or_create, MEMORY_NAME, data_size + 1024/* is it enough? */);	

	// make a region named "Matheader" and return its pointer
	SharedImageHeader* shared_image_header = msm.find_or_construct<SharedImageHeader>("MatHeader")();
	
	if (shared_image_header->type == NULL)
	{
		// make a unnamed shared memory region, its size is data_size
		const SharedImageHeader* shared_image_data_ptr;
		shared_image_data_ptr = (SharedImageHeader*)msm.allocate(data_size);

		// init the Shared Memory
		shared_image_header->size = captured_image.size();
		shared_image_header->type = captured_image.type();
		shared_image_header->isActive = 0;
		shared_image_header->total = imgFiles.size();

		// write the handler to an unnamed region to the Shared Memory
		shared_image_header->handle = msm.get_handle_from_address(shared_image_data_ptr);
	}
	else
	{
		shared_image_header->total += imgFiles.size();
	}

	cv::Mat shared;
	shared = cv::Mat(
		shared_image_header->size,
		shared_image_header->type,
		msm.get_address_from_handle(shared_image_header->handle));

	bool quit = false;

	for (size_t i = 0; i < imgFiles.size(); ++i) {
		int timeout = 0;
		while (true) {
			if (shared_image_header->isActive == 0 && shared_image_header->serverId == serverId) {
				string filename = imagePath + imgFiles[i];
				cv::Mat img;
				img = cv::imread(filename, -1);
				if (img.empty())
					cout << "Unable to decode image " << imgFiles[i] << endl;
				if (imageSize != 0)
					cv::resize(img, shared, cv::Size(imageSize, imageSize));
				else
					img.copyTo(shared);
				strcpy_s(shared_image_header->tag, imgFiles[i].c_str());
				shared_image_header->isActive = 1;
				break;
			}
			if ((shared_image_header->isActive == -1) || (timeout > 10000)) {
				quit = true;
				break;
			}
			timeout++;
			this_thread::sleep_for(chrono::milliseconds(1));
		}

		cout << imgFiles[i] << " : timeout = " << timeout << endl;
		if (quit)
			break;
	}

	return 0;
}
