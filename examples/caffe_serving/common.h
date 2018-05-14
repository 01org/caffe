#include <opencv2/core/core.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

typedef struct {
	cv::Size  size;
	int       type;
	int       isActive;
	int       serverId;
	char      tag[50];
	boost::interprocess::managed_shared_memory::handle_t handle;
	int       total;
} SharedImageHeader;

const char *MEMORY_NAME = "MatSharedMemory";