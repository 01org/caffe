#include <gflags/gflags.h>
#include <glog/logging.h>

#include <math.h>
#include <cassert>
#include <boost/filesystem.hpp>
#include <caffe/caffe.hpp>
#include "caffe/device.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/mqueue.hpp"

// using namespace caffe;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using caffe::device;
using std::ostringstream;
using namespace simple_mqueue;

DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(float_width, 16,
    "Optional; the Dtype used for client computation.");
DEFINE_string(host_ip, "127.0.0.1",
    "Requred; The host ip needed to connect to host,"
    "The default setting is 127.0.0.1.");
DEFINE_int32(port, 50000,
    "Required; The host port to connect,"
    "The default id is 50000");


template<typename Dtype>
class DNNClientContext : public simple_mqueue::BasicContext<Dtype, Dtype> {
public:
  DNNClientContext() {
    weight_initialized = false;
    input_msg_ = nullptr;
    output_msg_ = nullptr;

    /* Load the network. */
    net_.reset(new Net<Dtype>(FLAGS_model, caffe::TEST, Caffe::GetDefaultDevice()));
    net_->CopyTrainedLayersFrom(FLAGS_weights);
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
  }

  virtual ~DNNClientContext() {
    for (unsigned int i = 0; i < allocated_messages.size(); i++)
      delete allocated_messages[i];
  }

  void prepare_messages(vector<Message<Dtype> *> &msgs) {
    if (input_msg_ == nullptr) {
      Blob<Dtype>* input_layer = net_->input_blobs()[0];
      Blob<Dtype>* output_layer = net_->output_blobs()[0];
      input_msg_ = new Message<Dtype> (InputData, input_layer->shape(0),
                                       input_layer->shape(1), input_layer->shape(2),
                                       input_layer->shape(3));
      output_msg_ = new Message<Dtype> (OutputData, 1, 1, output_layer->shape(0),
                                        output_layer->shape(1));
      allocated_messages.push_back(input_msg_);
      allocated_messages.push_back(output_msg_);
    }
    msgs.push_back(input_msg_);
  }

  // FIXME zero copy ?
  int CopyWeight(Dtype *data, Blob<Dtype> &blob) {
    //memcpy(blob->mutable_cpu_data(), data, blob->count(0) * sizeof(Dtype));
    blob.set_cpu_data(data);
    return blob.count(0);
  }


  void handle_input_messages(vector<Message<Dtype> *> & msgs,
                             vector<Message<Dtype> *> & output_msgs) {

    Blob<Dtype>* input_blob = net_->input_blobs()[0];
    input_blob->Reshape(msgs[0]->memory()->shape()[0],
                msgs[0]->memory()->shape()[1],
                msgs[0]->memory()->shape()[2],
                msgs[0]->memory()->shape()[3]);
    input_blob->set_cpu_data(msgs[0]->mutable_memory()->mutable_data());
    net_->Reshape();
    net_->Forward();
    caffe::Caffe::Synchronize(caffe::Caffe::GetDefaultDevice()->id());

    Blob<Dtype> *out = net_->output_blobs()[0];
    output_msg_->mutable_memory()->set_data(out->count(0), out->mutable_cpu_data());
    output_msg_->mutable_memory()->reshape(out->shape(0), out->shape(1), out->shape(2), out->shape(3));
    output_msgs.push_back(output_msg_);
  }

  void handle_messages(vector<Message<Dtype> *> &msgs, vector<Message<Dtype> *> &output_msgs) {
    if (msgs[0]->id() == InitWeight) {
    } else if (msgs[0]->id() == InputData) {
      // Do inference here.
      handle_input_messages(msgs, output_msgs);
    }
  }

private:
  shared_ptr<Net<Dtype> > net_;
  vector<Message<Dtype> *> allocated_messages;
  bool weight_initialized;
  int batch_size_;
  Message<Dtype> * input_msg_;
  Message<Dtype> * output_msg_;
};

int main(int argc, char **argv) {

  if (std::getenv("HOME")) {
    std::stringstream cache_path;
    cache_path << std::getenv("HOME") << "/.cache/";
    const boost::filesystem::path& path = cache_path.str();
    const boost::filesystem::path& dir =
                 boost::filesystem::unique_path(path).string();
    if (!boost::filesystem::exists(dir))
      boost::filesystem::create_directories(dir);
    setenv("VIENNACL_CACHE_PATH", cache_path.str().c_str(), true);
  }

  caffe::GlobalInit(&argc, &argv);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::SetDevice(0);
  std::string host_ip = FLAGS_host_ip;
  uint16_t port = (uint16_t)FLAGS_port;
  int float_width = FLAGS_float_width;

  if (float_width != 16 && float_width != 32) {
    printf("Invalid float width %d\n", float_width);
    return -1;
  }

  printf("Host ip %s port %d\n", host_ip.c_str(), port);
  printf("Using fp%d\n", float_width);
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to forward.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to forward.";
  if (FLAGS_float_width == 16) {
    while (1) {
      DNNClientContext <half> client_ctx;
      simple_mqueue::Node<half, half> client_node(simple_mqueue::NodeTypeClient, &client_ctx);
      client_ctx.set_batch_size(16);
      client_node.set_host_addr(host_ip.c_str(), port);
      client_node.Run();
    }
  } else {
    while (1) {
      DNNClientContext <float> client_ctx;
      simple_mqueue::Node<float, float> client_node(simple_mqueue::NodeTypeClient, &client_ctx);
      client_ctx.set_batch_size(16);
      client_node.set_host_addr(host_ip.c_str(), port);
      client_node.Run();
    }
  }
  caffe::Caffe::TeardownDevice(0);

  return 0;
}
