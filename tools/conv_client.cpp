
#include <string>

#include "caffe/caffe.hpp"
#include "caffe/multi_node/msg_hub.hpp"
#include "caffe/multi_node/sk_sock.hpp"

#include "boost/unordered_map.hpp"
#include "caffe/multi_node/conv_node.hpp"
#include "caffe/multi_node/node_env.hpp"

using boost::unordered_map;
// using namespace caffe;

using caffe::NodeEnv;
using caffe::CONV_CLIENT;
using caffe::ModelRequest;
using caffe::ConvClient;
using caffe::Caffe;
// using caffe::client;

DEFINE_int32(threads, 1, "number of convolution client threads");
DEFINE_int32(zmq_cores, 2, "number of cores used by zeromq"
                            "0 means don't bind zmq threads to cores");

DEFINE_string(ip, "127.0.0.1", "the ip of the id and model server");
DEFINE_string(net_if, "", "the network interface to be used");
DEFINE_int32(id_port, 1955, "the tcp port of ID server");
DEFINE_int32(model_port, 1957, "the tcp port of model server");

DEFINE_int32(pub_port, 3001, "the tcp port to do broadcast");
DEFINE_int32(route_port, 3002, "the tcp port to listen");

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;
  google::InstallFailureSignalHandler();
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  Caffe::set_mode(Caffe::CPU);

  string id_server_addr = "tcp://";
  id_server_addr += FLAGS_ip;
  id_server_addr += ":";
  id_server_addr += boost::lexical_cast<string>(FLAGS_id_port);

  string model_server_addr = "tcp://";
  model_server_addr += FLAGS_ip;
  model_server_addr += ":";
  model_server_addr += boost::lexical_cast<string>(FLAGS_model_port);

  NodeEnv::set_model_server(model_server_addr);
  NodeEnv::set_id_server(id_server_addr);
  NodeEnv::set_node_role(CONV_CLIENT);

  ModelRequest rq;
  rq.mutable_node_info()->set_node_role(CONV_CLIENT);
  rq.mutable_node_info()->set_router_port(FLAGS_route_port);
  rq.mutable_node_info()->set_pub_port(FLAGS_pub_port);
  if (!FLAGS_net_if.empty()) {
    rq.mutable_node_info()->set_net_if(FLAGS_net_if);
  }
  NodeEnv::set_model_request(rq);

  NodeEnv::InitNode(FLAGS_threads, FLAGS_zmq_cores);

  LOG(INFO) << "conv node id: " << NodeEnv::Instance()->ID();

  // total threads equals worker thread + 1 parameter thread
  shared_ptr<ConvClient<float> >
  client(new ConvClient<float>(FLAGS_threads + 1));

  client->Init();
  client->Poll();

  return 0;
}

