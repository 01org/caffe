#ifndef __SIMPLE_MQUEUE_HPP__
#define __SIMPLE_MQUEUE_HPP__

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <vector>
#include <map>
#include <sys/epoll.h>
#include <cstddef>
#include <type_traits>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <iostream>
#include <unistd.h>
#include "caffe/3rdparty/half/half.hpp"

using namespace half_float;

namespace simple_mqueue {

enum NodeType {
  NodeTypeClient,
  NodeTypeHost
};

enum MessageID {
  InitWeight = 0,
  InputData = 1,
  OutputData = 2,
  ExitMessage = 9999,
};

enum MessageDataType {
  FP16 = 0,
  FP32 = 1,
  UNSUPPORTED = 9,
};

template <typename Dtype>
MessageDataType GetDataType(void)
{
  if (std::is_same<Dtype, half>::value)
    return FP16;
  if (std::is_same<Dtype, float>::value)
    return FP32;
  return UNSUPPORTED;
}

template <typename Dtype>
struct MessageHdr {

  MessageHdr(): id() {
  }

  MessageHdr(MessageID id): id(id) {
    data_type = GetDataType<Dtype>();
  }
  uint16_t data_type;
  uint16_t id;
  uint32_t shape[4];
  int len() {
    return shape[0] * shape[1] * shape[2] * shape[3];
  }
  uint32_t extra_shape_len;
};

template<typename Dtype>
class Memory {
public:
  Memory(int batch, int channel, int h, int w) {
    own_data_ = false;
    allocate(batch * channel * h * w);
    reshape(batch, channel, h, w);
  }
  Memory(int batch, int channel, int h, int w, Dtype *data) {
    data_ = data;
    data_len_ = batch * channel * h * w;
    own_data_ = false;
    reshape(batch, channel, h, w);
  }
  Memory(int len, Dtype *data) {
    data_ = data;
    data_len_ = len;
    own_data_ = false;
    reshape(1,1,1,len);
  }
  ~Memory(void) {
    if (own_data_)
      free(data_);
    own_data_ = false;
    data_ = NULL;
  }
  int data_len() const { return data_len_; }
  bool own_data() const { return own_data_; }
  const Dtype * data() const { return data_; }
  Dtype * mutable_data() { return data_; }
  void set_data(int len, Dtype *data) {
    if (data_ && own_data_)
      free(data_);
    data_ = data;
    own_data_ = false;
    data_len_ = len;
  }

  void set_data(Dtype *data) {
    if (data_ && own_data_)
      free(data_);
    data_ = data;
    own_data_ = false;
  }

  Dtype &at(int b, int c, int h, int w) {
    return data_[b * count(1) + c * count(2) + h * count(3) + w];
  }

  const Dtype &const_at(int b, int c, int h, int w) const {
    return data_[b * count(1) + c * count(2) + h * count(3) + w];
  }

  void allocate(int size) {
    if (data_ && own_data_)
      free(data_);
    data_ = (Dtype *)malloc(size * sizeof(Dtype));
    data_len_ = size;
    own_data_ = true;
  }

  int count(int start_id) const {
    int size = 1;
    for (unsigned int i = start_id ; i < shape_.size(); i++) {
      size *= shape_[i];
    }
    return size;
  }
  void reshape(int batch, int channel, int h, int w) {
    std::vector <int> shape;
    shape.push_back(batch);
    shape.push_back(channel);
    shape.push_back(h);
    shape.push_back(w);
    reshape(shape);
  }

  void reshape(std::vector <int> shape) {
    shape_.resize(4);
    for (unsigned int i = 0; i < 4; i++) {
      if (i < shape.size())
        shape_[i] = shape[i];
      else
        shape_[i] = 1;
    }
    if (count(0) > data_len_ && own_data_) {
      allocate(count(0));
      assert(data_ != NULL);
      if (data_ == NULL) {
        printf("fail to allocate buffer for %d.\n", count(0));
        exit(-1);
      }
    }
    if (count(0) > data_len_ && !own_data_) {
      printf("Try to reshape to larger size for an external memory block.\n");
      exit(-1);
    }
  }
  int batch_size() const { return shape_[0]; }
  const std::vector <int> &shape() const { return shape_; }
private:

  std::vector <int> shape_;
  int data_len_;
  bool own_data_;
  Dtype *data_;
};

template<typename DstDtype, typename SrcDtype>
void memory_copy(Memory <DstDtype> *dst, const SrcDtype * src) {
  DstDtype * dst_data = dst->mutable_data();
  for (int i = 0; i < dst->count(0); i++)
    dst_data[i] = src[i];
}

template void memory_copy<float, float>(Memory <float> *dst, const float *src);

template<typename Dtype>
class Message {
public:
  Message(MessageID id, int data_len) : id_(id) {
    mem_ = new Memory<Dtype>(1, 1, 1, data_len);
    extra_shape_ = NULL;
  }
  Message(MessageID id, int data_len, Dtype *data)
    : id_(id) {
    mem_ = new Memory<Dtype>(1, 1, 1, data_len, data);
    extra_shape_ = NULL;
  }
  Message(MessageID id, int batch, int channel, int h, int w, Dtype *data) {
    id_ = id;
    mem_ = new Memory<Dtype>(batch, channel, h, w, data);
    extra_shape_ = NULL;
  }
  Message(MessageID id, int batch, int channel, int h, int w) {
    id_ = id;
    mem_ = new Memory<Dtype>(batch, channel, h, w);
    extra_shape_ = NULL;
  }
  Message(MessageID id, int batch, int channel, int h, int w,
          int extra_shape_len) {
    id_ = id;
    mem_ = new Memory<Dtype>(batch, channel, h, w);
    extra_shape_ = new Memory<int>(extra_shape_len, 1, 1, 1);
  }

  ~Message(void) {
    if (mem_)
      delete mem_;
    if (extra_shape_)
      delete extra_shape_;
    mem_ = NULL;
    extra_shape_ = NULL;
  }
  void set_id(MessageID id) { id_ = id; }
  MessageID id() const { return id_; }
  const Memory<Dtype> *memory() const { return mem_; }
  Memory<Dtype> *mutable_memory() { return mem_; }
  const Memory<int> *extra_memory() const { return extra_shape_; }
  Memory<int> *mutable_extra_memory() { return extra_shape_; }
  bool has_extra_memory() const { return extra_shape_ != NULL; }
  void create_extra_memory(int len) {
    if (extra_shape_ != NULL)
      delete extra_shape_;
    extra_shape_ = new Memory<int>(len, 1, 1, 1);
  }
  void create_extra_memory(int len, int *data) {
    if (extra_shape_ != NULL)
      delete extra_shape_;
    extra_shape_ = new Memory<int>(len, 1, 1, 1, data);
  }

private:
  MessageID id_;
  Memory<Dtype> *mem_;  // contains data.
  Memory<int> *extra_shape_; // contains extra shape data.
};

struct BatchPair {
  BatchPair(int bi, int bs) : batch_index(bi), batch_size(bs) {}
  int batch_index;
  int batch_size;
};

template<typename Dtype, typename HostDtype>
class BasicContext{
public:

  BasicContext(void) {
  }
  ~BasicContext(void) {
  };
  virtual Message<Dtype> *prepare_message(MessageHdr<Dtype> hdr) { return NULL; };
  virtual Message<Dtype> *prepare_message(MessageID id) { return NULL; };
  virtual void prepare_messages(std::vector <Message<Dtype> *> &msgs) { };
  virtual void handle_messages(std::vector <Message<Dtype> *> &msg, std::vector <Message<Dtype> *> &output_messages) {};
  // The following two implementation is for LSTM work load on VCA card.
  virtual int get_prefer_batch_size(int total_bs, int client_num) {
    int bs = ((total_bs + client_num - 1) / client_num);
    // Make bs 4 aligned. 
    if (bs % 4 != 0 && (total_bs >= client_num * 8)) {
      if (total_bs - (bs & ~3) * client_num < 4)
        bs = bs & ~3;
      else
        bs = (bs + 3) & ~3;
    }
    // Set the minimum batch size to half of the batch size.
    if (bs < 4)
      bs = 4;
    return bs;
  }
  virtual int get_actual_batch_size(int prefer_bs, int total_bs) {
    if (prefer_bs >= total_bs)
      prefer_bs = total_bs;
    else if (total_bs - prefer_bs < 4)
      prefer_bs = total_bs;
    return prefer_bs;
  }
  void set_batch_size(int bs) {
    batch_size_ = bs;
  }

  std::vector <int> BatchSent;
  int batch_size_;
};

template<typename Dtype, typename HostDtype>
class Node {
public:
  Node(NodeType type,
       BasicContext<Dtype, HostDtype> * ctx) {
    type_ = type;
    ctx_ = ctx;
    weights_bytes_ = 0;
    sent_bytes_total_ = 0;
    recved_bytes_total_ = 0;
  }

  Node(NodeType type,
       BasicContext<Dtype, HostDtype> * ctx,
       int clients_num) {
    type_ = type;
    ctx_ = ctx;
    clients_num_ = clients_num;
    sent_bytes_total_ = 0;
    recved_bytes_total_ = 0;
    weights_bytes_ = 0;
  }

  void set_host_addr(const char * host_addr, int port) {
    if (inet_pton(AF_INET, host_addr, &host_addr_.sin_addr) < 0) {
      printf("invalid host addr %s \n", host_addr);
      exit(-1);
    }
    host_addr_.sin_port = htons(port);
    host_addr_.sin_family = AF_INET;
  }

  void Forward(const std::vector<Message<Dtype> *> input_messages, std::vector<Message<Dtype> *> output_messages) {
    unsigned int total_bs = input_messages[0]->memory()->batch_size();
    int bs = ctx_->get_prefer_batch_size(total_bs, sockets_.size());
    unsigned int batch_index = 0;
    std::vector<int> sockets = sockets_;
    unsigned int batch_processed = 0;
    std::vector <struct epoll_event> events;
    events.resize(clients_num_);
    ctx_->BatchSent.clear();
    while(batch_processed < total_bs) {
      if (sockets.size() > 0 && batch_index < total_bs) {
        int actual_bs = ctx_->get_actual_batch_size(bs, total_bs - batch_index);
        int socket = sockets.back();
        sockets.pop_back();
        SendSubMessages(input_messages, batch_index, actual_bs, socket);
        ctx_->BatchSent.push_back(actual_bs);
        BatchPair bp(batch_index, actual_bs);
        batch_index += actual_bs;
        sub_msgs_.insert(std::make_pair(socket, bp));
      } else {
        // epoll all sockets
        int event_count = epoll_wait(epfds_, &events[0], clients_num_ * output_messages.size(), 200);
        for (int i = 0; i < event_count; i++) {
          if (events[i].data.fd < 0)
            continue;
          if (events[i].events & EPOLLIN) {
            int socket = events[i].data.fd;
            auto it = sub_msgs_.find(socket);
            assert(it != sub_msgs_.end());
            BatchPair bp = it->second;
            ReceiveSubMessages(output_messages, bp.batch_index, bp.batch_size, socket);
            batch_processed += bp.batch_size;
            sub_msgs_.erase(it);
            sockets.insert(sockets.begin(), socket);
          }
        }
      }
    }
  }

  int Run() {
    if (type_ == NodeTypeHost) {
      printf("Initializing host node, prepare to get %d clients \n", clients_num_);
      // wait here to initialize all clients.
      epfds_ = epoll_create(256);
      int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
      int reuse_addr = 1;
      setsockopt(listen_fd,
                 SOL_SOCKET,
                 SO_REUSEADDR,
                 (const char*)&reuse_addr,
                 sizeof(int));
      if (bind(listen_fd, (sockaddr *)&host_addr_, sizeof(host_addr_)) < 0) {
        perror("failed to bind the address.\n");
        exit(-1);
      }
      if (listen(listen_fd, 64) < 0) {
        perror("failed to listen the server address and port.\n");
        exit(-1);
      }
      Message<Dtype> *msg = ctx_->prepare_message(InitWeight);
      while(sockets_.size() < clients_num_) {
        // accept all clients and register to epfd
        struct sockaddr_in client_addr;
        socklen_t client_len;
        char client_name[128];
        inet_ntop(AF_INET, (void*)&(host_addr_.sin_addr), client_name, 127);
        std::cout << "Listening :" << client_name << ". Port: "
                  << ntohs(host_addr_.sin_port) << std::endl;
        int conn_fd = accept(listen_fd, (sockaddr *)&client_addr, &client_len);
        inet_ntop(AF_INET, (void*)&(client_addr.sin_addr), client_name, 127);
        std::cout << "Got a client from :" << client_name << ". Port: "
                  << ntohs(client_addr.sin_port) << std::endl;
        if (conn_fd < 0) {
          perror("accept client failed.");
          exit(-1);
        }
        struct epoll_event event;
        event.data.fd = conn_fd;
        event.events = EPOLLIN | EPOLLET;
        epoll_ctl(epfds_, EPOLL_CTL_ADD, conn_fd, &event);
        sockets_.push_back(conn_fd);
        if (msg != nullptr) {
          SendMessage(msg, conn_fd);
          weights_bytes_ += msg->memory()->count(0) * sizeof(Dtype);
        }
      }
      close(listen_fd);
    } else if (type_ == NodeTypeClient) {
      // wait here to get weights or data from host.
      int conn_fd = socket(AF_INET, SOCK_STREAM, 0);
      if (conn_fd < 0) {
        perror("failed to create socket");
        exit(-1);
      }

      while (1) {
        if (connect(conn_fd, (struct sockaddr *)&host_addr_,
            sizeof(host_addr_)) < 0) {
          if (errno != ECONNREFUSED) {
            printf("err = %d ECO %d \n", errno, ECONNREFUSED);
            perror("failed to connect to host.");
            exit(-2);
          } else {
            printf("Host is not ready, retry after 1 second.\n");
            sleep(1);
          }
        } else {
          printf("Connected to host.\n");
          break;
        }
      }
      try {
        while(1) {
          std::vector <Message<Dtype> *> msgs;
          ctx_->prepare_messages(msgs);
          if (ReceiveMessages(msgs, conn_fd) < 0)
            break;
          std::vector <Message<Dtype> *> output_msgs;
          ctx_->handle_messages(msgs, output_msgs);
          SendMessages(output_msgs, conn_fd);
        }
        // graceful shutdown.
        close(conn_fd);
      } catch (int err) {
        close(conn_fd);
      }
    }
    return 0;
  }

  ~Node() {
    if (sockets_.size() != 0) {
      MessageHdr<Dtype> exit_msg(ExitMessage);
      std::cout << "Sending exit message to clients. " << std::endl;
      for (unsigned int i = 0; i < sockets_.size(); i++) {
        SendMessageHdr(exit_msg, sockets_[i]);
        close(sockets_[i]);
      }
      sockets_.clear();
      close(epfds_);
    }
    printf("Received %lu B, sent %lu B", recved_bytes_total_/*/1.e6*/, sent_bytes_total_/*/1.e6*/);
    if (type_ == NodeTypeHost) {
      printf("(including weights %lu B)", weights_bytes_/*/1.e6*/);
    }
    printf("\n");
  }

private:

  size_t sent_bytes_total_;
  size_t recved_bytes_total_;
  size_t weights_bytes_;

  ssize_t Send(int sockfd, const void *buf, size_t len, int flags) {
    size_t total_bytes_sent = 0;
    sent_bytes_total_ += len;
    while (total_bytes_sent < len) {
      ssize_t bytes_sent = send(sockfd, (const uint8_t*)buf + total_bytes_sent, len - total_bytes_sent, flags * 0);
      if (bytes_sent < 0) {
        perror("fatal error when sending data.\n");
        throw (errno);
      }
      total_bytes_sent += bytes_sent;
    }
    return total_bytes_sent;
  }

  ssize_t Recv(int sockfd, void *buf, size_t len, int flags) {
    size_t total_bytes_recved = 0;
    recved_bytes_total_ += len;

    while (total_bytes_recved < len) {
      ssize_t bytes_recved = recv(sockfd, (uint8_t*)buf + total_bytes_recved, len - total_bytes_recved, flags);
      if (bytes_recved < 0) {
        perror("fatal error when receiving data.\n");
        throw (errno);
      }
      total_bytes_recved += bytes_recved;
    }
    return total_bytes_recved;
  }

  void SendMessages(const std::vector <Message<Dtype> *> &messages,
                   int socket) {
    for (unsigned int i = 0; i < messages.size(); i++)
      SendMessage(messages[i], socket);
  }

  void SendMessage(const Message<Dtype> *message,
                   int socket) {
    MessageHdr<Dtype> hdr(message->id());
    hdr.shape[0] = message->memory()->shape()[0];
    hdr.shape[1] = message->memory()->shape()[1];
    hdr.shape[2] = message->memory()->shape()[2];
    hdr.shape[3] = message->memory()->shape()[3];
    hdr.extra_shape_len = message->has_extra_memory() ?
                          message->extra_memory()->count(0) : 0;
    int sent_bytes = Send(socket, &hdr, sizeof(hdr), MSG_MORE);
    assert(sent_bytes == sizeof(hdr));
    if (hdr.extra_shape_len > 0) {
      sent_bytes = Send(socket, message->extra_memory()->data(), hdr.extra_shape_len * sizeof(int), MSG_MORE);
      assert(sent_bytes == (int)(hdr.extra_shape_len * sizeof(int)));
    }
    sent_bytes = Send(socket, message->memory()->data(), hdr.len() * sizeof(Dtype), MSG_DONTWAIT);
    assert(sent_bytes == (int)(hdr.len() * sizeof(Dtype)));
  }

  void SendMessageHdr(MessageHdr<Dtype> hdr, int socket) {
    int sent_bytes = Send(socket, &hdr, sizeof(hdr), MSG_DONTWAIT);
    assert(sent_bytes == sizeof(hdr));
  }

  void SendSubMessages(const std::vector <Message<Dtype> *> &messages,
                      int batch_index,
                      int bs,
                      int socket) {
    for (unsigned int i = 0; i < messages.size(); i++)
      SendSubMessage(messages[i], batch_index, bs, socket);
  }

  void SendSubMessage(const Message<Dtype> *message,
                      int batch_index,
                      int bs,
                      int socket) {
    MessageHdr<Dtype> hdr(message->id());
    hdr.id = message->id();
    hdr.shape[0] = bs;
    hdr.shape[1] = message->memory()->shape()[1];
    hdr.shape[2] = message->memory()->shape()[2];
    hdr.shape[3] = message->memory()->shape()[3];
    hdr.extra_shape_len = message->has_extra_memory() ?
                          (message->extra_memory()->count(1) * bs) : 0;
    int sent_bytes = Send(socket, &hdr, sizeof(hdr), MSG_MORE);
    assert(sent_bytes == sizeof(hdr));
    if (hdr.extra_shape_len > 0) {
      sent_bytes = Send(socket, &message->extra_memory()->const_at(batch_index, 0, 0, 0),
           hdr.extra_shape_len * sizeof(int), MSG_MORE);
      assert(sent_bytes == (int)(hdr.extra_shape_len * sizeof(int)));
    }
    sent_bytes = Send(socket, &message->memory()->const_at(batch_index, 0, 0, 0),
         hdr.len() * sizeof(Dtype), MSG_DONTWAIT);
    assert(sent_bytes == (int)(hdr.len() * sizeof(Dtype)));
  }

  // Receive sub messages from client side.
  // The host side should prepare appropriate input messages.
  void ReceiveSubMessages(std::vector <Message<Dtype> *> &messages,
                         int batch_index,
                         int bs,
                         int socket) {
    for (unsigned int i = 0; i < messages.size(); i++) {
      MessageHdr<Dtype> hdr;
      int recv_bytes = Recv(socket, &hdr, sizeof(hdr), MSG_WAITALL);
      assert(recv_bytes == (int)sizeof(hdr));
      assert(hdr.data_type == GetDataType<Dtype>());

      if (hdr.len() != bs * messages[i]->memory()->count(1)) {
        messages[i]->mutable_memory()->reshape(messages[i]->memory()->shape()[0],
                                               hdr.shape[1],
                                               hdr.shape[2],
                                               hdr.shape[3]);

      }
      if (hdr.extra_shape_len != 0) {
        assert(messages[i]->has_extra_memory());
        int *shape_data = &messages[i]->mutable_extra_memory()->at(batch_index, 0, 0, 0);
        recv_bytes = Recv(socket, shape_data, hdr.extra_shape_len * sizeof(int), MSG_WAITALL);
        assert(recv_bytes == (int)(hdr.extra_shape_len * sizeof(int)));
      }

      Dtype *ptr = &messages[i]->mutable_memory()->at(batch_index, 0, 0, 0);
      recv_bytes = Recv(socket, ptr, hdr.len() * sizeof(Dtype), MSG_WAITALL);
      assert(recv_bytes == (int)(hdr.len() * sizeof(Dtype)));
    }
  }

  // Used by client side to receive messages from host side.
  int ReceiveMessages(std::vector <Message<Dtype> *> &messages,
                      int socket) {
    for (unsigned int i = 0; i < messages.size(); i++) {
      MessageHdr<Dtype> hdr;
      int recv_bytes = Recv(socket, &hdr, sizeof(MessageHdr<Dtype>), MSG_WAITALL);
      assert(recv_bytes == sizeof(MessageHdr<Dtype>));
      assert(hdr.data_type == GetDataType<Dtype>());
      if (hdr.id == ExitMessage) {
        if (i != 0)
          printf("Warning: got exit message with some unhandled data.\n");
        printf("got exit message from host, exiting. \n");
        return -1;
      }
      assert(hdr.id == messages[i]->id());
      if (hdr.extra_shape_len != 0) {
        if (!messages[i]->has_extra_memory())
          messages[i]->create_extra_memory(hdr.extra_shape_len);
        else
          messages[i]->mutable_extra_memory()->reshape(hdr.extra_shape_len, 1, 1, 1);
        int *shape_data = messages[i]->mutable_extra_memory()->mutable_data();
        recv_bytes = Recv(socket, shape_data, hdr.extra_shape_len * sizeof(int), MSG_WAITALL);
        assert(recv_bytes == (int)(hdr.extra_shape_len * sizeof(int)));
      }
      messages[i]->mutable_memory()->reshape(hdr.shape[0],
                                             hdr.shape[1],
                                             hdr.shape[2],
                                             hdr.shape[3]);
      recv_bytes = Recv(socket,
                        messages[i]->mutable_memory()->mutable_data(),
                        hdr.len() * sizeof(Dtype),
                        MSG_WAITALL);
      assert(recv_bytes == (int)(hdr.len() * sizeof(Dtype)));
    }
    return 0;
  }

  NodeType type_;
  std::vector<int> sockets_;
  std::map<int, BatchPair> sub_msgs_;
  int epfds_;
  BasicContext<Dtype, HostDtype> *ctx_;
  struct sockaddr_in host_addr_;
  unsigned int clients_num_;
};
} // namespace simple_mqueue
#endif
