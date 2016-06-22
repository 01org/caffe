#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template<typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  int_tp count = bottom[0]->count();
  Dtype dot;

  caffe_gpu_sub<Dtype>(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                       diff_.mutable_gpu_data());
  // Scale the error element-wise
  if (bottom.size() == 3) {
    caffe_gpu_mul<Dtype>(count, diff_.mutable_gpu_data(),
                         bottom[2]->gpu_data(), diff_.mutable_gpu_data());
  }
  caffe_gpu_dot<Dtype>(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / static_cast<Dtype>(bottom[0]->count(0)) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int_tp i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0]
          / static_cast<Dtype>(bottom[0]->count(0));

    caffe_gpu_axpby(bottom[i]->count(),                 // count
                    alpha,                              // alpha
                    diff_.gpu_data(),                   // a
                    Dtype(0),                           // beta
                    bottom[i]->mutable_gpu_diff());     // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
