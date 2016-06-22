#ifndef GREENTEA_IM2COL_HPP_
#define GREENTEA_IM2COL_HPP_
#ifdef USE_GREENTEA

#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/vector.hpp"

namespace caffe {

template<typename Dtype>
void im2col_gpu(const Dtype *data_im,
                const int_tp channels, const int_tp height,
                const int_tp width, const int_tp kernel_h,
                const int_tp kernel_w, const int_tp pad_h,
                const int_tp pad_w, const int_tp stride_h,
                const int_tp stride_w, const int_tp dilation_h,
                const int_tp dilation_w, Dtype *data_col);

template<typename Dtype>
void col2im_gpu(const Dtype *data_col,const int_tp channels,
                const int_tp height, const int_tp width,
                const int_tp patch_h, const int_tp patch_w,
                const int_tp pad_h, const int_tp pad_w,
                const int_tp stride_h, const int_tp stride_w,
                const int_tp dilation_h, const int_tp dilation_w,
                Dtype *data_im);

template<typename Dtype>
void im2col_nd_gpu(const Dtype* data_im, const int_tp num_spatial_axes,
                   const int_tp num_kernels, const int_tp* im_shape,
                   const int_tp* col_shape, const int_tp* kernel_shape,
                   const int_tp* pad, const int_tp* stride,
                   const int_tp* dilation, Dtype* data_col);

template<typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int_tp num_spatial_axes,
                   const int_tp im_size, const int_tp* im_shape,
                   const int_tp* col_shape, const int_tp* kernel_shape,
                   const int_tp* pad, const int_tp* stride,
                   const int_tp* dilation, Dtype* data_im);
}  // namespace caffe

#endif  // USE_GREENTEA
#endif  /* GREENTEA_IM2COL_HPP_ */
