/*
 * greentea_im2col.cpp
 *
 *  Created on: Apr 8, 2015
 *      Author: Fabian Tschopp
 */
#include "caffe/common.hpp"
#include "caffe/device.hpp"
#ifdef USE_GREENTEA
#include "caffe/util/im2col.hpp"

namespace caffe {
template<typename Dtype>
void im2col_gpu(const Dtype *data_im,
                const int_tp channels, const int_tp height,
                const int_tp width, const int_tp kernel_h,
                const int_tp kernel_w, const int_tp pad_h,
                const int_tp pad_w, const int_tp stride_h,
                const int_tp stride_w, const int_tp dilation_h,
                const int_tp dilation_w, Dtype *data_col) {
  int_tp height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1))
      / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1))
      / stride_w + 1;
  int_tp num_kernels = channels * height_col * width_col;

  ClState& clState = Caffe::cl_state();

  ClMemOff<Dtype> buf_data_im = clState.get_buffer_mem(data_im);
  ClMemOff<Dtype> buf_data_col = clState.get_buffer_mem(data_col);

  int dev_id = clState.get_mem_dev(buf_data_im.memobj);
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &prog = (Caffe::Get().GetDevice(dev_id, false))
                                  ->program();

  int offset_data_im = buf_data_im.offset;
  int offset_data_col = buf_data_col.offset;

  uint mem_base_address_align;
  clGetDeviceInfo(ctx.devices()[dev_id].id(), CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                  sizeof(uint), &mem_base_address_align, NULL);
  mem_base_address_align = (mem_base_address_align / 8) / sizeof(Dtype);

  size_t aligned_offset_data_im =
    static_cast<int>(offset_data_im /
    (static_cast<float>(mem_base_address_align))) * mem_base_address_align;
  cl_int offset_offset_data_im =
    ((offset_data_im / (static_cast<float>(mem_base_address_align))) -
    aligned_offset_data_im / mem_base_address_align) * mem_base_address_align;
  cl_mem bufDataIm = clState.create_subbuffer(data_im, aligned_offset_data_im);

  size_t aligned_offset_data_col =
    (static_cast<int>(offset_data_col /
    (static_cast<float>(mem_base_address_align)))) * mem_base_address_align;
  cl_int offset_offset_data_col =
    ((offset_data_col / (static_cast<float>(mem_base_address_align))) -
    aligned_offset_data_col / mem_base_address_align) *
    mem_base_address_align;
  cl_mem bufDataCol =
    clState.create_subbuffer(data_col, aligned_offset_data_col);

  viennacl::ocl::kernel &kernel = prog.get_kernel(CL_KERNEL_SELECT("im2col"));

  viennacl::ocl::enqueue(
    kernel(num_kernels, WrapHandle(bufDataIm, &ctx), offset_offset_data_im,
           height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
           stride_w, dilation_h, dilation_w, height_col, width_col,
           WrapHandle(bufDataCol, &ctx), offset_offset_data_col),
    ctx.get_queue());
}

// Explicit instantiation
template void im2col_gpu<float>(const float *data_im,
                                         const int_tp channels,
                                         const int_tp height,
                                         const int_tp width,
                                         const int_tp kernel_h,
                                         const int_tp kernel_w,
                                         const int_tp pad_h, const int_tp pad_w,
                                         const int_tp stride_h,
                                         const int_tp stride_w,
                                         const int_tp dilation_h,
                                         const int_tp dilation_w,
                                         float *data_col);

template void im2col_gpu<double>(const double *data_im,
                                         const int_tp channels,
                                         const int_tp height,
                                         const int_tp width,
                                         const int_tp kernel_h,
                                         const int_tp kernel_w,
                                         const int_tp pad_h, const int_tp pad_w,
                                         const int_tp stride_h,
                                         const int_tp stride_w,
                                         const int_tp dilation_h,
                                         const int_tp dilation_w,
                                         double *data_col);

template<typename Dtype>
void col2im_gpu(const Dtype *data_col, const int_tp channels,
                const int_tp height, const int_tp width,
                const int_tp kernel_h, const int_tp kernel_w,
                const int_tp pad_h, const int_tp pad_w,
                const int_tp stride_h, const int_tp stride_w,
                const int_tp dilation_h, const int_tp dilation_w,
                Dtype* data_im) {
  int_tp height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1))
    / stride_h + 1;
  int_tp width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1))
    / stride_w + 1;
  int_tp num_kernels = channels * height * width;

  ClState& clState = Caffe::cl_state();

  ClMemOff<Dtype> buf_data_im = clState.get_buffer_mem(data_im);
  ClMemOff<Dtype> buf_data_col = clState.get_buffer_mem(data_col);

  int dev_id = clState.get_mem_dev(buf_data_im.memobj);
  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &prog = (Caffe::Get().GetDevice(dev_id, false))
                                 ->program();

  int offset_data_im = buf_data_im.offset;
  int offset_data_col = buf_data_col.offset;

  uint mem_base_address_align;
  clGetDeviceInfo(ctx.devices()[dev_id].id(), CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                  sizeof(uint), &mem_base_address_align, NULL);
  mem_base_address_align = (mem_base_address_align / 8) / sizeof(Dtype);

  size_t aligned_offset_data_im =
    static_cast<int>(offset_data_im /
    (static_cast<float>(mem_base_address_align))) * mem_base_address_align;
  cl_int offset_offset_data_im =
    ((offset_data_im / (static_cast<float>(mem_base_address_align))) -
    aligned_offset_data_im / mem_base_address_align) * mem_base_address_align;
  cl_mem bufDataIm = clState.create_subbuffer(data_im, aligned_offset_data_im);

  size_t aligned_offset_data_col =
    (static_cast<int>(offset_data_col /
    (static_cast<float>(mem_base_address_align)))) * mem_base_address_align;
  cl_int offset_offset_data_col =
    ((offset_data_col / (static_cast<float>(mem_base_address_align))) -
    aligned_offset_data_col / mem_base_address_align) *
    mem_base_address_align;
  cl_mem bufDataCol =
    clState.create_subbuffer(data_col, aligned_offset_data_col);

  viennacl::ocl::kernel &kernel = prog.get_kernel(CL_KERNEL_SELECT("col2im"));

  viennacl::ocl::enqueue(
    kernel(num_kernels, WrapHandle(bufDataCol, &ctx), offset_offset_data_col,
           height, width, channels, kernel_h, kernel_w, pad_h, pad_w,
           stride_h, stride_w, dilation_h, dilation_w, height_col, width_col,
           WrapHandle(bufDataIm, &ctx), offset_offset_data_im),
    ctx.get_queue());
}

template void col2im_gpu<float>(const float *data_col,
                                         const int_tp channels,
                                         const int_tp height,
                                         const int_tp width,
                                         const int_tp patch_h,
                                         const int_tp patch_w,
                                         const int_tp pad_h, const int_tp pad_w,
                                         const int_tp stride_h,
                                         const int_tp stride_w,
                                         const int_tp dilation_h,
                                         const int_tp dilation_w,
                                         float *data_im);

template void col2im_gpu<double>(const double *data_col,
                                         const int_tp channels,
                                         const int_tp height,
                                         const int_tp width,
                                         const int_tp patch_h,
                                         const int_tp patch_w,
                                         const int_tp pad_h, const int_tp pad_w,
                                         const int_tp stride_h,
                                         const int_tp stride_w,
                                         const int_tp dilation_h,
                                         const int_tp dilation_w,
                                         double *data_im);

template<typename Dtype>
void im2col_nd_gpu(const Dtype* data_im, const int_tp num_spatial_axes,
                   const int_tp num_kernels, const int_tp* im_shape,
                   const int_tp* col_shape, const int_tp* kernel_shape,
                   const int_tp* pad, const int_tp* stride,
                   const int_tp* dilation, Dtype* data_col) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<Dtype> buf_data_im = clState.get_buffer_mem(data_im);
  ClMemOff<int_tp> buf_im_shape = clState.get_buffer_mem(im_shape);
  ClMemOff<int_tp> buf_col_shape = clState.get_buffer_mem(col_shape);
  ClMemOff<int_tp> buf_kernel_shape = clState.get_buffer_mem(kernel_shape);
  ClMemOff<int_tp> buf_pad = clState.get_buffer_mem(pad);
  ClMemOff<int_tp> buf_stride = clState.get_buffer_mem(stride);
  ClMemOff<int_tp> buf_dilation = clState.get_buffer_mem(dilation);
  ClMemOff<Dtype> buf_data_col = clState.get_buffer_mem(data_col);
  int dev_id = clState.get_mem_dev(buf_data_im.memobj);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &prog = (Caffe::Get().GetDevice(dev_id, false))
                                 ->program();
  viennacl::ocl::kernel &kernel = prog.get_kernel(
      CL_KERNEL_SELECT("im2col_nd"));

  viennacl::ocl::enqueue(
      kernel(num_kernels, num_spatial_axes, (int_tp)0,
             WrapHandle(buf_data_im.memobj, &ctx), (int_tp)buf_data_im.offset,
             WrapHandle(buf_im_shape.memobj, &ctx),
             WrapHandle(buf_col_shape.memobj, &ctx),
             WrapHandle(buf_kernel_shape.memobj, &ctx),
             WrapHandle(buf_pad.memobj, &ctx),
             WrapHandle(buf_stride.memobj, &ctx),
             WrapHandle(buf_dilation.memobj, &ctx),
             WrapHandle(buf_data_col.memobj, &ctx),
             (int_tp)buf_data_col.offset),
      ctx.get_queue());
}

template void im2col_nd_gpu<float>(const float* data_im,
                                   const int_tp num_spatial_axes,
                                   const int_tp num_kernels,
                                   const int_tp* im_shape,
                                   const int_tp* col_shape,
                                   const int_tp* kernel_shape,
                                   const int_tp* pad,
                                   const int_tp* stride,
                                   const int_tp* dilation,
                                   float* data_col);

template void im2col_nd_gpu<double>(const double* data_im,
                                    const int_tp num_spatial_axes,
                                    const int_tp num_kernels,
                                    const int_tp* im_shape,
                                    const int_tp* col_shape,
                                    const int_tp* kernel_shape,
                                    const int_tp* pad,
                                    const int_tp* stride,
                                    const int_tp* dilation,
                                    double* data_col);

template<typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int_tp num_spatial_axes,
                   const int_tp im_size, const int_tp* im_shape,
                   const int_tp* col_shape, const int_tp* kernel_shape,
                   const int_tp* pad, const int_tp* stride,
                   const int_tp* dilation, Dtype* data_im) {
  ClState& clState = Caffe::cl_state();
  ClMemOff<Dtype> buf_data_col = clState.get_buffer_mem(data_col);
  ClMemOff<int_tp> buf_im_shape = clState.get_buffer_mem(im_shape);
  ClMemOff<int_tp> buf_col_shape = clState.get_buffer_mem(col_shape);
  ClMemOff<int_tp> buf_kernel_shape = clState.get_buffer_mem(kernel_shape);
  ClMemOff<int_tp> buf_pad = clState.get_buffer_mem(pad);
  ClMemOff<int_tp> buf_stride = clState.get_buffer_mem(stride);
  ClMemOff<int_tp> buf_dilation = clState.get_buffer_mem(dilation);
  ClMemOff<Dtype> buf_data_im = clState.get_buffer_mem(data_im);

  int dev_id = clState.get_mem_dev(buf_data_col.memobj);

  viennacl::ocl::context &ctx = viennacl::ocl::get_context(dev_id);
  viennacl::ocl::program &prog = (Caffe::Get().GetDevice(dev_id, false))
                                 ->program();

  viennacl::ocl::kernel &kernel = prog.get_kernel(
      CL_KERNEL_SELECT("col2im_nd"));

  viennacl::ocl::enqueue(
    kernel(im_size, num_spatial_axes, (int_tp)0,
           WrapHandle(buf_data_col.memobj, &ctx), (int_tp)buf_data_col.offset,
           WrapHandle(buf_im_shape.memobj, &ctx),
           WrapHandle(buf_col_shape.memobj, &ctx),
           WrapHandle(buf_kernel_shape.memobj, &ctx),
           WrapHandle(buf_pad.memobj, &ctx),
           WrapHandle(buf_stride.memobj, &ctx),
           WrapHandle(buf_dilation.memobj, &ctx),
           WrapHandle(buf_data_im.memobj, &ctx),
           (int_tp)buf_data_im.offset),
    ctx.get_queue());
}

template void col2im_nd_gpu<float>(const float* data_col,
                                   const int_tp num_spatial_axes,
                                   const int_tp im_size,
                                   const int_tp* im_shape,
                                   const int_tp* col_shape,
                                   const int_tp* kernel_shape,
                                   const int_tp* pad,
                                   const int_tp* stride,
                                   const int_tp* dilation,
                                   float* data_im);

template void col2im_nd_gpu<double>(const double* data_col,
                                    const int_tp num_spatial_axes,
                                    const int_tp im_size,
                                    const int_tp* im_shape,
                                    const int_tp* col_shape,
                                    const int_tp* kernel_shape,
                                    const int_tp* pad,
                                    const int_tp* stride,
                                    const int_tp* dilation,
                                    double* data_im);

}  // namespace caffe
#endif
