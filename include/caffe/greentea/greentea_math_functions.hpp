/*
 * greentea_math_functions.hpp
 *
 *  Created on: Apr 6, 2015
 *      Author: fabian
 */

#ifndef GREENTEA_MATH_FUNCTIONS_HPP_
#define GREENTEA_MATH_FUNCTIONS_HPP_

#include "caffe/common.hpp"
#include "caffe/definitions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/util/math_functions.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/vector.hpp"

namespace caffe {

void caffe_gpu_memset(const uint_tp N, const int_tp alpha, void* X);

template<typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int_tp M, const int_tp N, const int_tp K,
                    const Dtype alpha, const Dtype* A, const Dtype* B,
                    const Dtype beta, Dtype* C);
                            
template<typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int_tp M,
                    const int_tp N, const Dtype alpha, const Dtype* A,
                    const Dtype* x, const Dtype beta, Dtype* y);
                            
template<typename Dtype>
void caffe_gpu_axpy(const int_tp N, const Dtype alpha, const Dtype* X,
                    Dtype* Y);
 
void caffe_gpu_memcpy(const uint_tp N, const void* X, void* Y);

template<typename Dtype>
void caffe_gpu_scal(const int_tp N, const Dtype alpha, Dtype *X);

template<typename Dtype>
void caffe_gpu_axpby(const int_tp N, const Dtype alpha, const Dtype* X,
                     const Dtype beta, Dtype* Y);
                             
template<typename Dtype>
void caffe_gpu_dot(const int_tp n, const Dtype* x, const Dtype* y, Dtype* out);
                           
template<typename Dtype>
void caffe_gpu_asum(const int_tp n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_scale(const int_tp n, const Dtype alpha, const Dtype *x,
                     Dtype* y);
                             
template<typename Dtype>
void caffe_gpu_set(const int_tp N, const Dtype alpha, Dtype* Y);

template<typename Dtype>
void caffe_gpu_sign(const int_tp n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sgnbit(const int_tp n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_add_scalar(const int_tp N, const Dtype alpha, Dtype *X);

template<typename Dtype>
void caffe_gpu_add(const int_tp N, const Dtype* a, const Dtype* b, Dtype* y);
                           
template<typename Dtype>
void caffe_gpu_sub(const int_tp N, const Dtype* a, const Dtype* b, Dtype* y);
                           
template<typename Dtype>
void caffe_gpu_mul(const int_tp N, const Dtype* a, const Dtype* b, Dtype* y);
                           
template<typename Dtype>
void caffe_gpu_div(const int_tp N, const Dtype* a, const Dtype* b, Dtype* y);
                           
template<typename Dtype>
void caffe_gpu_abs(const int_tp n, const Dtype* a, Dtype* y);

template<typename Dtype>
void caffe_gpu_exp(const int_tp n, const Dtype* a, Dtype* y);

template<typename Dtype>
void caffe_gpu_log(const int_tp n, const Dtype* a, Dtype* y);

template<typename Dtype>
void caffe_gpu_powx(const int_tp n, const Dtype* a, const Dtype b, Dtype* y);
                            
void caffe_gpu_rng_uniform(const int_tp n, unsigned int* r);

void caffe_gpu_rng_uniform(const int_tp n, unsigned long long* r);

template<typename Dtype>
void caffe_gpu_rng_uniform(const int_tp n, const Dtype a, const Dtype b,
                           Dtype* r);
                                   
template<typename Dtype>
void caffe_gpu_rng_gaussian(const int_tp n, const Dtype mu, const Dtype sigma,
                            Dtype* r);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

}  // namespace caffe

#endif  // USE GREENTEA
#endif  /* GREENTEA_MATH_FUNCTIONS_HPP_ */
