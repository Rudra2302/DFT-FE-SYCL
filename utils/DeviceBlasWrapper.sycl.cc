// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------


#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
#  include <DeviceBlasWrapper.h>
#  include <stdio.h>
#  include <vector>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <Exceptions.h>
#  include <sycl/sycl.hpp>
#  include <oneapi/mkl.hpp>
namespace dftfe
{
  namespace utils
  {
    namespace deviceBlasWrapper
    {
      deviceStream_t d_streamId{sycl::gpu_selector_v};
      auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
          try {
            std::rethrow_exception(e);
          } catch (dftfe::utils::deviceError_t const &e) {
            std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                      << e.what() << std::endl
                      << "Exception caught at file:" << __FILE__
                      << ", line:" << __LINE__ << std::endl;
          }
        }
      };

      deviceBlasStatus_t
      create(deviceStream_t d_streamId)
      {
        d_streamId = dftfe::utils::deviceStream_t{sycl::gpu_selector_v}; //, exception_handler);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      destroy(deviceBlasHandle_t handle)
      {
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      setStream(deviceBlasHandle_t handle, deviceStream_t stream)
      {
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      setMathMode(deviceBlasHandle_t handle, deviceBlasMath_t mathMode)
      {
        if(mathMode == dftfe::utils::DEVICEBLAS_TF32_TENSOR_OP_MATH)
            setenv("MKL_BLAS_COMPUTE_MODE", "FLOAT_TO_TF32", 1);
        else if(mathMode == dftfe::utils::DEVICEBLAS_DEFAULT_MATH)
            setenv("MKL_BLAS_COMPUTE_MODE", "STANDARD", 1);
        else if(mathMode == oneapi::mkl::blas::compute_mode::float_to_bf16)
            setenv("MKL_BLAS_COMPUTE_MODE", "FLOAT_TO_BF16", 1);
        else if(mathMode == oneapi::mkl::blas::compute_mode::float_to_bf16x2)
            setenv("MKL_BLAS_COMPUTE_MODE", "FLOAT_TO_BF16X2", 1);
        else if(mathMode == oneapi::mkl::blas::compute_mode::float_to_bf16x3)
            setenv("MKL_BLAS_COMPUTE_MODE", "FLOAT_TO_BF16X3", 1);
        else if(mathMode == oneapi::mkl::blas::compute_mode::complex_3m)
            setenv("MKL_BLAS_COMPUTE_MODE", "COMPLEX_3M", 1);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      copy(deviceStream_t     d_streamId,
           int                n,
           const double *     x,
           int                incx,
           double *           y,
           int                incy)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::copy(d_streamId, n, x, incx, y, incy);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      nrm2(deviceStream_t     d_streamId,
           int                n,
           const double *     x,
           int                incx,
           double *           result)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::nrm2(d_streamId, n, x, incx, result);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      dot(deviceStream_t     d_streamId,
          int                n,
          const double *     x,
          int                incx,
          const double *     y,
          int                incy,
          double *           result)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::dot(d_streamId, n, x, incx, y, incy, result);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      axpy(deviceStream_t     d_streamId,
           int                n,
           const double *     alpha,
           const double *     x,
           int                incx,
           double *           y,
           int                incy)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::axpy(d_streamId, n, alpha, x, incx, y, incy);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemm(deviceStream_t     d_streamId,
           deviceBlasOperation_t transa,
           deviceBlasOperation_t transb,
           int                   m,
           int                   n,
           int                   k,
           const double *        alpha,
           const double *        A,
           int                   lda,
           const double *        B,
           int                   ldb,
           const double *        beta,
           double *              C,
           int                   ldc)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm(d_streamId, transa, transb, m, n, k, alpha,
                                                                A, lda, B, ldb, beta, C, ldc);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemm(deviceStream_t     d_streamId,
           deviceBlasOperation_t transa,
           deviceBlasOperation_t transb,
           int                   m,
           int                   n,
           int                   k,
           const float *         alpha,
           const float *         A,
           int                   lda,
           const float *         B,
           int                   ldb,
           const float *         beta,
           float *               C,
           int                   ldc)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm(d_streamId, transa, transb, m, n, k, alpha,
                                                                A, lda, B, ldb, beta, C, ldc);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemm(deviceStream_t     d_streamId,
           deviceBlasOperation_t       transa,
           deviceBlasOperation_t       transb,
           int                         m,
           int                         n,
           int                         k,
           const std::complex<double> *alpha,
           const std::complex<double> *A,
           int                         lda,
           const std::complex<double> *B,
           int                         ldb,
           const std::complex<double> *beta,
           std::complex<double> *      C,
           int                         ldc)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm(d_streamId, transa, transb, m, n, k, alpha,
                                                                A, lda, B, ldb, beta, C, ldc);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemm(deviceStream_t     d_streamId,
           deviceBlasOperation_t      transa,
           deviceBlasOperation_t      transb,
           int                        m,
           int                        n,
           int                        k,
           const std::complex<float> *alpha,
           const std::complex<float> *A,
           int                        lda,
           const std::complex<float> *B,
           int                        ldb,
           const std::complex<float> *beta,
           std::complex<float> *      C,
           int                        ldc)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm(d_streamId, transa, transb, m, n, k, alpha,
                                                                A, lda, B, ldb, beta, C, ldc);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemmBatched(deviceStream_t     d_streamId,
                  deviceBlasOperation_t transa,
                  deviceBlasOperation_t transb,
                  int                   m,
                  int                   n,
                  int                   k,
                  const double *        alpha,
                  const double *        Aarray[],
                  int                   lda,
                  const double *        Barray[],
                  int                   ldb,
                  const double *        beta,
                  double *              Carray[],
                  int                   ldc,
                  int                   batchCount)
      {
        const long* group_size;
        // dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, &transa, &transb, m, n, k, alpha,
        //                                                                Aarray, lda, Barray, ldb, beta,
        //                                                                Carray, ldc, batchCount, group_size);
        // DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemmBatched(deviceStream_t     d_streamId,
                  deviceBlasOperation_t       transa,
                  deviceBlasOperation_t       transb,
                  int                         m,
                  int                         n,
                  int                         k,
                  const std::complex<double> *alpha,
                  const std::complex<double> *Aarray[],
                  int                         lda,
                  const std::complex<double> *Barray[],
                  int                         ldb,
                  const std::complex<double> *beta,
                  std::complex<double> *      Carray[],
                  int                         ldc,
                  int                         batchCount)
      {
        const long* group_size;
        // dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, &transa, &transb, m, n, k, alpha,
        //                                                                Aarray, lda, Barray, ldb, beta,
        //                                                                Carray, ldc, batchCount, group_size);
        // DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemmStridedBatched(deviceStream_t     d_streamId,
                         deviceBlasOperation_t transa,
                         deviceBlasOperation_t transb,
                         int                   m,
                         int                   n,
                         int                   k,
                         const double *        alpha,
                         const double *        A,
                         int                   lda,
                         long long int         strideA,
                         const double *        B,
                         int                   ldb,
                         long long int         strideB,
                         const double *        beta,
                         double *              C,
                         int                   ldc,
                         long long int         strideC,
                         int                   batchCount)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, transa, transb, m, n, k, alpha, 
                                                                        A, lda, strideA, 
                                                                        B, ldb, strideB, beta, 
                                                                        C, ldc, strideC, batchCount);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemmStridedBatched(deviceStream_t     d_streamId,
                         deviceBlasOperation_t transa,
                         deviceBlasOperation_t transb,
                         int                   m,
                         int                   n,
                         int                   k,
                         const float *         alpha,
                         const float *         A,
                         int                   lda,
                         long long int         strideA,
                         const float *         B,
                         int                   ldb,
                         long long int         strideB,
                         const float *         beta,
                         float *               C,
                         int                   ldc,
                         long long int         strideC,
                         int                   batchCount)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, transa, transb, m, n, k, alpha, 
                                                                        A, lda, strideA, 
                                                                        B, ldb, strideB, beta, 
                                                                        C, ldc, strideC, batchCount);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }


      deviceBlasStatus_t
      gemmStridedBatched(deviceStream_t     d_streamId,
                         deviceBlasOperation_t       transa,
                         deviceBlasOperation_t       transb,
                         int                         m,
                         int                         n,
                         int                         k,
                         const std::complex<double> *alpha,
                         const std::complex<double> *A,
                         int                         lda,
                         long long int               strideA,
                         const std::complex<double> *B,
                         int                         ldb,
                         long long int               strideB,
                         const std::complex<double> *beta,
                         std::complex<double> *      C,
                         int                         ldc,
                         long long int               strideC,
                         int                         batchCount)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, transa, transb, m, n, k, alpha, 
                                                                        A, lda, strideA, 
                                                                        B, ldb, strideB, beta, 
                                                                        C, ldc, strideC, batchCount);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemmStridedBatched(deviceStream_t     d_streamId,
                         deviceBlasOperation_t      transa,
                         deviceBlasOperation_t      transb,
                         int                        m,
                         int                        n,
                         int                        k,
                         const std::complex<float> *alpha,
                         const std::complex<float> *A,
                         int                        lda,
                         long long int              strideA,
                         const std::complex<float> *B,
                         int                        ldb,
                         long long int              strideB,
                         const std::complex<float> *beta,
                         std::complex<float> *      C,
                         int                        ldc,
                         long long int              strideC,
                         int                        batchCount)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, transa, transb, m, n, k, alpha, 
                                                                        A, lda, strideA, 
                                                                        B, ldb, strideB, beta, 
                                                                        C, ldc, strideC, batchCount);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemv(deviceStream_t     d_streamId,
           deviceBlasOperation_t trans,
           int                   m,
           int                   n,
           const double *        alpha,
           const double *        A,
           int                   lda,
           const double *        x,
           int                   incx,
           const double *        beta,
           double *              y,
           int                   incy)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(d_streamId, trans, m, n, alpha,
                                                                A, lda, x, incx, beta, y, incy);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemv(deviceStream_t     d_streamId,
           deviceBlasOperation_t trans,
           int                   m,
           int                   n,
           const float *         alpha,
           const float *         A,
           int                   lda,
           const float *         x,
           int                   incx,
           const float *         beta,
           float *               y,
           int                   incy)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(d_streamId, trans, m, n, alpha,
                                                                A, lda, x, incx, beta, y, incy);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemv(deviceStream_t     d_streamId,
           deviceBlasOperation_t       trans,
           int                         m,
           int                         n,
           const std::complex<double> *alpha,
           const std::complex<double> *A,
           int                         lda,
           const std::complex<double> *x,
           int                         incx,
           const std::complex<double> *beta,
           std::complex<double> *      y,
           int                         incy)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(d_streamId, trans, m, n, alpha,
                                                                A, lda, x, incx, beta, y, incy);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

      deviceBlasStatus_t
      gemv(deviceStream_t     d_streamId,
           deviceBlasOperation_t      trans,
           int                        m,
           int                        n,
           const std::complex<float> *alpha,
           const std::complex<float> *A,
           int                        lda,
           const std::complex<float> *x,
           int                        incx,
           const std::complex<float> *beta,
           std::complex<float> *      y,
           int                        incy)
      {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(d_streamId, trans, m, n, alpha,
                                                                A, lda, x, incx, beta, y, incy);
        DEVICEBLAS_API_CHECK(event);
        return dftfe::utils::deviceBlasSuccess;
      }

    } // namespace deviceBlasWrapper
  }   // namespace utils
} // namespace dftfe
#endif
