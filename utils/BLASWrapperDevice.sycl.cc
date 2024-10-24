// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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
//

#include <BLASWrapper.h>
#include <deviceKernelsGeneric.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "BLASWrapperDeviceKernels.sycl.cc"

namespace dftfe
{
  namespace linearAlgebra
  {

    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::BLASWrapper()
    {
      d_streamId = dftfe::utils::deviceStream_t{sycl::gpu_selector_v};
    }

    dftfe::utils::deviceStream_t &
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::getDeviceStream()
    {
      return d_streamId;
    }

    dftfe::utils::deviceStream_t &
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::getDeviceBlasHandle()
    {
      return d_streamId;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::destroy()
    {
      dftfe::utils::deviceBlasStatus_t status = {};
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::setStream(
      dftfe::utils::deviceStream_t streamId)
    {
      dftfe::utils::deviceBlasStatus_t status = {};
      return status;
    }

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

    // dftfe::utils::deviceBlasStatus_t
    // dftfe::utils::deviceBlasStatus_t create(dftfe::utils::deviceStream_t d_streamId)
    // {
    //   d_streamId = sycl::queue{sycl::gpu_selector_v, exception_handler};
    //   return dftfe::utils::deviceBlasSuccess;
    // }

    dftfe::utils::deviceBlasStatus_t setMathMode(dftfe::utils::deviceBlasMath_t mathMode){
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

    template <typename ValueType>
    ValueType* device_allocation(dftfe::utils::deviceStream_t d_streamId, size_type n)
    {
        ValueType* A_device = sycl::malloc_device<ValueType>(n,d_streamId);
        d_streamId.wait();
        return A_device;
    }

    template <typename ValueType>
    void device_copy(dftfe::utils::deviceStream_t d_streamId, ValueType* A_device, const ValueType* A, int n)
    {
        dftfe::utils::deviceEvent_t event = d_streamId.memcpy(A_device,A,sizeof(ValueType)*n);
        DEVICE_API_CHECK(event);
    }

    template <typename ValueType>
    ValueType* device_allocation_copy(dftfe::utils::deviceStream_t d_streamId, const ValueType* A, int n)
    {
        ValueType* A_device = sycl::malloc_device<ValueType>(n,d_streamId);
        dftfe::utils::deviceEvent_t event = d_streamId.memcpy(A_device,A,sizeof(ValueType)*n);
        DEVICE_API_CHECK(event);
        return A_device;
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
        const unsigned int n,
        const double *     x,
        const unsigned int incx,
        double *           y,
        const unsigned int incy)
    {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::copy(d_streamId, n, x, incx, y, incy);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
        const unsigned int          n,
        const std::complex<double> *x,
        const unsigned int          incx,
        std::complex<double> *      y,
        const unsigned int          incy)
    {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::copy(d_streamId, n, x, incx, y, incy);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
        const unsigned int    n,
        const float          *x,
        const unsigned int    incx,
        float                *y,
        const unsigned int    incy)
    {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::copy(d_streamId, n, x, incx, y, incy);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
        const unsigned int    n,
        const std::complex<float>      *x,
        const unsigned int    incx,
        std::complex<float>            *y,
        const unsigned int    incy)
    {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::copy(d_streamId, n, x, incx, y, incy);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char  transA,
      const char  transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const std::complex<double> *  alpha,
      const std::complex<double> *  A,
      const unsigned int lda,
      const std::complex<double> *  B,
      const unsigned int ldb,
      const std::complex<double> *  beta,
      std::complex<double> *        C,
      const unsigned int ldc)
    {
        dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm(d_streamId, transa, transb, m, n, k, alpha,
                                                                A, lda, B, ldb, beta, C, ldc);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char  transA,
      const char  transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const double *  alpha,
      const double *  A,
      const unsigned int lda,
      const double *  B,
      const unsigned int ldb,
      const double *  beta,
      double *        C,
      const unsigned int ldc)
    {
        dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm(d_streamId, transa, transb, m, n, k, alpha,
                                                                A, lda, B, ldb, beta, C, ldc);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char  transA,
      const char  transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const float *  alpha,
      const float *  A,
      const unsigned int lda,
      const float *  B,
      const unsigned int ldb,
      const float *  beta,
      float *        C,
      const unsigned int ldc)
    {
        dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm(d_streamId, transa, transb, m, n, k, alpha,
                                                                A, lda, B, ldb, beta, C, ldc);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char  transA,
      const char  transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const std::complex<float> *  alpha,
      const std::complex<float> *  A,
      const unsigned int lda,
      const std::complex<float> *  B,
      const unsigned int ldb,
      const std::complex<float> *  beta,
      std::complex<float> *        C,
      const unsigned int ldc)
    {
        dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm(d_streamId, transa, transb, m, n, k, alpha,
                                                                A, lda, B, ldb, beta, C, ldc);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char  transA,
      const unsigned int m,
      const unsigned int n,
      const double *  alpha,
      const double *  A,
      const unsigned int lda,
      const double *  x,
      const unsigned int incx,
      const double *  beta,
      double *        y,
      const unsigned int incy)
    {
        dftfe::utils::deviceBlasOperation_t transa;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(d_streamId, transa, m, n, alpha,
                                                                A, lda, x, incx, beta, y, incy);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char  transA,
      const unsigned int m,
      const unsigned int n,
      const float *  alpha,
      const float *  A,
      const unsigned int lda,
      const float *  x,
      const unsigned int incx,
      const float *  beta,
      float *        y,
      const unsigned int incy)
    {
        dftfe::utils::deviceBlasOperation_t transa;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(d_streamId, transa, m, n, alpha,
                                                                A, lda, x, incx, beta, y, incy);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char  transA,
      const unsigned int m,
      const unsigned int n,
      const std::complex<double> *   alpha,
      const std::complex<double> *  A,
      const unsigned int lda,
      const std::complex<double> *  x,
      const unsigned int incx,
      const std::complex<double> *  beta,
      std::complex<double> *        y,
      const unsigned int incy)
    {
        dftfe::utils::deviceBlasOperation_t transa;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(d_streamId, transa, m, n, alpha,
                                                                A, lda, x, incx, beta, y, incy);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char  transA,
      const unsigned int m,
      const unsigned int n,
      const std::complex<float> *  alpha,
      const std::complex<float> *  A,
      const unsigned int lda,
      const std::complex<float> *  x,
      const unsigned int incx,
      const std::complex<float> *  beta,
      std::complex<float> *        y,
      const unsigned int incy)
    {
        dftfe::utils::deviceBlasOperation_t transa;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(d_streamId, transa, m, n, alpha,
                                                                A, lda, x, incx, beta, y, incy);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int n,
      const double *  alpha,
      const double *  x,
      const unsigned int incx,
      double *        y,
      const unsigned int incy)
    {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::axpy(d_streamId, n, alpha, x, incx, y, incy);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int n,
      const std::complex<double> *  alpha,
      const std::complex<double> *  x,
      const unsigned int incx,
      std::complex<double> *        y,
      const unsigned int incy)
    {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::axpy(d_streamId, n, alpha, x, incx, y, incy);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::add(
      double *               y,
      const double *         x,
      const double           alpha,
      const dftfe::size_type size)
    {
      xaxpy(size, &alpha, x, 1, y, 1);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int n,
      const double *  x,
      const unsigned int incx,
      const double *  y,
      const unsigned int incy,
      double *        result)
    {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::dot(d_streamId, n, x, incx, y, incy, result);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int n,
      const std::complex<double> *  x,
      const unsigned int incx,
      const std::complex<double> *  y,
      const unsigned int incy,
      std::complex<double> *        result)
    {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::dotu(d_streamId, n, x, incx, y, incy, result);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int n,
      const std::complex<double> *  x,
      const unsigned int incx,
      const std::complex<double> *  y,
      const unsigned int incy,
      const MPI_Comm &   mpi_communicator,
      std::complex<double> *        result)
    {
        std::complex<double> localResult(0.0, 0.0);
        *result = std::complex<double>(0.0, 0.0);
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::dotu(d_streamId, n, x, incx, y, incy, &localResult);
        DEVICE_API_CHECK(event);
        MPI_Allreduce(&localResult,
                    result,
                    1,
                    dataTypes::mpi_type_id(result),
                    MPI_SUM,
                    mpi_communicator);
        
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int n,
      const double *  x,
      const unsigned int incx,
      const double *  y,
      const unsigned int incy,
      const MPI_Comm &   mpi_communicator,
      double *        result)
    {
        double localResult                      = 0.0;
        *result                                 = 0.0;
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::dot(d_streamId, n, x, incx, y, incy, &localResult);
        DEVICE_API_CHECK(event);
        MPI_Allreduce(
          &localResult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const unsigned int          n,
      const double *           x,
      const unsigned int          incx,
      const MPI_Comm &   mpi_communicator,
      double *                    result)
    {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::nrm2(d_streamId, n, x, incx, result);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const unsigned int          n,
      const std::complex<double> *           x,
      const unsigned int          incx,
      const MPI_Comm &   mpi_communicator,
      double *                    result)
    {
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::nrm2(d_streamId, n, x, incx, result);
        DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
      void BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      ValueType1 *           x,
      const ValueType2       alpha,
      const dftfe::size_type n)
    {
        const unsigned int incx = 1;
        dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::scal(d_streamId, n, alpha, x, incx);
        DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char                 transA,
      const char                 transB,
      const unsigned int         m,
      const unsigned int         n,
      const unsigned int         k,
      const double           *alpha,
      const double           *A,
      const unsigned int         lda,
      long long int              strideA,
      const double           *B,
      const unsigned int         ldb,
      long long int              strideB,
      const double           *beta,
      double                 *C,
      const unsigned int         ldc,
      long long int              strideC,
      const int                  batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        // dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, transa, transb, m, n, k, alpha, 
        //                                                                 A, lda, strideA, 
        //                                                                 B, ldb, strideB, beta, 
        //                                                                 C, ldc, strideC, batchCount);
        // DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char                 transA,
      const char                 transB,
      const unsigned int         m,
      const unsigned int         n,
      const unsigned int         k,
      const std::complex<double>           *alpha,
      const std::complex<double>           *A,
      const unsigned int         lda,
      long long int              strideA,
      const std::complex<double>           *B,
      const unsigned int         ldb,
      long long int              strideB,
      const std::complex<double>           *beta,
      std::complex<double>                 *C,
      const unsigned int         ldc,
      long long int              strideC,
      const int                  batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        // dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, transa, transb, m, n, k, alpha, 
        //                                                                 A, lda, strideA, 
        //                                                                 B, ldb, strideB, beta, 
        //                                                                 C, ldc, strideC, batchCount);
        // DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char                 transA,
      const char                 transB,
      const unsigned int         m,
      const unsigned int         n,
      const unsigned int         k,
      const std::complex<float>           *alpha,
      const std::complex<float>           *A,
      const unsigned int         lda,
      long long int              strideA,
      const std::complex<float>           *B,
      const unsigned int         ldb,
      long long int              strideB,
      const std::complex<float>           *beta,
      std::complex<float>                 *C,
      const unsigned int         ldc,
      long long int              strideC,
      const int                  batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        // dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, transa, transb, m, n, k, alpha, 
        //                                                                 A, lda, strideA, 
        //                                                                 B, ldb, strideB, beta, 
        //                                                                 C, ldc, strideC, batchCount);
        // DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char                 transA,
      const char                 transB,
      const unsigned int         m,
      const unsigned int         n,
      const unsigned int         k,
      const float           *alpha,
      const float           *A,
      const unsigned int         lda,
      long long int              strideA,
      const float           *B,
      const unsigned int         ldb,
      long long int              strideB,
      const float           *beta,
      float                 *C,
      const unsigned int         ldc,
      long long int              strideC,
      const int                  batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

        // dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, transa, transb, m, n, k, alpha, 
        //                                                                 A, lda, strideA, 
        //                                                                 B, ldb, strideB, beta, 
        //                                                                 C, ldc, strideC, batchCount);
        // DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char         transA,
      const char         transB,
      const unsigned int    m,
      const unsigned int    n,
      const unsigned int    k,
      const std::complex<double> *  alpha,
      const std::complex<double> *  A[],
      const unsigned int    lda,
      const std::complex<double> *  B[],
      const unsigned int    ldb,
      const std::complex<double> *  beta,
      std::complex<double> *        C[],
      const unsigned int    ldc,
      const int          batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
        const long* group_size;
      // dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, &transa, &transb, m, n, k, alpha,
      //                                                                  A, lda, B, ldb, beta,
      //                                                                  C, ldc, batchCount, group_size);
      // DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char         transA,
      const char         transB,
      const unsigned int    m,
      const unsigned int    n,
      const unsigned int    k,
      const double *  alpha,
      const double *  A[],
      const unsigned int    lda,
      const double *  B[],
      const unsigned int    ldb,
      const double *  beta,
      double *        C[],
      const unsigned int    ldc,
      const int          batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
        const long* group_size;
      // dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, &transa, &transb, m, n, k, alpha,
      //                                                                  A, lda, B, ldb, beta,
      //                                                                  C, ldc, batchCount, group_size);
      // DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char         transA,
      const char         transB,
      const unsigned int    m,
      const unsigned int    n,
      const unsigned int    k,
      const std::complex<float> *  alpha,
      const std::complex<float> *  A[],
      const unsigned int    lda,
      const std::complex<float> *  B[],
      const unsigned int    ldb,
      const std::complex<float> *  beta,
      std::complex<float> *        C[],
      const unsigned int    ldc,
      const int          batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
        const long* group_size;
      // dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, &transa, &transb, m, n, k, alpha,
      //                                                                  A, lda, B, ldb, beta,
      //                                                                  C, ldc, batchCount, group_size);
      // DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char         transA,
      const char         transB,
      const unsigned int    m,
      const unsigned int    n,
      const unsigned int    k,
      const float *  alpha,
      const float *  A[],
      const unsigned int    lda,
      const float *  B[],
      const unsigned int    ldb,
      const float *  beta,
      float *        C[],
      const unsigned int    ldc,
      const int          batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
        const long* group_size;
      // dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemm_batch(d_streamId, &transa, &transb, m, n, k, alpha,
      //                                                                  A, lda, B, ldb, beta,
      //                                                                  C, ldc, batchCount, group_size);
      // DEVICE_API_CHECK(event);
    }

    // BlasWrapperDeviceKernels.sycl.cpp kernels used
    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(
        const unsigned int n,
        const ValueType2   alpha,
        const ValueType1 * x,
        const ValueType2   beta,
        ValueType1 *       y)
    {
        size_type total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=](sycl::nd_item<1> ind){
            dftfe::axpbyDeviceKernel(ind, n, x, y, alpha, beta);
        });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueType0,
              typename ValueType1,
              typename ValueType2,
              typename ValueType3,
              typename ValueType4>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(
      const unsigned int m,
      const unsigned int n,
      const ValueType0   alpha,
      const ValueType1 * A,
      const ValueType2 * B,
      const ValueType3 * D,
      ValueType4 *       C)
    {
        size_type total_workitems = ((n * m / dftfe::utils::DEVICE_BLOCK_SIZE) + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=](sycl::nd_item<1> ind){
            dftfe::ApaBDDeviceKernel(ind, m, n, alpha, A, B, D, C);
        });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyComplexArrToRealArrs(
      const dftfe::size_type  size,
      const ValueTypeComplex *complexArr,
      ValueTypeReal          *realArr,
      ValueTypeReal          *imagArr)
    {
        size_type total_workitems = (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=](sycl::nd_item<1> ind){
            dftfe::copyComplexArrToRealArrsDeviceKernel(ind, size, complexArr, realArr, imagArr);
        });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyRealArrsToComplexArr(
      const dftfe::size_type size,
      const ValueTypeReal *  realArr,
      const ValueTypeReal *  imagArr,
      ValueTypeComplex *     complexArr)
    {
        size_type total_workitems = (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=](sycl::nd_item<1> ind){
            dftfe::copyRealArrsToComplexArrDeviceKernel(ind, size, realArr, imagArr, complexArr);
        });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyValueType1ArrToValueType2Arr(
      const size_type           size,
      const ValueType1 *     valueType1Arr,
      ValueType2 *           valueType2Arr)
    {
      size_type total_workitems = (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
          dftfe::copyValueType1ArrToValueType2ArrDeviceKernel(ind, size, valueType1Arr, valueType2Arr);
      });
      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const size_type                          contiguousBlockSize,
      const size_type                          numContiguousBlocks,
      const ValueType1                        *copyFromVec,
      ValueType2                              *copyToVecBlock,
      const dftfe::global_size_type           *copyFromVecStartingContiguousBlockIds)
    {
        size_type total_workitems = ((numContiguousBlocks * contiguousBlockSize) / 
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;

        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyToBlockDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks,
                                                    copyFromVec, copyToVecBlock,
                                                    copyFromVecStartingContiguousBlockIds);
        });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const size_type          contiguousBlockSize,
      const size_type          numContiguousBlocks,
      const ValueType      *addFromVec,
      ValueType            *addToVec,
      const global_size_type         *addToVecStartingContiguousBlockIds)
    {
        size_type total_workitems = ((numContiguousBlocks * contiguousBlockSize) / 
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::axpyStridedBlockAtomicAddDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks, 
                                                          addFromVec, addToVec, addToVecStartingContiguousBlockIds);
          });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2, typename ValueType3>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const size_type                   contiguousBlockSize,
      const size_type                   numContiguousBlocks,
      const ValueType1               a,
      const ValueType1 *             s,
      const ValueType2 *             addFromVec,
      ValueType3 *                   addToVec,
      const global_size_type *                 addToVecStartingContiguousBlockIds)
    {
        size_type total_workitems = ((numContiguousBlocks * contiguousBlockSize) / 
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::axpyStridedBlockAtomicAddDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks, a, s,
                                                          addFromVec, addToVec, addToVecStartingContiguousBlockIds);
          });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
      const size_type       contiguousBlockSize,
      const size_type       numContiguousBlocks,
      const ValueType1  *copyFromVecBlock,
      ValueType2        *copyToVec,
      const global_size_type      *copyFromVecStartingContiguousBlockIds)
    {
        size_type total_workitems = ((numContiguousBlocks * contiguousBlockSize) / 
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyFromBlockDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks,
                                                      copyFromVecBlock, copyToVec,
                                                      copyFromVecStartingContiguousBlockIds);
          });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlockConstantStride(
      const size_type blockSizeTo,
      const size_type blockSizeFrom,
      const size_type numBlocks,
      const size_type startingId,
      const ValueType1 *     copyFromVec,
      ValueType2 *           copyToVec)
    {
        size_type total_workitems = ((numBlocks * blockSizeTo) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyToBlockConstantStrideDeviceKernel(ind, blockSizeTo, blockSizeFrom, numBlocks,
                                                                  startingId, copyFromVec, copyToVec);
          });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlockConstantStride(
      const size_type        blockSizeTo,
      const size_type        blockSizeFrom,
      const size_type        numBlocks,
      const size_type        startingId,
      const ValueType1 *  copyFromVec,
      ValueType2 *        copyToVec)
    { 
        size_type total_workitems = ((numBlocks * blockSizeFrom) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyFromBlockConstantStrideDeviceKernel( ind, blockSizeTo, blockSizeFrom, numBlocks,
                                                                  startingId, copyFromVec, copyToVec);
        });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
      const size_type blockSize,
      const size_type strideTo,
      const size_type strideFrom,
      const size_type numBlocks,
      const size_type startingToId,
      const size_type startingFromId,
      const ValueType1 *     copyFromVec,
      ValueType2 *           copyToVec)
    {
        size_type total_workitems = ((numBlocks * blockSize) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyConstantStrideDeviceKernel(ind, blockSize, strideTo, strideFrom, numBlocks, 
                                                          startingToId, startingFromId, copyFromVec, copyToVec);
          });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
      const size_type           contiguousBlockSize,
      const size_type           numContiguousBlocks,
      const ValueType1       a,
      const ValueType1 *     s,
      const ValueType2 *     copyFromVec,
      ValueType2 *           copyToVecBlock,
      const global_size_type          *copyFromVecStartingContiguousBlockIds)
    {
        size_type total_workitems = ((numContiguousBlocks * contiguousBlockSize) / 
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyToBlockScaleDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks, a, s, 
                                                        copyFromVec, copyToVecBlock, 
                                                        copyFromVecStartingContiguousBlockIds);
        });

        DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const size_type           contiguousBlockSize,
      const size_type           numContiguousBlocks,
      const ValueType1       a,
      const ValueType1 *     s,
      ValueType2 *           x)
    {
        size_type total_workitems = ((numContiguousBlocks * contiguousBlockSize) / 
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedBlockScaleDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks, a, s, x);
        });

        DEVICE_API_CHECK(event);
    }

#include "./BLASWrapperDevice.inst.cc"
  } // linearAlgebra
} // dftfe