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
//
// @author Sambit Das, Gourab Panigrahi
//


#include <deviceKernelsGeneric.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <DeviceBlasWrapper.h>
#include <dftUtils.h>
#include <headers.h>
#include "BLASWrapperDeviceKernels.sycl.cc"

namespace dftfe
{
  namespace utils
  {
    namespace deviceKernelsGeneric
    {
      void
      setupDevice()
      {
        // dftfe::utils::deviceBlasWrapper::create(dftfe::utils::deviceBlasWrapper::d_streamId);
        // dftfe::utils::deviceBlasWrapper::d_streamId = dftfe::utils::deviceStream_t{sycl::gpu_selector_v};
        int n_devices = 0;
        dftfe::utils::getDeviceCount(&n_devices);
        // std::cout<< "Number of Devices "<<n_devices<<std::endl;
        int device_id =
          dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) % n_devices;
        // std::cout<<"Device Id: "<<device_id<<" Task Id
        // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
        dftfe::utils::setDevice(device_id);
        // int device = 0;
        // dftfe::utils::getDevice(&device);
        // std::cout<< "Device Id currently used is "<<device<< " for taskId:
        // "<<dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;
        dftfe::utils::deviceReset();

        // #ifdef DFTFE_WITH_DEVICE_AMD
        //         dftfe::utils::deviceBlasWrapper::initialize();
        // #endif
      }


      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyComplexArrToRealArrsDevice(const dftfe::size_type  size,
                                     const ValueTypeComplex *complexArr,
                                     ValueTypeReal *         realArr,
                                     ValueTypeReal *         imagArr)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=](sycl::nd_item<1> ind){
            dftfe::copyComplexArrToRealArrsDeviceKernel(ind, 
                                            size, complexArr, realArr, imagArr);
        });
        DEVICE_API_CHECK(event);
#endif
      }



      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const ValueTypeReal *  realArr,
                                     const ValueTypeReal *  imagArr,
                                     ValueTypeComplex *     complexArr)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=](sycl::nd_item<1> ind){
            dftfe::copyRealArrsToComplexArrDeviceKernel(ind, size, realArr, imagArr, complexArr);
        });
        DEVICE_API_CHECK(event);
#endif
      }

      template <typename ValueType1, typename ValueType2>
      void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const ValueType1 *     valueType1Arr,
                                       ValueType2 *           valueType2Arr,
                                       deviceStream_t   streamId)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceEvent_t event = streamId.parallel_for(
                                                sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                                [=] (sycl::nd_item<1> ind){
            dftfe::copyValueType1ArrToValueType2ArrDeviceKernel(ind, size, valueType1Arr, valueType2Arr);
        });
        DEVICE_API_CHECK(event);
#endif
      }

      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1 *             copyFromVec,
        ValueType2 *                   copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((numContiguousBlocks * contiguousBlockSize) / 
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyToBlockDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks,
                                                    copyFromVec, copyToVecBlock,
                                                    copyFromVecStartingContiguousBlockIds);
        });
        DEVICE_API_CHECK(event);
#endif
      }


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1 *             copyFromVecBlock,
        ValueType2 *                   copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((numContiguousBlocks * contiguousBlockSize) / 
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyFromBlockDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks,
                                                      copyFromVecBlock, copyToVec,
                                                      copyFromVecStartingContiguousBlockIds);
          });
        DEVICE_API_CHECK(event);
#endif
      }


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const ValueType1 *     copyFromVec,
                                       ValueType2 *           copyToVec)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((numBlocks * blockSizeTo) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyToBlockConstantStrideDeviceKernel(ind, blockSizeTo, blockSizeFrom, numBlocks,
                                                                  startingId, copyFromVec, copyToVec);
          });

        DEVICE_API_CHECK(event);
#endif
      }

      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyConstantStride(const dftfe::size_type blockSize,
                                const dftfe::size_type strideTo,
                                const dftfe::size_type strideFrom,
                                const dftfe::size_type numBlocks,
                                const dftfe::size_type startingToId,
                                const dftfe::size_type startingFromId,
                                const ValueType1 *     copyFromVec,
                                ValueType2 *           copyToVec)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((numBlocks * blockSize) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyConstantStrideDeviceKernel(ind, blockSize, strideTo, strideFrom, numBlocks, 
                                                          startingToId, startingFromId, copyFromVec, copyToVec);
          });
        DEVICE_API_CHECK(event);
#endif
      }


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const ValueType1 *     copyFromVec,
                                         ValueType2 *           copyToVec)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((numBlocks * blockSizeFrom) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedCopyFromBlockConstantStrideDeviceKernel( ind, blockSizeTo, blockSizeFrom, numBlocks,
                                                                  startingId, copyFromVec, copyToVec);
        });
        DEVICE_API_CHECK(event);
#endif
      }

      template <typename ValueType1, typename ValueType2>
      void
      axpby(const dftfe::size_type n,
            const ValueType1 *     x,
            ValueType1 *           y,
            const ValueType2       a,
            const ValueType2       b)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=](sycl::nd_item<1> ind){
            dftfe::axpbyDeviceKernel(ind, n, x, y, a, b);
        });

        DEVICE_API_CHECK(event);
#endif
      }

      template <typename ValueType>
      void
      axpyStridedBlockAtomicAdd(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType *              addFromVec,
        ValueType *                    addToVec,
        const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((numContiguousBlocks * contiguousBlockSize) / 
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::axpyStridedBlockAtomicAddDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks, 
                                                          addFromVec, addToVec, addToVecStartingContiguousBlockIds);
          });

        DEVICE_API_CHECK(event);
#endif
      }

      template <typename ValueType>
      void
      axpyStridedBlockAtomicAdd(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType *              addFromVec,
        double *                       addToVecReal,
        double *                       addToVecImag,
        const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((contiguousBlockSize * numContiguousBlocks) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::axpyStridedBlockAtomicAddDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks,
                                                            addFromVec, addToVecReal, addToVecImag,
                                                            addToVecStartingContiguousBlockIds);
          });

        DEVICE_API_CHECK(event);
#endif
      }

      template <typename ValueType1, typename ValueType2>
      void
      ascal(const dftfe::size_type n, ValueType1 *x, const ValueType2 a)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::ascalDeviceKernel(ind, n, x, a);
          });

        DEVICE_API_CHECK(event);
#endif
      }

      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockScale(const dftfe::size_type contiguousBlockSize,
                        const dftfe::size_type numContiguousBlocks,
                        const ValueType1       a,
                        const ValueType1 *     s,
                        ValueType2 *           x)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((numContiguousBlocks * contiguousBlockSize) / 
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::stridedBlockScaleDeviceKernel(ind, contiguousBlockSize, numContiguousBlocks, a, s, x);
        });

        DEVICE_API_CHECK(event);
#endif
      }


      template <typename ValueType>
      void
      sadd(ValueType *            y,
           ValueType *            x,
           const ValueType        beta,
           const dftfe::size_type size)
      {
        const dftfe::size_type gridSize =
          (size / dftfe::utils::DEVICE_BLOCK_SIZE) +
          (size % dftfe::utils::DEVICE_BLOCK_SIZE == 0 ? 0 : 1);
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = (gridSize / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=] (sycl::nd_item<1> ind){
            dftfe::saddKernel(ind, y, x, beta, size);
        });

        DEVICE_API_CHECK(event);
#endif
      }

      void
      add(double *                          y,
          const double *                    x,
          const double                      alpha,
          const dftfe::size_type            size,
          dftfe::utils::deviceStream_t d_streamId)
      {
        dftfe::size_type incx = 1, incy = 1;
        // dftfe::utils::deviceBlasWrapper::axpy(deviceBlasWrapper::d_streamId, size, &alpha, x, incx, y, incy);
      }

      double
      l2_norm(const double *                    x,
              const dftfe::size_type            size,
              const MPI_Comm &                  mpi_communicator,
              dftfe::utils::deviceStream_t d_streamId)
      {
        dftfe::size_type incx = 1;
        double           local_nrm, nrm = 0;

        // dftfe::utils::deviceBlasWrapper::nrm2(
        //   deviceBlasHandle, size, x, incx, &local_nrm);

        // local_nrm *= local_nrm;
        // MPI_Allreduce(
        //   &local_nrm, &nrm, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

        return std::sqrt(nrm);
      }

      double
      dot(const double *                    x,
          const double *                    y,
          const dftfe::size_type            size,
          const MPI_Comm &                  mpi_communicator,
          dftfe::utils::deviceStream_t streamId)
      {
        dftfe::size_type incx = 1, incy = 1;
        double           local_sum, sum = 0;

        // dftfe::utils::deviceBlasWrapper::dot(size, x, incx, y, incy, &local_sum);
        // MPI_Allreduce(
        //   &local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

        return sum;
      }



      template void
      copyComplexArrToRealArrsDevice(const dftfe::size_type      size,
                                     const std::complex<double> *complexArr,
                                     double *                    realArr,
                                     double *                    imagArr);

      template void
      copyComplexArrToRealArrsDevice(const dftfe::size_type     size,
                                     const std::complex<float> *complexArr,
                                     float *                    realArr,
                                     float *                    imagArr);

      template void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const double *         realArr,
                                     const double *         imagArr,
                                     std::complex<double> * complexArr);

      template void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const float *          realArr,
                                     const float *          imagArr,
                                     std::complex<float> *  complexArr);
      template void
      copyComplexArrToRealArrsDevice(const dftfe::size_type     size,
                                     const std::complex<float> *complexArr,
                                     double *                   realArr,
                                     double *                   imagArr);

      template void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const double *         realArr,
                                     const double *         imagArr,
                                     std::complex<float> *  complexArr);

      template void
      sadd(double *               y,
           double *               x,
           const double           beta,
           const dftfe::size_type size);

      // for axpby
      template void
      axpby(const dftfe::size_type n,
            const double *         x,
            double *               y,
            const double           a,
            const double           b);

      template void
      axpby(const dftfe::size_type n,
            const float *          x,
            float *                y,
            const float            a,
            const float            b);

      template void
      axpby(const dftfe::size_type      n,
            const std::complex<double> *x,
            std::complex<double> *      y,
            const std::complex<double>  a,
            const std::complex<double>  b);

      template void
      axpby(const dftfe::size_type     n,
            const std::complex<float> *x,
            std::complex<float> *      y,
            const std::complex<float>  a,
            const std::complex<float>  b);


      template void
      axpby(const dftfe::size_type      n,
            const std::complex<double> *x,
            std::complex<double> *      y,
            const double                a,
            const double                b);

      template void
      axpby(const dftfe::size_type     n,
            const std::complex<float> *x,
            std::complex<float> *      y,
            const double               a,
            const double               b);


      // for ascal
      template void
      ascal(const dftfe::size_type n, double *x, const double a);

      template void
      ascal(const dftfe::size_type n, float *x, const float a);

      template void
      ascal(const dftfe::size_type     n,
            std::complex<double> *     x,
            const std::complex<double> a);

      template void
      ascal(const dftfe::size_type    n,
            std::complex<float> *     x,
            const std::complex<float> a);

      template void
      ascal(const dftfe::size_type n, std::complex<double> *x, double a);

      template void
      ascal(const dftfe::size_type n, std::complex<float> *x, double a);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       double *               valueType2Arr,
                                       const deviceStream_t   streamId);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const float *          valueType1Arr,
                                       float *                valueType2Arr,
                                       const deviceStream_t   streamId);

      template void
      copyValueType1ArrToValueType2Arr(
        const dftfe::size_type      size,
        const std::complex<double> *valueType1Arr,
        std::complex<double> *      valueType2Arr,
        const deviceStream_t        streamId);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type     size,
                                       const std::complex<float> *valueType1Arr,
                                       std::complex<float> *      valueType2Arr,
                                       const deviceStream_t       streamId);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       float *                valueType2Arr,
                                       const deviceStream_t   streamId);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const float *          valueType1Arr,
                                       double *               valueType2Arr,
                                       const deviceStream_t   streamId);

      template void
      copyValueType1ArrToValueType2Arr(
        const dftfe::size_type      size,
        const std::complex<double> *valueType1Arr,
        std::complex<float> *       valueType2Arr,
        const deviceStream_t        streamId);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type     size,
                                       const std::complex<float> *valueType1Arr,
                                       std::complex<double> *     valueType2Arr,
                                       const deviceStream_t       streamId);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       std::complex<float> *  valueType2Arr,
                                       const deviceStream_t   streamId);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const float *          valueType1Arr,
                                       std::complex<float> *  valueType2Arr,
                                       const deviceStream_t   streamId);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       std::complex<double> * valueType2Arr,
                                       const deviceStream_t   streamId);

      template void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const float *          valueType1Arr,
                                       std::complex<double> * valueType2Arr,
                                       const deviceStream_t   streamId);


      // strided copy to block
      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double *                 copyFromVec,
        double *                       copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const float *                  copyFromVec,
        float *                        copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double> *   copyFromVec,
        std::complex<double> *         copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<float> *    copyFromVec,
        std::complex<float> *          copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double *                 copyFromVec,
        float *                        copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const float *                  copyFromVec,
        double *                       copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double> *   copyFromVec,
        std::complex<float> *          copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<float> *    copyFromVec,
        std::complex<double> *         copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      // strided copy from block
      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double *                 copyFromVecBlock,
        double *                       copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const float *                  copyFromVecBlock,
        float *                        copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double> *   copyFromVecBlock,
        std::complex<double> *         copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<float> *    copyFromVecBlock,
        std::complex<float> *          copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double *                 copyFromVecBlock,
        float *                        copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const float *                  copyFromVecBlock,
        double *                       copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double> *   copyFromVecBlock,
        std::complex<float> *          copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<float> *    copyFromVecBlock,
        std::complex<double> *         copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      // strided copy to block constant stride
      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const double *         copyFromVec,
                                       double *               copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const float *          copyFromVec,
                                       float *                copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<double> *      copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type     blockSizeTo,
                                       const dftfe::size_type     blockSizeFrom,
                                       const dftfe::size_type     numBlocks,
                                       const dftfe::size_type     startingId,
                                       const std::complex<float> *copyFromVec,
                                       std::complex<float> *      copyToVec);


      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const double *         copyFromVec,
                                       float *                copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const float *          copyFromVec,
                                       double *               copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<float> *       copyToVec);

      template void
      stridedCopyToBlockConstantStride(const dftfe::size_type     blockSizeTo,
                                       const dftfe::size_type     blockSizeFrom,
                                       const dftfe::size_type     numBlocks,
                                       const dftfe::size_type     startingId,
                                       const std::complex<float> *copyFromVec,
                                       std::complex<double> *     copyToVec);

      // strided copy from block constant stride
      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const double *         copyFromVec,
                                         double *               copyToVec);

      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const float *          copyFromVec,
                                         float *                copyToVec);

      template void
      stridedCopyFromBlockConstantStride(
        const dftfe::size_type      blockSizeTo,
        const dftfe::size_type      blockSizeFrom,
        const dftfe::size_type      numBlocks,
        const dftfe::size_type      startingId,
        const std::complex<double> *copyFromVec,
        std::complex<double> *      copyToVec);

      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const std::complex<float> *copyFromVec,
                                         std::complex<float> *      copyToVec);

      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const double *         copyFromVec,
                                         float *                copyToVec);

      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const float *          copyFromVec,
                                         double *               copyToVec);

      template void
      stridedCopyFromBlockConstantStride(
        const dftfe::size_type      blockSizeTo,
        const dftfe::size_type      blockSizeFrom,
        const dftfe::size_type      numBlocks,
        const dftfe::size_type      startingId,
        const std::complex<double> *copyFromVec,
        std::complex<float> *       copyToVec);

      template void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const std::complex<float> *copyFromVec,
                                         std::complex<double> *     copyToVec);
      // strided copy  constant stride
      template void
      stridedCopyConstantStride(const dftfe::size_type blockSize,
                                const dftfe::size_type strideTo,
                                const dftfe::size_type strideFrom,
                                const dftfe::size_type numBlocks,
                                const dftfe::size_type startingToId,
                                const dftfe::size_type startingFromId,
                                const double *         copyFromVec,
                                double *               copyToVec);

      template void
      stridedCopyConstantStride(const dftfe::size_type blockSize,
                                const dftfe::size_type strideTo,
                                const dftfe::size_type strideFrom,
                                const dftfe::size_type numBlocks,
                                const dftfe::size_type startingToId,
                                const dftfe::size_type startingFromId,
                                const float *          copyFromVec,
                                float *                copyToVec);

      template void
      stridedCopyConstantStride(const dftfe::size_type      blockSize,
                                const dftfe::size_type      strideTo,
                                const dftfe::size_type      strideFrom,
                                const dftfe::size_type      numBlocks,
                                const dftfe::size_type      startingToId,
                                const dftfe::size_type      startingFromId,
                                const std::complex<double> *copyFromVec,
                                std::complex<double> *      copyToVec);

      template void
      stridedCopyConstantStride(const dftfe::size_type     blockSize,
                                const dftfe::size_type     strideTo,
                                const dftfe::size_type     strideFrom,
                                const dftfe::size_type     numBlocks,
                                const dftfe::size_type     startingToId,
                                const dftfe::size_type     startingFromId,
                                const std::complex<float> *copyFromVec,
                                std::complex<float> *      copyToVec);


      template void
      stridedCopyConstantStride(const dftfe::size_type blockSize,
                                const dftfe::size_type strideTo,
                                const dftfe::size_type strideFrom,
                                const dftfe::size_type numBlocks,
                                const dftfe::size_type startingToId,
                                const dftfe::size_type startingFromId,
                                const double *         copyFromVec,
                                float *                copyToVec);

      template void
      stridedCopyConstantStride(const dftfe::size_type blockSize,
                                const dftfe::size_type strideTo,
                                const dftfe::size_type strideFrom,
                                const dftfe::size_type numBlocks,
                                const dftfe::size_type startingToId,
                                const dftfe::size_type startingFromId,
                                const float *          copyFromVec,
                                double *               copyToVec);

      template void
      stridedCopyConstantStride(const dftfe::size_type      blockSize,
                                const dftfe::size_type      strideTo,
                                const dftfe::size_type      strideFrom,
                                const dftfe::size_type      numBlocks,
                                const dftfe::size_type      startingToId,
                                const dftfe::size_type      startingFromId,
                                const std::complex<double> *copyFromVec,
                                std::complex<float> *       copyToVec);

      template void
      stridedCopyConstantStride(const dftfe::size_type     blockSize,
                                const dftfe::size_type     strideTo,
                                const dftfe::size_type     strideFrom,
                                const dftfe::size_type     numBlocks,
                                const dftfe::size_type     startingToId,
                                const dftfe::size_type     startingFromId,
                                const std::complex<float> *copyFromVec,
                                std::complex<double> *     copyToVec);

      // stridedBlockScale
      template void
      stridedBlockScale(const dftfe::size_type contiguousBlockSize,
                        const dftfe::size_type numContiguousBlocks,
                        const double           a,
                        const double *         s,
                        double *               x);

      template void
      stridedBlockScale(const dftfe::size_type contiguousBlockSize,
                        const dftfe::size_type numContiguousBlocks,
                        const float            a,
                        const float *          s,
                        float *                x);

      template void
      stridedBlockScale(const dftfe::size_type      contiguousBlockSize,
                        const dftfe::size_type      numContiguousBlocks,
                        const std::complex<double>  a,
                        const std::complex<double> *s,
                        std::complex<double> *      x);

      template void
      stridedBlockScale(const dftfe::size_type     contiguousBlockSize,
                        const dftfe::size_type     numContiguousBlocks,
                        const std::complex<float>  a,
                        const std::complex<float> *s,
                        std::complex<float> *      x);

      template void
      stridedBlockScale(const dftfe::size_type contiguousBlockSize,
                        const dftfe::size_type numContiguousBlocks,
                        const double           a,
                        const double *         s,
                        float *                x);

      template void
      stridedBlockScale(const dftfe::size_type contiguousBlockSize,
                        const dftfe::size_type numContiguousBlocks,
                        const float            a,
                        const float *          s,
                        double *               x);

      template void
      stridedBlockScale(const dftfe::size_type      contiguousBlockSize,
                        const dftfe::size_type      numContiguousBlocks,
                        const std::complex<double>  a,
                        const std::complex<double> *s,
                        std::complex<float> *       x);

      template void
      stridedBlockScale(const dftfe::size_type     contiguousBlockSize,
                        const dftfe::size_type     numContiguousBlocks,
                        const std::complex<float>  a,
                        const std::complex<float> *s,
                        std::complex<double> *     x);

      template void
      stridedBlockScale(const dftfe::size_type contiguousBlockSize,
                        const dftfe::size_type numContiguousBlocks,
                        const double           a,
                        const double *         s,
                        std::complex<double> * x);

      template void
      stridedBlockScale(const dftfe::size_type contiguousBlockSize,
                        const dftfe::size_type numContiguousBlocks,
                        const double           a,
                        const double *         s,
                        std::complex<float> *  x);

      // axpyStridedBlockAtomicAdd
      template void
      axpyStridedBlockAtomicAdd(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double *                 addFromVec,
        double *                       addToVec,
        const dftfe::global_size_type *addToVecStartingContiguousBlockIds);

      template void
      axpyStridedBlockAtomicAdd(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double> *   addFromVec,
        std::complex<double> *         addToVec,
        const dftfe::global_size_type *addToVecStartingContiguousBlockIds);

      template void
      axpyStridedBlockAtomicAdd(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const double *                 addFromVec,
        double *                       addToVecReal,
        double *                       addToVecImag,
        const dftfe::global_size_type *addToVecStartingContiguousBlockIds);

      template void
      axpyStridedBlockAtomicAdd(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const std::complex<double> *   addFromVec,
        double *                       addToVecReal,
        double *                       addToVecImag,
        const dftfe::global_size_type *addToVecStartingContiguousBlockIds);

    } // namespace deviceKernelsGeneric
  }   // namespace utils
} // namespace dftfe
