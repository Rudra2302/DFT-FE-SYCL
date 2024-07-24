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
#  include <DeviceAPICalls.h>
#  include <stdio.h>
#  include <vector>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <Exceptions.h>

namespace dftfe
{
  namespace utils
  {
    namespace
    {
      template <typename ValueType>
      void setValueKernel(sycl::nd_item<1> ind, ValueType *devPtr, ValueType value, std::size_t size)
      {
        const size_t globalThreadId = ind.get_global_id(0);
        size_t n_workgroups = ind.get_group_range(0);
        size_t n_workitems = ind.get_local_range(0);

        for (size_t idx = globalThreadId; idx < size;
            idx += n_workgroups * n_workitems)
          {
            devPtr[idx] = value;
          }
      }
    } // namespace

    deviceError_t
    deviceReset()
    {
      // deviceError_t err = cudaDeviceReset();
      // DEVICE_API_CHECK(err);
      return {};
    }


    deviceError_t
    deviceMemGetInfo(std::size_t *free, std::size_t *total)
    {
      // std::error_code x;
      // deviceError_t err(x);
      dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      try{
          *free = queue.get_device().get_info<sycl::info::device::local_mem_size>();
          *total = queue.get_device().get_info<sycl::info::device::global_mem_size>();
      }
      catch(const deviceError_t &e)
      {
          return e;
      }
      return {};
    }

    deviceError_t
    getDeviceCount(int *count)
    {
      // deviceError_t err = cudaGetDeviceCount(count);
      // DEVICE_API_CHECK(err);
      return {};
    }

    deviceError_t
    getDevice(int *deviceId)
    {
      // deviceError_t err = cudaGetDevice(deviceId);
      // DEVICE_API_CHECK(err);
      return {};
    }

    deviceError_t
    setDevice(int deviceId)
    {
      // deviceError_t err = cudaSetDevice(deviceId);
      // DEVICE_API_CHECK(err);
      return {};
    }

    deviceError_t
    deviceMalloc(void **devPtr, std::size_t size)
    {
        dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
        try {
            *devPtr = sycl::malloc_device(size,queue);
            queue.wait();
        } 
        catch (const dftfe::utils::deviceError_t &e) 
        {
            return e;
        }
        return {};
    }

    deviceError_t
    deviceMemset(void *devPtr, int value, std::size_t count)
    {
      dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      try {
          queue.memset(devPtr,value,count);
          queue.wait();
      } 
      catch (const dftfe::utils::deviceError_t &e) 
      {
          return e;
      }
      return {};
    }

    template <typename ValueType>
    void
    deviceSetValue(ValueType *devPtr, ValueType value, std::size_t size)
    {
        dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
        size_t total_workitems = (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) * dftfe::utils::DEVICE_BLOCK_SIZE;
        deviceEvent_t event = queue.parallel_for(
                                    sycl::nd_range<1>(total_workitems,
                                    dftfe::utils::DEVICE_BLOCK_SIZE), 
                                    [=](sycl::nd_item<1> ind){
            setValueKernel(ind, devPtr, value, size);
        });
        DEVICE_API_CHECK(event);
    }

    template void
    deviceSetValue(int *devPtr, int value, std::size_t size);

    template void
    deviceSetValue(long int *devPtr, long int value, std::size_t size);

    template void
    deviceSetValue(size_type *devPtr, size_type value, std::size_t size);

    template void
    deviceSetValue(global_size_type *devPtr,
                   global_size_type  value,
                   std::size_t       size);

    template void
    deviceSetValue(double *devPtr, double value, std::size_t size);

    template void
    deviceSetValue(float *devPtr, float value, std::size_t size);

    template void
    deviceSetValue(std::complex<float> *devPtr,
                   std::complex<float>  value,
                   std::size_t          size);

    template void
    deviceSetValue(std::complex<double> *devPtr,
                   std::complex<double>  value,
                   std::size_t           size);

    deviceError_t
    deviceFree(void *devPtr)
    {
      dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      try {
          sycl::free(devPtr, queue);
      } 
      catch (const dftfe::utils::deviceError_t &e) 
      {
          return e;
      }
      return {};
    }

    deviceError_t
    deviceHostMalloc(void **hostPtr, std::size_t size)
    {
      dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      try {
          *hostPtr = sycl::malloc_host(size,queue);
          queue.wait();
      } 
      catch (const dftfe::utils::deviceError_t &e) 
      {
          return e;
      }
      return {};
    }

    deviceError_t
    deviceHostFree(void *hostPtr)
    {
      dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      try {
          sycl::free(hostPtr, queue);
      } 
      catch (const dftfe::utils::deviceError_t &e) 
      {
          return e;
      }
      return {};
    }

    deviceError_t
    deviceMemcpyD2H(void *dst, const void *src, std::size_t count)
    {
      dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      try {
          queue.memcpy(dst, src, count);
      } 
      catch (const dftfe::utils::deviceError_t &e) 
      {
          return e;
      }
      return {};
    }

    deviceError_t
    deviceMemcpyD2D(void *dst, const void *src, std::size_t count)
    {
      dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      try {
          queue.memcpy(dst, src, count);
      } 
      catch (const dftfe::utils::deviceError_t &e) 
      {
          return e;
      }
      return {};
    }
    deviceError_t
    deviceMemcpyH2D(void *dst, const void *src, std::size_t count)
    {
      dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      try {
          queue.memcpy(dst, src, count);
      } 
      catch (const dftfe::utils::deviceError_t &e) 
      {
          return e;
      }
      return {};
    }

    deviceError_t
    deviceMemcpyD2H_2D(void *      dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height)
    {
      // dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      // deviceEvent_t event = queue.sycl::_V1::queue::ext_oneapi_memcpy2d(dst, dpitch, src, spitch, width, height);
      // DEVICE_API_CHECK(event);
      return {};
    }


    deviceError_t
    deviceMemcpyD2D_2D(void *      dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height)
    {
      // dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      // deviceEvent_t event = queue.sycl::_V1::queue::ext_oneapi_memcpy2d(dst, dpitch, src, spitch, width, height);
      // DEVICE_API_CHECK(event);
      return {};
    }

    deviceError_t
    deviceMemcpyH2D_2D(void *      dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height)
    {
      // dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      // deviceEvent_t event = queue.sycl::_V1::queue::ext_oneapi_memcpy2d(dst, dpitch, src, spitch, width, height);
      // DEVICE_API_CHECK(event);
      return {};
    }

    deviceError_t
    deviceSynchronize()
    {
      dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      try {
          queue.wait();
      } 
      catch (const dftfe::utils::deviceError_t &e) {
          return e;
      }
      return {};
    }

    deviceError_t
    deviceMemcpyAsyncD2H(void *         dst,
                         const void *   src,
                         std::size_t    count,
                         deviceStream_t stream)
    {
      try {
          stream.memcpy(dst, src, count);
      } 
      catch (const dftfe::utils::deviceError_t &e) 
      {
          return e;
      }
      return {};
    }

    deviceError_t
    deviceMemcpyAsyncD2D(void *         dst,
                         const void *   src,
                         std::size_t    count,
                         deviceStream_t stream)
    {
      try {
          stream.memcpy(dst, src, count);
      } 
      catch (const dftfe::utils::deviceError_t &e) 
      {
          return e;
      }
      return {};
    }

    deviceError_t
    deviceMemcpyAsyncH2D(void *         dst,
                         const void *   src,
                         std::size_t    count,
                         deviceStream_t stream)
    {
      try {
          stream.memcpy(dst, src, count);
      } 
      catch (const dftfe::utils::deviceError_t &e) 
      {
          return e;
      }
      return {};
    }

    deviceError_t
    deviceStreamCreate(deviceStream_t *pStream, const bool nonBlocking)
    {
      // pStream = dftfe::utils::deviceStream_t{sycl::gpu_selector_v};
      // if (!nonBlocking)
      //   {
      //     deviceError_t err = cudaStreamCreate(pStream);
      //     DEVICE_API_CHECK(err);
      //     return err;
      //   }
      // else
      //   {
      //     int priority;
      //     cudaDeviceGetStreamPriorityRange(NULL, &priority);
      //     deviceError_t err =
      //       cudaStreamCreateWithPriority(pStream,
      //                                    cudaStreamNonBlocking,
      //                                    priority);
      //     DEVICE_API_CHECK(err);
      //     return err;
      //   }

      return {};
    }

    deviceError_t
    deviceStreamDestroy(deviceStream_t stream)
    {
      return {};
    }

    deviceError_t
    deviceStreamSynchronize(deviceStream_t stream)
    {
      dftfe::utils::deviceStream_t queue{sycl::gpu_selector_v};
      try {
          queue.wait();
      } 
      catch (const dftfe::utils::deviceError_t &e) {
          return e;
      }
      return {};
    }

    deviceError_t
    deviceEventCreate(deviceEvent_t *pEvent)
    {
      return {};
    }

    deviceError_t
    deviceEventDestroy(deviceEvent_t event)
    {
      return {};
    }

    deviceError_t
    deviceEventRecord(deviceEvent_t event, deviceStream_t stream)
    {
      // deviceError_t err = cudaEventRecord(event, stream);
      // DEVICE_API_CHECK(err);
      // return err;

      return {};
    }

    deviceError_t
    deviceEventSynchronize(deviceEvent_t event)
    {
      event.wait_and_throw();
      return {};
    }

    deviceError_t
    deviceStreamWaitEvent(deviceStream_t stream,
                          deviceEvent_t  event,
                          unsigned int   flags)
    {
      event.wait_and_throw();
      stream.wait_and_throw();
      return {};
    }

  } // namespace utils
} // namespace dftfe
#endif
