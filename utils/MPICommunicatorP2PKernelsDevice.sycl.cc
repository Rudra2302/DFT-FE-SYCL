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


/*
 * @author Sambit Das.
 */

#ifdef DFTFE_WITH_DEVICE
#  include <DeviceKernelLauncherConstants.h>
#  include <DeviceDataTypeOverloads.h>
#  include <MPICommunicatorP2PKernels.h>
#  include <Exceptions.h>
#  include <complex>
#  include <algorithm>
#  include <deviceKernelsGeneric.h>


namespace dftfe
{
  namespace utils
  {
    namespace
    {
      template <typename ValueType1, typename ValueType2>
      void gatherSendBufferDeviceKernel(
        sycl::nd_item<1>     ind,
        const size_type      totalFlattenedSize,
        const size_type      blockSize,
        const ValueType1 *   dataArray,
        const size_type *    ownedLocalIndicesForTargetProcs,
        ValueType2 *         sendBuffer)
      {        
        const size_type globalId = ind.get_global_id(0);
        size_type n_workgroups = ind.get_group_range(0);
        size_type n_workitems = ind.get_local_range(0);

        for (size_type i = globalId; i < totalFlattenedSize; i += n_workgroups * n_workitems)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            sendBuffer[i] =
              dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                        intraBlockId];
          }
      }

      template <>
      void
      gatherSendBufferDeviceKernel(
        sycl::nd_item<1>                             ind,
        const size_type                              totalFlattenedSize,
        const size_type                              blockSize,
        const dftfe::utils::deviceDoubleComplex     *dataArray,
        const size_type *                            ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceFloatComplex           *sendBuffer)
      {
        const size_type globalId = ind.get_global_id(0);
        size_type n_workgroups = ind.get_group_range(0);
        size_type n_workitems = ind.get_local_range(0);

        for (size_type i = globalId; i < totalFlattenedSize; i += n_workgroups * n_workitems)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;
            sendBuffer[i].real(
              dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                        intraBlockId].real());
            sendBuffer[i].imag(
              dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                        intraBlockId].imag());
          }
      }

      template <typename ValueType1, typename ValueType2>
      void
      accumAddFromRecvBufferDeviceKernel(
        sycl::nd_item<1>     ind,
        const size_type      totalFlattenedSize,
        const size_type      blockSize,
        const ValueType1 *   recvBuffer,
        const size_type *    ownedLocalIndicesForTargetProcs,
        ValueType2 *         dataArray)
      {
        const size_type globalId = ind.get_global_id(0);
        size_type n_workgroups = ind.get_group_range(0);
        size_type n_workitems = ind.get_local_range(0);

        for (size_type i = globalId; i < totalFlattenedSize; i += n_workgroups * n_workitems)
          {
            const size_type  blockId      = i / blockSize;
            const size_type  intraBlockId = i - blockId * blockSize;
            const ValueType2 recvVal      = recvBuffer[i];

            auto atomic_add = sycl::atomic_ref<ValueType2, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        (dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize + intraBlockId]);
            atomic_add += recvVal;
          }
      }

      template <>
      void
      accumAddFromRecvBufferDeviceKernel(
        sycl::nd_item<1>                        ind,
        const size_type                         totalFlattenedSize,
        const size_type                         blockSize,
        const dftfe::utils::deviceFloatComplex *recvBuffer,
        const size_type *                       ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceFloatComplex *      dataArray)
      {
        const size_type globalId = ind.get_global_id(0);
        size_type n_workgroups = ind.get_group_range(0);
        size_type n_workitems = ind.get_local_range(0);

        for (size_type i = globalId; i < totalFlattenedSize; i += n_workgroups * n_workitems)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            auto atomic_add_real = sycl::atomic_ref<float, sycl::memory_order::relaxed, 
                                            sycl::memory_scope::device, 
                                            sycl::access::address_space::global_space>
                                            (reinterpret_cast<float*>(&dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize + intraBlockId])[0]);
            atomic_add_real += recvBuffer[i].real();
                                
            auto atomic_add_imag = sycl::atomic_ref<float, sycl::memory_order::relaxed, 
                                                sycl::memory_scope::device, 
                                                sycl::access::address_space::global_space>
                                                (reinterpret_cast<float*>(&dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize + intraBlockId])[1]);
            atomic_add_imag += recvBuffer[i].imag();
          }
      }

      template <>
      void
      accumAddFromRecvBufferDeviceKernel(
        sycl::nd_item<1>                             ind,
        const size_type                              totalFlattenedSize,
        const size_type                              blockSize,
        const dftfe::utils::deviceDoubleComplex     *recvBuffer,
        const size_type *                            ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceDoubleComplex           *dataArray)
      {
        const size_type globalId = ind.get_global_id(0);
        size_type n_workgroups = ind.get_group_range(0);
        size_type n_workitems = ind.get_local_range(0);

        for (size_type i = globalId; i < totalFlattenedSize; i += n_workgroups * n_workitems)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            auto atomic_add_real = sycl::atomic_ref<double, sycl::memory_order::relaxed, 
                                            sycl::memory_scope::device, 
                                            sycl::access::address_space::global_space>
                                            (reinterpret_cast<double*>(&dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize + intraBlockId])[0]);
            atomic_add_real += recvBuffer[i].real();
                                
            auto atomic_add_imag = sycl::atomic_ref<double, sycl::memory_order::relaxed, 
                                                sycl::memory_scope::device, 
                                                sycl::access::address_space::global_space>
                                                (reinterpret_cast<double*>(&dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize + intraBlockId])[1]);
            atomic_add_imag += recvBuffer[i].imag();
          }
      }

      template <>
      void
      accumAddFromRecvBufferDeviceKernel(
        sycl::nd_item<1>                        ind,
        const size_type                         totalFlattenedSize,
        const size_type                         blockSize,
        const dftfe::utils::deviceFloatComplex *recvBuffer,
        const size_type *                       ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceDoubleComplex *     dataArray)
      {
        const size_type globalId = ind.get_global_id(0);
        size_type n_workgroups = ind.get_group_range(0);
        size_type n_workitems = ind.get_local_range(0);

        for (size_type i = globalId; i < totalFlattenedSize; i += n_workgroups * n_workitems)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;
            const double recvValReal = recvBuffer[i].real();
            const double recvValImag = recvBuffer[i].imag();

            auto atomic_add_real = sycl::atomic_ref<double, sycl::memory_order::relaxed, 
                                            sycl::memory_scope::device, 
                                            sycl::access::address_space::global_space>
                                            (reinterpret_cast<double*>(&dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize + intraBlockId])[0]);
            atomic_add_real += recvValReal;
                                
            auto atomic_add_imag = sycl::atomic_ref<double, sycl::memory_order::relaxed, 
                                                sycl::memory_scope::device, 
                                                sycl::access::address_space::global_space>
                                                (reinterpret_cast<double*>(&dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize + intraBlockId])[1]);
            atomic_add_imag += recvValImag;
          }
      }

    } // namespace

    template <typename ValueType>
    template <typename ValueTypeComm>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<ValueTypeComm, utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t deviceCommStream)
    {
// #  ifdef DFTFE_WITH_DEVICE_LANG_SYCL
//         size_type total_workitems = ((ownedLocalIndicesForTargetProcs.size() *
//                                       blockSize) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
//                                       dftfe::utils::DEVICE_BLOCK_SIZE;
//         dftfe::utils::deviceEvent_t event = deviceCommStream.parallel_for(
//                                                 sycl::nd_range<1>(total_workitems,
//                                                       dftfe::utils::DEVICE_BLOCK_SIZE), 
//                                                 [=](sycl::nd_item<1> ind){
//             gatherSendBufferDeviceKernel<ValueType,ValueTypeComm>(ind, 
//                                          ownedLocalIndicesForTargetProcs.size() * blockSize,
//                                          blockSize,
//                                          dataArray.data(),
//                                          ownedLocalIndicesForTargetProcs.data(),
//                                          sendBuffer.data());
//         });
//         deviceCommStream.wait();
// #  endif
    }

    template <typename ValueType>
    template <typename ValueTypeComm>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueTypeComm, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream)
    {
// #  ifdef DFTFE_WITH_DEVICE_LANG_SYCL
//         size_type total_workitems = ((ownedLocalIndicesForTargetProcs.size() * blockSize) / 
//                                       dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
//                                       dftfe::utils::DEVICE_BLOCK_SIZE;
//         dftfe::utils::deviceEvent_t event = deviceCommStream.parallel_for(
//                                                 sycl::nd_range<1>(total_workitems,
//                                                       dftfe::utils::DEVICE_BLOCK_SIZE), 
//                                                 [=](sycl::nd_item<1> ind){
//             accumAddFromRecvBufferDeviceKernel<ValueType,ValueTypeComm>(ind, ownedLocalIndicesForTargetProcs.size() * blockSize,
//                                             blockSize,
//                                             recvBuffer.data(),
//                                             ownedLocalIndicesForTargetProcs.data(),
//                                             dataArray.data());
//         });
//         deviceCommStream.wait();
// #  endif
    }

    
    template <typename ValueType>
    template <typename ValueType1, typename ValueType2>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const size_type              blockSize,
        const ValueType1 *           type1Array,
        ValueType2 *                 type2Array,
        dftfe::utils::deviceStream_t deviceCommStream)
    {
      dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
        blockSize, type1Array, type2Array, deviceCommStream);
    }

    template class MPICommunicatorP2PKernels<double,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<float,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<dftfe::utils::deviceDoubleComplex,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<dftfe::utils::deviceFloatComplex,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<double, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<double, utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t                       deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<double, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<float, utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t                      deviceCommStream);

    template void
    MPICommunicatorP2PKernels<float, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<float, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<float, utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t                      deviceCommStream);

    template void
    MPICommunicatorP2PKernels<dftfe::utils::deviceDoubleComplex,
                              utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<dftfe::utils::deviceDoubleComplex, utils::MemorySpace::DEVICE>
          &dataArray,
        const MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<dftfe::utils::deviceDoubleComplex, utils::MemorySpace::DEVICE>
          &                          sendBuffer,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<dftfe::utils::deviceDoubleComplex,
                              utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<dftfe::utils::deviceDoubleComplex, utils::MemorySpace::DEVICE>
          &dataArray,
        const MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<dftfe::utils::deviceFloatComplex, utils::MemorySpace::DEVICE>
          &                          sendBuffer,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<dftfe::utils::deviceFloatComplex, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<dftfe::utils::deviceFloatComplex, utils::MemorySpace::DEVICE>
          &dataArray,
        const MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<dftfe::utils::deviceFloatComplex, utils::MemorySpace::DEVICE>
          &                          sendBuffer,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<double, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<float, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<dftfe::utils::deviceDoubleComplex,
                              utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<dftfe::utils::deviceDoubleComplex, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<dftfe::utils::deviceDoubleComplex, dftfe::utils::MemorySpace::DEVICE>
          &                          dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<dftfe::utils::deviceDoubleComplex,
                              utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<dftfe::utils::deviceFloatComplex, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<dftfe::utils::deviceDoubleComplex, dftfe::utils::MemorySpace::DEVICE>
          &                          dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<dftfe::utils::deviceFloatComplex, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<dftfe::utils::deviceFloatComplex, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<dftfe::utils::deviceFloatComplex, dftfe::utils::MemorySpace::DEVICE>
          &                          dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);
    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const size_type              blockSize,
        const double *               type1Array,
        float *                      type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const size_type              blockSize,
        const float *                type1Array,
        double *                     type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<dftfe::utils::deviceDoubleComplex,
                              dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const size_type              blockSize,
        const dftfe::utils::deviceDoubleComplex * type1Array,
        dftfe::utils::deviceFloatComplex *        type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<dftfe::utils::deviceDoubleComplex,
                              dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const size_type              blockSize,
        const dftfe::utils::deviceFloatComplex *  type1Array,
        dftfe::utils::deviceDoubleComplex *       type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);
    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const size_type              blockSize,
        const float *                type1Array,
        float *                      type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<dftfe::utils::deviceFloatComplex,
                              dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const size_type              blockSize,
        const dftfe::utils::deviceFloatComplex *  type1Array,
        dftfe::utils::deviceFloatComplex *        type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

  } // namespace utils
} // namespace dftfe
#endif