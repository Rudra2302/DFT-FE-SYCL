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

#include <FEBasisOperationsKernelsInternal.h>


namespace dftfe
{
  namespace
  {
    template <typename ValueType>
    void
    reshapeFromNonAffineDeviceKernel(sycl::nd_item<1>       ind, 
                                     const dftfe::size_type           numVecs,
                                     const dftfe::size_type           numQuads,
                                     const dftfe::size_type           numCells,
                                     const ValueType *      copyFromVec,
                                     ValueType *            copyToVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
        const dftfe::size_type numberEntries = numQuads * numCells * numVecs * 3;
        dftfe::size_type n_workgroups = ind.get_group_range(0);
        dftfe::size_type n_workitems = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex  = index / numVecs;
          dftfe::size_type iVec        = index - blockIndex * numVecs;
          dftfe::size_type blockIndex2 = blockIndex / numQuads;
          dftfe::size_type iQuad       = blockIndex - blockIndex2 * numQuads;
          dftfe::size_type iCell       = blockIndex2 / 3;
          dftfe::size_type iDim        = blockIndex2 - iCell * 3;
          dftfe::utils::copyValue(
            copyToVec + index,
            copyFromVec[iVec + iDim * numVecs + iQuad * 3 * numVecs +
                        iCell * 3 * numQuads * numVecs]);
        }
    }

    template <typename ValueType>
    void
    reshapeToNonAffineDeviceKernel(sycl::nd_item<1>       ind,
                                   const dftfe::size_type           numVecs,
                                   const dftfe::size_type           numQuads,
                                   const dftfe::size_type           numCells,
                                   const ValueType *      copyFromVec,
                                   ValueType *            copyToVec)
    {
        const dftfe::size_type globalThreadId = ind.get_global_id(0);
        const dftfe::size_type numberEntries = numQuads * numCells * numVecs * 3;
        dftfe::size_type n_workgroups = ind.get_group_range(0);
        dftfe::size_type n_workitems = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex  = index / numVecs;
          dftfe::size_type iVec        = index - blockIndex * numVecs;
          dftfe::size_type blockIndex2 = blockIndex / numQuads;
          dftfe::size_type iQuad       = blockIndex - blockIndex2 * numQuads;
          dftfe::size_type iCell       = blockIndex2 / 3;
          dftfe::size_type iDim        = blockIndex2 - iCell * 3;
          dftfe::utils::copyValue(copyToVec + iVec + iDim * numVecs +
                                    iQuad * 3 * numVecs +
                                    iCell * 3 * numQuads * numVecs,
                                  copyFromVec[index]);
        }
    }
  } // namespace
  namespace basis
  {
    namespace FEBasisOperationsKernelsInternal
    {
      template <typename ValueType>
      void
      reshapeFromNonAffineLayoutDevice(const dftfe::size_type numVecs,
                                       const dftfe::size_type numQuads,
                                       const dftfe::size_type numCells,
                                       const ValueType *      copyFromVec,
                                       ValueType *            copyToVec)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        dftfe::size_type total_workitems = ((numVecs * numCells * numQuads * 3) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                [=](sycl::nd_item<1> ind){
            reshapeFromNonAffineDeviceKernel(ind, numVecs, numQuads, numCells, copyFromVec, copyToVec);
        });
        stream.wait();
#endif
      }


      template <typename ValueType>
      void
      reshapeToNonAffineLayoutDevice(const dftfe::size_type numVecs,
                                     const dftfe::size_type numQuads,
                                     const dftfe::size_type numCells,
                                     const ValueType *      copyFromVec,
                                     ValueType *            copyToVec)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        dftfe::size_type total_workitems = ((numVecs * numCells * numQuads * 3) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), [=](sycl::nd_item<1> ind){
            reshapeFromNonAffineDeviceKernel(ind, numVecs, numQuads, numCells, copyFromVec, copyToVec);
        });
        stream.wait();
#endif
      }


      template void
      reshapeFromNonAffineLayoutDevice(const dftfe::size_type numVecs,
                                       const dftfe::size_type numQuads,
                                       const dftfe::size_type numCells,
                                       const double *         copyFromVec,
                                       double *               copyToVec);
      template void
      reshapeFromNonAffineLayoutDevice(const dftfe::size_type      numVecs,
                                       const dftfe::size_type      numQuads,
                                       const dftfe::size_type     numCells,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<double> *      copyToVec);

      template void
      reshapeToNonAffineLayoutDevice(const dftfe::size_type numVecs,
                                     const dftfe::size_type numQuads,
                                     const dftfe::size_type numCells,
                                     const double *         copyFromVec,
                                     double *               copyToVec);
      template void
      reshapeToNonAffineLayoutDevice(const dftfe::size_type      numVecs,
                                     const dftfe::size_type      numQuads,
                                     const dftfe::size_type      numCells,
                                     const std::complex<double> *copyFromVec,
                                     std::complex<double> *      copyToVec);

    } // namespace FEBasisOperationsKernelsInternal
  }   // namespace basis
} // namespace dftfe
