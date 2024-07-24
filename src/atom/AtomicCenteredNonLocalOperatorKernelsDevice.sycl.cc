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
// @author Kartick Ramakrishnan
//

#include <AtomicCenteredNonLocalOperatorKernelsDevice.h>
#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>

namespace dftfe
{
  namespace
  {
    template <typename ValueType>
    void
    copyFromParallelNonLocalVecToAllCellsVecKernel(
      sycl::nd_item<1>   ind,
      const unsigned int numWfcs,
      const unsigned int numNonLocalCells,
      const unsigned int maxSingleAtomPseudoWfc,
      const ValueType *  sphericalFnTimesWfcParallelVec,
      ValueType *        sphericalFnTimesWfcAllCellsVec,
      const int *        indexMapPaddedToParallelVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numNonLocalCells * maxSingleAtomPseudoWfc * numWfcs;
      const unsigned int n_workgroups = ind.get_group_range(0);
      const unsigned int n_workitems = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          const unsigned int blockIndex      = index / numWfcs;
          const unsigned int intraBlockIndex = index % numWfcs;
          const int mappedIndex = indexMapPaddedToParallelVec[blockIndex];
          if (mappedIndex != -1)
            sphericalFnTimesWfcAllCellsVec[index] =
              sphericalFnTimesWfcParallelVec[mappedIndex * numWfcs +
                                             intraBlockIndex];
        }
    }

    template <typename ValueType>
    void
    copyToDealiiParallelNonLocalVecKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  numWfcs,
      const unsigned int  totalPseudoWfcs,
      const ValueType *   sphericalFnTimesWfcParallelVec,
      ValueType *         sphericalFnTimesWfcDealiiParallelVec,
      const unsigned int *indexMapDealiiParallelNumbering)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries  = totalPseudoWfcs * numWfcs;
      const unsigned int n_workgroups = ind.get_group_range(0);
      const unsigned int n_workitems = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          const unsigned int blockIndex      = index / numWfcs;
          const unsigned int intraBlockIndex = index % numWfcs;
          const unsigned int mappedIndex =
            indexMapDealiiParallelNumbering[blockIndex];

          sphericalFnTimesWfcDealiiParallelVec[mappedIndex * numWfcs +
                                               intraBlockIndex] =
            sphericalFnTimesWfcParallelVec[index];
        }
    }
    template <typename ValueType>
    void
    addNonLocalContributionDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const ValueType *   xVec,
      ValueType *         yVec,
      const unsigned int *xVecToyVecBlockIdMap)
    {
      const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
      const dealii::types::global_dof_index numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      const unsigned int n_workgroups = ind.get_group_range(0);
      const unsigned int n_workitems = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dealii::types::global_dof_index blockIndex =
            index / contiguousBlockSize;
          dealii::types::global_dof_index intraBlockIndex =
            index % contiguousBlockSize;
          yVec[xVecToyVecBlockIdMap[blockIndex] * contiguousBlockSize +
               intraBlockIndex] =
            dftfe::utils::add(
              yVec[xVecToyVecBlockIdMap[blockIndex] * contiguousBlockSize +
                   intraBlockIndex],
              xVec[index]);
        }
    }

  } // namespace

  namespace AtomicCenteredNonLocalOperatorKernelsDevice
  {
    template <typename ValueType>
    void
    copyFromParallelNonLocalVecToAllCellsVec(
      const unsigned int numWfcs,
      const unsigned int numNonLocalCells,
      const unsigned int maxSingleAtomContribution,
      const ValueType *  sphericalFnTimesWfcParallelVec,
      ValueType *        sphericalFnTimesWfcAllCellsVec,
      const int *        indexMapPaddedToParallelVec)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((numWfcs + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE * numNonLocalCells *
                                    maxSingleAtomContribution) * dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=](sycl::nd_item<1> ind){
            copyFromParallelNonLocalVecToAllCellsVecKernel(ind, numWfcs,
                                                            numNonLocalCells,
                                                            maxSingleAtomContribution,
                                                            sphericalFnTimesWfcParallelVec,
                                                            sphericalFnTimesWfcAllCellsVec,
                                                            indexMapPaddedToParallelVec);
        });
        stream.wait();
#endif
    }
    template <typename ValueType>
    void
    copyToDealiiParallelNonLocalVec(
      const unsigned int  numWfcs,
      const unsigned int  totalEntries,
      const ValueType *   sphericalFnTimesWfcParallelVec,
      ValueType *         sphericalFnTimesWfcDealiiParallelVec,
      const unsigned int *indexMapDealiiParallelNumbering)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((numWfcs + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE * totalEntries) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=](sycl::nd_item<1> ind){
            copyToDealiiParallelNonLocalVecKernel(ind, numWfcs,
                                                    totalEntries,
                                                    sphericalFnTimesWfcParallelVec,
                                                    sphericalFnTimesWfcDealiiParallelVec,
                                                    indexMapDealiiParallelNumbering);
        });
        stream.wait();
#endif
    }

    template <typename ValueType>
    void
    addNonLocalContribution(
      const unsigned int numberCellsForAtom,
      const unsigned int numberNodesPerElement,
      const unsigned int numberWfc,
      const unsigned int numberCellsTraversed,
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
        &        nonLocalContribution,
      ValueType *TotalContribution,
      const dftfe::utils::MemoryStorage<unsigned int,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellNodeIdMapNonLocalToLocal)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((numberWfc + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE * numberCellsForAtom *
                                    numberNodesPerElement) * dftfe::utils::DEVICE_BLOCK_SIZE;
        auto x1 = numberWfc;
        auto x2 = numberCellsForAtom * numberNodesPerElement;
        auto x3 = nonLocalContribution.begin() + numberCellsTraversed * numberNodesPerElement * numberWfc;
        auto x4 = TotalContribution;
        auto x5 = cellNodeIdMapNonLocalToLocal.begin() + numberCellsTraversed * numberNodesPerElement;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                              sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                              [=](sycl::nd_item<1> ind){
            addNonLocalContributionDeviceKernel(ind, x1, x2, x3, x4, x5);
        });
        stream.wait();
#endif
    }



    template void
    copyToDealiiParallelNonLocalVec(
      const unsigned int  numWfcs,
      const unsigned int  totalEntries,
      const double *      sphericalFnTimesWfcParallelVec,
      double *            sphericalFnTimesWfcDealiiParallelVec,
      const unsigned int *indexMapDealiiParallelNumbering);

    template void
    copyToDealiiParallelNonLocalVec(
      const unsigned int          numWfcs,
      const unsigned int          totalEntries,
      const dftfe::utils::deviceDoubleComplex *sphericalFnTimesWfcParallelVec,
      dftfe::utils::deviceDoubleComplex *      sphericalFnTimesWfcDealiiParallelVec,
      const unsigned int *        indexMapDealiiParallelNumbering);

    template void
    copyToDealiiParallelNonLocalVec(
      const unsigned int          numWfcs,
      const unsigned int          totalEntries,
      const float *sphericalFnTimesWfcParallelVec,
      float *      sphericalFnTimesWfcDealiiParallelVec,
      const unsigned int *        indexMapDealiiParallelNumbering);

    template void
    copyToDealiiParallelNonLocalVec(
      const unsigned int          numWfcs,
      const unsigned int          totalEntries,
      const dftfe::utils::deviceFloatComplex *sphericalFnTimesWfcParallelVec,
      dftfe::utils::deviceFloatComplex *      sphericalFnTimesWfcDealiiParallelVec,
      const unsigned int *        indexMapDealiiParallelNumbering);

    template void
    copyFromParallelNonLocalVecToAllCellsVec(
      const unsigned int numWfcs,
      const unsigned int numNonLocalCells,
      const unsigned int maxSingleAtomContribution,
      const double *     sphericalFnTimesWfcParallelVec,
      double *           sphericalFnTimesWfcAllCellsVec,
      const int *        indexMapPaddedToParallelVec);

    template void
    copyFromParallelNonLocalVecToAllCellsVec(
      const unsigned int numWfcs,
      const unsigned int numNonLocalCells,
      const unsigned int maxSingleAtomContribution,
      const float *     sphericalFnTimesWfcParallelVec,
      float *           sphericalFnTimesWfcAllCellsVec,
      const int *        indexMapPaddedToParallelVec);

    template void
    copyFromParallelNonLocalVecToAllCellsVec(
      const unsigned int          numWfcs,
      const unsigned int          numNonLocalCells,
      const unsigned int          maxSingleAtomContribution,
      const dftfe::utils::deviceDoubleComplex *sphericalFnTimesWfcParallelVec,
      dftfe::utils::deviceDoubleComplex *      sphericalFnTimesWfcAllCellsVec,
      const int *                 indexMapPaddedToParallelVec);

    template void
    copyFromParallelNonLocalVecToAllCellsVec(
      const unsigned int          numWfcs,
      const unsigned int          numNonLocalCells,
      const unsigned int          maxSingleAtomContribution,
      const dftfe::utils::deviceFloatComplex *sphericalFnTimesWfcParallelVec,
      dftfe::utils::deviceFloatComplex *      sphericalFnTimesWfcAllCellsVec,
      const int *                 indexMapPaddedToParallelVec);


    template void
    addNonLocalContribution(
      const unsigned int numberCellsForAtom,
      const unsigned int numberNodesPerElement,
      const unsigned int numberWfc,
      const unsigned int numberCellsTraversed,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &     nonLocalContribution,
      double *TotalContribution,
      const dftfe::utils::MemoryStorage<unsigned int,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellNodeIdMapNonLocalToLocal);

    template void
    addNonLocalContribution(
      const unsigned int numberCellsForAtom,
      const unsigned int numberNodesPerElement,
      const unsigned int numberWfc,
      const unsigned int numberCellsTraversed,
      const dftfe::utils::MemoryStorage<dftfe::utils::deviceDoubleComplex,
                                        dftfe::utils::MemorySpace::DEVICE>
        &                   nonLocalContribution,
      dftfe::utils::deviceDoubleComplex *TotalContribution,
      const dftfe::utils::MemoryStorage<unsigned int,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellNodeIdMapNonLocalToLocal);

    template void
    addNonLocalContribution(
      const unsigned int numberCellsForAtom,
      const unsigned int numberNodesPerElement,
      const unsigned int numberWfc,
      const unsigned int numberCellsTraversed,
      const dftfe::utils::MemoryStorage<float,
                                        dftfe::utils::MemorySpace::DEVICE>
        &                   nonLocalContribution,
      float *TotalContribution,
      const dftfe::utils::MemoryStorage<unsigned int,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellNodeIdMapNonLocalToLocal);

    template void
    addNonLocalContribution(
      const unsigned int numberCellsForAtom,
      const unsigned int numberNodesPerElement,
      const unsigned int numberWfc,
      const unsigned int numberCellsTraversed,
      const dftfe::utils::MemoryStorage<dftfe::utils::deviceFloatComplex,
                                        dftfe::utils::MemorySpace::DEVICE>
        &                   nonLocalContribution,
      dftfe::utils::deviceFloatComplex *TotalContribution,
      const dftfe::utils::MemoryStorage<unsigned int,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellNodeIdMapNonLocalToLocal);

  } // namespace AtomicCenteredNonLocalOperatorKernelsDevice

} // namespace dftfe
