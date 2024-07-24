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
// @author Sambit Das
//

// source file for electron density related computations
#include <constants.h>
#include <densityFirstOrderResponseCalculator.h>
#include <dftUtils.h>
#include <DataTypeOverloads.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>

namespace dftfe
{
  namespace
  {
    void
    computeRhoResponseFromInterpolatedValues(
      sycl::nd_item<1>   ind,
      const unsigned int numVectors,
      const unsigned int numCells,
      const unsigned int nQuadsPerCell,
      const double *     wfc,
      const double *     wfcPrime,
      double *           rhoResponseHamCellsWfcContributions,
      double *           rhoResponseFermiEnergyCellsWfcContributions)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numEntriesPerCell = numVectors * nQuadsPerCell;
      const unsigned int numberEntries     = numEntriesPerCell * numCells;
      const unsigned int n_workgroups = ind.get_group_range(0);
      const unsigned int n_workitems = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          const double psi                                   = wfc[index];
          const double psiPrime                              = wfcPrime[index];
          rhoResponseFermiEnergyCellsWfcContributions[index] = psi * psi;
          rhoResponseHamCellsWfcContributions[index]         = psi * psiPrime;
        }
    }

    void
    computeRhoResponseFromInterpolatedValues(
      sycl::nd_item<1>                         ind,
      const unsigned int                       numVectors,
      const unsigned int                       numCells,
      const unsigned int                       nQuadsPerCell,
      const dftfe::utils::deviceDoubleComplex *wfc,
      const dftfe::utils::deviceDoubleComplex *wfcPrime,
      double *rhoResponseHamCellsWfcContributions,
      double *rhoResponseFermiEnergyCellsWfcContributions)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numEntriesPerCell = numVectors * nQuadsPerCell;
      const unsigned int numberEntries     = numEntriesPerCell * numCells;
      const unsigned int n_workgroups = ind.get_group_range(0);
      const unsigned int n_workitems = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          const dftfe::utils::deviceDoubleComplex psi      = wfc[index];
          const dftfe::utils::deviceDoubleComplex psiPrime = wfcPrime[index];
          rhoResponseFermiEnergyCellsWfcContributions[index] =
            psi.real() * psi.real() + psi.imag() * psi.imag();
          rhoResponseHamCellsWfcContributions[index] =
            psi.real() * psiPrime.real() + psi.imag() * psiPrime.imag();
        }
    }
  } // namespace
  template <typename NumberType>
  void
  computeRhoResponseFromInterpolatedValues(
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      &                                         BLASWrapperPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    onesVec,
    double *                                    partialOccupPrimeVec,
    NumberType *                                wfcQuadPointData,
    NumberType *                                wfcPrimeQuadPointData,
    double *rhoResponseHamCellsWfcContributions,
    double *rhoResponseFermiEnergyCellsWfcContributions,
    double *rhoResponseHam,
    double *rhoResponseFermiEnergy)
  {
    const unsigned int cellsBlockSize   = cellRange.second - cellRange.first;
    const unsigned int vectorsBlockSize = vecRange.second - vecRange.first;
    const unsigned int nQuadsPerCell    = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells           = basisOperationsPtr->nCells();
    const double       scalarCoeffAlphaRho = 1.0;
    const double       scalarCoeffBetaRho  = 1.0;
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize) * dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream = BLASWrapperPtr->getDeviceStream();
        dftfe::utils::deviceEvent_t event = stream.parallel_for(sycl::nd_range<1>(
                                            total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                            [=](sycl::nd_item<1> ind){
            computeRhoResponseFromInterpolatedValues(ind, vectorsBlockSize,
                                                        cellsBlockSize,
                                                        nQuadsPerCell,
                                                        wfcQuadPointData,
                                                        wfcPrimeQuadPointData,
                                                        rhoResponseHamCellsWfcContributions,
                                                        rhoResponseFermiEnergyCellsWfcContributions);
        });
        DEVICE_API_CHECK(event);
#endif
    BLASWrapperPtr->xgemv('T',
                          vectorsBlockSize,
                          cellsBlockSize * nQuadsPerCell,
                          &scalarCoeffAlphaRho,
                          rhoResponseHamCellsWfcContributions,
                          vectorsBlockSize,
                          onesVec,
                          1,
                          &scalarCoeffBetaRho,
                          rhoResponseHam + cellRange.first * nQuadsPerCell,
                          1);

    BLASWrapperPtr->xgemv('T',
                          vectorsBlockSize,
                          cellsBlockSize * nQuadsPerCell,
                          &scalarCoeffAlphaRho,
                          rhoResponseFermiEnergyCellsWfcContributions,
                          vectorsBlockSize,
                          partialOccupPrimeVec,
                          1,
                          &scalarCoeffBetaRho,
                          rhoResponseFermiEnergy +
                            cellRange.first * nQuadsPerCell,
                          1);
  }
  template void
  computeRhoResponseFromInterpolatedValues(
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      &                                         BLASWrapperPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    onesVec,
    double *                                    partialOccupVec,
    dataTypes::number *                         wfcQuadPointData,
    dataTypes::number *                         wfcPrimeQuadPointData,
    double *rhoResponseHamCellsWfcContributions,
    double *rhoResponseFermiEnergyCellsWfcContributions,
    double *rhoResponseHam,
    double *rhoResponseFermiEnergy);

} // namespace dftfe
