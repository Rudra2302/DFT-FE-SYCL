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

// source file for force related computations
#if defined(DFTFE_WITH_DEVICE)
#  include "deviceKernelsGeneric.h"
#  include "dftfeDataTypes.h"
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceAPICalls.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <forceWfcContractionsDeviceKernels.h>
// #  include <DeviceBlasWrapper.h>

namespace dftfe
{
  namespace forceDeviceKernels
  {
    namespace
    {
      void
      computeELocWfcEshelbyTensorContributions(
        sycl::nd_item<1>   ind,
        const unsigned int contiguousBlockSize,
        const unsigned int numContiguousBlocks,
        const unsigned int numQuads,
        const double *     psiQuadValues,
        const double *     gradPsiQuadValues,
        const double *     eigenValues,
        const double *     partialOccupancies,
        double *           eshelbyTensor)
      {
        const unsigned int globalThreadId = ind.get_global_id(0);
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;
        const unsigned int n_workgroups = ind.get_group_range(0);
        const unsigned int n_workitems = ind.get_local_range(0);

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex = index / contiguousBlockSize;
            const unsigned int intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            const unsigned int blockIndex2  = blockIndex / 9;
            const unsigned int eshelbyIndex = blockIndex - 9 * blockIndex2;
            const unsigned int cellIndex    = blockIndex2 / numQuads;
            const unsigned int quadId = blockIndex2 - cellIndex * numQuads;
            const unsigned int tempIndex =
              (cellIndex)*numQuads * contiguousBlockSize +
              quadId * contiguousBlockSize + intraBlockIndex;
            const unsigned int tempIndex2 =
              (cellIndex)*numQuads * contiguousBlockSize * 3 +
              quadId * contiguousBlockSize + intraBlockIndex;
            const double psi      = psiQuadValues[tempIndex];
            const double gradPsiX = gradPsiQuadValues[tempIndex2];
            const double gradPsiY =
              gradPsiQuadValues[tempIndex2 + numQuads * contiguousBlockSize];
            const double gradPsiZ =
              gradPsiQuadValues[tempIndex2 +
                                2 * numQuads * contiguousBlockSize];
            const double eigenValue = eigenValues[intraBlockIndex];
            const double partOcc    = partialOccupancies[intraBlockIndex];

            const double identityFactor =
              partOcc * (gradPsiX * gradPsiX + gradPsiY * gradPsiY +
                         gradPsiZ * gradPsiZ) -
              2.0 * partOcc * eigenValue * psi * psi;

            if (eshelbyIndex == 0)
              eshelbyTensor[index] =
                -2.0 * partOcc * gradPsiX * gradPsiX + identityFactor;
            else if (eshelbyIndex == 1)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiX * gradPsiY;
            else if (eshelbyIndex == 2)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiX * gradPsiZ;
            else if (eshelbyIndex == 3)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiY * gradPsiX;
            else if (eshelbyIndex == 4)
              eshelbyTensor[index] =
                -2.0 * partOcc * gradPsiY * gradPsiY + identityFactor;
            else if (eshelbyIndex == 5)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiY * gradPsiZ;
            else if (eshelbyIndex == 6)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiZ * gradPsiX;
            else if (eshelbyIndex == 7)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiZ * gradPsiY;
            else if (eshelbyIndex == 8)
              eshelbyTensor[index] =
                -2.0 * partOcc * gradPsiZ * gradPsiZ + identityFactor;
          }
      }


      void
      computeELocWfcEshelbyTensorContributions(
        sycl::nd_item<1>                         ind,
        const unsigned int                       contiguousBlockSize,
        const unsigned int                       numContiguousBlocks,
        const unsigned int                       numQuads,
        const dftfe::utils::deviceDoubleComplex *psiQuadValues,
        const dftfe::utils::deviceDoubleComplex *gradPsiQuadValues,
        const double *                           eigenValues,
        const double *                           partialOccupancies,
        const double                             kcoordx,
        const double                             kcoordy,
        const double                             kcoordz,
        double *                                 eshelbyTensor,
        const bool                               addEk)
      {
        const unsigned int globalThreadId = ind.get_global_id(0);
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;
        const unsigned int n_workgroups = ind.get_group_range(0);
        const unsigned int n_workitems = ind.get_local_range(0);

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex = index / contiguousBlockSize;
            const unsigned int intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            const unsigned int blockIndex2  = blockIndex / 9;
            const unsigned int eshelbyIndex = blockIndex - 9 * blockIndex2;
            const unsigned int cellIndex    = blockIndex2 / numQuads;
            const unsigned int quadId = blockIndex2 - cellIndex * numQuads;
            const unsigned int tempIndex =
              (cellIndex)*numQuads * contiguousBlockSize +
              quadId * contiguousBlockSize + intraBlockIndex;
            const unsigned int tempIndex2 =
              (cellIndex)*numQuads * contiguousBlockSize * 3 +
              quadId * contiguousBlockSize + intraBlockIndex;
            const dftfe::utils::deviceDoubleComplex psi =
              psiQuadValues[tempIndex];
            const dftfe::utils::deviceDoubleComplex psiConj =
              dftfe::utils::conj(psiQuadValues[tempIndex]);
            const dftfe::utils::deviceDoubleComplex gradPsiX =
              gradPsiQuadValues[tempIndex2];
            const dftfe::utils::deviceDoubleComplex gradPsiY =
              gradPsiQuadValues[tempIndex2 + numQuads * contiguousBlockSize];
            const dftfe::utils::deviceDoubleComplex gradPsiZ =
              gradPsiQuadValues[tempIndex2 +
                                2 * numQuads * contiguousBlockSize];
            const dftfe::utils::deviceDoubleComplex gradPsiXConj =
              dftfe::utils::conj(gradPsiQuadValues[tempIndex2]);
            const dftfe::utils::deviceDoubleComplex gradPsiYConj =
              dftfe::utils::conj(
                gradPsiQuadValues[tempIndex2 + numQuads * contiguousBlockSize]);
            const dftfe::utils::deviceDoubleComplex gradPsiZConj =
              dftfe::utils::conj(
                gradPsiQuadValues[tempIndex2 +
                                  2 * numQuads * contiguousBlockSize]);
            const double eigenValue = eigenValues[intraBlockIndex];
            const double partOcc    = partialOccupancies[intraBlockIndex];

            const double identityFactor =
              partOcc *
              ((dftfe::utils::mult(gradPsiXConj, gradPsiX).real() +
                dftfe::utils::mult(gradPsiYConj, gradPsiY).real() +
                dftfe::utils::mult(gradPsiZConj, gradPsiZ).real()) +
               2.0 * (kcoordx * dftfe::utils::mult(psiConj, gradPsiX).imag() +
                      kcoordy * dftfe::utils::mult(psiConj, gradPsiY).imag() +
                      kcoordz * dftfe::utils::mult(psiConj, gradPsiZ).imag()) +
               (kcoordx * kcoordx + kcoordy * kcoordy + kcoordz * kcoordz -
                2.0 * eigenValue) *
                 dftfe::utils::mult(psiConj, psi).real());
            if (addEk)
              {
                if (eshelbyIndex == 0)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiX).real() +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).imag() *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).imag() *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).real() *
                      kcoordx * kcoordx +
                    identityFactor;
                else if (eshelbyIndex == 1)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiY).real() +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).imag() *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).imag() *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).real() *
                      kcoordx * kcoordy;
                else if (eshelbyIndex == 2)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiZ).real() +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).imag() *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).imag() *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).real() *
                      kcoordx * kcoordz;
                else if (eshelbyIndex == 3)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiX).real() +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).imag() *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).imag() *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).real() *
                      kcoordy * kcoordx;
                else if (eshelbyIndex == 4)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiY).real() +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).imag() *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).imag() *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).real() *
                      kcoordy * kcoordy +
                    identityFactor;
                else if (eshelbyIndex == 5)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiZ).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).imag() *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).imag() *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).real() *
                      kcoordy * kcoordz;
                else if (eshelbyIndex == 6)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiX).real() +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).imag() *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).imag() *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).real() *
                      kcoordz * kcoordx;
                else if (eshelbyIndex == 7)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiY).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).imag() *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).imag() *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).real() *
                      kcoordz * kcoordy;
                else if (eshelbyIndex == 8)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiZ).real() +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).imag() *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).imag() *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).real() *
                      kcoordz * kcoordz +
                    identityFactor;
              }
            else
              {
                if (eshelbyIndex == 0)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiX).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).imag() *
                      kcoordx +
                    identityFactor;
                else if (eshelbyIndex == 1)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiY).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).imag() *
                      kcoordy;
                else if (eshelbyIndex == 2)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiZ).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).imag() *
                      kcoordz;
                else if (eshelbyIndex == 3)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiX).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).imag() *
                      kcoordx;
                else if (eshelbyIndex == 4)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiY).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).imag() *
                      kcoordy +
                    identityFactor;
                else if (eshelbyIndex == 5)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiZ).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).imag() *
                      kcoordz;
                else if (eshelbyIndex == 6)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiX).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).imag() *
                      kcoordx;
                else if (eshelbyIndex == 7)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiY).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).imag() *
                      kcoordy;
                else if (eshelbyIndex == 8)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiZ).real() -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).imag() *
                      kcoordz +
                    identityFactor;
              }
          }
      }


      void
      nlpContractionContributionPsiIndexDeviceKernel(
        sycl::nd_item<1>    ind,
        const unsigned int  numPsi,
        const unsigned int  numQuadsNLP,
        const unsigned int  totalNonTrivialPseudoWfcs,
        const unsigned int  startingId,
        const double *      projectorKetTimesVectorPar,
        const double *      gradPsiOrPsiQuadValuesNLP,
        const double *      partialOccupancies,
        const unsigned int *nonTrivialIdToElemIdMap,
        const unsigned int *projecterKetTimesFlattenedVectorLocalIds,
        double *            nlpContractionContribution)
      {
        const unsigned int globalThreadId = ind.get_global_id(0);
        const unsigned int numberEntries =
          totalNonTrivialPseudoWfcs * numQuadsNLP * numPsi;
        const unsigned int n_workgroups = ind.get_group_range(0);
        const unsigned int n_workitems = ind.get_local_range(0);

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex  = index / numPsi;
            const unsigned int wfcId       = index - blockIndex * numPsi;
            unsigned int       pseudoWfcId = blockIndex / numQuadsNLP;
            const unsigned int quadId = blockIndex - pseudoWfcId * numQuadsNLP;
            pseudoWfcId += startingId;
            nlpContractionContribution[index] =
              partialOccupancies[wfcId] *
              gradPsiOrPsiQuadValuesNLP[nonTrivialIdToElemIdMap[pseudoWfcId] *
                                          numQuadsNLP * numPsi +
                                        quadId * numPsi + wfcId] *
              projectorKetTimesVectorPar
                [projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId] *
                   numPsi +
                 wfcId];
          }
      }

      void
      nlpContractionContributionPsiIndexDeviceKernel(
        sycl::nd_item<1>                         ind,
        const unsigned int                       numPsi,
        const unsigned int                       numQuadsNLP,
        const unsigned int                       totalNonTrivialPseudoWfcs,
        const unsigned int                       startingId,
        const dftfe::utils::deviceDoubleComplex *projectorKetTimesVectorPar,
        const dftfe::utils::deviceDoubleComplex *gradPsiOrPsiQuadValuesNLP,
        const double *                           partialOccupancies,
        const unsigned int *                     nonTrivialIdToElemIdMap,
        const unsigned int *projecterKetTimesFlattenedVectorLocalIds,
        dftfe::utils::deviceDoubleComplex *nlpContractionContribution)
      {
        const unsigned int globalThreadId = ind.get_global_id(0);
        const unsigned int numberEntries =
          totalNonTrivialPseudoWfcs * numQuadsNLP * numPsi;
        const unsigned int n_workgroups = ind.get_group_range(0);
        const unsigned int n_workitems = ind.get_local_range(0);

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex  = index / numPsi;
            const unsigned int wfcId       = index - blockIndex * numPsi;
            unsigned int       pseudoWfcId = blockIndex / numQuadsNLP;
            const unsigned int quadId = blockIndex - pseudoWfcId * numQuadsNLP;
            pseudoWfcId += startingId;

            const dftfe::utils::deviceDoubleComplex temp = dftfe::utils::mult(
              dftfe::utils::conj(
                gradPsiOrPsiQuadValuesNLP[nonTrivialIdToElemIdMap[pseudoWfcId] *
                                            numQuadsNLP * numPsi +
                                          quadId * numPsi + wfcId]),
              projectorKetTimesVectorPar
                [projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId] *
                   numPsi +
                 wfcId]);
            nlpContractionContribution[index] =
              std::complex<double>(partialOccupancies[wfcId] * temp.real(),
                                   partialOccupancies[wfcId] * temp.imag());
          }
      }

    } // namespace

    template <typename ValueType>
    void
    nlpContractionContributionPsiIndex(
      const unsigned int  wfcBlockSize,
      const unsigned int  blockSizeNlp,
      const unsigned int  numQuadsNLP,
      const unsigned int  startingIdNlp,
      const ValueType *   projectorKetTimesVectorPar,
      const ValueType *   gradPsiOrPsiQuadValuesNLP,
      const double *      partialOccupancies,
      const unsigned int *nonTrivialIdToElemIdMap,
      const unsigned int *projecterKetTimesFlattenedVectorLocalIds,
      ValueType *         nlpContractionContribution)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((wfcBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE * numQuadsNLP * blockSizeNlp) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                    sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                    [=](sycl::nd_item<1> ind){
            nlpContractionContributionPsiIndexDeviceKernel(ind, 
                                                            wfcBlockSize,
                                                            numQuadsNLP,
                                                            blockSizeNlp,
                                                            startingIdNlp,
                                                            projectorKetTimesVectorPar,
                                                            gradPsiOrPsiQuadValuesNLP,
                                                            partialOccupancies,
                                                            nonTrivialIdToElemIdMap,
                                                            projecterKetTimesFlattenedVectorLocalIds,
                                                            nlpContractionContribution);
        });
        DEVICE_API_CHECK(event);
#  endif
    }


    template <typename ValueType>
    void
    computeELocWfcEshelbyTensorContributions(const unsigned int wfcBlockSize,
                                             const unsigned int cellsBlockSize,
                                             const unsigned int numQuads,
                                             const ValueType *  psiQuadValues,
                                             const ValueType *gradPsiQuadValues,
                                             const double *   eigenValues,
                                             const double *partialOccupancies,
#  ifdef USE_COMPLEX
                                             const double kcoordx,
                                             const double kcoordy,
                                             const double kcoordz,
#  endif
                                             double *eshelbyTensorContributions
#  ifdef USE_COMPLEX
                                             ,
                                             const bool addEk
#  endif
    )
    {
#ifdef DFTFE_WITH_DEVICE_LANG_SYCL
        size_t total_workitems = ((wfcBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE * numQuads * cellsBlockSize) * 
                                    dftfe::utils::DEVICE_BLOCK_SIZE;
        dftfe::utils::deviceStream_t stream{sycl::gpu_selector_v};
        dftfe::utils::deviceEvent_t event = stream.parallel_for(
                                    sycl::nd_range<1>(total_workitems,dftfe::utils::DEVICE_BLOCK_SIZE), 
                                    [=](sycl::nd_item<1> ind){
            computeELocWfcEshelbyTensorContributions(ind, 
                                                            wfcBlockSize,
                                                            cellsBlockSize * numQuads * 9,
                                                            numQuads,
                                                            psiQuadValues,
                                                            gradPsiQuadValues,
                                                            eigenValues,
                                                            partialOccupancies,
#    ifdef USE_COMPLEX
        kcoordx,
        kcoordy,
        kcoordz,
#    endif
        eshelbyTensorContributions
#    ifdef USE_COMPLEX
        ,
        addEk
#    endif
        );
      });
      DEVICE_API_CHECK(event);
#  endif
    }

    template void
    nlpContractionContributionPsiIndex(
      const unsigned int       wfcBlockSize,
      const unsigned int       blockSizeNlp,
      const unsigned int       numQuadsNLP,
      const unsigned int       startingIdNlp,
      const dataTypes::number *projectorKetTimesVectorPar,
      const dataTypes::number *gradPsiOrPsiQuadValuesNLP,
      const double *           partialOccupancies,
      const unsigned int *     nonTrivialIdToElemIdMap,
      const unsigned int *     projecterKetTimesFlattenedVectorLocalIds,
      dataTypes::number *      nlpContractionContribution);

    template void
    computeELocWfcEshelbyTensorContributions(
      const unsigned int       wfcBlockSize,
      const unsigned int       cellsBlockSize,
      const unsigned int       numQuads,
      const dataTypes::number *psiQuadValues,
      const dataTypes::number *gradPsiQuadValues,
      const double *           eigenValues,
      const double *           partialOccupancies,
#  ifdef USE_COMPLEX
      const double kcoordx,
      const double kcoordy,
      const double kcoordz,
#  endif
      double *eshelbyTensor
#  ifdef USE_COMPLEX
      ,
      const bool addEk
#  endif
    );

  } // namespace forceDeviceKernels
} // namespace dftfe
#endif
