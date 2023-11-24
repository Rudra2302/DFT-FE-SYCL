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

#include <linearAlgebraOperations.h>
namespace dftfe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                  nodalData,
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients) const
    {
      interpolateKernel(nodalData,
                        quadratureValues,
                        quadratureGradients,
                        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      integrateWithBasis(
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData) const
    {
      integrateWithBasisKernel(quadratureValues,
                               quadratureGradients,
                               nodalData,
                               std::pair<unsigned int, unsigned int>(0,
                                                                     d_nCells));
    }


    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                  nodalData,
        ValueTypeBasisCoeff *cellNodalDataPtr) const
    {
      extractToCellNodalDataKernel(
        nodalData,
        cellNodalDataPtr,
        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      accumulateFromCellNodalData(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &nodalData) const
    {
      accumulateFromCellNodalDataKernel(
        cellNodalDataPtr,
        nodalData,
        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }
    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      interpolateKernel(
        const dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                dftfe::utils::MemorySpace::HOST>
          &                                         nodalValues,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        {
          extractToCellNodalDataKernel(
            nodalValues,
            tempCellNodalData.data(),
            std::pair<unsigned int, unsigned int>(iCell, iCell + 1));
          interpolateKernel(tempCellNodalData.data(),
                            quadratureValues,
                            quadratureGradients,
                            std::pair<unsigned int, unsigned int>(iCell,
                                                                  iCell + 1));
        }
    }
    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      interpolateKernel(
        const ValueTypeBasisCoeff *                 cellNodalValues,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      const ValueTypeBasisCoeff scalarCoeffAlpha = ValueTypeBasisCoeff(1.0),
                                scalarCoeffBeta  = ValueTypeBasisCoeff(0.0);
      const char transA = 'N', transB = 'N';

      if (quadratureValues != NULL)
        {
          d_BLASWrapperPtr->xgemmStridedBatched(
            transA,
            transB,
            d_nVectors,
            d_nQuadsPerCell[d_quadratureIndex],
            d_nDofsPerCell,
            &scalarCoeffAlpha,
            cellNodalValues,
            d_nVectors,
            d_nVectors * d_nDofsPerCell,
            d_shapeFunctionData.find(d_quadratureID)->second.data(),
            d_nDofsPerCell,
            0,
            &scalarCoeffBeta,
            quadratureValues,
            d_nVectors,
            d_nVectors * d_nQuadsPerCell[d_quadratureIndex],
            cellRange.second - cellRange.first);
        }
      if (quadratureGradients != NULL)
        {
          const unsigned int d_nQuadsPerCellTimesThree =
            d_nQuadsPerCell[d_quadratureIndex] * 3;

          d_BLASWrapperPtr->xgemmStridedBatched(
            transA,
            transB,
            d_nVectors,
            d_nQuadsPerCellTimesThree,
            d_nDofsPerCell,
            &scalarCoeffAlpha,
            cellNodalValues,
            d_nVectors,
            d_nVectors * d_nDofsPerCell,
            d_shapeFunctionGradientDataInternalLayout.find(d_quadratureID)
              ->second.data(),
            d_nDofsPerCell,
            0,
            &scalarCoeffBeta,
            areAllCellsCartesian ? quadratureGradients :
                                   tempQuadratureGradientsData.data(),
            d_nVectors,
            d_nVectors * d_nQuadsPerCell[d_quadratureIndex] * 3,
            cellRange.second - cellRange.first);
          if (areAllCellsCartesian)
            {
              const unsigned int d_nQuadsPerCellTimesnVectors =
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors;
              const unsigned int one = 1;
              for (unsigned int iCell = cellRange.first;
                   iCell < cellRange.second;
                   ++iCell)
                {
                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    d_BLASWrapperPtr->xscal(
                      quadratureGradients +
                        d_nQuadsPerCell[d_quadratureIndex] * d_nVectors * 3 *
                          (iCell - cellRange.first) +
                        d_nQuadsPerCell[d_quadratureIndex] * d_nVectors * iDim,
                      *(d_inverseJacobianData.find(0)->second.data() +
                        3 * iCell + iDim),
                      d_nQuadsPerCellTimesnVectors);
                }
            }
          else if (areAllCellsAffine)
            {
              const unsigned int d_nQuadsPerCellTimesnVectors =
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors;
              const unsigned int three = 3;
              d_BLASWrapperPtr->xgemmStridedBatched(
                transA,
                transB,
                d_nQuadsPerCellTimesnVectors,
                three,
                three,
                &scalarCoeffAlpha,
                tempQuadratureGradientsData.data(),
                d_nQuadsPerCellTimesnVectors,
                d_nQuadsPerCellTimesnVectors * 3,
                d_inverseJacobianData.find(0)->second.data() +
                  9 * cellRange.first,
                three,
                9,
                &scalarCoeffBeta,
                quadratureGradients,
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors,
                d_nVectors * d_nQuadsPerCell[d_quadratureIndex] * 3,
                cellRange.second - cellRange.first);
            }
          else
            {
              const unsigned int three = 3;
              d_BLASWrapperPtr->xgemmStridedBatched(
                transA,
                transB,
                d_nVectors,
                three,
                three,
                &scalarCoeffAlpha,
                tempQuadratureGradientsData.data(),
                d_nVectors,
                d_nVectors * 3,
                d_inverseJacobianData.find(d_quadratureID)->second.data() +
                  9 * cellRange.first * d_nQuadsPerCell[d_quadratureIndex],
                3,
                9,
                &scalarCoeffBeta,
                tempQuadratureGradientsDataNonAffine.data(),
                d_nVectors,
                d_nVectors * 3,
                (cellRange.second - cellRange.first) *
                  d_nQuadsPerCell[d_quadratureIndex]);
              for (unsigned int iCell = cellRange.first;
                   iCell < cellRange.second;
                   ++iCell)
                {
                  for (unsigned int iQuad = 0;
                       iQuad < d_nQuadsPerCell[d_quadratureIndex];
                       ++iQuad)

                    {
                      for (unsigned int iDim = 0; iDim < 3; ++iDim)
                        std::memcpy(
                          quadratureGradients +
                            d_nVectors * 3 * d_nQuadsPerCell[d_quadratureIndex] *
                              (iCell - cellRange.first) +
                            d_nVectors * d_nQuadsPerCell[d_quadratureIndex] *
                              iDim +
                            d_nVectors * iQuad,
                          tempQuadratureGradientsDataNonAffine.data() +
                            d_nVectors * 3 * iQuad + d_nVectors * iDim,
                          d_nVectors * sizeof(ValueTypeBasisCoeff));
                    }
                }
            }
        }
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      integrateWithBasisKernel(
        const ValueTypeBasisCoeff *quadratureValues,
        const ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        cellNodalData, tempQuadratureGradientsData,
        tempQuadratureGradientsDataNonAffine;
      cellNodalData.resize(d_nVectors * d_nDofsPerCell * d_nCells);
      if (quadratureGradients != NULL)
        tempQuadratureGradientsData.resize(3 * d_nVectors *
                                           d_nQuadsPerCell[d_quadratureIndex]);

      if (quadratureGradients != NULL)
        tempQuadratureGradientsDataNonAffine.resize(
          areAllCellsAffine ?
            0 :
            (3 * d_nVectors * d_nQuadsPerCell[d_quadratureIndex]));



      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        {
          const ValueTypeBasisCoeff scalarCoeffAlpha = ValueTypeBasisCoeff(1.0),
                                    scalarCoeffBeta  = ValueTypeBasisCoeff(0.0);
          const char transA = 'N', transB = 'T';
          d_BLASWrapperPtr->xgemm(
            transA,
            transB,
            d_nVectors,
            d_nDofsPerCell,
            d_nQuadsPerCell[d_quadratureIndex],
            &scalarCoeffAlpha,
            quadratureValues + d_nQuadsPerCell[d_quadratureIndex] * iCell,
            d_nVectors,
            d_shapeFunctionData.find(d_quadratureID)->second.data(),
            d_nQuadsPerCell[d_quadratureIndex],
            &scalarCoeffBeta,
            cellNodalData.data() + d_nDofsPerCell * iCell,
            d_nVectors);
          if (quadratureGradients != NULL)
            {
              if (areAllCellsCartesian)
                {
                  const unsigned int d_nQuadsPerCellTimesnVectors =
                    d_nQuadsPerCell[d_quadratureIndex] * d_nVectors;
                  const unsigned int one = 1;
                  std::memcpy(tempQuadratureGradientsData.data(),
                              quadratureGradients +
                                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors *
                                  3 * iCell,
                              3 * d_nQuadsPerCellTimesnVectors *
                                sizeof(ValueTypeBasisCoeff));
                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    {
                      d_BLASWrapperPtr->xscal(
                        tempQuadratureGradientsData.data() +
                          d_nQuadsPerCell[d_quadratureIndex] * d_nVectors * iDim,
                        *(d_inverseJacobianData.find(0)->second.data() +
                          3 * iCell + iDim),
                        d_nQuadsPerCellTimesnVectors);
                    }
                }
              else if (areAllCellsAffine)
                {
                  const unsigned int d_nQuadsPerCellTimesnVectors =
                    d_nQuadsPerCell[d_quadratureIndex] * d_nVectors;
                  const unsigned int three = 3;
                  d_BLASWrapperPtr->xgemm(
                    transA,
                    transB,
                    d_nQuadsPerCellTimesnVectors,
                    three,
                    three,
                    &scalarCoeffAlpha,
                    quadratureGradients +
                      d_nQuadsPerCell[d_quadratureIndex] * d_nVectors * 3 * iCell,
                    d_nQuadsPerCellTimesnVectors,
                    d_inverseJacobianData.find(0)->second.data() + 9 * iCell,
                    three,
                    &scalarCoeffBeta,
                    tempQuadratureGradientsData.data(),
                    d_nQuadsPerCellTimesnVectors);
                }
              else
                {
                  for (unsigned int iQuad = 0;
                       iQuad < d_nQuadsPerCell[d_quadratureIndex];
                       ++iQuad)
                    for (unsigned int iDim = 0; iDim < 3; ++iDim)
                      std::memcpy(tempQuadratureGradientsDataNonAffine.data() +
                                    d_nVectors * 3 * iQuad + d_nVectors * iDim,
                                  quadratureGradients +
                                    d_nVectors * 3 *
                                      d_nQuadsPerCell[d_quadratureIndex] * iCell +
                                    d_nVectors *
                                      d_nQuadsPerCell[d_quadratureIndex] * iDim +
                                    d_nVectors * iQuad,
                                  d_nVectors * sizeof(ValueTypeBasisCoeff));
                  const unsigned int three = 3;
                  for (unsigned int iQuad = 0;
                       iQuad < d_nQuadsPerCell[d_quadratureIndex];
                       ++iQuad)
                    d_BLASWrapperPtr->xgemm(
                      transA,
                      transB,
                      d_nVectors,
                      three,
                      three,
                      &scalarCoeffAlpha,
                      tempQuadratureGradientsDataNonAffine.data() +
                        d_nVectors * 3 * iQuad,
                      d_nVectors,
                      d_inverseJacobianData.find(d_quadratureID)
                          ->second.data() +
                        9 * d_nQuadsPerCell[d_quadratureIndex] * iCell + 9 * iQuad,
                      three,
                      &scalarCoeffBeta,
                      tempQuadratureGradientsData.data() +
                        d_nVectors * 3 * iQuad,
                      d_nVectors);
                }
              const unsigned int d_nQuadsPerCellTimesThree =
                d_nQuadsPerCell[d_quadratureIndex] * 3;
              d_BLASWrapperPtr->xgemm(transA,
                                      transB,
                                      d_nVectors,
                                      d_nQuadsPerCellTimesThree,
                                      d_nDofsPerCell,
                                      &scalarCoeffAlpha,
                                      tempQuadratureGradientsData.data(),
                                      d_nVectors,
                                      d_shapeFunctionGradientDataInternalLayout
                                        .find(d_quadratureID)
                                        ->second.data(),
                                      d_nDofsPerCell,
                                      &scalarCoeffBeta,
                                      cellNodalData.data() +
                                        d_nDofsPerCell * iCell,
                                      d_nVectors);
            }
          accumulateFromCellNodalDataKernel(
            cellNodalData.data(),
            nodalData,
            std::pair<unsigned int, unsigned int>(iCell, iCell + 1));
        }
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      extractToCellNodalDataKernel(
        const dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        ValueTypeBasisCoeff *                       cellNodalDataPtr,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        for (unsigned int iDof = 0; iDof < d_nDofsPerCell; ++iDof)
          {
            std::memcpy(cellNodalDataPtr +
                          (iCell - cellRange.first) * d_nVectors *
                            d_nDofsPerCell +
                          iDof * d_nVectors,
                        nodalData.data() +
                          d_flattenedCellDofIndexToProcessDofIndexMap
                            [iCell * d_nDofsPerCell + iDof],
                        d_nVectors * sizeof(ValueTypeBasisCoeff));
          }
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::HOST>::
      accumulateFromCellNodalDataKernel(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::HOST>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        for (unsigned int iDof = 0; iDof < d_nDofsPerCell; ++iDof)
          std::transform(
            cellNodalDataPtr + iCell * d_nVectors * d_nDofsPerCell +
              iDof * d_nVectors,
            cellNodalDataPtr + iCell * d_nVectors * d_nDofsPerCell +
              iDof * d_nVectors + d_nVectors,
            nodalData.data() + d_flattenedCellDofIndexToProcessDofIndexMap
                                 [iCell * d_nDofsPerCell + iDof],
            nodalData.data() + d_flattenedCellDofIndexToProcessDofIndexMap
                                 [iCell * d_nDofsPerCell + iDof],
            std::plus<ValueTypeBasisCoeff>());
    }
  } // namespace basis
} // namespace dftfe
