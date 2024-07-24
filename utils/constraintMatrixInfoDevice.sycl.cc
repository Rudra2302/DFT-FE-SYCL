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
// @author  Sambit Das, Phani Motamarri
//

#include <constraintMatrixInfo.h>
#include <deviceKernelsGeneric.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>
#include <dftUtils.h>

namespace dftfe
{
  // Declare dftUtils functions
  namespace dftUtils
  {
    namespace
    {
      void
      distributeKernel(
        sycl::nd_item<1>    ind,
        const unsigned int  contiguousBlockSize,
        double *            xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            xVec[xVecStartingIdRow + intraBlockIndex] =
              inhomogenities[blockIndex];
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                xVec[xVecStartingIdRow + intraBlockIndex] +=
                  constraintColumnValuesAllRowsUnflattened
                    [startingColumnNumber + i] *
                  xVec[xVecStartingIdColumn + intraBlockIndex];
              }
          }
      }


      void
      distributeKernel(
        sycl::nd_item<1>    ind,
        const unsigned int  contiguousBlockSize,
        float *             xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {            
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            xVec[xVecStartingIdRow + intraBlockIndex] =
              inhomogenities[blockIndex];
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                xVec[xVecStartingIdRow + intraBlockIndex] +=
                  constraintColumnValuesAllRowsUnflattened
                    [startingColumnNumber + i] *
                  xVec[xVecStartingIdColumn + intraBlockIndex];
              }
          }
      }

      void
      scaleConstraintsKernel(
        sycl::nd_item<1>    ind,
        const double *      xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        double *            constraintColumnValuesAllRowsUnflattened)
      {
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries = numConstraints;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[index];
            const unsigned int numberColumns = constraintRowSizes[index];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[index];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId;
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                constraintColumnValuesAllRowsUnflattened[startingColumnNumber +
                                                         i] *=
                  xVec[constrainedColumnId];
              }
          }
      }


      void
      distributeKernel(
        sycl::nd_item<1>    ind,
        const unsigned int                 contiguousBlockSize,
        dftfe::utils::deviceDoubleComplex *xVec,
        const unsigned int *               constraintLocalRowIdsUnflattened,
        const unsigned int                 numConstraints,
        const unsigned int *               constraintRowSizes,
        const unsigned int *               constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            dftfe::utils::copyValue(xVec + xVecStartingIdRow + intraBlockIndex,
                                    inhomogenities[blockIndex]);
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                dftfe::utils::copyValue(
                  xVec + xVecStartingIdRow + intraBlockIndex,
                  dftfe::utils::add(
                    xVec[xVecStartingIdRow + intraBlockIndex],
                    dftfe::utils::makeComplex(
                      xVec[xVecStartingIdColumn + intraBlockIndex].real() *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i],
                      xVec[xVecStartingIdColumn + intraBlockIndex].imag() *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i])));
              }
          }
      }


      void
      distributeKernel(
        sycl::nd_item<1>    ind,
        const unsigned int                contiguousBlockSize,
        dftfe::utils::deviceFloatComplex *xVec,
        const unsigned int *              constraintLocalRowIdsUnflattened,
        const unsigned int                numConstraints,
        const unsigned int *              constraintRowSizes,
        const unsigned int *              constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            dftfe::utils::copyValue(xVec + xVecStartingIdRow + intraBlockIndex,
                                    inhomogenities[blockIndex]);
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                dftfe::utils::copyValue(
                  xVec + xVecStartingIdRow + intraBlockIndex,
                  dftfe::utils::add(
                    xVec[xVecStartingIdRow + intraBlockIndex],
                    dftfe::utils::makeComplex(
                      xVec[xVecStartingIdColumn + intraBlockIndex].real() *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i],
                      xVec[xVecStartingIdColumn + intraBlockIndex].imag() *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i])));
              }
          }
      }

      void
      distributeSlaveToMasterKernelAtomicAdd(
        sycl::nd_item<1>    ind,
        const unsigned int  contiguousBlockSize,
        double *            xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened)
      {
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                
                auto atomic_add = sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        (xVec[xVecStartingIdColumn + intraBlockIndex]);
                atomic_add += constraintColumnValuesAllRowsUnflattened
                              [startingColumnNumber + i] *
                            xVec[xVecStartingIdRow + intraBlockIndex];
              }
            xVec[xVecStartingIdRow + intraBlockIndex] = 0.0;
          }
      }


      void
      distributeSlaveToMasterKernelAtomicAdd(
        sycl::nd_item<1>    ind,
        const unsigned int                 contiguousBlockSize,
        dftfe::utils::deviceDoubleComplex *xVec,
        const unsigned int *               constraintLocalRowIdsUnflattened,
        const unsigned int                 numConstraints,
        const unsigned int *               constraintRowSizes,
        const unsigned int *               constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened)
      {
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const dealii::types::global_dof_index xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const dealii::types::global_dof_index xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                const dftfe::utils::deviceDoubleComplex tempComplval =
                  dftfe::utils::mult(constraintColumnValuesAllRowsUnflattened
                                       [startingColumnNumber + i],
                                     xVec[xVecStartingIdRow + intraBlockIndex]);
                
                auto atomic_add_real = sycl::atomic_ref<float, sycl::memory_order::relaxed, 
                                                    sycl::memory_scope::device, 
                                                    sycl::access::address_space::global_space>
                                                    (reinterpret_cast<float*>(&xVec[xVecStartingIdColumn + intraBlockIndex])[0]);
                atomic_add_real += tempComplval.real();
                                    
                auto atomic_add_imag = sycl::atomic_ref<float, sycl::memory_order::relaxed, 
                                                    sycl::memory_scope::device, 
                                                    sycl::access::address_space::global_space>
                                                    (reinterpret_cast<float*>(&xVec[xVecStartingIdColumn + intraBlockIndex])[1]);
                atomic_add_imag += tempComplval.imag();
              }
            xVec[xVecStartingIdRow + intraBlockIndex].real(0.0);
            xVec[xVecStartingIdRow + intraBlockIndex].imag(0.0);
          }
      }


      void
      setzeroKernel(sycl::nd_item<1>    ind,
                    const unsigned int  contiguousBlockSize,
                    double *            xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            xVec[constraintLocalRowIdsUnflattened[blockIndex] *
                   contiguousBlockSize +
                 intraBlockIndex]              = 0;
          }
      }

      void
      setzeroKernel(sycl::nd_item<1>    ind,
                    const unsigned int  contiguousBlockSize,
                    float *             xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            xVec[constraintLocalRowIdsUnflattened[blockIndex] *
                   contiguousBlockSize +
                 intraBlockIndex]              = 0;
          }
      }

      void
      setzeroKernel(sycl::nd_item<1>    ind,
                    const unsigned int                 contiguousBlockSize,
                    dftfe::utils::deviceDoubleComplex *xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            dftfe::utils::copyValue(
              xVec +
                constraintLocalRowIdsUnflattened[blockIndex] *
                  contiguousBlockSize +
                intraBlockIndex,
              0.0);
          }
      }


      void
      setzeroKernel(sycl::nd_item<1>    ind,
                    const unsigned int                contiguousBlockSize,
                    dftfe::utils::deviceFloatComplex *xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const dealii::types::global_dof_index globalThreadId = ind.get_global_id(0);
        const dealii::types::global_dof_index numberEntries =
          numConstraints * contiguousBlockSize;
        const dealii::types::global_dof_index n_workgroups = ind.get_group_range(0);
        const dealii::types::global_dof_index n_workitems = ind.get_local_range(0);

        for (dealii::types::global_dof_index index = globalThreadId;
             index < numberEntries;
             index += n_workgroups * n_workitems)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            dftfe::utils::copyValue(
              xVec +
                constraintLocalRowIdsUnflattened[blockIndex] *
                  contiguousBlockSize +
                intraBlockIndex,
              0.0);
          }
      }
    } // namespace

    // constructor
    //
    constraintMatrixInfo<
      dftfe::utils::MemorySpace::DEVICE>::constraintMatrixInfo()
    {}

    //
    // destructor
    //
    constraintMatrixInfo<
      dftfe::utils::MemorySpace::DEVICE>::~constraintMatrixInfo()
    {}


    //
    // store constraintMatrix row data in STL vector
    //
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::initialize(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &                                      partitioner,
      const dealii::AffineConstraints<double> &constraintMatrixData,
      const bool                               useInhomogeneties)

    {
      clear();
      const dealii::IndexSet &locally_owned_dofs =
        partitioner->locally_owned_range();
      const dealii::IndexSet &ghost_dofs = partitioner->ghost_indices();

      dealii::types::global_dof_index     count = 0;
      std::vector<std::set<unsigned int>> slaveToMasterSet;
      for (dealii::IndexSet::ElementIterator it = locally_owned_dofs.begin();
           it != locally_owned_dofs.end();
           ++it)
        {
          if (constraintMatrixData.is_constrained(*it))
            {
              const dealii::types::global_dof_index lineDof = *it;
              d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
              if (useInhomogeneties)
                d_inhomogenities.push_back(
                  constraintMatrixData.get_inhomogeneity(lineDof));
              else
                d_inhomogenities.push_back(0.0);
              const std::vector<
                std::pair<dealii::types::global_dof_index, double>> *rowData =
                constraintMatrixData.get_constraint_entries(lineDof);
              d_rowSizes.push_back(rowData->size());
              d_rowSizesAccumulated.push_back(count);
              count += rowData->size();
              std::set<unsigned int> columnIds;
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  Assert((*rowData)[j].first < partitioner->size(),
                         dealii::ExcMessage("Index out of bounds"));
                  const unsigned int columnId =
                    partitioner->global_to_local((*rowData)[j].first);
                  d_columnIdsLocal.push_back(columnId);
                  d_columnValues.push_back((*rowData)[j].second);
                  columnIds.insert(columnId);
                }
              slaveToMasterSet.push_back(columnIds);
            }
        }


      for (dealii::IndexSet::ElementIterator it = ghost_dofs.begin();
           it != ghost_dofs.end();
           ++it)
        {
          if (constraintMatrixData.is_constrained(*it))
            {
              const dealii::types::global_dof_index lineDof = *it;
              d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
              if (useInhomogeneties)
                d_inhomogenities.push_back(
                  constraintMatrixData.get_inhomogeneity(lineDof));
              else
                d_inhomogenities.push_back(0.0);
              const std::vector<
                std::pair<dealii::types::global_dof_index, double>> *rowData =
                constraintMatrixData.get_constraint_entries(lineDof);
              d_rowSizes.push_back(rowData->size());
              d_rowSizesAccumulated.push_back(count);
              count += rowData->size();
              std::set<unsigned int> columnIds;
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  Assert((*rowData)[j].first < partitioner->size(),
                         dealii::ExcMessage("Index out of bounds"));
                  const unsigned int columnId =
                    partitioner->global_to_local((*rowData)[j].first);
                  d_columnIdsLocal.push_back(columnId);
                  d_columnValues.push_back((*rowData)[j].second);
                  columnIds.insert(columnId);
                }
              slaveToMasterSet.push_back(columnIds);
            }
        }

      d_rowIdsLocalDevice.resize(d_rowIdsLocal.size());
      d_rowIdsLocalDevice.copyFrom(d_rowIdsLocal);

      d_columnIdsLocalDevice.resize(d_columnIdsLocal.size());
      d_columnIdsLocalDevice.copyFrom(d_columnIdsLocal);

      d_columnValuesDevice.resize(d_columnValues.size());
      d_columnValuesDevice.copyFrom(d_columnValues);

      d_inhomogenitiesDevice.resize(d_inhomogenities.size());
      d_inhomogenitiesDevice.copyFrom(d_inhomogenities);

      d_rowSizesDevice.resize(d_rowSizes.size());
      d_rowSizesDevice.copyFrom(d_rowSizes);

      d_rowSizesAccumulatedDevice.resize(d_rowSizesAccumulated.size());
      d_rowSizesAccumulatedDevice.copyFrom(d_rowSizesAccumulated);

      d_numConstrainedDofs = d_rowIdsLocal.size();
    }

    template <typename NumberType>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedDeviceVec<NumberType> &fieldVector) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      const unsigned int blockSize = fieldVector.numVectors();
// #ifdef DFTFE_WITH_DEVICE_LANG_SYCL
//         size_t total_workitems = ((blockSize * d_numConstrainedDofs +
//                                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
//                                         dftfe::utils::DEVICE_BLOCK_SIZE) * dftfe::utils::DEVICE_BLOCK_SIZE;
//         dftfe::utils::deviceEvent_t event = dftfe::utils::deviceBlasWrapper::d_streamId.parallel_for(sycl::nd_range<1>(total_workitems,
//                              dftfe::utils::DEVICE_BLOCK_SIZE), [=](sycl::nd_item<1> ind){
//             distributeKernel(ind,
//                              blockSize,
//                              fieldVector.begin(),
//                              d_rowIdsLocalDevice.begin(),
//                              d_numConstrainedDofs,
//                              d_rowSizesDevice.begin(),
//                              d_rowSizesAccumulatedDevice.begin(),
//                              d_columnIdsLocalDevice.begin(),
//                              d_columnValuesDevice.begin(),
//                              d_inhomogenitiesDevice.begin());
//         });
//         event.wait();
// #endif
    }


    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      initializeScaledConstraints(
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &invSqrtMassVec)
    {
      if (d_numConstrainedDofs == 0)
        return;
// #ifdef DFTFE_WITH_DEVICE_LANG_SYCL
//         size_t total_workitems = ((d_numConstrainedDofs +
//                                     (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
//                                      dftfe::utils::DEVICE_BLOCK_SIZE) * dftfe::utils::DEVICE_BLOCK_SIZE;
//         dftfe::utils::deviceEvent_t event = dftfe::utils::deviceBlasWrapper::d_streamId.parallel_for(
//                                                 sycl::nd_range<1>(total_workitems,
//                                                 dftfe::utils::DEVICE_BLOCK_SIZE), 
//                                                 [=](sycl::nd_item<1> ind){
//             scaleConstraintsKernel( ind,
//                                     invSqrtMassVec.data(),
//                                     d_rowIdsLocalDevice.begin(),
//                                     d_numConstrainedDofs,
//                                     d_rowSizesDevice.begin(),
//                                     d_rowSizesAccumulatedDevice.begin(),
//                                     d_columnIdsLocalDevice.begin(),
//                                     d_columnValuesDevice.begin());
//         });
//         DEVICE_API_CHECK(event);
// #endif
    }
    //
    // set the constrained degrees of freedom to values so that constraints
    // are satisfied for flattened array
    //
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(
        distributedDeviceVec<double> &fieldVector) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      const unsigned int blockSize = fieldVector.numVectors();
// #ifdef DFTFE_WITH_DEVICE_LANG_SYCL
//         size_t total_workitems = ((blockSize * d_numConstrainedDofs +
//                                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
//                                         dftfe::utils::DEVICE_BLOCK_SIZE) * dftfe::utils::DEVICE_BLOCK_SIZE;
//         dftfe::utils::deviceEvent_t event = dftfe::utils::deviceBlasWrapper::d_streamId.parallel_for(
//                                                   sycl::nd_range<1>(total_workitems,
//                                                   dftfe::utils::DEVICE_BLOCK_SIZE), 
//                                                   [=](sycl::nd_item<1> ind){
//             distributeSlaveToMasterKernelAtomicAdd( ind,
//                                                     blockSize,
//                                                     fieldVector.begin(),
//                                                     d_rowIdsLocalDevice.begin(),
//                                                     d_numConstrainedDofs,
//                                                     d_rowSizesDevice.begin(),
//                                                     d_rowSizesAccumulatedDevice.begin(),
//                                                     d_columnIdsLocalDevice.begin(),
//                                                     d_columnValuesDevice.begin());
//         });
//         DEVICE_API_CHECK(event);
// #endif
    }

    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(
        distributedDeviceVec<float> &fieldVector) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      const unsigned int blockSize = fieldVector.numVectors();
// #ifdef DFTFE_WITH_DEVICE_LANG_SYCL
//         size_t total_workitems = ((blockSize * d_numConstrainedDofs +
//                                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
//                                         dftfe::utils::DEVICE_BLOCK_SIZE) * dftfe::utils::DEVICE_BLOCK_SIZE;
//         dftfe::utils::deviceEvent_t event = dftfe::utils::deviceBlasWrapper::d_streamId.parallel_for(
//                                                   sycl::nd_range<1>(total_workitems,
//                                                   dftfe::utils::DEVICE_BLOCK_SIZE), 
//                                                   [=](sycl::nd_item<1> ind){
//             distributeSlaveToMasterKernelAtomicAdd( ind,
//                                                     blockSize,
//                                                     fieldVector.begin(),
//                                                     d_rowIdsLocalDevice.begin(),
//                                                     d_numConstrainedDofs,
//                                                     d_rowSizesDevice.begin(),
//                                                     d_rowSizesAccumulatedDevice.begin(),
//                                                     d_columnIdsLocalDevice.begin(),
//                                                     d_columnValuesDevice.begin());
//         });
//         DEVICE_API_CHECK(event);
// #endif
    }


    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(
        distributedDeviceVec<std::complex<double>> &fieldVector) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      const unsigned int blockSize = fieldVector.numVectors();
// #ifdef DFTFE_WITH_DEVICE_LANG_SYCL
//         size_t total_workitems = ((blockSize * d_numConstrainedDofs +
//                                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
//                                         dftfe::utils::DEVICE_BLOCK_SIZE) * dftfe::utils::DEVICE_BLOCK_SIZE;
//         dftfe::utils::deviceEvent_t event = dftfe::utils::deviceBlasWrapper::d_streamId.parallel_for(
//                                                 sycl::nd_range<1>(total_workitems,
//                                                 dftfe::utils::DEVICE_BLOCK_SIZE), 
//                                                 [=](sycl::nd_item<1> ind){
//             distributeSlaveToMasterKernelAtomicAdd( ind,
//                                                     blockSize,
//                                                     fieldVector.begin(),
//                                                     d_rowIdsLocalDevice.begin(),
//                                                     d_numConstrainedDofs,
//                                                     d_rowSizesDevice.begin(),
//                                                     d_rowSizesAccumulatedDevice.begin(),
//                                                     d_columnIdsLocalDevice.begin(),
//                                                     d_columnValuesDevice.begin());
//         });
//         DEVICE_API_CHECK(event);
// #endif
    }

    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(
        distributedDeviceVec<std::complex<float>> &fieldVector) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      const unsigned int blockSize = fieldVector.numVectors();
// #ifdef DFTFE_WITH_DEVICE_LANG_SYCL
//         size_t total_workitems = ((blockSize * d_numConstrainedDofs +
//                                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
//                                         dftfe::utils::DEVICE_BLOCK_SIZE) * dftfe::utils::DEVICE_BLOCK_SIZE;
//         dftfe::utils::deviceEvent_t event = dftfe::utils::deviceBlasWrapper::d_streamId.parallel_for(
//                                                 sycl::nd_range<1>(total_workitems,
//                                                 dftfe::utils::DEVICE_BLOCK_SIZE), 
//                                                 [=](sycl::nd_item<1> ind){
//             distributeSlaveToMasterKernelAtomicAdd( ind,
//                                                     blockSize,
//                                                     fieldVector.begin(),
//                                                     d_rowIdsLocalDevice.begin(),
//                                                     d_numConstrainedDofs,
//                                                     d_rowSizesDevice.begin(),
//                                                     d_rowSizesAccumulatedDevice.begin(),
//                                                     d_columnIdsLocalDevice.begin(),
//                                                     d_columnValuesDevice.begin());
//         });
//         DEVICE_API_CHECK(event);
// #endif
    }


    template <typename NumberType>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedDeviceVec<NumberType> &fieldVector) const
    {
      if (d_numConstrainedDofs == 0)
        return;

      const unsigned int blockSize          = fieldVector.numVectors();
      const unsigned int numConstrainedDofs = d_rowIdsLocal.size();
// #ifdef DFTFE_WITH_DEVICE_LANG_SYCL
//         size_t total_workitems = ((blockSize * numConstrainedDofs +
//                                         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
//                                         dftfe::utils::DEVICE_BLOCK_SIZE) * dftfe::utils::DEVICE_BLOCK_SIZE;
//         dftfe::utils::deviceEvent_t event = dftfe::utils::deviceBlasWrapper::d_streamId.parallel_for(
//                                                 sycl::nd_range<1>(total_workitems,
//                                                 dftfe::utils::DEVICE_BLOCK_SIZE), 
//                                                 [=](sycl::nd_item<1> ind){
//             setzeroKernel(ind,
//                           blockSize,
//                           fieldVector.begin(),
//                           d_rowIdsLocalDevice.begin(),
//                           numConstrainedDofs);
//         });
//         DEVICE_API_CHECK(event);
// #endif
    }

    //
    //
    // clear the data variables
    //
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::clear()
    {
      d_rowIdsLocal.clear();
      d_columnIdsLocal.clear();
      d_columnValues.clear();
      d_inhomogenities.clear();
      d_rowSizes.clear();
      d_rowSizesAccumulated.clear();
      d_rowIdsLocalBins.clear();
      d_columnIdsLocalBins.clear();
      d_columnValuesBins.clear();
      d_binColumnSizesAccumulated.clear();
      d_binColumnSizes.clear();

      d_rowIdsLocalDevice.clear();
      d_columnIdsLocalDevice.clear();
      d_columnValuesDevice.clear();
      d_inhomogenitiesDevice.clear();
      d_rowSizesDevice.clear();
      d_rowSizesAccumulatedDevice.clear();
      d_rowIdsLocalBinsDevice.clear();
      d_columnIdsLocalBinsDevice.clear();
      d_columnValuesBinsDevice.clear();
    }


    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedCPUVec<double> &fieldVector) const
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }

    template <typename T>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedCPUVec<T> &fieldVector,
      const unsigned int    blockSize) const
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }

    template <typename T>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      distribute_slave_to_master(distributedCPUVec<T> &fieldVector,
                                 const unsigned int    blockSize) const
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }

    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::
      initializeScaledConstraints(
        const distributedCPUVec<double> &invSqrtMassVec)
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }

    template <typename T>
    void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedCPUVec<T> &fieldVector,
      const unsigned int    blockSize) const
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedDeviceVec<double> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedDeviceVec<std::complex<double>> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedDeviceVec<float> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::distribute(
      distributedDeviceVec<std::complex<float>> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedDeviceVec<double> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedDeviceVec<std::complex<double>> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedDeviceVec<float> &fieldVector) const;

    template void
    constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>::set_zero(
      distributedDeviceVec<std::complex<float>> &fieldVector) const;

    template class constraintMatrixInfo<dftfe::utils::MemorySpace::DEVICE>;

  } // namespace dftUtils
} // namespace dftfe
