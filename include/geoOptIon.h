// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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

/** @file geoOptIon.h
 *  @brief This calls calls the relaxation solver for the atomic relaxations and acts as an interface between the
 *  solver and the force class. Currently we have option of one solver: Polak–Ribière nonlinear CG solver 
 *  with secant based line search. In future releases, we will have more options like BFGS solver.
 *
 *  @author Sambit Das
 */

#ifndef geoOptIon_H_
#define geoOptIon_H_
#include "solverFunction.h"
#include "constants.h"

using namespace dealii;
template <unsigned int FEOrder> class dftClass;
//
//Define geoOptIon class
//
template <unsigned int FEOrder>
class geoOptIon : public solverFunction
{
public:
/** @brief Constructor.
 *
 *  @param _dftPtr pointer to dftClass
 *  @param mpi_comm_replica mpi_communicator of the current pool
 */      
  geoOptIon(dftClass<FEOrder>* _dftPtr,  MPI_Comm &mpi_comm_replica);

/**
 * @brief initializes the data member d_relaxationFlags.
 *
 */    
  void init();

/**
 * @brief starts the atomic force relaxation.
 *
 */    
  void run();  

/**
 * @brief writes the current fem mesh. The mesh changes as atoms move.
 *
 */   
  void writeMesh(std::string meshFileName);

/**
 * @brief Obtain number of unknowns (total number of force components to be relaxed).
 *
 * @return int Number of unknowns.
 */   
  int getNumberUnknowns() const ;

/**
 * @brief Compute function gradient (aka forces).
 *
 * @param gradient STL vector for gradient values.
 */    
  void gradient(std::vector<double> & gradient);

/**
 * @brief Update atomic positions.
 *
 * @param solution displacement of the atoms with respect to their current position. 
 * The size of the solution vector is equal to the number of unknowns.
 */    
  void update(const std::vector<double> & solution);

  /// not implemented
  double value() const;

  /// not implemented
  void value(std::vector<double> & functionValue);

  /// not implemented
  void precondition(std::vector<double>       & s,
	            const std::vector<double> & gradient) const;  

  /// not implemented
  void solution(std::vector<double> & solution);

  /// not implemented
  std::vector<int> getUnknownCountFlag() const;

private:

  /// storage for relaxation flags for each global atom.
  /// each atom has three flags corresponding to three components (0- no relax, 1- relax)
  std::vector<int> d_relaxationFlags;

  /// maximum force component to be relaxed
  double d_maximumAtomForceToBeRelaxed;

  /// total number of calls to update()
  int d_totalUpdateCalls;

  /// pointer to dft class
  dftClass<FEOrder>* dftPtr;

  /// parallel communication objects
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  /// conditional stream object
  dealii::ConditionalOStream   pcout;
};

#endif
