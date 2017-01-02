//Include header files
#include "../../include2/headers.h"
#include "../../include2/dft.h"
#include "../../utils/fileReaders.cc"
#include "../poisson/poisson.cc"
#include "../eigen/eigen.cc"
#include "mesh.cc"
#include "init.cc"
#include "psiInitialGuess.cc"
#include "energy.cc"
#include "charge.cc"
#include "density.cc"
#include "locatenodes.cc"
#include "createBins.cc"
#include "mixingschemes.cc"
#include "chebyshev.cc"
 
//dft constructor
dftClass::dftClass():
  triangulation (MPI_COMM_WORLD),
  FE (QGaussLobatto<1>(FEOrder+1)),
    dofHandler (triangulation),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  poisson(this),
  eigen(this),
  numElectrons(0),
  numBaseLevels(0),
  numLevels(0),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times),
  bLow(0.0),
  a0(lowerEndWantedSpectrum)
{

}

void convertToCellCenteredCartesianCoordinates(std::vector<std::vector<double> > & atomLocations,
					       std::vector<std::vector<double> > & latticeVectors)
{

  //
  //
  //
  /*for (int i = 0; i < atomLocations.size(); ++i)
    {
      cartX[i] = atomLocations[i][2]*latticeVectors[0][0] + atomLocations[i][3]*latticeVectors[1][0] + atomLocations[i][4]*latticeVectors[2][0];
      }*/

}


void dftClass::set(){
  //
  //read coordinates
  //
  unsigned int numberColumnsCoordinatesFile = 5;

  /*#ifdef ENABLE_PERIODIC_BC

  //
  //read fractionalCoordinates of atoms in periodic case
  //
  readFile(numberColumnsCoordinatesFile, atomLocations, coordinatesFile);
  pcout << "number of atoms: " << atomLocations.size() << "\n";

  //
  //find unique atom types
  //
  for (std::vector<std::vector<double> >::iterator it=atomLocations.begin(); it<atomLocations.end(); it++){
    atomTypes.insert((unsigned int)((*it)[0]));
  }

  //
  //read lattice Vectors
  //
  std::vector<std::vector<double> > latticeVectors;
  unsigned int numberColumnsLatticeVectorsFile = 3;
  readFile(numberColumnsLatticeVectorsFile,latticeVectors,latticeVectorsFile);
  for(int i = 0; i < latticeVectors.size(); ++i)
    {
      pcout<<"Lattice Vectors: "<<latticeVectors[i][0]<<" "<<latticeVectors[i][1]<<" "<<latticeVectors[i][2]<<"\n";
    }

  //
  //find cell-centered cartesian coordinates
  //
  convertToCellCenteredCartesianCoordinates(atomLocations,
					    latticeVectors);
  
#else
  readFile(numberColumnsCoordinatesFile, atomLocations, coordinatesFile);
  pcout << "number of atoms: " << atomLocations.size() << "\n";
  //find unique atom types
  for (std::vector<std::vector<double> >::iterator it=atomLocations.begin(); it<atomLocations.end(); it++){
    atomTypes.insert((unsigned int)((*it)[0]));
  }
  #endif*/

  readFile(numberColumnsCoordinatesFile, atomLocations, coordinatesFile);
  pcout << "number of atoms: " << atomLocations.size() << "\n";
  //find unique atom types
  for (std::vector<std::vector<double> >::iterator it=atomLocations.begin(); it<atomLocations.end(); it++){
    atomTypes.insert((unsigned int)((*it)[0]));
  }
 
  pcout << "number of atoms types: " << atomTypes.size() << "\n";


  
  //estimate total number of wave functions
  determineOrbitalFilling();  
  //numEigenValues=waveFunctionsVector.size();
  pcout << "num of eigen values: " << numEigenValues << std::endl; 
  //set size of eigenvalues and eigenvectors data structures
  eigenValues.resize(numEigenValues);
  for (unsigned int i=0; i<numEigenValues; ++i){
    eigenVectors.push_back(new vectorType);
    PSI.push_back(new vectorType);
    tempPSI.push_back(new vectorType);
    tempPSI2.push_back(new vectorType);
    tempPSI3.push_back(new vectorType);
  } 
}

//dft run
void dftClass::run (){
  pcout << "number of MPI processes: "
	<< Utilities::MPI::n_mpi_processes(mpi_communicator)
	<< std::endl;
  //read coordinates file
  set();
  
  //generate mesh
  //if meshFile provided, pass to mesh()
  mesh();

  //initialize
  init();
 
  //solve
  computing_timer.enter_section("dft solve"); 

  /*
  //temp check
  poisson.solve(poisson.phiTotRhoIn,1,rhoInValues);
  std::cout << poisson.phiTotRhoIn.linfty_norm() << std::endl;
  DataOut<3> data_out;
  data_out.attach_dof_handler (dofHandler);
  data_out.add_data_vector (poisson.phiTotRhoIn, "solution");
  data_out.build_patches (4);
  std::ofstream output ("poisson.vtu");
  data_out.write_vtu (output);
  exit(-1);
  */

  //
  //phiExt with nuclear charge
  //
  int numberBins = d_boundaryFlag.size();
  int numberGlobalCharges = atomLocations.size();
  
  //int constraintMatrixId = 2;
  //poisson.solve(poisson.phiExt,constraintMatrixId);
 

  poisson.phiExt = 0;

  pcout<<"Size of support points: "<<d_supportPoints.size()<<std::endl;

  std::map<dealii::types::global_dof_index, int>::iterator iterMap;
  for(int iBin = 0; iBin < numberBins; ++iBin)
    {
      int constraintMatrixId = iBin + 2;
      poisson.solve(poisson.vselfBinScratch,constraintMatrixId);

      std::set<int> & atomsInBinSet = d_bins[iBin];
      std::vector<int> atomsInCurrentBin(atomsInBinSet.begin(),atomsInBinSet.end());
      int numberGlobalAtomsInBin = atomsInCurrentBin.size();

      std::vector<int> & imageIdsOfAtomsInCurrentBin = d_imageIdsInBins[iBin];
      int numberImageAtomsInBin = imageIdsOfAtomsInCurrentBin.size();

      std::map<dealii::types::global_dof_index, int> & boundaryNodeMap = d_boundaryFlag[iBin];
      std::map<types::global_dof_index,Point<3> >::iterator iterNodalCoorMap;

      int inNodes =0, outNodes = 0;
      for(iterNodalCoorMap = d_supportPoints.begin(); iterNodalCoorMap != d_supportPoints.end(); ++iterNodalCoorMap)
	{
	  if(poisson.vselfBinScratch.in_local_range(iterNodalCoorMap->first))
	    {
	      //
	      //get the vertex Id
	      //
	      Point<3> nodalCoor = iterNodalCoorMap->second;

	      //
	      //get the boundary flag for iVertex for current bin
	      //
	      int boundaryFlag;
	      iterMap = boundaryNodeMap.find(iterNodalCoorMap->first);
	      if(iterMap != boundaryNodeMap.end())
		{
		  boundaryFlag = iterMap->second;
		}
	      else
		{
		  std::cout<<"Could not find boundaryNode Map for the given dof:"<<std::endl;
		  exit(-1);
		}

	      //
	      //go through all atoms in a given bin
	      //
	      for(int iCharge = 0; iCharge < numberGlobalAtomsInBin+numberImageAtomsInBin; ++iCharge)
		{
		  //
		  //get the globalChargeId corresponding to iCharge in the current bin
		  //check what changes for periodic calculation
		  int chargeId;
		  if(iCharge < numberGlobalAtomsInBin)
		    chargeId = atomsInCurrentBin[iCharge];
		  else
		    chargeId = imageIdsOfAtomsInCurrentBin[iCharge-numberGlobalAtomsInBin]+numberGlobalCharges;

		  //std::cout<<"Charge Id in BinId: "<<chargeId<<" "<<iBin<<std::endl;

		  
		  double vSelf;
		  if(boundaryFlag == chargeId)
		    {
		      vSelf = poisson.vselfBinScratch(iterNodalCoorMap->first);
		      inNodes++;
		    }
		  else
		    {
		      Point<3> atomCoor(0.0,0.0,0.0);
		      double nuclearCharge;
		      if(iCharge < numberGlobalAtomsInBin)
			{
			  atomCoor[0] = atomLocations[chargeId][2];
			  atomCoor[1] = atomLocations[chargeId][3];
			  atomCoor[2] = atomLocations[chargeId][4];
			  
			  if(isPseudopotential)
			    nuclearCharge = atomLocations[chargeId][1];
			  else
			    nuclearCharge = atomLocations[chargeId][0];
			  
			}
		      else
			{
			  //fill this up
			}

		      const double r = nodalCoor.distance(atomCoor);
		      vSelf = -nuclearCharge/r;
		      outNodes++;
		    }

		  //store updated value in phiExt which is sumVself

		  poisson.phiExt(iterNodalCoorMap->first)+= vSelf;

		}//charge loop
	    
	    }

	}//Vertexloop

      //
      //store Vselfs for atoms in bin
      //
      for (std::map<unsigned int, double>::iterator it = d_atomsInBin[iBin].begin(); it != d_atomsInBin[iBin].end(); ++it)
	{
	  std::vector<double> temp(2,0.0);
	  temp[0] = it->second;//charge;
	  temp[1] = poisson.vselfBinScratch(it->first);//vself 
	  d_localVselfs.push_back(temp);
	}


      //   std::cout << "In: " << inNodes << "  Out: " << outNodes << "\n";
    }//bin loop

    poisson.phiExt.compress(VectorOperation::insert);
    poisson.phiExt.update_ghost_values();

    //
    //print the norms of phiExt (in periodic case L2 norm of phiExt field does not match. check later)
    //
    pcout<<"Peak Value of phiext: "<<poisson.phiExt.linfty_norm()<<std::endl;
    pcout<<"L2 Norm Value of phiext: "<<poisson.phiExt.l2_norm()<<std::endl;

  //
  //postprocess the data
  //
  //const ConstraintMatrix * constraintMatrix = d_constraintsVector[constraintMatrixId];

  //
  //Modify the phi value based on constraintValue
  //
  /*for(types::global_dof_index i = 0; i < poisson.phiExt.size(); ++i)
    {
      if(locally_relevant_dofs.is_element(i))
	{
	  if(constraintMatrix->is_constrained(i))
	    {
	      poisson.phiExt(i) = constraintMatrix->get_inhomogeneity(i);
	    }
	}
	}*/

  //std::cout<<"L2 Norm of Phi Ext: "<<poisson.phiExt.l2_norm()<<std::endl;
  //exit(-1);
  
  /*DataOut<3> data_out;
  data_out.attach_dof_handler (dofHandler);
  data_out.add_data_vector (poisson.phiExt, "solution");
  data_out.build_patches (4);
  std::ofstream output ("poisson.vtu");
  data_out.write_vtu (output);*/
  
  //exit(-1);
  //Begin SCF iteration
  unsigned int scfIter=0;
  double norm=1.0;
  while ((norm > 1.0e-13) && (scfIter < numSCFIterations)){
    if(this_mpi_process==0) printf("\n\nBegin SCF Iteration:%u\n", scfIter+1);
    //Mixing scheme
    if (scfIter>0){
      if (scfIter==1) norm=mixing_simple();
      else norm=mixing_anderson();
      if(this_mpi_process==0) printf("Mixing Scheme: iter:%u, norm:%12.6e\n", scfIter+1, norm);
    }
    //phiTot with rhoIn
    int constraintMatrixId = 1;
    poisson.solve(poisson.phiTotRhoIn,constraintMatrixId,rhoInValues);
    std::cout<<"L2 Norm of Phi out Tot L2  : "<<poisson.phiTotRhoIn.l2_norm()<<std::endl;
    std::cout<<"L2 Norm of Phi out Tot Linf: "<<poisson.phiTotRhoIn.linfty_norm()<<std::endl;
    //visualise
    DataOut<3> data_out;
    data_out.attach_dof_handler (dofHandler);
    data_out.add_data_vector (poisson.phiTotRhoIn, "solution");
    data_out.build_patches (4);
    std::ofstream output ("poisson.vtu");
    data_out.write_vtu (output);
    //eigen solve
    eigen.computeVEff(rhoInValues, poisson.phiTotRhoIn); 
    chebyshevSolver();
    //fermi energy
    compute_fermienergy();
    //rhoOut
    compute_rhoOut();
    //compute integral rhoOut
    double integralRhoOut=totalCharge(rhoOutValues);
    char buffer[100];
    sprintf(buffer, "Number of Electrons: %18.16e \n", integralRhoOut);
    pcout << buffer;
    //phiTot with rhoOut
    poisson.solve(poisson.phiTotRhoOut,constraintMatrixId, rhoOutValues);
    pcout<<"L2 Norm of Phi out Tot L2  : "<<poisson.phiTotRhoOut.l2_norm()<<std::endl;
    pcout<<"L2 Norm of Phi out Tot Linf: "<<poisson.phiTotRhoOut.linfty_norm()<<std::endl;
    //energy
    compute_energy();
    pcout<<"SCF iteration: " << scfIter+1 << " complete\n";
    scfIter++;
  }
  computing_timer.exit_section("dft solve"); 
}

