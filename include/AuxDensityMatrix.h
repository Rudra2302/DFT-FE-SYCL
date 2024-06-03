//
// Created by Arghadwip Paul.
//

#ifndef DFTFE_AUXDM_AUXDENSITYMATRIX_H
#define DFTFE_AUXDM_AUXDENSITYMATRIX_H

#include <vector>
#include <utility>
#include <map>

namespace dftfe
{
  enum class DensityDescriptorDataAttributes
  {
    valuesTotal,
    valuesSpinUp,
    valuesSpinDown,
    gradValueSpinUp,
    gradValueSpinDown,
    hessianSpinUp,
    hessianSpinDown,
    laplacianSpinUp,
    laplacianSpinDown
  };

  class AuxDensityMatrix
  {
  public:
    // Virtual destructor
    virtual ~AuxDensityMatrix();

    // Pure virtual functions

    virtual void
    applyLocalOperations(const std::vector<double> &    Points,
                         std::map<DensityDescriptorDataAttributes,
                                  std::vector<double>> &densityData) = 0;
    virtual void
    reinitAuxDensityMatrix(
      const std::vector<std::pair<std::string, std::vector<double>>>
        &                        atomCoords,
      const std::string &        auxBasisFile,
      const std::vector<double> &quadpts,
      const int                  nSpin,
      const int                  maxDerOrder) = 0;

    virtual void
    evalOverlapMatrixStart(const std::vector<double> &quadWt) = 0;

    virtual void
    evalOverlapMatrixEnd() = 0;

    /**
     *
     * @param projectionInputs is a map from string to inputs needed
     *                          for projection.
     * eg - projectionInputs["quadpts"],
     *      projectionInputs["quadWt"],
     *      projectionInputs["psiFunc"],
     *      projectionInputs["fValues"]
     *
     * psiFunc The SCF wave function or eigen function in FE Basis.
     *                psiFunc(quad_index, wfc_index),
     *                quad_index is fastest.
     * fValues The SCF eigen values.
     *                fValues(wfc_index),
     *
     */

    virtual void
    projectDensityMatrixStart(
      std::unordered_map<std::string, std::vector<double>>
        &projectionInputs) = 0;

    /**
     *
     * @param projectionInputs - same as above
     * @param iSpin - 0 (up) or 1 (down) required to fill d_DM
     */

    virtual void
    projectDensityMatrixEnd(
      std::unordered_map<std::string, std::vector<double>> &projectionInputs,
      int                                                   iSpin) = 0;

    /**
     * @brief Projects the quadrature density to aux basis (L2 projection).
     *
     *
     * @param Qpts The quadrature points.
     * @param QWt The quadrature weights.
     * @param nQ The number of quadrature points.
     * @param densityVals density values at quad points with spin index the
     * slowest index followed by the quad index
     * @param gradDensityVals gradient density values at quad points with spin index the
     * slowest index, followed by quad index, and finally the dimension index
     */
    virtual void
    projectDensity(const std::vector<double> &Qpts,
                   const std::vector<double> &QWt,
                   const int                  nQ,
                   const std::vector<double> &densityVals,
                   const std::vector<double> &gradDensityVals) = 0;
  };
} // namespace dftfe

#endif // DFTFE_AUXDM_AUXDENSITYMATRIX_H
