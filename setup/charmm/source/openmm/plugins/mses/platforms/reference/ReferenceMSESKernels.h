#ifndef REFERENCE_MSES_KERNELS_H_
#define REFERENCE_MSES_KERNELS_H_

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "MSESKernels.h"
#include "MSESForce.h"
#include "openmm/Platform.h"
#include "openmm/reference/RealVec.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace OpenMMMSES;
using namespace OpenMM;
using namespace std;

namespace OpenMMMSES {

/**
 * This kernel is invoked by MSESForce to calculate the forces acting 
 * on the system and the energy of the system.
 */
class ReferenceCalcMSESForceKernel : public CalcMSESForceKernel 
{
public:
   ReferenceCalcMSESForceKernel(std::string name, const OpenMM::Platform& platform) : CalcMSESForceKernel(name, platform) {
   }

   /**
    * Initialize the kernel.
    * 
    * @param system     the System this kernel will be applied to
    * @param force      the MSESForce this kernel will be used for
    */
   void initialize(const OpenMM::System& system, const MSESForce& force);
   /**
    * Execute the kernel to calculate the forces and/or energy.
    *
    * @param context        the context in which to execute this kernel
    * @return the potential energy due to the force
    */
   double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
   /**
    * Copy changed parameters over to a context.
    *
    * @param context    the context to copy parameters to
    * @param force      the MSESForce to copy the parameters from
    */
   void copyParametersToContext(OpenMM::ContextImpl& context, const MSESForce& force);

   void initMSES(const MSESForce& force);
   void executeMSES(double& ener, std::vector<RealVec>& pos, std::vector<RealVec>& force);

private:
   int numDistPair;
   std::vector<int> at1Vec,at2Vec,at3Vec,at4Vec;
   std::vector<double> kcVec,fmaxVec,dcutVec,sexpVec;

};

} // namespace

#endif /*REFERENCE_MSES_KERNELS_H_*/
