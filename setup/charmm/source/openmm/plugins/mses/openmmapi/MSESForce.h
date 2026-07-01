#ifndef OPENMM_MSESFORCE_H_
#define OPENMM_MSESFORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <vector>

namespace OpenMMMSES 
{

/*
 * define force class
*/
class MSESForce : public OpenMM::Force 
{
public:
   MSESForce();
   int addDistPair(int at1, int at2, int at3, int at4, double kc, double fmax, double dcut, double sexp);
   int getNumDistPair() const;
   void getDistPairParameters(int index, int& at1, int& at2, int& at3, int& at4, double& kc, double& fmax, double& dcut, double& sexp) const;
   void setDistPairParameters(int index, int at1, int at2, int at3, int at4, double kc, double fmax, double dcut, double sexp);
   void updateParametersInContext(OpenMM::Context& context);
   void setUsesPeriodicBoundaryConditions(bool periodic);
   bool usesPeriodicBoundaryConditions() const;

protected:
   OpenMM::ForceImpl* createImpl() const;

private:
   class distPairInfo;
   std::vector<distPairInfo> distPair;
   bool usePeriodic;
};

class MSESForce::distPairInfo
{
public:
   int at1,at2,at3,at4;
   double kc,fmax,dcut,sexp;
   distPairInfo() {at1=0; at2=0; at3=0; at4=0; kc=0.f; fmax=0.f; dcut=0.f; sexp=0.f;}
   distPairInfo(int at1, int at2, int at3, int at4, double kc, double fmax, double dcut, double sexp) :
   at1(at1),at2(at2),at3(at3),at4(at4),kc(kc),fmax(fmax),dcut(dcut),sexp(sexp) {}
};
 
} // namespace

#endif /*OPENMM_MSESFORCE_H_*/

