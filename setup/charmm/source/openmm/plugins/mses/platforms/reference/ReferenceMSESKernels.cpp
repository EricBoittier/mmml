
#include "ReferenceMSESKernels.h"

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

void ReferenceCalcMSESForceKernel::initialize(const System& system, const MSESForce& force) {
    // initial MSES coupling term
    initMSES(force);
}

double ReferenceCalcMSESForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);

    // execute MSES coupling term
    double ener=0.f;
    executeMSES(ener,pos,force);
    
    return ener;
}

void ReferenceCalcMSESForceKernel::copyParametersToContext(ContextImpl& context, const MSESForce& force) {
    if (force.getNumDistPair() != numDistPair)
        throw OpenMMException("updateParametersInContext: The number of MSES Distance-based pair interaction has changed");
    for (int i = 0; i < force.getNumDistPair(); i++) {
        int p1, p2, p3, p4;
        force.getDistPairParameters(i, p1, p2, p3, p4, kcVec[i], fmaxVec[i], dcutVec[i], sexpVec[i]);
        if (p1 != at1Vec[i] || p2!=at2Vec[i] || p3!=at3Vec[i] || p4!=at4Vec[i])
            throw OpenMMException("updateParametersInContext: A MSES interaction index has changed");
    }
}

// MSES coupling term
void ReferenceCalcMSESForceKernel::initMSES(const MSESForce& force)
{
    numDistPair = force.getNumDistPair();
    at1Vec.resize(numDistPair);
    at2Vec.resize(numDistPair);
    at3Vec.resize(numDistPair);
    at4Vec.resize(numDistPair);
    kcVec.resize(numDistPair);
    fmaxVec.resize(numDistPair);
    dcutVec.resize(numDistPair);
    sexpVec.resize(numDistPair);
    for (int i=0; i<numDistPair; i++) {
       force.getDistPairParameters(i,at1Vec[i],at2Vec[i],at3Vec[i],at4Vec[i],kcVec[i],fmaxVec[i],dcutVec[i],sexpVec[i]);
    }

}
void ReferenceCalcMSESForceKernel::executeMSES(double& ener, vector<RealVec>& pos, vector<RealVec>& force)
{
    // calculate MSES coupling term
    // msescCalcEnerForc(param,pos,energy,force);
    int i,at1,at2,at3,at4;
    RealVec force1,force3;
    RealVec dvat12,dvat34;
    RealOpenMM kc,fmax,dcut,sexp,softA,softB;
    RealOpenMM drat12,drat34,dist,absDist,invAbsDist,invPowAbsDistSexp;
    RealOpenMM x1, x2;

    ener = 0.f;
    for (i=0; i<numDistPair; i++)
    {
       at1 = at1Vec[i];
       at2 = at2Vec[i];
       at3 = at3Vec[i];
       at4 = at4Vec[i];
       kc = kcVec[i];
       fmax = fmaxVec[i];
       dcut = dcutVec[i];
       sexp = sexpVec[i];
       dvat12 = pos[at1] - pos[at2];
       dvat34 = pos[at3] - pos[at4];
       drat12 = dvat12.dot(dvat12);
       drat34 = dvat34.dot(dvat34);
       drat12 = sqrt(drat12);
       drat34 = sqrt(drat34);
       dist = drat12 - drat34;
       absDist = abs(dist);
       invAbsDist = 1.0f/absDist;
       invPowAbsDistSexp = 1.0f/pow(absDist,sexp);

       x1 = dcut*dcut;
       x2 = 1.0f/sexp;
       softA = x1*(0.5f+x2) - fmax*dcut*(1.0f+x2);
       softB = pow(dcut,sexp+1.0f)*(fmax-dcut)*x2;

       x1 = 0.5f*kc*absDist*absDist;
       x2 = kc*(softA + softB*invPowAbsDistSexp + fmax*absDist);
       x1 = (absDist<=dcut) ? x1 : x2;
       ener += x1; 

       x1 = kc*dist;
       x2 = kc*(fmax - softB*sexp*invPowAbsDistSexp*invAbsDist)*dist*invAbsDist;
       x1 = (absDist<=dcut) ? x1 : x2;
       force1 = dvat12/drat12 * x1; 
       force3 = dvat34/drat34 * x1; 
       force[at1] -= force1;
       force[at2] += force1;
       force[at3] += force3;
       force[at4] -= force3;
    }
    
} // msesc


