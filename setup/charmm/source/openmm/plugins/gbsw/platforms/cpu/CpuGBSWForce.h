#ifndef OPENMM_CPU_GBSW_FORCE_H__
#define OPENMM_CPU_GBSW_FORCE_H__

#include "openmm/cpu/AlignedArray.h"
#include "openmm/internal/ThreadPool.h"
#include "openmm/internal/vectorize.h"
#include <vector>

namespace OpenMMGBSW {

class CpuGBSWForce {
public:
    class ComputeTask;
    CpuGBSWForce();

    /**
     * Set the force to use a cutoff.
     * 
     * @param distance    the cutoff distance
     */
    void setUseCutoff(float distance);

    /**
     * 
     * Set the force to use periodic boundary conditions.  This requires that a cutoff has
     * already been set, and the smallest side of the periodic box is at least twice the cutoff
     * distance.
     *
     * @param boxSize             the X, Y, and Z widths of the periodic box
     */
    void setPeriodic(float* periodicBoxSize);

    /**
     * Set the solute dielectric constant.
     */
    void setSoluteDielectric(float dielectric);

    /**
     * Set the solvent dielectric constant.
     */
    void setSolventDielectric(float dielectric);
    
    /**
     * Set the surface area energy.
     */
    void setSurfaceAreaEnergy(float energy);
    
    /**
     * Get the per-particle parameters (offset radius, scaled radius).
     */
    const std::vector<std::pair<float, float> >& getParticleParameters() const;
    
    /**
     * Set the per-particle parameters (offset radius, scaled radius).
     */
    void setParticleParameters(const std::vector<std::pair<float, float> >& params);

    /**
     * 
     * Calculate LJ Coulomb pair ixn
     *
     * @param posq             atom coordinates and charges
     * @param forces           force array (forces added)
     * @param totalEnergy      total energy
     * @param threads          the thread pool to use
     */
    void computeForce(const OpenMM::AlignedArray<float>& posq, std::vector<OpenMM::AlignedArray<float> >& threadForce, double* totalEnergy, OpenMM::ThreadPool& threads);

    /**
     * This routine contains the code executed by each thread.
     */
    void threadComputeForce(OpenMM::ThreadPool& threads, int threadIndex);

private:
    bool cutoff;
    bool periodic;
    float periodicBoxSize[3];
    float cutoffDistance, soluteDielectric, solventDielectric, surfaceAreaFactor;
    std::vector<std::pair<float, float> > particleParams;        
    OpenMM::AlignedArray<float> bornRadii;
    std::vector<OpenMM::AlignedArray<float> > threadBornForces;
    OpenMM::AlignedArray<float> gbswChain;
    std::vector<double> threadEnergy;
    std::vector<float> logTable;
    float logDX, logDXInv;
    // The following variables are used to make information accessible to the individual threads.
    float const* posq;
    std::vector<OpenMM::AlignedArray<float> >* threadForce;
    bool includeEnergy;
    void* atomicCounter;
  
    static const int NUM_TABLE_POINTS;
    static const float TABLE_MIN;
    static const float TABLE_MAX;

    /**
     * Compute the displacement and squared distance between a collection of points, optionally using
     * periodic boundary conditions.
     */
    void getDeltaR(const fvec4& posI, const fvec4& x, const fvec4& y, const fvec4& z, fvec4& dx, fvec4& dy, fvec4& dz, fvec4& r2, bool periodic, const fvec4& boxSize, const fvec4& invBoxSize) const;
    
    /**
     * Evaluate log(x) using a lookup table for speed.
     */
    fvec4 fastLog(const fvec4& x);
};

} // namespace OpenMMGBSW

// ---------------------------------------------------------------------------------------

#endif // OPENMM_CPU_GBSW_FORCE_H__
