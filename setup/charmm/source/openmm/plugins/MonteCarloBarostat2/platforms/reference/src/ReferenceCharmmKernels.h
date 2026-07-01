#ifndef REFERENCE_CHARMM_KERNELS_H_
#define REFERENCE_CHARMM_KERNELS_H_

#include "openmm/CharmmKernels.h"
#include "ReferenceMonteCarloBarostat2.h"
#include <openmm/reference/ReferencePlatform.h>

namespace OpenMM {

/**
 * This kernel is invoked by MonteCarloBarostat2 to adjust the periodic box volume
 */
class ReferenceApplyMonteCarloBarostatKernel2 : public ApplyMonteCarloBarostatKernel2 {
public:
    ReferenceApplyMonteCarloBarostatKernel2(std::string name, const Platform& platform) : ApplyMonteCarloBarostatKernel2(name, platform), barostat(NULL) {
    }
    ~ReferenceApplyMonteCarloBarostatKernel2();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param barostat   the MonteCarloBarostat this kernel will be used for
     */
    void initialize(const System& system, const MonteCarloBarostat2& barostat);
    /**
     * Attempt a Monte Carlo step, scaling particle positions (or cluster centers) by a specified value.
     * This is called BEFORE the periodic box size is modified.  It should begin by translating each particle
     * or cluster into the first periodic box, so that coordinates will still be correct after the box size
     * is changed.
     *
     * @param context    the context in which to execute this kernel
     * @param scale      the scale factor by which to multiply particle positions
     */
    void scaleCoordinates(ContextImpl& context, Vec3& scale);
    /**
     * Reject the most recent Monte Carlo step, restoring the particle positions to where they were before
     * scaleCoordinates() was last called.
     *
     * @param context    the context in which to execute this kernel
     */
    void restoreCoordinates(ContextImpl& context);
private:
    ReferenceMonteCarloBarostat2* barostat;
};

}
#endif

