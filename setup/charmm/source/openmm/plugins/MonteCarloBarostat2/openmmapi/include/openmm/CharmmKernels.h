#ifndef CHARMM_KERNELS_H_
#define CHARMM_KERNELS_H_

#include "OpenMM.h"
#include "openmm/MonteCarloBarostat2.h"

/*
 * Abstract interfaces for kernels implemented on various platforms.
 */

namespace OpenMM {

/**
 * This kernel is invoked by MonteCarloBarostat2 to adjust the periodic box volume
 */
class ApplyMonteCarloBarostatKernel2 : public KernelImpl {
public:
    static std::string Name() {
        return "ApplyMonteCarloBarostat2";
    }
    ApplyMonteCarloBarostatKernel2(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param barostat   the MonteCarloBarostat this kernel will be used for
     */
    virtual void initialize(const System& system, const MonteCarloBarostat2& barostat) = 0;
    /**
     * Attempt a Monte Carlo step, scaling particle positions (or cluster centers) by a specified value.
     * This is called BEFORE the periodic box size is modified.  It should begin by translating each particle
     * or cluster into the first periodic box, so that coordinates will still be correct after the box size
     * is changed.
     *
     * @param context    the context in which to execute this kernel
     * @param scale      the scale factor by which to multiply particle positions
     */
    virtual void scaleCoordinates(ContextImpl& context, Vec3& scale) = 0;
    /**
     * Reject the most recent Monte Carlo step, restoring the particle positions to where they were before
     * scaleCoordinates() was last called.
     *
     * @param context    the context in which to execute this kernel
     */
    virtual void restoreCoordinates(ContextImpl& context) = 0;
};

}

#endif /* CHARMMKERNELS_H_ */

