#ifndef OPENCL_CHARMM_KERNELS_H_
#define OPENCL_CHARMM_KERNELS_H_

#include "openmm/CharmmKernels.h"
#include <openmm/opencl/OpenCLContext.h>

namespace OpenMM {

/**
 * This kernel is invoked by MonteCarloBarostat2 to adjust the periodic box volume
 */
class OpenCLApplyMonteCarloBarostatKernel2 : public ApplyMonteCarloBarostatKernel2 {
public:
    OpenCLApplyMonteCarloBarostatKernel2(std::string name, const Platform& platform, OpenCLContext& cl) : ApplyMonteCarloBarostatKernel2(name, platform), cl(cl),
            hasInitializedKernels(false), savedPositions(NULL), moleculeAtoms(NULL), moleculeStartIndex(NULL) {
    }

    ~OpenCLApplyMonteCarloBarostatKernel2();

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param barostat   the MonteCarloBarostat2 this kernel will be used for
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
    OpenCLContext& cl;
    bool hasInitializedKernels;
    int numMolecules;
    OpenCLArray* savedPositions;
    OpenCLArray* moleculeAtoms;
    OpenCLArray* moleculeStartIndex;
    cl::Kernel kernel;
    std::vector<int> lastAtomOrder;
};

}
#endif

