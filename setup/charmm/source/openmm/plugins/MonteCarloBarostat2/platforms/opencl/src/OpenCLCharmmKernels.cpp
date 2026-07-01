#include "OpenCLCharmmKernels.h"
#include "OpenCLCharmmKernelSources.h"
#include <openmm/opencl/OpenCLArray.h>
#include "openmm/internal/ContextImpl.h"

using namespace std;

namespace OpenMM {

/* copied from OpenCLKernels.cpp */
static void setPeriodicBoxSizeArg(OpenCLContext& cl, cl::Kernel& kernel, int index) {
    if (cl.getUseDoublePrecision())
        kernel.setArg<mm_double4>(index, cl.getPeriodicBoxSizeDouble());
    else
        kernel.setArg<mm_float4>(index, cl.getPeriodicBoxSize());
}

/* copied from OpenCLKernels.cpp */
static void setInvPeriodicBoxSizeArg(OpenCLContext& cl, cl::Kernel& kernel, int index) {
    if (cl.getUseDoublePrecision())
        kernel.setArg<mm_double4>(index, cl.getInvPeriodicBoxSizeDouble());
    else
        kernel.setArg<mm_float4>(index, cl.getInvPeriodicBoxSize());
}

OpenCLApplyMonteCarloBarostatKernel2::~OpenCLApplyMonteCarloBarostatKernel2() {
    if (savedPositions != NULL)
        delete savedPositions;
    if (moleculeAtoms != NULL)
        delete moleculeAtoms;
    if (moleculeStartIndex != NULL)
        delete moleculeStartIndex;
}

void OpenCLApplyMonteCarloBarostatKernel2::initialize(const System& system, const MonteCarloBarostat2& thermostat) {
    savedPositions = OpenCLArray::create<mm_float4>(cl, cl.getPaddedNumAtoms(), "savedPositions");
    cl::Program program = cl.createProgram(OpenCLCharmmKernelSources::monteCarloBarostat2);
    kernel = cl::Kernel(program, "scalePositions2");
}

void OpenCLApplyMonteCarloBarostatKernel2::scaleCoordinates(ContextImpl& context, Vec3& scale) {
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;

        // Create the arrays with the molecule definitions.

        vector<vector<int> > molecules = context.getMolecules();
        numMolecules = molecules.size();
        moleculeAtoms = OpenCLArray::create<int>(cl, cl.getNumAtoms(), "moleculeAtoms");
        moleculeStartIndex = OpenCLArray::create<int>(cl, numMolecules+1, "moleculeStartIndex");
        vector<int> atoms(moleculeAtoms->getSize());
        vector<int> startIndex(moleculeStartIndex->getSize());
        int index = 0;
        for (int i = 0; i < numMolecules; i++) {
            startIndex[i] = index;
            for (int j = 0; j < (int) molecules[i].size(); j++)
                atoms[index++] = molecules[i][j];
        }
        startIndex[numMolecules] = index;
        moleculeAtoms->upload(atoms);
        moleculeStartIndex->upload(startIndex);

        // Initialize the kernel arguments.

        kernel.setArg<cl_int>(1, numMolecules);
        kernel.setArg<cl::Buffer>(4, cl.getPosq().getDeviceBuffer());
        kernel.setArg<cl::Buffer>(5, moleculeAtoms->getDeviceBuffer());
        kernel.setArg<cl::Buffer>(6, moleculeStartIndex->getDeviceBuffer());
    }
    cl.getQueue().enqueueCopyBuffer(cl.getPosq().getDeviceBuffer(), savedPositions->getDeviceBuffer(), 0, 0, cl.getPosq().getSize()*sizeof(mm_float4));
    Vec3 scale_Vec3 = scale;
    if (cl.getUseDoublePrecision()){
        mm_double4 scale_double4;
        scale_double4 = mm_double4(scale_Vec3[0], scale_Vec3[1], scale_Vec3[2], 0);
    kernel.setArg<mm_double4>(0, scale_double4);
    }else{
        mm_float4 scale_float4;
        scale_float4 = mm_float4(scale_Vec3[0], scale_Vec3[1], scale_Vec3[2], 0);
    kernel.setArg<mm_float4>(0, scale_float4);
    }
    setPeriodicBoxSizeArg(cl, kernel, 2);
    setInvPeriodicBoxSizeArg(cl, kernel, 3);
    cl.executeKernel(kernel, cl.getNumAtoms());
    for (int i = 0; i < (int) cl.getPosCellOffsets().size(); i++)
        cl.getPosCellOffsets()[i] = mm_int4(0, 0, 0, 0);
    lastAtomOrder = cl.getAtomIndex();
}

void OpenCLApplyMonteCarloBarostatKernel2::restoreCoordinates(ContextImpl& context) {
    if (cl.getAtomsWereReordered()) {
        // The atoms were reordered since we saved the positions, so we need to fix them.

        const vector<int> atomOrder = cl.getAtomIndex();
        int numAtoms = cl.getNumAtoms();
        if (cl.getUseDoublePrecision()) {
            mm_double4* pos = (mm_double4*) cl.getPinnedBuffer();
            savedPositions->download(pos);
            vector<mm_double4> fixedPos(cl.getPaddedNumAtoms());
            for (int i = 0; i < numAtoms; i++)
                fixedPos[lastAtomOrder[i]] = pos[i];
            for (int i = 0; i < numAtoms; i++)
                pos[i] = fixedPos[atomOrder[i]];
            cl.getPosq().upload(pos);
        }
        else {
            mm_float4* pos = (mm_float4*) cl.getPinnedBuffer();
            savedPositions->download(pos);
            vector<mm_float4> fixedPos(cl.getPaddedNumAtoms());
            for (int i = 0; i < numAtoms; i++)
                fixedPos[lastAtomOrder[i]] = pos[i];
            for (int i = 0; i < numAtoms; i++)
                pos[i] = fixedPos[atomOrder[i]];
            cl.getPosq().upload(pos);
        }
    }
    else
        cl.getQueue().enqueueCopyBuffer(savedPositions->getDeviceBuffer(), cl.getPosq().getDeviceBuffer(), 0, 0, cl.getPosq().getSize()*sizeof(mm_float4));
}

}

