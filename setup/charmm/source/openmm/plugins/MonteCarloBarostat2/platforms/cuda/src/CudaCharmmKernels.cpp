#include "CudaCharmmKernels.h"
#include "CudaCharmmKernelSources.h"
#include "openmm/internal/ContextImpl.h"

using namespace std;

namespace OpenMM {

CudaApplyMonteCarloBarostatKernel2::~CudaApplyMonteCarloBarostatKernel2() {
    cu.setAsCurrent();
    if (savedPositions != NULL)
        delete savedPositions;
    if (moleculeAtoms != NULL)
        delete moleculeAtoms;
    if (moleculeStartIndex != NULL)
        delete moleculeStartIndex;
}

void CudaApplyMonteCarloBarostatKernel2::initialize(const System& system, const MonteCarloBarostat2& thermostat) {
    cu.setAsCurrent();
    savedPositions = new CudaArray(cu, cu.getPaddedNumAtoms(), cu.getUseDoublePrecision() ? sizeof(double4) : sizeof(float4), "savedPositions");
    CUmodule module = cu.createModule(CudaCharmmKernelSources::monteCarloBarostat2);
    kernel = cu.getKernel(module, "scalePositions2");
}

void CudaApplyMonteCarloBarostatKernel2::scaleCoordinates(ContextImpl& context, Vec3& scale) {
    cu.setAsCurrent();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;

        // Create the arrays with the molecule definitions.

        vector<vector<int> > molecules = context.getMolecules();
        numMolecules = molecules.size();
        moleculeAtoms = CudaArray::create<int>(cu, cu.getNumAtoms(), "moleculeAtoms");
        moleculeStartIndex = CudaArray::create<int>(cu, numMolecules+1, "moleculeStartIndex");
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

    }
    int bytesToCopy = cu.getPosq().getSize()*(cu.getUseDoublePrecision() ? sizeof(double4) : sizeof(float4));
    CUresult result = cuMemcpyDtoD(savedPositions->getDevicePointer(), cu.getPosq().getDevicePointer(), bytesToCopy);
    if (result != CUDA_SUCCESS) {
        std::stringstream m;
        m<<"Error saving positions for MC barostat: "<<cu.getErrorString(result)<<" ("<<result<<")";
        throw OpenMMException(m.str());
    }
    Vec3 scale_Vec3 = scale;
    double4 scale_double4 = make_double4(scale[0], scale[1], scale[2], 0.0);
    void* args[] = {&scale_double4, &numMolecules, cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            &cu.getPosq().getDevicePointer(), &moleculeAtoms->getDevicePointer(), &moleculeStartIndex->getDevicePointer()};
    cu.executeKernel(kernel, args, cu.getNumAtoms());
    for (int i = 0; i < (int) cu.getPosCellOffsets().size(); i++)
#ifdef OPENMM_COMMON
        cu.getPosCellOffsets()[i] = mm_int4(0, 0, 0, 0);
#else
        cu.getPosCellOffsets()[i] = make_int4(0, 0, 0, 0);
#endif
    lastAtomOrder = cu.getAtomIndex();
}

void CudaApplyMonteCarloBarostatKernel2::restoreCoordinates(ContextImpl& context) {
    cu.setAsCurrent();
    if (cu.getAtomsWereReordered()) {
        // The atoms were reordered since we saved the positions, so we need to fix them.

        const vector<int> atomOrder = cu.getAtomIndex();
        int numAtoms = cu.getNumAtoms();
        if (cu.getUseDoublePrecision()) {
            double4* pos = (double4*) cu.getPinnedBuffer();
            savedPositions->download(pos);
            vector<double4> fixedPos(cu.getPaddedNumAtoms());
            for (int i = 0; i < numAtoms; i++)
                fixedPos[lastAtomOrder[i]] = pos[i];
            for (int i = 0; i < numAtoms; i++)
                pos[i] = fixedPos[atomOrder[i]];
            cu.getPosq().upload(pos);
        }
        else {
            float4* pos = (float4*) cu.getPinnedBuffer();
            savedPositions->download(pos);
            vector<float4> fixedPos(cu.getPaddedNumAtoms());
            for (int i = 0; i < numAtoms; i++)
                fixedPos[lastAtomOrder[i]] = pos[i];
            for (int i = 0; i < numAtoms; i++)
                pos[i] = fixedPos[atomOrder[i]];
            cu.getPosq().upload(pos);
        }
    }
    else {
        int bytesToCopy = cu.getPosq().getSize()*(cu.getUseDoublePrecision() ? sizeof(double4) : sizeof(float4));
        CUresult result = cuMemcpyDtoD(cu.getPosq().getDevicePointer(), savedPositions->getDevicePointer(), bytesToCopy);
        if (result != CUDA_SUCCESS) {
            std::stringstream m;
            m<<"Error restoring positions for MC barostat: "<<cu.getErrorString(result)<<" ("<<result<<")";
            throw OpenMMException(m.str());
        }
    }
}

}

