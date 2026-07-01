#include "ReferenceCharmmKernels.h"
#include "openmm/internal/ContextImpl.h"

using namespace std;

namespace OpenMM {

/* copied from ReferenceKernels.cpp */
static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

/* copied from ReferenceKernels.cpp */
static RealVec& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *(RealVec*) data->periodicBoxSize;
}

ReferenceApplyMonteCarloBarostatKernel2::~ReferenceApplyMonteCarloBarostatKernel2() {
    if (barostat)
        delete barostat;
}

void ReferenceApplyMonteCarloBarostatKernel2::initialize(const System& system, const MonteCarloBarostat2& barostat) {
}

void ReferenceApplyMonteCarloBarostatKernel2::scaleCoordinates(ContextImpl& context, Vec3& scale) {
    if (barostat == NULL)
        barostat = new ReferenceMonteCarloBarostat2(context.getSystem().getNumParticles(), context.getMolecules());
    vector<RealVec>& posData = extractPositions(context);
    RealVec& boxSize = extractBoxSize(context);
    barostat->applyBarostat(posData, boxSize, scale);
}

void ReferenceApplyMonteCarloBarostatKernel2::restoreCoordinates(ContextImpl& context) {
    vector<RealVec>& posData = extractPositions(context);
    barostat->restorePositions(posData);
}

}

