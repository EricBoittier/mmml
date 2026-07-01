/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

/**
 * This tests the reference implementation of GBSWForce.
 */

#include "GBSWForce.h"

#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/LangevinIntegrator.h"
#include "openmm/NonbondedForce.h"
#include "openmm/opencl/OpenCLPlatform.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/internal/AssertionUtilities.h"
#include "sfmt/SFMT.h"

#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace OpenMMGBSW;
using namespace std;

const double TOL = 1e-5;

void testSingleParticle() {
    ReferencePlatform platform;
    System system;
    system.addParticle(2.0);
    LangevinIntegrator integrator(0, 0.1, 0.01);
    GBSWForce* forceField = new GBSWForce();
    forceField->addParticle(0.5, 0.15);
    system.addForce(forceField);
    Context context(system, integrator, platform);
    vector<Vec3> positions(1);
    positions[0] = Vec3(0, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Energy);
    double bornRadius = 0.15;
    double eps0 = EPSILON0;
    double bornEnergy = (-0.5*0.5/(8*PI_M*eps0)) * 
        (1.0/forceField->getSoluteDielectric()-1.0/forceField->getSolventDielectric())/bornRadius;
    ASSERT_EQUAL_TOL(bornEnergy, state.getPotentialEnergy(), 0.1);
    
    // Change the parameters and see if it is still correct.
    
    forceField->setParticleParameters(0, 0.4, 0.25);
    forceField->updateParametersInContext(context);
    state = context.getState(State::Energy);
    bornRadius = 0.25;
    bornEnergy = (-0.4*0.4/(8*PI_M*eps0)) * 1.1 *
        (1.0/forceField->getSoluteDielectric()-1.0/forceField->getSolventDielectric())/bornRadius;
    ASSERT_EQUAL_TOL((bornEnergy), state.getPotentialEnergy(), 0.1);
}

void testGlobalSettings() {
    ReferencePlatform platform;
    System system;
    system.addParticle(2.0);
    LangevinIntegrator integrator(0, 0.1, 0.01);
    GBSWForce* forceField = new GBSWForce();
    forceField->addParticle(0.5, 0.15);
    const double soluteDielectric = 2.1;
    const double solventDielectric = 35.0;
    const double surfaceAreaEnergy = 0.75;
    forceField->setSoluteDielectric(soluteDielectric);
    forceField->setSolventDielectric(solventDielectric);
    forceField->setSurfaceAreaEnergy(surfaceAreaEnergy);
    system.addForce(forceField);
    Context context(system, integrator, platform);
    vector<Vec3> positions(1);
    positions[0] = Vec3(0, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Energy);
    double bornRadius = 0.15-0.009; // dielectric offset
    double eps0 = EPSILON0;
    double bornEnergy = (-0.5*0.5/(8*PI_M*eps0))*(1.0/soluteDielectric-1.0/solventDielectric)/bornRadius;
    double extendedRadius = 0.15+0.14; // probe radius
    double nonpolarEnergy = 4*PI_M*surfaceAreaEnergy*extendedRadius*extendedRadius*std::pow(0.15/bornRadius, 6.0);
    ASSERT_EQUAL_TOL((bornEnergy), state.getPotentialEnergy(), 0.1);
}

void testCutoffAndPeriodic() {
    ReferencePlatform platform;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    LangevinIntegrator integrator(0, 0.1, 0.01);
    GBSWForce* gbsw = new GBSWForce();
    NonbondedForce* nonbonded = new NonbondedForce();
    gbsw->addParticle(-1, 0.15);
    nonbonded->addParticle(-1, 1, 0);
    gbsw->addParticle(1, 0.15);
    nonbonded->addParticle(1, 1, 0);
    const double cutoffDistance = 3.0;
    const double boxSize = 10.0;
    nonbonded->setCutoffDistance(cutoffDistance);
    gbsw->setCutoffDistance(cutoffDistance);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    system.addForce(gbsw);
    system.addForce(nonbonded);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(2, 0, 0);

    // Calculate the forces for both cutoff and periodic with two different atom positions.
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffNonPeriodic);
    gbsw->setNonbondedMethod(GBSWForce::CutoffNonPeriodic);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State state1 = context.getState(State::Forces);
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    gbsw->setNonbondedMethod(GBSWForce::CutoffPeriodic);
    context.reinitialize();
    context.setPositions(positions);
    State state2 = context.getState(State::Forces);
    positions[1][0]+= boxSize;
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffNonPeriodic);
    gbsw->setNonbondedMethod(GBSWForce::CutoffNonPeriodic);
    context.reinitialize();
    context.setPositions(positions);
    State state3 = context.getState(State::Forces);
    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    gbsw->setNonbondedMethod(GBSWForce::CutoffPeriodic);
    context.reinitialize();
    context.setPositions(positions);
    State state4 = context.getState(State::Forces);

    // All forces should be identical, exception state3 which should be zero.

    ASSERT_EQUAL_VEC(state1.getForces()[0], state2.getForces()[0], 0.1);
    ASSERT_EQUAL_VEC(state1.getForces()[1], state2.getForces()[1], 0.1);
    ASSERT_EQUAL_VEC(state1.getForces()[0], state4.getForces()[0], 0.1);
    ASSERT_EQUAL_VEC(state1.getForces()[1], state4.getForces()[1], 0.1);
    ASSERT_EQUAL_VEC(state3.getForces()[0], Vec3(0, 0, 0), 0.1);
    ASSERT_EQUAL_VEC(state3.getForces()[1], Vec3(0, 0, 0), 0.1);
}

void testForce() {
    ReferencePlatform platform;
    const int numParticles = 10;
    System system;
    LangevinIntegrator integrator(0, 0.1, 0.01);
    GBSWForce* forceField = new GBSWForce();
    for (int i = 0; i < numParticles; ++i) {
        system.addParticle(1.0);
        forceField->addParticle(i%2 == 0 ? -1 : 1, 0.15);
    }
    system.addForce(forceField);
    Context context(system, integrator, platform);
    
    // Set random positions for all the particles.
    
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);

    for (int i = 0; i < numParticles; ++i)
        positions[i] = Vec3(5.0*genrand_real2(sfmt), 5.0*genrand_real2(sfmt), 5.0*genrand_real2(sfmt));
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    
    // Take a small step in the direction of the energy gradient and see whether the potential energy changes by the expected amount.
    
    double norm = 0.0;
    for (int i = 0; i < numParticles; ++i) {
        Vec3 f = state.getForces()[i];
        norm += f[0]*f[0] + f[1]*f[1] + f[2]*f[2];
    }
    norm = std::sqrt(norm);
    const double delta = 1e-2;
    double step = 0.5*delta/norm;
    vector<Vec3> positions2(numParticles), positions3(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        Vec3 p = positions[i];
        Vec3 f = state.getForces()[i];
        positions2[i] = Vec3(p[0]-f[0]*step, p[1]-f[1]*step, p[2]-f[2]*step);
        positions3[i] = Vec3(p[0]+f[0]*step, p[1]+f[1]*step, p[2]+f[2]*step);
    }
    context.setPositions(positions2);
    State state2 = context.getState(State::Energy);
    context.setPositions(positions3);
    State state3 = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(norm, (state2.getPotentialEnergy()-state3.getPotentialEnergy())/delta, 0.01)
}

int main() {
    try {
        testSingleParticle();
        testGlobalSettings();
        testCutoffAndPeriodic();
        testForce();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
