/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

/**
 * This tests the CUDA implementation of GBSWForce.
 */

#include "GBSWForce.h"

#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/LangevinIntegrator.h"
#include "openmm/NonbondedForce.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/internal/AssertionUtilities.h"
#include "sfmt/SFMT.h"

#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace OpenMMGBSW;
using namespace std;

CudaPlatform platform;

const double TOL = 1e-5;

void testSingleParticle() {
    
    // make a system of one atom with radius 0.15 nm.
    // calculate its energy using GBSW
    System system;
    system.addParticle(2.0);
    LangevinIntegrator integrator(0, 0.1, 0.01);
    GBSWForce* gbsw = new GBSWForce();
    NonbondedForce* nonbonded = new NonbondedForce();
    gbsw->addParticle(0.5, 0.15);
    nonbonded->addParticle(0.5, 1, 0);
    system.addForce(gbsw);
    system.addForce(nonbonded);
    Context context(system, integrator, platform);
    vector<Vec3> positions(1);
    positions[0] = Vec3(0, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Energy);
    
    // manually calculate the energy. with one atom, only the self-term is needed
    double bornRadius = 0.15;
    double eps0 = EPSILON0;
    double tau = (1.0/gbsw->getSoluteDielectric()-1.0/gbsw->getSolventDielectric()) / (8*PI_M*eps0);
    double bornEnergy = -0.5*0.5 * tau /bornRadius;
        
//    printf("energy %f calculated    %f ideal\n", state.getPotentialEnergy(), bornEnergy);
    ASSERT_EQUAL_TOL(bornEnergy, state.getPotentialEnergy(), 0.03);
    
    // Change the parameters and see if it is still correct.
    
    gbsw->setParticleParameters(0, 0.4, 0.25);
    gbsw->updateParametersInContext(context);
    state = context.getState(State::Energy);
    bornRadius = 0.25;
    bornEnergy = -0.4*0.4 * tau /bornRadius;
    
//    printf("energy %f calculated  with %f Born R    %f ideal  with %f Born R \n", 
//        state.getPotentialEnergy(), -0.4*0.4 * tau / state.getPotentialEnergy(), 
//        bornEnergy, 0.25);
    ASSERT_EQUAL_TOL(bornEnergy, state.getPotentialEnergy(), 0.09);
}

void testCutoffAndPeriodic() {
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

    ASSERT_EQUAL_VEC(state1.getForces()[0], state2.getForces()[0], 0.01);
    ASSERT_EQUAL_VEC(state1.getForces()[1], state2.getForces()[1], 0.01);
    ASSERT_EQUAL_VEC(state1.getForces()[0], state4.getForces()[0], 0.01);
    ASSERT_EQUAL_VEC(state1.getForces()[1], state4.getForces()[1], 0.01);
    ASSERT_EQUAL_VEC(state3.getForces()[0], Vec3(0, 0, 0), 0.01);
    ASSERT_EQUAL_VEC(state3.getForces()[1], Vec3(0, 0, 0), 0.01);
}

void testForce(int numParticles, NonbondedForce::NonbondedMethod method, GBSWForce::NonbondedMethod method2) {
    ReferencePlatform reference;
    System system;
    GBSWForce* gbsw = new GBSWForce();
    NonbondedForce* nonbonded = new NonbondedForce();
    for (int i = 0; i < numParticles; ++i) {
        system.addParticle(1.0);
        double charge = i%2 == 0 ? -1 : 1;
        gbsw->addParticle(charge, 0.15);
        nonbonded->addParticle(charge, 1, 0);
    }
    nonbonded->setNonbondedMethod(method);
    gbsw->setNonbondedMethod(method2);
    nonbonded->setCutoffDistance(3.0);
    gbsw->setCutoffDistance(3.0);
    int grid = (int) floor(0.5+pow(numParticles, 1.0/3.0));
    if (method == NonbondedForce::CutoffPeriodic) {
        double boxSize = (grid+1)*1.1;
        system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    }
    system.addForce(gbsw);
    system.addForce(nonbonded);
    LangevinIntegrator integrator1(0, 0.1, 0.01);
    LangevinIntegrator integrator2(0, 0.1, 0.01);
    Context context(system, integrator1, platform);
    Context refContext(system, integrator2, reference);

    // Set random (but uniformly distributed) positions for all the particles.

    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    
    double spacing = 0.2; // this was 1.1, which makes a crazy not-dense system
    
    for (int i = 0; i < grid; i++)
        for (int j = 0; j < grid; j++)
            for (int k = 0; k < grid; k++)
                positions[i*grid*grid+j*grid+k] = Vec3(i*spacing, j*spacing, k*spacing);
    for (int i = 0; i < numParticles; ++i)
        positions[i] = positions[i] + Vec3(0.5*genrand_real2(sfmt), 0.5*genrand_real2(sfmt), 0.5*genrand_real2(sfmt));
    context.setPositions(positions);
    refContext.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    State refState = refContext.getState(State::Forces | State::Energy);

    // Make sure the CUDA and Reference platforms agree.

    double norm = 0.0;
    double diff = 0.0;
    for (int i = 0; i < numParticles; ++i) {
        Vec3 f = state.getForces()[i];
        norm += f[0]*f[0] + f[1]*f[1] + f[2]*f[2];
        Vec3 delta = f-refState.getForces()[i];
        diff += delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
    }
    norm = std::sqrt(norm) / (double)numParticles;
    diff = std::sqrt(diff) / (double)numParticles;
    
//    printf("%d particles  CUDA energy %f    Reference energy %f   normal  %f    difference %f\n", 
//        numParticles, state.getPotentialEnergy(), refState.getPotentialEnergy(), norm, diff);
    
    double dot = 0.0;
    double magA = 0.0;
    double magB = 0.0;
    for (int i = 0; i < numParticles; ++i) {
        Vec3 f = state.getForces()[i];
        Vec3 fr = refState.getForces()[i];
        
        dot += f[0]*fr[0] + f[1]*fr[1] + f[2]*fr[2];
        magA += f[0]*f[0] + f[1]*f[1] + f[2]*f[2];
        magB += fr[0]*fr[0] + fr[1]*fr[1] + fr[2]*fr[2];
    }
    
//    printf("%d particles  cosine %f   magnitude B %f    magnitude A %f\n", 
//        numParticles, dot / (std::sqrt(magA)* std::sqrt(magB)), std::sqrt(magA), std::sqrt(magB));
    
    ASSERT_EQUAL_TOL(1.0, dot / (std::sqrt(magA)* std::sqrt(magB)), 0.02); // dot-product comparison
    ASSERT_EQUAL_TOL(0.0, diff, 0.2*norm);
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), refState.getPotentialEnergy(), 0.01);

    // Take a small step in the direction of the energy gradient and see whether the potential energy changes by the expected amount.
    // (This doesn't work with cutoffs, since the energy changes discontinuously at the cutoff distance.)
    
    // this finite difference method doesn't work well with GBSW, since the energy
    // function derivative is not a continuous
    
//    if (method == NonbondedForce::NoCutoff) {
//        const double delta = 0.0000003;
//        double step = 0.5*delta/norm;
//        vector<Vec3> positions2(numParticles), positions3(numParticles);
//        for (int i = 0; i < numParticles; ++i) {
//            Vec3 p = positions[i];
//            Vec3 f = state.getForces()[i];
////            printf("%d    %f   %f   %f\n",i, -f[0]*step,-f[1]*step,-f[2]*step);
//            positions2[i] = Vec3(p[0]-f[0]*step, p[1]-f[1]*step, p[2]-f[2]*step);
//            positions3[i] = Vec3(p[0]+f[0]*step, p[1]+f[1]*step, p[2]+f[2]*step);
//        }
//        context.setPositions(positions2);
//        State state2 = context.getState(State::Energy);
//        context.setPositions(positions3);
//        State state3 = context.getState(State::Energy);
//        
//        printf("%f    %f   - %f   %f   %f\n", state2.getPotentialEnergy(), 
//                state3.getPotentialEnergy(), (state2.getPotentialEnergy()-state3.getPotentialEnergy()), norm, delta);
//        
//        ASSERT_EQUAL_TOL(norm, (state2.getPotentialEnergy()-state3.getPotentialEnergy())/0.005, 0.07 * (1 + delta))
//    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 1)
            platform.setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        testSingleParticle();
        testCutoffAndPeriodic();
        for (int i = 5; i < 11; i++) {
            testForce(i*i*i, NonbondedForce::NoCutoff, GBSWForce::NoCutoff);
            testForce(i*i*i, NonbondedForce::CutoffNonPeriodic, GBSWForce::CutoffNonPeriodic);
            testForce(i*i*i, NonbondedForce::CutoffPeriodic, GBSWForce::CutoffPeriodic);
        }
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}

