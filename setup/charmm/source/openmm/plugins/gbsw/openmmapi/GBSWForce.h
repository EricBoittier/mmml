#ifndef OPENMM_GBSWFORCEFIELD_H_
#define OPENMM_GBSWFORCEFIELD_H_

/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBSW                                 *
 * -------------------------------------------------------------------------- */

#include "openmm/Force.h"

#include <vector>

namespace OpenMMGBSW {

/**
 * This class implements an implicit solvation force using the GBSW model.
 * 
 * To use this class, create a GBSWForce object, then call addParticle() once for each particle in the
 * System to define its parameters.  The number of particles for which you define GBSW parameters must
 * be exactly equal to the number of particles in the System, or else an exception will be thrown when you
 * try to create a Context.  After a particle has been added, you can modify its force field parameters
 * by calling setParticleParameters().  This will have no effect on Contexts that already exist unless you
 * call updateParametersInContext().
 * 
 * When using this Force, the System should also include a NonbondedForce, and both objects must specify
 * identical charges for all particles.  Otherwise, the results will not be correct.  Furthermore, if the
 * nonbonded method is set to CutoffNonPeriodic or CutoffPeriodic, you should call setReactionFieldDielectric(1.0)
 * on the NonbondedForce to turn off the reaction field approximation, which does not produce correct results
 * when combined with GBSW.
 */

class GBSWForce : public OpenMM::Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,
        /**
         * Interactions beyond the cutoff distance are ignored.
         */
        CutoffNonPeriodic = 1,
        /**
         * Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of
         * each other particle.  Interactions beyond the cutoff distance are ignored.
         */
        CutoffPeriodic = 2,
    };
    /**
     * Create a GBSWForce.
     */
    GBSWForce();
    /**
     * Initialize CPHMD force with lambda dynamics.
     * 
     * @param useCPHMD         boolean of whether or not CPHMD is active
     * @param nTitratingGroups number of titrating groups
     * @param globalPH         pH of this system
     * @param timestepTheta    dynamics timestep
     * @param temperaturePH    temperature of the theta dimension
     * @param massTheta        mass of atoms in the theta dimension
     * @param outputFrequency  how many timesteps between recording lambda values
     */
    void addCPHMDForce( double pH, double T_theta, double mTheta, double ts_theta,
        double beta, int outFreq, char* fileName ) {
        useCPHMD = true;
        globalPH = pH;
        tempTheta = T_theta;
        timestepTheta = ts_theta;
        massTheta = mTheta;
        pHbeta = beta;
        outputFrequency = outFreq;
        lambdaOutFile = fileName;
    }
    /**
     * Ask whether or not using CPHMD (lambda dynamics)
     */
    bool usingCPHMD() const {
        return useCPHMD;
    }
  
    /**
     * Do we need to integrate within the GBSW Force for CPHMD?
     */
    void setDoingDynamics(OpenMM::Context & context, bool val);
  
    /**
     * Get the number of titrating groups in the system.
     */
    int getNumTitratingGroups() const {
        return titratingGroups.size();
    }
    /**
     * Add CPHMD parameters for a new titrating group. This should be called once
     * for each titrating group in the system. When it is called for the i'th time,
     * it specifies parameters for the i'th group.
     *
     * @param resPKA1   the pKa of tautomer 1 of this residue or group of atoms
     * @param resPKA2   the pKa of tautomer 2 of this residue or group of atoms
     * @param barrier1  the tautomer 1 energy barrier parameter beta for protonation
     * @param barrier2  the tautomer 2 energy barrier parameter beta for protonation
     * @param a0        the lambda^2 * x^2 term of the model potential (Umod)
     * @param a1        the lambda^2 * x   term of the model potential (Umod)
     * @param a2        the lambda   * x^2 term of the model potential (Umod)
     * @param a3        the lambda   * x   term of the model potential (Umod)
     * @param a4        the lambda^2       term of the model potential (Umod)
     * @param a5        the            x^2 term of the model potential (Umod)
     * @param a6        the lambda         term of the model potential (Umod)
     * @param a7        the            x   term of the model potential (Umod)
     * @param a8        the                term of the model potential (Umod)
     */
    int addTitratingGroupParameters(
        double resPKA1, double resPKA2, double barrier1, double barrier2,
        double a0, double a1, double a2, double a3, double a4, double a5, 
        double a6, double a7, double a8);
    /**
     * Get CPHMD parameters for a new titrating group. This should be called once
     * for each titrating group in the system. When it is called for the i'th time,
     * it specifies parameters for the i'th group.
     *
     * @param index     the index of the titrating group for which to get parameters
     * @param resPKA1   the pKa of tautomer 1 of this residue or group of atoms
     * @param resPKA2   the pKa of tautomer 2 of this residue or group of atoms
     * @param barrier1  the tautomer 1 energy barrier parameter beta for protonation
     * @param barrier2  the tautomer 2 energy barrier parameter beta for protonation
     * @param a0        the lambda^2 * x^2 term of the model potential (Umod)
     * @param a1        the lambda^2 * x   term of the model potential (Umod)
     * @param a2        the lambda   * x^2 term of the model potential (Umod)
     * @param a3        the lambda   * x   term of the model potential (Umod)
     * @param a4        the lambda^2       term of the model potential (Umod)
     * @param a5        the            x^2 term of the model potential (Umod)
     * @param a6        the lambda         term of the model potential (Umod)
     * @param a7        the            x   term of the model potential (Umod)
     * @param a8        the                term of the model potential (Umod)
     */
    void getTitratingGroupParameters(int index, 
        double& resPKA1, double& resPKA2, double& barrier1, double& barrier2,
        double& a0, double& a1, double& a2, double& a3, double& a4, double& a5, 
        double& a6, double& a7, double& a8) const;
    /**
     * Get CPHMD parameters for a new titrating group. This should be called once
     * for each titrating group in the system. When it is called for the i'th time,
     * it specifies parameters for the i'th group.
     *
     * @param index     the index of the titrating group for which to get parameters
     * @param resPKA1   the pKa of tautomer 1 of this residue or group of atoms
     * @param resPKA2   the pKa of tautomer 2 of this residue or group of atoms
     * @param barrier1  the tautomer 1 energy barrier parameter beta for protonation
     * @param barrier2  the tautomer 2 energy barrier parameter beta for protonation
     * @param a0        the lambda^2 * x^2 term of the model potential (Umod)
     * @param a1        the lambda^2 * x   term of the model potential (Umod)
     * @param a2        the lambda   * x^2 term of the model potential (Umod)
     * @param a3        the lambda   * x   term of the model potential (Umod)
     * @param a4        the lambda^2       term of the model potential (Umod)
     * @param a5        the            x^2 term of the model potential (Umod)
     * @param a6        the lambda         term of the model potential (Umod)
     * @param a7        the            x   term of the model potential (Umod)
     * @param a8        the                term of the model potential (Umod)
     */
    void setTitratingGroupParameters(int index, 
        double resPKA1, double resPKA2, double barrier1, double barrier2,
        double a0, double a1, double a2, double a3, double a4, double a5, 
        double a6, double a7, double a8);
    /**
     * Add a nonbonded exception for the CPHMD calculation (such as a 1-4 interaction)
     * 
     * @param atom1         atom 1's atom index
     * @param atom2         atom 2's atom index
     */
    int addNonbondedException( int atom1, int atom2 );
    /**
     * Get a nonbonded exception for the CPHMD calculation (such as a 1-4 interaction)
     * 
     * @param atom1         atom 1's atom index
     * @param atom2         atom 2's atom index
     */
    void getNonbondedException( int index, int& atom1, int& atom2 ) const;
    /**
     * Get the number of nonbonded pair exceptions
     */
    int getNumNonbondedExceptions() const {
        return nonbondedFixes.size();
    }
    /**
     * Get the number of particles in the system.
     */
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Add the GBSW parameters for a particle.  This should be called once for each particle
     * in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.
     *
     * @param charge         the charge of the particle, measured in units of the proton charge
     * @param radius         the GBSW radius of the particle, measured in nm
     */
    int addParticle(double charge, double radius);
    /**
     * Get the force field parameters for a particle.
     * 
     * @param index          the index of the particle for which to get parameters
     * @param charge         the charge of the particle, measured in units of the proton charge
     * @param radius         the GBSW radius of the particle, measured in nm
     */
    void getParticleParameters(int index, double& charge, double& radius) const;
    /**
     * Set the force field parameters for a particle.
     * 
     * @param index          the index of the particle for which to set parameters
     * @param charge         the charge of the particle, measured in units of the proton charge
     * @param radius         the GBSW radius of the particle, measured in nm
     */
    void setParticleParameters(int index, double charge, double radius);
    /**
     * Add the CPHMD parameters for a particle.  This should be called once for each particle
     * in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.
     *
     * @param titrateResID   the residue ID of this particle if it titrates
     * @param refChargeState1 the lambda-dynamics partial charge at neutral pH when x=0
     * @param refChargeState2 the lambda-dynamics partial charge at neutral pH when x=1
     * @param chargeState1   the lambda-dynamics partial charge state when x=0
     * @param chargeState2   the lambda-dynamics partial charge state when x=1
     */
    int addCphmdParameters(int titrateResID, double refChargeState1, double refChargeState2,
        double chargeState1, double chargeState2);
    /**
     * Get the CPHMD force field parameters for a particle.
     * 
     * @param index          the index of the particle for which to get parameters
     * @param titrateResID   the residue ID of this particle if it titrates
     * @param refChargeState1 the lambda-dynamics partial charge at neutral pH when x=0
     * @param refChargeState2 the lambda-dynamics partial charge at neutral pH when x=1
     * @param chargeState1   the lambda-dynamics partial charge state when x=0
     * @param chargeState2   the lambda-dynamics partial charge state when x=1
     */
    void getCphmdParameters(int index, int& titrateResID, double& refChargeState1, 
        double& refChargeState2, double& chargeState1, double& chargeState2) const;
    /**
     * Set the CPHMD force field parameters for a particle.
     * 
     * @param index          the index of the particle for which to set parameters
     * @param titrateResID   the residue ID of this particle if it titrates
     * @param refChargeState1 the lambda-dynamics partial charge at neutral pH when x=0
     * @param refChargeState2 the lambda-dynamics partial charge at neutral pH when x=1
     * @param chargeState1   the lambda-dynamics partial charge state when x=0
     * @param chargeState2   the lambda-dynamics partial charge state when x=1
     */
    void setCphmdParameters(int index, int titrateResID, double refChargeState1, 
        double refChargeState2, double chargeState1, double chargeState2);
    /**
     * Get the name of the lambda output file
     */
    char* getLambdaOutputFile() const {
        return lambdaOutFile;
    }
    /**
     * Set the name of the lambda output file
     */
    void setLambdaOutputFile( char* tmp ) {
        lambdaOutFile = tmp;
    }
    /**
     * Get the pH of the system
     */
    double getSystemPH() const {
        return globalPH;
    }
    /**
     * Set the pH of the system
     */
    void setSystemPH( double tmp ) {
        globalPH = tmp;
    }
    /**
     * Get the temperature of theta (lambda) dynamics
     */
    double getThetaTemp() const {
        return tempTheta;
    }
    /**
     * Set the temperature of theta (lambda) dynamics
     */
    void setThetaTemp( double tmp ) {
        tempTheta = tmp;
    }
    /**
     * Get the timestep of theta (lambda) dynamics
     */
    double getThetaTimestep() const {
        return timestepTheta;
    }
    /**
     * Set the timestep of theta (lambda) dynamics
     */
    void setThetaTimestep( double tmp ) {
        timestepTheta = tmp;
    }
    /**
     * Get the mass of atoms in the theta dimension
     */
    double getThetaMass() const {
        return massTheta;
    }
    /**
     * Set the mass of atoms in the theta dimension
     */
    void setThetaMass( double tmp ) {
        massTheta = tmp;
    }
    /**
     * Get how many timesteps between recording lambda values
     */
    int getLambdaOutputFrequency() const {
        return outputFrequency;
    }
    /**
     * Set how many timesteps between recording lambda values
     */
    void setLambdaOutputFrequency( int tmp ) {
        outputFrequency = tmp;
    }
    /**
     * Get the Langevan dynamics beta-factor (scales random force)
     */
    double getPHbeta() const {
        return pHbeta;
    }
    /**
     * Set the Langevan dynamics beta-factor (scales random force)
     */
    void setPHbeta( double tmp ) {
        pHbeta = tmp;
    }
    /**
     * Get the dielectric constant for the solvent.
     */
    double getSolventDielectric() const {
        return solventDielectric;
    }
    /**
     * Set the dielectric constant for the solvent.
     */
    void setSolventDielectric(double dielectric) {
        solventDielectric = dielectric;
    }
    /**
     * Get the dielectric constant for the solute.
     */
    double getSoluteDielectric() const {
        return soluteDielectric;
    }
    /**
     * Set the dielectric constant for the solute.
     */
    void setSoluteDielectric(double dielectric) {
        soluteDielectric = dielectric;
    }
    /**
     * Get the energy scale for the surface energy term, measured in kJ/mol/nm^2.
     */
    double getSurfaceAreaEnergy() const {
        return surfaceAreaEnergy;
    }
    /**
     * Set the energy scale for the surface energy term, measured in kJ/mol/nm^2.
     */
    void setSurfaceAreaEnergy(double energy) {
        surfaceAreaEnergy = energy;
    }
    /**
     * get the phenomenological constant AA0 for the GBSW delta-G0 term.
     */
    double getAA0() const {
        return AA0;
    }
    /**
     * set the phenomenological constant AA0 for the GBSW delta-G0 term.
     */
    void setAA0(double value) {
        AA0 = value;
    }
    /**
     * get the phenomenological constant AA1 for the GBSW delta-G1 term.
     */
    double getAA1() const {
        return AA1;
    }
    /**
     * set the phenomenological constant AA1 for the GBSW delta-G1 term.
     */
    void setAA1(double value) {
        AA1 = value;
    }
    /**
     * Get number of gaussian-legendre radii used for integration for Born R calculation.
     */
    int getNumGauLegRad() const {
        return numRadii;
    }
    /**
     * Set number of gaussian-legendre radii used for integration for Born R calculation.
     */
    void setNumGauLegRad(int number) {
        numRadii = number;
    }
    /**
     * Get number of lebedev angles used for integration for Born R calculation.
     */
    int getNumLebAng() const {
        return numAngles;
    }
    /**
     * Set number of lebedev angles used for integration for Born R calculation.
     */
    void setNumLebAng(int number) {
        numAngles = number;
    }
    /**
     * Get the Debye-Huckel length that adds a salt concentration to GBSW.
     */
    double getDebyeHuckelLength() const {
        return kappa;
    }
    /**
     * Set the Debye-Huckel length that adds a salt concentration to GBSW.
     */
    void setDebyeHuckelLength(double length) {
        kappa = length;
    }
    /**
     * Get the switching length that blurs the dielectric boundary in nm.
     */
    double getSwitchingLength() const {
        return swLen;
    }
    /**
     * Set the switching length that blurs the dielectric boundary in nm.
     */
    void setSwitchingLength(double length) {
        swLen = length;
    }
    /**
     * Get the thickness of the membrane about the XY-plane
     */
    double getMembraneThickness() const {
        return membraneThickness;
    }
    /**
     * Set the thickness of the membrane about the XY-plane
     */
    void setMembraneThickness( double tmp ) {
        membraneThickness = tmp;
    }
    /**
     * Get the switching function length of the membrane about the XY-plane
     */
    double getMembraneSwLen() const {
        return membraneSwLen;
    }
    /**
     * Set the switching function length of the membrane about the XY-plane
     */
    void setMembraneSwLen( double tmp ) {
        membraneSwLen = tmp;
    }
    /**
     * Get the method used for handling long range nonbonded interactions.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long range nonbonded interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);
    /**
     * Update the particle parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to
     * reinitialize it.  Simply call setParticleParameters() to modify this object's parameters, then call
     * updateParametersInState() to copy them over to the Context.
     * 
     * The only information this method updates is the values of per-particle parameters.  All other aspects
     * of the Force (the nonbonded method, the cutoff distance, etc.) are unaffected and can only be changed
     * by reinitializing the Context.  Furthermore, this method cannot be used to add new particles, only to
     * change the parameters of existing ones.
     */
    void updateParametersInContext(OpenMM::Context& context);
    /**
     * Copy lambda information to arrays: lambda positions, velocities, and forces
     *
     * @param context            the context to copy parameters to
     * @param LambdaPosVelForce  the array of the position, velocity, and force of lambdas
     */
    void getLambdaState(OpenMM::Context& context, std::vector<double>& LambdaState);
    /**
     * Set lambda information to arrays: lambda positions, velocities, and forces
     *
     * @param context            the context to copy parameters from
     * @param LambdaPosVelForce  the array of the position, velocity, and force of lambdas
     */
    void setLambdaState(OpenMM::Context& context, std::vector<double>& LambdaState);
    /**
     * Set reaction field dielectric parameter
     *
     * @param rfDielectric       reaction field dielectric used for OpenMM coulombic interaction
     */
    double getReactionFieldDielectric() const;
    /**
     * Set reaction field dielectric parameter
     *
     * @param rfDielectric       reaction field dielectric used for OpenMM coulombic interaction
     */
    void setReactionFieldDielectric(double tmp);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == GBSWForce::CutoffPeriodic;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    class ParticleCphmdInfo;
    class TitratingGroupInfo;
    class NonbondedExceptionFixes;
    NonbondedMethod nonbondedMethod;
    bool useCPHMD;
    double cutoffDistance, solventDielectric, soluteDielectric, surfaceAreaEnergy,
        swLen, AA0, AA1, kappa, tempTheta, timestepTheta, globalPH, massTheta, pHbeta,
        membraneThickness, membraneSwLen, rfDielectric;
    int numAngles, numRadii, outputFrequency;
    char* lambdaOutFile;
    std::vector<ParticleInfo> particles;
    std::vector<ParticleCphmdInfo> cphmdInfo;
    std::vector<TitratingGroupInfo> titratingGroups;
    std::vector<NonbondedExceptionFixes> nonbondedFixes;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class GBSWForce::ParticleInfo {
public:
    double charge, radius;
    ParticleInfo() {
        charge = radius = 0.0;
    }
    ParticleInfo(double charge, double radius) :
        charge(charge), radius(radius) {
    }
};

/**
 * This is an internal class used to record titrating group information about a residue.
 * @private
 */
class GBSWForce::TitratingGroupInfo {
public:
    double resPKA1, resPKA2, barrier1, barrier2, a0, a1, a2, a3, a4, a5, a6, a7, a8;
    TitratingGroupInfo() {
        resPKA1 = resPKA2 = barrier1 = barrier2 = 
        a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = 0.0;
    }
    TitratingGroupInfo( double resPKA1, double resPKA2, double barrier1, double barrier2,
        double a0, double a1, double a2, double a3, 
        double a4, double a5, double a6, double a7, double a8 ) :
        resPKA1(resPKA1), resPKA2(resPKA2), barrier1(barrier1), barrier2(barrier2), 
        a0(a0), a1(a1), a2(a2), a3(a3), a4(a4), a5(a5), a6(a6), a7(a7), a8(a8) {
    }
};

/**
 * This is an internal class used to record CPHMD information about a particle.
 * @private
 */
class GBSWForce::ParticleCphmdInfo {
public:
    int titrateResID;
    double refChargeState1, refChargeState2, chargeState1, chargeState2;
    ParticleCphmdInfo() {
        refChargeState1 = refChargeState2 = chargeState1 = chargeState2 = 0.0;
        titrateResID = 0;
    }
    ParticleCphmdInfo(int titrateResID, 
        double refChargeState1, double refChargeState2, 
        double chargeState1, double chargeState2) :
        titrateResID(titrateResID), 
        refChargeState1(refChargeState1), refChargeState2(refChargeState2),
        chargeState1(chargeState1), chargeState2(chargeState2) {
    }
};

/**
 * This is an internal class used to record nonbonded exception fixes
 * the CHARMM interface uses this to include the 1-4 nonbonded interactions
 * @private
 */
class GBSWForce::NonbondedExceptionFixes {
public:
    int atom1, atom2;
    NonbondedExceptionFixes() {
        atom1 = atom2 = 0;
    }
    NonbondedExceptionFixes( int atom1, int atom2 ) :
        atom1(atom1), atom2(atom2) {
    }
};

} // namespace OpenMMGBSW

#endif /*OPENMM_GBSWFORCEFIELD_H_*/
