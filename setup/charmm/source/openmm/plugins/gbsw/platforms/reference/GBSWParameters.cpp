#include <math.h>
#include <sstream>
#include <string.h>

#include "GBSWParameters.h"

#include "openmm/OpenMMException.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/SimTKOpenMMRealType.h"

using std::vector;
using namespace OpenMM;
using namespace OpenMMGBSW;

/**---------------------------------------------------------------------------------------

    GBSWParameters constructor 

    @param numberOfAtoms       number of atoms
    @param gbswType             GBSW type (Eq. 7 or 8 in paper)

    --------------------------------------------------------------------------------------- */

GBSWParameters::GBSWParameters(int numberOfAtoms, GBSWParameters::GBSWType gbswType) :
                              _numberOfAtoms(numberOfAtoms),
                              _solventDielectric( 80.0 ),
                              _soluteDielectric( 1.0 ),
                              _electricConstant(-0.5*ONE_4PI_EPS0),
                              _probeRadius(0.14),
                              _pi4Asolv(28.3919551),
                              _gbswType(gbswType),
                              _cutoff(false),
                              _periodic(false) {

    _atomicRadii.resize(numberOfAtoms);
    setGBSWTypeParameters(gbswType);
}

/**---------------------------------------------------------------------------------------

    GBSWParameters destructor 

    --------------------------------------------------------------------------------------- */

GBSWParameters::~GBSWParameters() {
}

/**---------------------------------------------------------------------------------------

   Get number of atoms

   @return number of atoms

   --------------------------------------------------------------------------------------- */

int GBSWParameters::getNumberOfAtoms() const {
    return _numberOfAtoms;
}

/**---------------------------------------------------------------------------------------

    Get GBSW type

    @return GBSW type

    --------------------------------------------------------------------------------------- */

GBSWParameters::GBSWType GBSWParameters::getGBSWType() const {
    return _gbswType;
}

/**---------------------------------------------------------------------------------------

    Set GBSW type specific parameters

    @param gbswType GBSW type (GBSWTypeI or GBSWTypeII -- Eq. 7 or 8)

    --------------------------------------------------------------------------------------- */

void GBSWParameters::setGBSWTypeParameters( GBSWParameters::GBSWType gbswType ){
    _maxAtomsInVoxel = 25;
    _maxGradientsPerAtom = 160;
    _deltaR          = 0.15f;
    _switchLen       = 0.03f;
    _PBradiusMod     = 0.9520f;
    _RBuffer         = _switchLen + sqrtf(3.0f) * _deltaR / 2.0f;
    _AA0             = -0.1801f;
    _AA1             = 1.81745f;
    _Rmin            = 0.05f;
    _numGauLegRad    = 24;
    _numLebAng       = 50;
    _gbswType = gbswType;
}

/**---------------------------------------------------------------------------------------

    Get the PB modifier for atom radii before they are used to calc the Born Radii

    @return _PBradiusMod

    --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getPBradiusMod( void ) const {
    return _PBradiusMod;
}


/**---------------------------------------------------------------------------------------

    Get number of gaussian-legendre radii used for integration for Born R calculation

    @return _numGauLegRad

    --------------------------------------------------------------------------------------- */

int GBSWParameters::getNumGauLegRad( void ) const {
    return _numGauLegRad;
}

/**---------------------------------------------------------------------------------------

    Get number of lebedev angles used for integration for Born R calculation
    
    @return _numLebAng

    --------------------------------------------------------------------------------------- */

int GBSWParameters::getNumLebAng( void ) const {
    return _numLebAng;
}

/**---------------------------------------------------------------------------------------

    Get minimum radius of integration parameter for GBSW Born R calculation

    @return _Rmin

    --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getRmin( void ) const {
    return _Rmin;
}

/**---------------------------------------------------------------------------------------

    Get aa0 parameter for GBSW Born R calculation

    @return _AA0

    --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getAA0( void ) const {
    return _AA0;
}

/**---------------------------------------------------------------------------------------

    Get aa1 parameter for GBSW Born R calculation

    @return _AA1

    --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getAA1( void ) const {
    return _AA1;
}

/**---------------------------------------------------------------------------------------

    Get buffer radius around an atom when fitting into lookup table

    @return _RBuffer

    --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getSwitchLen( void ) const {
    return _switchLen;
}

/**---------------------------------------------------------------------------------------

    Get buffer radius around an atom when fitting into lookup table

    @return _RBuffer

    --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getRBuffer( void ) const {
    return _RBuffer;
}

/**---------------------------------------------------------------------------------------

    Get maximum number of atoms resident in a voxel

    @return _deltaR

    --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getDeltaR( void ) const {
    return _deltaR;
}

/**---------------------------------------------------------------------------------------

    Get maximum number of gradient terms allowed per atom

    @return _maxGradientsPerAtom

    --------------------------------------------------------------------------------------- */

int GBSWParameters::getMaxGradientsPerAtom( void ) const {
    return _maxGradientsPerAtom;
}

/**---------------------------------------------------------------------------------------

    Get maximum number of atoms resident in a voxel

    @return _maxAtomsInVoxel

    --------------------------------------------------------------------------------------- */

int GBSWParameters::getMaxAtomsInVoxel( void ) const {
    return _maxAtomsInVoxel;
}

/**---------------------------------------------------------------------------------------

   Get solvent dielectric

   @return solvent dielectric

   --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getSolventDielectric( void ) const {
    return _solventDielectric;
}

/**---------------------------------------------------------------------------------------

   Set solvent dielectric

   @param solventDielectric solvent dielectric

   --------------------------------------------------------------------------------------- */

void GBSWParameters::setSolventDielectric( RealOpenMM solventDielectric ){
    _solventDielectric = solventDielectric;
}
/**---------------------------------------------------------------------------------------

   Get solute dielectric

   @return soluteDielectric

   --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getSoluteDielectric( void ) const {
    return _soluteDielectric;
}

/**---------------------------------------------------------------------------------------

   Set solute dielectric

   @param soluteDielectric solute dielectric

   --------------------------------------------------------------------------------------- */

void GBSWParameters::setSoluteDielectric( RealOpenMM soluteDielectric ){
    _soluteDielectric = soluteDielectric;
}

/**---------------------------------------------------------------------------------------

   Get electric constant 

   @return electricConstant

   --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getElectricConstant( void ) const {
    return _electricConstant;
}

/**---------------------------------------------------------------------------------------

   Get probe radius 

   @return probeRadius

   --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getProbeRadius( void ) const {
    return _probeRadius;
}

/**---------------------------------------------------------------------------------------

   Set probe radius  

   @param probeRadius   probe radius

   --------------------------------------------------------------------------------------- */

void GBSWParameters::setProbeRadius( RealOpenMM probeRadius ){
    _probeRadius = probeRadius;
}

/**---------------------------------------------------------------------------------------

   Get pi*4*Asolv:  used in ACE approximation for nonpolar term  
         ((RealOpenMM) M_PI)*4.0f*0.0049*1000.0; (Still) 
         ((RealOpenMM) M_PI)*4.0f*0.0054*1000.0; (GBSW) 

   @return pi4Asolv

   --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getPi4Asolv( void ) const {
    return _pi4Asolv;
}


/**---------------------------------------------------------------------------------------

    Get AtomicRadii array

    @return array of atomic radii

    --------------------------------------------------------------------------------------- */

const std::vector<RealOpenMM>& GBSWParameters::getAtomicRadii( void ) const {
    return _atomicRadii;
}

/**---------------------------------------------------------------------------------------

    Set AtomicRadii array

    @param atomicRadii vector of atomic radii

    --------------------------------------------------------------------------------------- */

void GBSWParameters::setAtomicRadii( const std::vector<RealOpenMM>& atomicRadii ){

    if( atomicRadii.size() == _atomicRadii.size() ){
        for( unsigned int ii = 0; ii < atomicRadii.size(); ii++ ){
            _atomicRadii[ii] = atomicRadii[ii];
        }   
    } else {
        std::stringstream msg;
        msg << "GBSWParameters: input size for atomic radii does not agree w/ current size: input=";
        msg << atomicRadii.size();
        msg << " current size=" << _atomicRadii.size();
        throw OpenMM::OpenMMException(msg.str());
    }   

}

/**---------------------------------------------------------------------------------------

      Set the force to use a cutoff.

      @param distance            the cutoff distance

      --------------------------------------------------------------------------------------- */

void GBSWParameters::setUseCutoff( RealOpenMM distance ) {

     _cutoff         = true;
     _cutoffDistance = distance;
}

/**---------------------------------------------------------------------------------------

      Get whether to use a cutoff.

      --------------------------------------------------------------------------------------- */

bool GBSWParameters::getUseCutoff() const {
     return _cutoff;
}

/**---------------------------------------------------------------------------------------

      Get the cutoff distance.

      --------------------------------------------------------------------------------------- */

RealOpenMM GBSWParameters::getCutoffDistance() const {
     return _cutoffDistance;
}

/**---------------------------------------------------------------------------------------

      Set the force to use periodic boundary conditions.  This requires that a cutoff has
      also been set, and the smallest side of the periodic box is at least twice the cutoff
      distance.

      @param boxSize             the X, Y, and Z widths of the periodic box

      --------------------------------------------------------------------------------------- */


void GBSWParameters::setPeriodic(const OpenMM::RealVec& boxSize) {

     assert(_cutoff);

     assert(boxSize[0] >= 2.0*_cutoffDistance);
     assert(boxSize[1] >= 2.0*_cutoffDistance);
     assert(boxSize[2] >= 2.0*_cutoffDistance);

     _periodic           = true;
     _periodicBoxSize[0] = boxSize[0];
     _periodicBoxSize[1] = boxSize[1];
     _periodicBoxSize[2] = boxSize[2];
}

/**---------------------------------------------------------------------------------------

      Get whether to use periodic boundary conditions.

      --------------------------------------------------------------------------------------- */

bool GBSWParameters::getPeriodic() {
     return _periodic;
}

/**---------------------------------------------------------------------------------------

      Get the periodic box dimension

      --------------------------------------------------------------------------------------- */

const RealOpenMM* GBSWParameters::getPeriodicBox() {
     return _periodicBoxSize;
}
