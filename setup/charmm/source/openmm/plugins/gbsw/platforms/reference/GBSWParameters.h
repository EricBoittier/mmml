
#ifndef __GBSWParameters_H__
#define __GBSWParameters_H__

#include "openmm/reference/RealVec.h"
#include "openmm/reference/SimTKOpenMMRealType.h"

#include <vector>

namespace OpenMMGBSW {

class GBSWParameters {

   public:

       // GBSW types

       enum GBSWType { GBSWTypeI, GBSWTypeII };

   private:

      // GBSW constants & parameters
   
      int _numberOfAtoms;
      int _maxAtomsInVoxel;
      int _maxGradientsPerAtom;
      int _numGauLegRad;
      int _numLebAng;
      
      RealOpenMM _AA0;
      RealOpenMM _AA1;
      RealOpenMM _Rmin;
      RealOpenMM _switchLen;
      RealOpenMM _PBradiusMod;
      RealOpenMM _RBuffer;
      RealOpenMM _deltaR;
      RealOpenMM _solventDielectric;
      RealOpenMM _soluteDielectric;
      RealOpenMM _electricConstant;
      RealOpenMM _probeRadius;
      RealOpenMM _pi4Asolv;
      GBSWType _gbswType;

      // scaled radius factors (S_kk in HCT paper)

      std::vector<RealOpenMM> _atomicRadii;

      // cutoff and periodic boundary conditions
      
      bool _cutoff;
      bool _periodic;
      RealOpenMM _periodicBoxSize[3];
      RealOpenMM _cutoffDistance;

      /**---------------------------------------------------------------------------------------
      
         Set solvent dielectric 
      
         @param dielectricOffset         solvent dielectric

         --------------------------------------------------------------------------------------- */
      
      void setDielectricOffset(RealOpenMM dielectricOffset);

   public:

      /**---------------------------------------------------------------------------------------
      
         GBSWParameters constructor 
      
         @param numberOfAtoms       number of atoms
      
         --------------------------------------------------------------------------------------- */
      
       GBSWParameters( int numberOfAtoms, GBSWParameters::GBSWType gbswType = GBSWTypeII );

      /**---------------------------------------------------------------------------------------
      
         GBSWParameters destructor 
      
         --------------------------------------------------------------------------------------- */
      
       ~GBSWParameters( );

      /**---------------------------------------------------------------------------------------
      
         Get number of atoms
      
         @return number of atoms
      
         --------------------------------------------------------------------------------------- */

      int getNumberOfAtoms( void ) const;

      /**---------------------------------------------------------------------------------------
      
         Get electric constant
      
         @return electric constant
      
         --------------------------------------------------------------------------------------- */

      RealOpenMM getElectricConstant( void ) const;

      /**---------------------------------------------------------------------------------------
      
         Get probe radius (Simbios) 
      
         @return probeRadius
      
         --------------------------------------------------------------------------------------- */

      RealOpenMM getProbeRadius( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Set probe radius (Simbios) 
      
         @param probeRadius   probe radius
      
         --------------------------------------------------------------------------------------- */

      void setProbeRadius(RealOpenMM probeRadius);

      /**---------------------------------------------------------------------------------------
      
         Get pi4Asolv:  used in ACE approximation for nonpolar term  
            ((RealOpenMM) M_PI)*4.0f*0.0049f*1000.0f; (Simbios) 
      
         @return pi4Asolv
      
         --------------------------------------------------------------------------------------- */

      RealOpenMM getPi4Asolv() const;

      /**---------------------------------------------------------------------------------------
      
         Set pi4Asolv
      
         --------------------------------------------------------------------------------------- */

      void setPi4Asolv(RealOpenMM pi4Asolv);

      /**---------------------------------------------------------------------------------------
      
         Get solvent dielectric
      
         @return solvent dielectric
      
         --------------------------------------------------------------------------------------- */

      RealOpenMM getSolventDielectric() const;

      /**---------------------------------------------------------------------------------------
      
         Set solvent dielectric
      
         @param solventDielectric solvent dielectric
      
         --------------------------------------------------------------------------------------- */

      void setSolventDielectric(RealOpenMM solventDielectric);

      /**---------------------------------------------------------------------------------------
      
         Get solute dielectric
      
         @return soluteDielectric
      
         --------------------------------------------------------------------------------------- */

      RealOpenMM getSoluteDielectric() const;

      /**---------------------------------------------------------------------------------------
      
         Set solute dielectric
      
         @param soluteDielectric solute dielectric
      
         --------------------------------------------------------------------------------------- */

      void setSoluteDielectric(RealOpenMM soluteDielectric);

      /**---------------------------------------------------------------------------------------
      
         Get GBSW type
      
         @return GBSW type
      
         --------------------------------------------------------------------------------------- */
      
      GBSWParameters::GBSWType getGBSWType() const;
      
      /**---------------------------------------------------------------------------------------
      
         Set GBSW type specific parameters
      
         @param gbswType GBSW type (GBSWTypeI or GBSWTypeII -- Eq. 7 or 8)
      
         --------------------------------------------------------------------------------------- */
      
      void setGBSWTypeParameters(GBSWParameters::GBSWType gbswType);
      
      /**---------------------------------------------------------------------------------------
      
         Get the PB modifier for atom radii before they are used to calc the Born Radii
      
         @return PB radius modifier
      
         --------------------------------------------------------------------------------------- */
      
      RealOpenMM getPBradiusMod( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Get number of gaussian-legendre radii used for integration for Born R calculation
      
         @return number of radii
      
         --------------------------------------------------------------------------------------- */
      
      int getNumGauLegRad( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Get number of lebadev angles used for integration for Born R calculation
      
         @return number of angles
      
         --------------------------------------------------------------------------------------- */
      
      int getNumLebAng( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Get minimum radius of integration parameter for GBSW Born R calculation
      
         @return minimum radius
      
         --------------------------------------------------------------------------------------- */
      
      RealOpenMM getRmin( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Get aa0 parameter for GBSW Born R calculation
      
         @return AA0 parameter
      
         --------------------------------------------------------------------------------------- */
      
      RealOpenMM getAA0( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Get aa1 parameter for GBSW Born R calculation
      
         @return AA1 parameter
      
         --------------------------------------------------------------------------------------- */
      
      RealOpenMM getAA1( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Get switching function length for GBSW calculation
      
         @return switching function length
      
         --------------------------------------------------------------------------------------- */
      
      RealOpenMM getSwitchLen( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Get buffer radius around an atom when fitting into lookup table
      
         @return buffer radius
      
         --------------------------------------------------------------------------------------- */
      
      RealOpenMM getRBuffer( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Get voxel width of the lookup grid
      
         @return voxel width of the lookup grid
      
         --------------------------------------------------------------------------------------- */
      
      RealOpenMM getDeltaR( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Get maximum number of gradient terms allowed per atom

         @return maximum number of gradient terms from an atom
      
         --------------------------------------------------------------------------------------- */
      
      int getMaxGradientsPerAtom( void ) const;
      
      /**---------------------------------------------------------------------------------------
      
         Get maximum number of atoms resident in a voxel

         @return maximum number of atoms lookup stores in a voxel
      
         --------------------------------------------------------------------------------------- */
      
      int getMaxAtomsInVoxel( void ) const;
        
      /**---------------------------------------------------------------------------------------
      
         Get AtomicRadii array w/ dielectric offset applied
      
         @return array of atom volumes
      
         --------------------------------------------------------------------------------------- */

      const std::vector<RealOpenMM>& getAtomicRadii( void ) const;

      /**---------------------------------------------------------------------------------------
      
         Set AtomicRadii array
      
         @param atomicRadii vector of atomic radii
      
         --------------------------------------------------------------------------------------- */

      void setAtomicRadii( const std::vector<RealOpenMM>& atomicRadii );


      /**---------------------------------------------------------------------------------------

         Set the force to use a cutoff.

         @param distance            the cutoff distance

         --------------------------------------------------------------------------------------- */

      void setUseCutoff( RealOpenMM distance );

      /**---------------------------------------------------------------------------------------

         Get whether to use a cutoff.

         --------------------------------------------------------------------------------------- */

      bool getUseCutoff() const;

      /**---------------------------------------------------------------------------------------

         Get the cutoff distance.

         --------------------------------------------------------------------------------------- */

      RealOpenMM getCutoffDistance() const;

      /**---------------------------------------------------------------------------------------

         Set the force to use periodic boundary conditions.  This requires that a cutoff has
         already been set, and the smallest side of the periodic box is at least twice the cutoff
         distance.

         @param boxSize             the X, Y, and Z widths of the periodic box

         --------------------------------------------------------------------------------------- */

      void setPeriodic(const OpenMM::RealVec& boxSize);

      /**---------------------------------------------------------------------------------------

         Get whether to use periodic boundary conditions.

         --------------------------------------------------------------------------------------- */

      bool getPeriodic();

      /**---------------------------------------------------------------------------------------

         Get the periodic box vectors

         --------------------------------------------------------------------------------------- */

      const RealOpenMM* getPeriodicBox();

};

} // namespace OpenMMGBSW

#endif // __GBSWParameters_H__
