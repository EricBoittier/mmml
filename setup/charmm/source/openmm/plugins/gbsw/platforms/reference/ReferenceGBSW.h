#ifndef __ReferenceGBSW_H__
#define __ReferenceGBSW_H__

#include "GBSWParameters.h"

namespace OpenMMGBSW {

struct localForce {
    int atomNum;
    RealOpenMM dx, dy, dz;
};

class ReferenceGBSW {

   private:

      // quadratures used for volumetric integration
      
      std::vector<RealOpenMM> _QuadR;
      std::vector<RealOpenMM> _QuadWrad;
      std::vector<RealOpenMM> _QuadX;
      std::vector<RealOpenMM> _QuadY;
      std::vector<RealOpenMM> _QuadZ;
      std::vector<RealOpenMM> _QuadWang;
      
      
      // GBSW parameters

      GBSWParameters* _gbswParameters;
      
      // atom lookup table for volumetric integration
      
      std::vector< std::vector<int> > _lookupTable;
      

      // arrays containing GBSW chain derivative 
      
      std::vector<int> _nForceIncrimentAtoms;
      std::vector< std::vector<localForce> > _gbswChain;

      // flag to signal whether ACE approximation
      // is to be included

      int _includeAceApproximation;


   public:

      /**---------------------------------------------------------------------------------------
      
         Constructor
      
         @param implicitSolventParameters    ImplicitSolventParameters reference
      
         @return CpuImplicitSolvent object
      
         --------------------------------------------------------------------------------------- */

       ReferenceGBSW( GBSWParameters* gbswParameters );

      /**---------------------------------------------------------------------------------------
      
         Destructor
      
         --------------------------------------------------------------------------------------- */

       ~ReferenceGBSW( );

      /**---------------------------------------------------------------------------------------
      
         Return GBSWParameters
      
         @return GBSWParameters
      
         --------------------------------------------------------------------------------------- */

      GBSWParameters* getGBSWParameters( void ) const;

      /**---------------------------------------------------------------------------------------
      
         Set ImplicitSolventParameters
      
         @param ImplicitSolventParameters
      
         --------------------------------------------------------------------------------------- */

      void setGBSWParameters( GBSWParameters* gbswParameters );
 
      /**---------------------------------------------------------------------------------------
      
         Return flag signalling whether AceApproximation for nonpolar term is to be included
      
         @return flag
      
         --------------------------------------------------------------------------------------- */

      int includeAceApproximation( void ) const;

      /**---------------------------------------------------------------------------------------
      
         Set flag indicating whether AceApproximation is to be included
      
         @param includeAceApproximation new includeAceApproximation value
      
         --------------------------------------------------------------------------------------- */

      void setIncludeAceApproximation( int includeAceApproximation );
      
      /**---------------------------------------------------------------------------------------
      
         Return the Gaussuan-Legendre quadrature radii and weights
      
         @return array _QuadR
         @return array _QuadWrad
      
         --------------------------------------------------------------------------------------- */
      
      std::vector<RealOpenMM>& getQuadR( int numRadii );
      std::vector<RealOpenMM>& getQuadWrad( int numRadii );
      
      /**---------------------------------------------------------------------------------------
      
         Return the Lebadev quadrature angles (X, Y, Z coordinate terms and the angular weights)
      
         @return array _QuadX
         @return array _QuadY
         @return array _QuadZ
         @return array _QuadWang
      
         --------------------------------------------------------------------------------------- */
      
      std::vector<RealOpenMM>& getQuadX( int numAngles );
      std::vector<RealOpenMM>& getQuadY( int numAngles );
      std::vector<RealOpenMM>& getQuadZ( int numAngles );
      std::vector<RealOpenMM>& getQuadWang( int numAngles );
      
      /**---------------------------------------------------------------------------------------
      
         Return GBSW atom lookup table for volumetric integration
      
         @return 2D jagged array
      
         --------------------------------------------------------------------------------------- */
      
      std::vector< std::vector<int> >& getLookupTable( void );
      
      /**---------------------------------------------------------------------------------------
      
         Return number of gradient points in chain derivative per atom
      
         @return array
      
         --------------------------------------------------------------------------------------- */
      
      std::vector<int>& getnForceIncrimentAtoms( void );
      
      /**---------------------------------------------------------------------------------------
      
         Return GBSW chain derivative: size = _implicitSolventParameters->getNumberOfAtoms()
      
         @return array
      
         --------------------------------------------------------------------------------------- */
      
      std::vector< std::vector<localForce> >& getGBSWChain( void );
      
      /**---------------------------------------------------------------------------------------
      
         Get lookup table based on GBSW 

         @param atomCoordinates   atomic coordinates
         @param lookupTable       atom lookup table, IDs where in space atoms are
         @param bornRadii         output array of Born radii
      
         --------------------------------------------------------------------------------------- */

      void computeLookupTable( const std::vector<OpenMM::RealVec>& atomCoordinates, std::vector< std::vector<int> >& lookupTable );
      
      /**---------------------------------------------------------------------------------------
      
         Get Born radii based on GBSW 

         @param atomCoordinates   atomic coordinates
         @param lookupTable       atom lookup table, IDs where in space atoms are
         @param bornRadii         output array of Born radii
      
         --------------------------------------------------------------------------------------- */
      
      void computeBornRadii( const std::vector<OpenMM::RealVec>& atomCoordinates, 
                             const std::vector< std::vector<int> >& lookupTable, 
                             std::vector<RealOpenMM>& bornRadii, 
                             std::vector<int>& nForceIncrimentAtoms,
                             std::vector< std::vector<localForce> >& gbswChain );
      
      /**---------------------------------------------------------------------------------------
        
         Get nonpolar solvation force constribution via ACE approximation
        
         @param gbswParameters parameters
         @param vdwRadii                  Vdw radii
         @param bornRadii                 Born radii
         @param energy                    energy (output): value is incremented from input value 
         @param forces                    forces: values are incremented from input values
        
            --------------------------------------------------------------------------------------- */
        
      void computeAceNonPolarForce( const GBSWParameters* gbswParameters, 
                                    const std::vector<RealOpenMM>& bornRadii, 
                                    RealOpenMM* energy, std::vector<RealOpenMM>& forces ) const;
        
      /**---------------------------------------------------------------------------------------
      
         Get Born energy and forces based on GBSW 
      
         @param atomCoordinates   atomic coordinates
         @param partialCharges    partial charges
         @param forces            forces
      
         --------------------------------------------------------------------------------------- */
      
      RealOpenMM computeBornEnergyForces( const std::vector<OpenMM::RealVec>& atomCoordinates,
                                          const std::vector<RealOpenMM>& partialCharges, 
                                          std::vector<OpenMM::RealVec>& forces );
      
    /**---------------------------------------------------------------------------------------
    
        Print GBSW parameters, radii, forces, ...
    
        @param atomCoordinates     atomic coordinates
        @param partialCharges      partial charges
        @param bornRadii           Born radii (may be empty)
        @param bornForces          Born forces (may be empty)
        @param forces              forces (may be empty)
        @param idString            id string (who is calling)
        @param log                 log file
    
        --------------------------------------------------------------------------------------- */
    
    void printGBSW( const std::vector<OpenMM::RealVec>& atomCoordinates,
                   const std::vector<RealOpenMM>& partialCharges,
                   const std::vector<RealOpenMM>& bornRadii,
                   const std::vector<RealOpenMM>& bornForces,
                   const std::vector<OpenMM::RealVec>& forces,
                   const std::string& idString, FILE* log );
    
};

// ---------------------------------------------------------------------------------------

} // namespace OpenMM

#endif // __ReferenceGBSW_H__
