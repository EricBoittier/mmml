#include "ReferenceGBSW.h"

#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferenceForce.h"

#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <cmath>
#include <cstdio>

using namespace OpenMM;
using namespace OpenMMGBSW;
using namespace std;

/**---------------------------------------------------------------------------------------

    ReferenceGBSW constructor

    gbswParameters      gbswParameters object
    
    --------------------------------------------------------------------------------------- */

ReferenceGBSW::ReferenceGBSW( GBSWParameters* gbswParameters ) : 
    _gbswParameters(gbswParameters), _includeAceApproximation(1) {
    _nForceIncrimentAtoms.resize(_gbswParameters->getNumberOfAtoms());
    _gbswChain.resize(_gbswParameters->getNumberOfAtoms());
    for ( int i = 0; i < _gbswParameters->getNumberOfAtoms(); i++ )
        _gbswChain[i].resize(_gbswParameters->getMaxGradientsPerAtom());
}

/**---------------------------------------------------------------------------------------

    ReferenceGBSW destructor

    --------------------------------------------------------------------------------------- */

ReferenceGBSW::~ReferenceGBSW( ){
}

/**---------------------------------------------------------------------------------------

    Get GBSWParameters reference

    @return GBSWParameters reference

    --------------------------------------------------------------------------------------- */

GBSWParameters* ReferenceGBSW::getGBSWParameters() const {
    return _gbswParameters;
}


/**---------------------------------------------------------------------------------------

    Set GBSWParameters reference

    @param GBSWParameters reference

    --------------------------------------------------------------------------------------- */

void ReferenceGBSW::setGBSWParameters(  GBSWParameters* gbswParameters ){
    _gbswParameters = gbswParameters;
}

/**---------------------------------------------------------------------------------------

   Return flag signalling whether AceApproximation for nonpolar term is to be included

   @return flag

   --------------------------------------------------------------------------------------- */

int ReferenceGBSW::includeAceApproximation( void ) const {
    return _includeAceApproximation;
}

/**---------------------------------------------------------------------------------------

   Set flag indicating whether AceApproximation is to be included

   @param includeAceApproximation new includeAceApproximation value

   --------------------------------------------------------------------------------------- */

void ReferenceGBSW::setIncludeAceApproximation( int includeAceApproximation ){
    _includeAceApproximation = includeAceApproximation;
}

/**---------------------------------------------------------------------------------------
    
    GAUSSIAN-LEGENDRE QUADRATURE:  24-POINT RADIAL GRID
    
    The following 2 functions produce the radii and their corresponding
    weights to appropriately scale the unit sphere integration of the Levadev
    quadrature. This enables volumetric integration. The radii and weights
    are defined by two second-order gaussian-legendre polynomials[1] covering
    radii from 0.05 nm to 0.1 nm, and 0.1 nm to 2.0 nm. The result yields
    an enhanced sampling at lower radii.
    
    [1] M. Abramowitz, and I.A. Stegun
        "Handbook of Mathematical Functions with Formulas, Graphs, and 
         Mathematical Tables"
        New York: Dover, (1965), "Chapter 25.4, Integration"
    
    @return array _QuadR
    @return array _QuadWrad

    --------------------------------------------------------------------------------------- */

std::vector<RealOpenMM>& ReferenceGBSW::getQuadR( int numRadii ){
    
    // currently only the 24-radii double-volume quadrature is supported.
    if ( numRadii != 24 ) {
        printf("\n ERROR!!  ONLY 24-RADII QUADRATURE SUPPORTED!!\n\n");
        return _QuadR;
    }
    
    // simply load up the data into the vector and return it
    const RealOpenMM tmp [] = {
        0.0523455038515334, 0.0615382672473579,  0.075, 
        0.0884617327526421, 0.0976544961484666,  0.107213498348595,  
        0.137802255471911,  0.192001891565923,   0.268421076289714,  
        0.365082131531532,  0.479481960571403,   0.608657795692837,  
        0.749264105034552,  0.897659286641786,   1.05,  
        1.20234071335821,   1.35073589496545,    1.49134220430716,   
        1.6205180394286,    1.73491786846847,    1.83157892371029,  
        1.90799810843408,   1.96219774452809,    1.99278650165141 };
    
    _QuadR.resize(numRadii);
    for ( int i = 0; i < numRadii; i++ ) _QuadR[i] = tmp[i];
    
    return _QuadR;
}

std::vector<RealOpenMM>& ReferenceGBSW::getQuadWrad( int numRadii ){
    
    // currently only the 24-radii double-volume quadrature is supported.
    if ( numRadii != 24 ) {
        printf("\n ERROR!!  ONLY 24-RADII QUADRATURE SUPPORTED!!\n\n");
        return _QuadR;
    }
    
    // simply load up the data into the vector and return it
    const RealOpenMM tmp [] = {
        0.00592317212640454, 0.0119657167624842,  0.0142222222222222, 
        0.0119657167624842,  0.00592317212640454, 0.0184886988182386,
        0.0425735154274146,  0.0655923156007592,  0.0869155205413276,
        0.105988313269961,   0.122316264412369,   0.135476367064926,
        0.145127739962567,   0.151020401224257,   0.153001727356345,
        0.151020401224257,   0.145127739962567,   0.135476367064926,
        0.122316264412369,   0.105988313269961,   0.0869155205413276,
        0.0655923156007592,  0.0425735154274146,  0.0184886988182386 };
    
    _QuadWrad.resize(numRadii);
    for ( int i = 0; i < numRadii; i++ ) _QuadWrad[i] = tmp[i];
    
    return _QuadWrad;
}

/**---------------------------------------------------------------------------------------

    LEBEDEV QUADRATURE:  50-POINT ANGULAR GRID
    
    The following 4 functions produce the XYZ coordinates and their 
    corresponding weights in accordance with polar Lebedev grids [1-6] for 
    integration on a unit sphere. The 50 points come from more coarse 
    quadratures of 6, 8, 12, and 24 points, each of those being defined by 
    the maxima of spherical harmonic fomulae. 
    
    Users of this code are asked to include reference [1] in their
    publications, and in the user- and programmers-manuals
    describing their codes.
    
    [1] V.I. Lebedev, and D.N. Laikov
        "A quadrature formula for the sphere of the 131st
         algebraic order of accuracy"
        Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
    
    [2] V.I. Lebedev
        "A quadrature formula for the sphere of 59th algebraic
         order of accuracy"
        Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286.
    
    [3] V.I. Lebedev, and A.L. Skorokhodov
        "Quadrature formulas of orders 41, 47, and 53 for the sphere"
        Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592.
    
    [4] V.I. Lebedev
        "Spherical quadrature formulas exact to orders 25-29"
        Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
    
    [5] V.I. Lebedev
        "Quadratures on a sphere"
        Computational Mathematics and Mathematical Physics, Vol. 16,
        1976, pp. 10-24.
    
    [6] V.I. Lebedev
        "Values of the nodes and weights of ninth to seventeenth
         order Gauss-Markov quadrature formulae invariant under the
         octahedron group with inversion"
        Computational Mathematics and Mathematical Physics, Vol. 15,
        1975, pp. 44-51.

    @return array _QuadX
    @return array _QuadY
    @return array _QuadZ
    @return array _QuadWang

    --------------------------------------------------------------------------------------- */

std::vector<RealOpenMM>& ReferenceGBSW::getQuadX( int numAngles ){
    
    // currently only the 50 angle quadrature is supported.
    if ( numAngles != 50 ) {
        printf("\n ERROR!!  ONLY 50-ANGLE QUADRATURE SUPPORTED!!\n\n");
        return _QuadX;
    }
    
    // simply load up the data into the vector and return it
    const RealOpenMM tmp [] = {
        1.0,               -1.0,                0.0,
        0.0,                0.0,                0.0,
        0.0,                0.0,                0.0,
        0.0,                0.707106781186548, -0.707106781186548,  
        0.707106781186548, -0.707106781186548,  0.707106781186548,
       -0.707106781186548,  0.707106781186548, -0.707106781186548,
        0.577350269189626, -0.577350269189626,  0.577350269189626,
       -0.577350269189626,  0.577350269189626, -0.577350269189626,
        0.577350269189626, -0.577350269189626,  0.301511344577764,
       -0.301511344577764,  0.301511344577764, -0.301511344577764,
        0.301511344577764, -0.301511344577764,  0.301511344577764,
       -0.301511344577764,  0.301511344577764, -0.301511344577764,
        0.301511344577764, -0.301511344577764,  0.301511344577764,
       -0.301511344577764,  0.301511344577764, -0.301511344577764,
        0.904534033733291, -0.904534033733291,  0.904534033733291,
       -0.904534033733291,  0.904534033733291, -0.904534033733291,
        0.904534033733291, -0.904534033733291 };
    
    _QuadX.resize(numAngles);
    for ( int i = 0; i < numAngles; i++ ) _QuadX[i] = tmp[i];
    
    return _QuadX;
}

std::vector<RealOpenMM>& ReferenceGBSW::getQuadY( int numAngles ){
    
    // currently only the 50 angle quadrature is supported.
    if ( numAngles != 50 ) {
        printf("\n ERROR!!  ONLY 50-ANGLE QUADRATURE SUPPORTED!!\n\n");
        return _QuadY;
    }
    
    // simply load up the data into the vector and return it
    const RealOpenMM tmp [] = {
        0.0,                0.0,                1.0, 
       -1.0,                0.0,                0.0,
        0.707106781186548, -0.707106781186548,  0.707106781186548,
       -0.707106781186548,  0.0,                0.0, 
        0.0,                0.0,                0.707106781186548, 
        0.707106781186548, -0.707106781186548, -0.707106781186548, 
        0.577350269189626,  0.577350269189626, -0.577350269189626, 
       -0.577350269189626,  0.577350269189626,  0.577350269189626,
       -0.577350269189626, -0.577350269189626,  0.301511344577764, 
        0.301511344577764, -0.301511344577764, -0.301511344577764, 
        0.301511344577764,  0.301511344577764, -0.301511344577764,
       -0.301511344577764,  0.904534033733291,  0.904534033733291,
       -0.904534033733291, -0.904534033733291,  0.904534033733291,
        0.904534033733291, -0.904534033733291, -0.904534033733291,
        0.301511344577764,  0.301511344577764, -0.301511344577764, 
       -0.301511344577764,  0.301511344577764,  0.301511344577764, 
       -0.301511344577764, -0.301511344577764 };
    
    _QuadY.resize(numAngles);
    for ( int i = 0; i < numAngles; i++ ) _QuadY[i] = tmp[i];
    
    return _QuadY;
}

std::vector<RealOpenMM>& ReferenceGBSW::getQuadZ( int numAngles ){
    
    // currently only the 50 angle quadrature is supported.
    if ( numAngles != 50 ) {
        printf("\n ERROR!!  ONLY 50-ANGLE QUADRATURE SUPPORTED!!\n\n");
        return _QuadZ;
    }
    
    // simply load up the data into the vector and return it
    const RealOpenMM tmp [] = {
        0.0,                0.0,                0.0,
        0.0,                1.0,               -1.0,
        0.707106781186548,  0.707106781186548, -0.707106781186548,
       -0.707106781186548,  0.707106781186548,  0.707106781186548,
       -0.707106781186548, -0.707106781186548,  0.0,
        0.0,                0.0,                0.0,
        0.577350269189626,  0.577350269189626,  0.577350269189626, 
        0.577350269189626, -0.577350269189626, -0.577350269189626, 
       -0.577350269189626, -0.577350269189626,  0.904534033733291, 
        0.904534033733291,  0.904534033733291,  0.904534033733291,
       -0.904534033733291, -0.904534033733291, -0.904534033733291,
       -0.904534033733291,  0.301511344577764,  0.301511344577764,
        0.301511344577764,  0.301511344577764, -0.301511344577764,
       -0.301511344577764, -0.301511344577764, -0.301511344577764,
        0.301511344577764,  0.301511344577764,  0.301511344577764,
        0.301511344577764, -0.301511344577764, -0.301511344577764,
       -0.301511344577764, -0.301511344577764 };
    
    _QuadZ.resize(numAngles);
    for ( int i = 0; i < numAngles; i++ ) _QuadZ[i] = tmp[i];
    
    return _QuadZ;
}

std::vector<RealOpenMM>& ReferenceGBSW::getQuadWang( int numAngles ){
    
    // currently only the 50 angle quadrature is supported.
    if ( numAngles != 50 ) {
        printf("\n ERROR!!  ONLY 50-ANGLE QUADRATURE SUPPORTED!!\n\n");
        return _QuadWang;
    }
    
    // simply load up the data into the vector and return it
    const RealOpenMM tmp [] = {
        0.0126984126984127, 0.0126984126984127, 0.0126984126984127,
        0.0126984126984127, 0.0126984126984127, 0.0126984126984127,
        0.0225749559082892, 0.0225749559082892, 0.0225749559082892,
        0.0225749559082892, 0.0225749559082892, 0.0225749559082892,
        0.0225749559082892, 0.0225749559082892, 0.0225749559082892,
        0.0225749559082892, 0.0225749559082892, 0.0225749559082892,
        0.02109375,         0.02109375,         0.02109375, 
        0.02109375,         0.02109375,         0.02109375,  
        0.02109375,         0.02109375,         0.0201733355379189,  
        0.0201733355379189, 0.0201733355379189, 0.0201733355379189, 
        0.0201733355379189, 0.0201733355379189, 0.0201733355379189,
        0.0201733355379189, 0.0201733355379189, 0.0201733355379189, 
        0.0201733355379189, 0.0201733355379189, 0.0201733355379189, 
        0.0201733355379189, 0.0201733355379189, 0.0201733355379189, 
        0.0201733355379189, 0.0201733355379189, 0.0201733355379189, 
        0.0201733355379189, 0.0201733355379189, 0.0201733355379189,
        0.0201733355379189, 0.0201733355379189 };
    
    _QuadWang.resize(numAngles);
    for ( int i = 0; i < numAngles; i++ ) _QuadWang[i] = tmp[i];
    
    return _QuadWang;
}

/**---------------------------------------------------------------------------------------

    Return GBSW atom lookup table for volumetric integration

    @return 2D jagged array

    --------------------------------------------------------------------------------------- */

std::vector< std::vector<int> >& ReferenceGBSW::getLookupTable( void ){
    return _lookupTable;
}

/**---------------------------------------------------------------------------------------

    Return number of gradient points in chain derivative: size = _gbswParameters->getNumberOfAtoms()

    @return array

    --------------------------------------------------------------------------------------- */

std::vector<int>& ReferenceGBSW::getnForceIncrimentAtoms( void ){
    return _nForceIncrimentAtoms;
}

/**---------------------------------------------------------------------------------------

    Return GBSW chain derivative: size = _gbswParameters->getNumberOfAtoms()

    @return array

    --------------------------------------------------------------------------------------- */

std::vector< std::vector<localForce> >& ReferenceGBSW::getGBSWChain( void ){
    return _gbswChain;
}

/**---------------------------------------------------------------------------------------

    Get Born radii based on paper:
    
       Journal of Computational Chemistry, 24, 14:1691-1702 (2003) (GBSW paper)

    @param atomCoordinates     atomic coordinates
    @param bornRadii           output array of Born radii

    --------------------------------------------------------------------------------------- */


void ReferenceGBSW::computeLookupTable( const std::vector<RealVec>& atomCoordinates, 
    std::vector< std::vector<int> >& lookupTable ){
    
    // ---------------------------------------------------------------------------------------
    
    static const RealOpenMM zero         = static_cast<RealOpenMM>( 0.0 );
    static const RealOpenMM half         = static_cast<RealOpenMM>( 0.5 );
    static const RealOpenMM one          = static_cast<RealOpenMM>( 1.0 );
    static const RealOpenMM minusOne     = static_cast<RealOpenMM>( -1.0 );
    static const RealOpenMM two          = static_cast<RealOpenMM>( 2.0 );
    static const RealOpenMM bigNegative  = static_cast<RealOpenMM>( -99999999999.0 );
    static const RealOpenMM bigPositive  = static_cast<RealOpenMM>( 99999999999.0 );
    
    // ---------------------------------------------------------------------------------------
    // get useful parameters for the calculation
    
    GBSWParameters* gbswParameters              = getGBSWParameters();

    int numberOfAtoms                           = gbswParameters->getNumberOfAtoms();
    const std::vector<RealOpenMM>& atomicRadii  = gbswParameters->getAtomicRadii();
    const RealOpenMM* periodicBox          = gbswParameters->getPeriodicBox();
    
    int maxAtomsInVoxel                         = gbswParameters->getMaxAtomsInVoxel(); // 25 atoms
    RealOpenMM swLen                            = gbswParameters->getSwitchLen(); // smoothing fn width
    RealOpenMM PBradiusMod                      = gbswParameters->getPBradiusMod(); // PB radius modifier
    RealOpenMM deltaR                           = gbswParameters->getDeltaR(); // width of lookup voxel
    RealOpenMM RBuffer                          = gbswParameters->getRBuffer(); // diagonal length of lookup voxel
    
    // ---------------------------------------------------------------------------------------
    // calculate system values and extrema
    
    int gridDim[3], numGridLoc, atomI, gridLoc;
    RealOpenMM maxX, minX, maxY, minY, maxZ, minZ, XI, YI, ZI, radiusI;
    RealVec transVec;
    
    if (_gbswParameters->getPeriodic()) {
        
        // save grid dimensions and translation vector to fit atoms onto grid
        gridDim[0] = ceil( periodicBox[0] / deltaR );
        gridDim[1] = ceil( periodicBox[1] / deltaR );
        gridDim[2] = ceil( periodicBox[2] / deltaR );
        
        transVec[0] = atomCoordinates[0][0] * minusOne;
        transVec[1] = atomCoordinates[0][1] * minusOne;
        transVec[2] = atomCoordinates[0][2] * minusOne;
        
    } else {
        
        maxX = bigNegative; maxY = bigNegative; maxZ = bigNegative;
        minX = bigPositive; minY = bigPositive; minZ = bigPositive;
        
        // find extrema
        for ( atomI = 0; atomI < numberOfAtoms; atomI++ ) {
            
            XI = atomCoordinates[atomI][0];
            YI = atomCoordinates[atomI][1];
            ZI = atomCoordinates[atomI][2];
            radiusI = atomicRadii[atomI];
            
            // modulate radius to be consistent with switching function
            if (radiusI != zero)
                radiusI = (radiusI + swLen) * PBradiusMod;
            
            // check for maximum
            if (XI + radiusI > maxX)  maxX = XI + radiusI;
            if (YI + radiusI > maxY)  maxY = YI + radiusI;
            if (ZI + radiusI > maxZ)  maxZ = ZI + radiusI;
            
            // check for minimum
            if (XI - radiusI < minX)  minX = XI - radiusI;
            if (YI - radiusI < minY)  minY = YI - radiusI;
            if (ZI - radiusI < minZ)  minZ = ZI - radiusI;
        }
        
        // save grid dimensions and translation vector to fit atoms onto grid
        gridDim[0] = floor((maxX - minX + two*RBuffer) / deltaR);
        gridDim[1] = floor((maxY - minY + two*RBuffer) / deltaR);
        gridDim[2] = floor((maxZ - minZ + two*RBuffer) / deltaR);
        
        transVec[0] = half * ((RealOpenMM)gridDim[0]-one) * deltaR - (maxX + minX) * half;
        transVec[1] = half * ((RealOpenMM)gridDim[1]-one) * deltaR - (maxY + minY) * half;
        transVec[2] = half * ((RealOpenMM)gridDim[2]-one) * deltaR - (maxZ + minZ) * half;
    }
    
    // ---------------------------------------------------------------------------------------
    // check lookup table size, and reset it
    
    numGridLoc = gridDim[0]*gridDim[1]*gridDim[2];
    
    // is lookup big enough?
    if ( lookupTable.size() < numGridLoc )
        lookupTable.resize( numGridLoc+100, std::vector<int>( maxAtomsInVoxel , 0 ) );
    
    // reset number of neighbors in lookup table
    for ( gridLoc = 0; gridLoc < numGridLoc; gridLoc++ )
        lookupTable[gridLoc][0] = 0;
    
    // ---------------------------------------------------------------------------------------
    // fill lookup table
    
    int i, j, k, ip, jp, kp, gridR, numLocalAtoms,
        gridMinX, gridMaxX, gridMinY, gridMaxY, gridMinZ, gridMaxZ;
    RealOpenMM radiusBuffer_2, r, r2, dx, dy, dz;
    
    // main atom loop
    for ( atomI = 0; atomI < numberOfAtoms; atomI++ ) {
        
        radiusI = atomicRadii[atomI];
        if (radiusI != zero)
            radiusI = (radiusI + swLen) * PBradiusMod;
        
        // only add atoms with a non-zero radius to the grid
        if ( radiusI > zero ) {
            
            XI = atomCoordinates[atomI][0];
            YI = atomCoordinates[atomI][1];
            ZI = atomCoordinates[atomI][2];
            
            // location and radius of atom I in 3D grid
            i = floor((XI + transVec[0]) / deltaR);
            j = floor((YI + transVec[1]) / deltaR);
            k = floor((ZI + transVec[2]) / deltaR);
            gridR = floor((radiusI + RBuffer) / deltaR) + 1;
            
            // calc max and min XYZ of search in the grid
            gridMinX = i - gridR;
            gridMaxX = i + gridR + 1;
            gridMinY = j - gridR;
            gridMaxY = j + gridR + 1;
            gridMinZ = k - gridR;
            gridMaxZ = k + gridR + 1;
            
            // periodic! so every ijk must fit on the grid somewhere
            if (_gbswParameters->getPeriodic()) {
                
                radiusBuffer_2 = (radiusI + RBuffer) * (radiusI + RBuffer);
                
                // search if grids locations are near enough to the atom
                for ( i = gridMinX; i < gridMaxX; i++ ) {
                    dx = (RealOpenMM)i * deltaR - XI - transVec[0];
                for ( j = gridMinY; j < gridMaxY; j++ ) {
                    dy = (RealOpenMM)j * deltaR - YI - transVec[1];
                for ( k = gridMinZ; k < gridMaxZ; k++ ) {
                    dz = (RealOpenMM)k * deltaR - ZI - transVec[2];
                    
                    // final check if grid is near or inside atom
                    r2 = dx*dx + dy*dy + dz*dz;
                    if ( r2 <= radiusBuffer_2 ) {
                        
                        // enforce periodic boundaries onto ijk (save as ip, jp, kp)
                        ip = i; jp = j; kp = k;
                        if (i < 0) ip += gridDim[0]; else if (i >= gridDim[0]) ip += gridDim[0];
                        if (j < 0) jp += gridDim[1]; else if (j >= gridDim[1]) jp += gridDim[1];
                        if (k < 0) kp += gridDim[2]; else if (k >= gridDim[2]) kp += gridDim[2];
                        
                        gridLoc = ip +
                                  jp * gridDim[0] +
                                  kp * gridDim[0] * gridDim[1];
                        
                        // add atom I to the proper location in the lookup table
                        numLocalAtoms = lookupTable[gridLoc][0] + 1;
                        lookupTable[gridLoc][0] ++;
                        
                        // add more space to the lookup if needed
                        if (lookupTable[gridLoc].size() < numLocalAtoms + 1)
                            lookupTable[gridLoc].resize(numLocalAtoms + 3);
                        
                        // append atomI's index to the lookup
                        lookupTable[gridLoc][numLocalAtoms] = atomI;
                        
                    } // grid nearby atom check
                } // k loop
                } // j loop
                } // i loop
                
            } else {
                
                radiusBuffer_2 = (radiusI + RBuffer) * (radiusI + RBuffer);
                
                // nonperiodic bounds lookup: bracket ijk to a local gridspace
                if ( gridMinX < 0 ) gridMinX = 0;
                if ( gridMaxX > gridDim[0] ) gridMaxX = gridDim[0];
                if ( gridMinY < 0 ) gridMinY = 0;
                if ( gridMaxY > gridDim[1] ) gridMaxY = gridDim[1];
                if ( gridMinZ < 0 ) gridMinZ = 0;
                if ( gridMaxZ > gridDim[2] ) gridMaxZ = gridDim[2];
                
                // search if grids locations are near enough to the atom
                for ( i = gridMinX; i < gridMaxX; i++ ) {
                    dx = (RealOpenMM)i * deltaR - XI - transVec[0];
                for ( j = gridMinY; j < gridMaxY; j++ ) {
                    dy = (RealOpenMM)j * deltaR - YI - transVec[1];
                for ( k = gridMinZ; k < gridMaxZ; k++ ) {
                    dz = (RealOpenMM)k * deltaR - ZI - transVec[2];
                    
                    
                    // final check if grid is near or inside atom
                    r2 = dx*dx + dy*dy + dz*dz;
                    if ( r2 <= radiusBuffer_2 ) {
                        
                        gridLoc = i +
                                  j * gridDim[0] +
                                  k * gridDim[0] * gridDim[1];
                        
                        // add atom I to the proper location in the lookup table
                        numLocalAtoms = lookupTable[gridLoc][0] + 1;
                        lookupTable[gridLoc][0] ++;
                        
                        // add more space to the lookup if needed
                        if (lookupTable[gridLoc].size() < numLocalAtoms + 1)
                            lookupTable[gridLoc].resize(numLocalAtoms + 3);
                        
                        // append atomI's index to the lookup
                        lookupTable[gridLoc][numLocalAtoms] = atomI;
                        
                    } // grid nearby atom check
                } // k loop
                } // j loop
                } // i loop
                
            } // periodic bounds check
        } // nonzero radius check
    } // main atom loop
    
    // normally we would sort the lookup table here for a faster Born R calculation
    // since this is a reference forcefield, we won't bother
    
}


typedef struct {
    RealOpenMM r, sf;
    int radInt, angInt;
} GradientIncriments;


void ReferenceGBSW::computeBornRadii( const std::vector<RealVec>& atomCoordinates, 
                                const std::vector< std::vector<int> >& lookupTable, 
                                std::vector<RealOpenMM>& bornRadii, 
                                std::vector<int>& nForceIncrimentAtoms,
                                std::vector< std::vector<localForce> >& gbswChain ){
    
    // ---------------------------------------------------------------------------------------
    
    static const RealOpenMM zero         = static_cast<RealOpenMM>( 0.0 );
    static const RealOpenMM one          = static_cast<RealOpenMM>( 1.0 );
    static const RealOpenMM minusOne     = static_cast<RealOpenMM>( -1.0 );
    static const RealOpenMM half         = static_cast<RealOpenMM>( 0.5 );
    static const RealOpenMM fourth       = static_cast<RealOpenMM>( 0.25 );
    static const RealOpenMM threeFourths = static_cast<RealOpenMM>( 0.75 );
    static const RealOpenMM two          = static_cast<RealOpenMM>( 2.0 );
    static const RealOpenMM bigNegative  = static_cast<RealOpenMM>( -99999999999.0 );
    static const RealOpenMM bigPositive  = static_cast<RealOpenMM>( 99999999999.0 );
    
    // ---------------------------------------------------------------------------------------
    // get useful parameters for the calculation
    
    GBSWParameters* gbswParameters              = getGBSWParameters();
    
    int numberOfAtoms                           = gbswParameters->getNumberOfAtoms();
    const std::vector<RealOpenMM>& atomicRadii         = gbswParameters->getAtomicRadii();
    const RealOpenMM* periodicBox          = gbswParameters->getPeriodicBox();
    
    int maxGradientsPerAtom                     = gbswParameters->getMaxGradientsPerAtom(); // 160 gradient components
    int maxAtomsInVoxel                         = gbswParameters->getMaxAtomsInVoxel(); // 25 atoms
    RealOpenMM deltaR                           = gbswParameters->getDeltaR(); // 0.15 nm
    RealOpenMM RBuffer                          = gbswParameters->getRBuffer(); // 0.15990 nm
    
    RealOpenMM swLen                            = gbswParameters->getSwitchLen(); // 0.03 nm
    RealOpenMM PBradiusMod                      = gbswParameters->getPBradiusMod(); // 0.9520
    RealOpenMM AA0                              = gbswParameters->getAA0(); // -0.1801
    RealOpenMM AA1                              = gbswParameters->getAA1(); // 1.81745
    RealOpenMM Rmin                             = gbswParameters->getRmin(); // 0.05 nm
    int numGauLegRad                            = gbswParameters->getNumGauLegRad(); // 24 radii
    int numLebAng                               = gbswParameters->getNumLebAng(); // 50 angles
    
    RealOpenMM PRECALC1    = threeFourths / swLen;
    RealOpenMM PRECALC2    = fourth / (swLen * swLen * swLen);
    RealOpenMM PRECALC3    = fourth / (Rmin * Rmin * Rmin * Rmin);
    RealOpenMM PRECALC4    = threeFourths / (swLen * swLen * swLen);
    
    // ---------------------------------------------------------------------------------------
    
    // radial quadrature data (gaussian-legendre)
    const std::vector<RealOpenMM>& QuadR            = getQuadR(numGauLegRad);
    const std::vector<RealOpenMM>& QuadWrad         = getQuadWrad(numGauLegRad);
    
    // angular quadrature data (lebadev)
    const std::vector<RealOpenMM>& QuadX            = getQuadX(numLebAng);
    const std::vector<RealOpenMM>& QuadY            = getQuadY(numLebAng);
    const std::vector<RealOpenMM>& QuadZ            = getQuadZ(numLebAng);
    const std::vector<RealOpenMM>& QuadWang         = getQuadWang(numLebAng);
        
    // ---------------------------------------------------------------------------------------
    
    int gridDim[3], numGridLoc, atomI, atomJ;
    RealOpenMM maxX, minX, maxY, minY, maxZ, minZ, XI, YI, ZI, radiusI, radiusJ;
    RealVec transVec;
    
    if (_gbswParameters->getPeriodic()) {
        
        // save grid dimensions and translation vector to fit atoms onto grid
        gridDim[0] = ceil( periodicBox[0] / deltaR );
        gridDim[1] = ceil( periodicBox[1] / deltaR );
        gridDim[2] = ceil( periodicBox[2] / deltaR );
        
        // we need last +0.5 * dgp to use sqrt(3)/2 in filling the lookup table
        transVec[0] = atomCoordinates[0][0] * minusOne + deltaR * half;
        transVec[1] = atomCoordinates[0][1] * minusOne + deltaR * half;
        transVec[2] = atomCoordinates[0][2] * minusOne + deltaR * half;
        
    } else {
        
        maxX = bigNegative; maxY = bigNegative; maxZ = bigNegative;
        minX = bigPositive; minY = bigPositive; minZ = bigPositive;
        
        // find extrema
        for ( atomI = 0; atomI < numberOfAtoms; atomI++ ) {
            
            XI = atomCoordinates[atomI][0];
            YI = atomCoordinates[atomI][1];
            ZI = atomCoordinates[atomI][2];
            radiusI = atomicRadii[atomI];
            
            // modulate radius to be consistent with switching function
            if (radiusI != zero)
                radiusI = (radiusI + swLen) * PBradiusMod;
            radiusI += swLen;
            
            // check for maximum
            if (XI + radiusI > maxX)  maxX = XI + radiusI;
            if (YI + radiusI > maxY)  maxY = YI + radiusI;
            if (ZI + radiusI > maxZ)  maxZ = ZI + radiusI;
            
            // check for minimum
            if (XI - radiusI < minX)  minX = XI - radiusI;
            if (YI - radiusI < minY)  minY = YI - radiusI;
            if (ZI - radiusI < minZ)  minZ = ZI - radiusI;
        }
        
        // save grid dimensions and translation vector to fit atoms onto grid
        gridDim[0] = floor((maxX - minX + two*RBuffer - two*swLen) / deltaR);
        gridDim[1] = floor((maxY - minY + two*RBuffer - two*swLen) / deltaR);
        gridDim[2] = floor((maxZ - minZ + two*RBuffer - two*swLen) / deltaR);
        
        // THIS TRANSLATION VECTOR IS DIFFERENT FROM THE LOOKUP TABLE!
        // we need last +0.5 * dgp to use sqrt(3)/2 in filling the lookup table
        // transVec is the realspace location of the grid center
        // this moves all atoms as a reference from the center of the grid
        transVec[0] = half * (RealOpenMM)gridDim[0] * deltaR - (maxX + minX) * half;
        transVec[1] = half * (RealOpenMM)gridDim[1] * deltaR - (maxY + minY) * half;
        transVec[2] = half * (RealOpenMM)gridDim[2] * deltaR - (maxZ + minZ) * half;
    }
    
    // ---------------------------------------------------------------------------------------
    // run atomic density intrgration to calculate Born Radii
    
    int i, j, k, gridLoc, numLocalAtoms, atomCounter, sfAtomCount, gradientAtomCount, sfIdx;
    RealOpenMM radGauLeg, angSum, switchFn, volSum_AA0, volSum_AA1, r, r2quadpt, 
        sf, rad2, rad5, wtLeb, wtGauLeg, allSwitchFn[numGauLegRad][numLebAng];
    RealVec quadPt, atomIxyz, atomJxyz;
    RealOpenMM deltaXYZ[4];
    
    // stuff to hold the force gradient calculations
    localForce thisForce;
    std::vector <localForce> localForceIncriments(maxGradientsPerAtom);
    GradientIncriments thisIncriment;
    std::vector <GradientIncriments> allGradIncriments(maxGradientsPerAtom);
    
    
    for ( atomI = 0; atomI < numberOfAtoms; atomI++ ) {
        
        atomIxyz = atomCoordinates[atomI];
        
        volSum_AA0        = zero;
        volSum_AA1        = zero;
        
        gradientAtomCount = 0;
        
        // 3D integration quadrature: radial shell loop (Gaussian-Legendre quadrature)
        for( int radInt = 0; radInt < numGauLegRad; radInt++ ){
            
            // initialize radius and angular sums
            radGauLeg = QuadR[radInt]; // Gaussian-Legendre quadrature
            wtGauLeg  = QuadWrad[radInt];
            angSum    = zero;
            
            // 3D integration quadrature: angular loop (Lebadev quadrature)
            for( int angInt = 0; angInt < numLebAng; angInt++ ) {
                
                // switching function product begins at 1 (outside all atoms)
                switchFn = one;
                sfAtomCount = 0;
                
                // generate quadrature point XYZ
                quadPt[0] = radGauLeg * QuadX[angInt] + atomIxyz[0];
                quadPt[1] = radGauLeg * QuadY[angInt] + atomIxyz[1];
                quadPt[2] = radGauLeg * QuadZ[angInt] + atomIxyz[2];
                wtLeb = QuadWang[angInt];
                
                // periodic boundary method
                if (_gbswParameters->getPeriodic()) {
                    
                    // find location in gridspace
                    i = floor((quadPt[0] + transVec[0]) / deltaR);
                    j = floor((quadPt[1] + transVec[1]) / deltaR);
                    k = floor((quadPt[2] + transVec[2]) / deltaR);
                    
                    // enforce periodic bounds
                    if (i < 0) i += gridDim[0]; else if (i >= gridDim[0]) i -= gridDim[0];
                    if (j < 0) j += gridDim[1]; else if (j >= gridDim[1]) j -= gridDim[1];
                    if (k < 0) k += gridDim[2]; else if (k >= gridDim[2]) k -= gridDim[2];
                    
                    gridLoc = i +
                              j * gridDim[0] +
                              k * gridDim[0] * gridDim[1];
                    numLocalAtoms = lookupTable[gridLoc][0];
                    
                    // are atoms at this location in gridspace?
                    if ( numLocalAtoms > 0 ) {
                    
                    // if so calculate the atomic density (switching function)
                    atomCounter = 0;
                    while (atomCounter < numLocalAtoms) {
                        atomCounter ++;
                        
                        atomJ     = lookupTable[gridLoc][atomCounter];
                        radiusJ   = atomicRadii[atomJ];
                        if (radiusJ != zero) radiusJ = (radiusJ + swLen) * PBradiusMod;
                        atomJxyz = atomCoordinates[atomJ];
                        
                        // find distance between quadrature point and atom
                        deltaXYZ[0] = atomJxyz[0] - quadPt[0];
                        deltaXYZ[0] -= (RealOpenMM)(floor(deltaXYZ[0]/periodicBox[0]+0.5)*periodicBox[0]);
                        deltaXYZ[1] = atomJxyz[1] - quadPt[1];
                        deltaXYZ[1] -= (RealOpenMM)(floor(deltaXYZ[1]/periodicBox[1]+0.5)*periodicBox[1]);
                        deltaXYZ[2] = atomJxyz[2] - quadPt[2];
                        deltaXYZ[2] -= (RealOpenMM)(floor(deltaXYZ[2]/periodicBox[2]+0.5)*periodicBox[2]);
                        deltaXYZ[3] = DOT3( deltaXYZ, deltaXYZ );
                        r = SQRT( deltaXYZ[3] );
                        
                        // is quad pt inside this atom, outside, or in between?
                        if ( r < radiusJ + swLen ) {
                        if ( r < radiusJ - swLen ) {
                            switchFn = zero;                // totally inside atom
                            atomCounter = numLocalAtoms; // exit loop
                            sfAtomCount = 0;
                        } else {
                            
                            r2quadpt = r;
                            r -= radiusJ; // inside smooth switching function
                            sf = half + PRECALC1 * r - PRECALC2 * r * r * r;
                            switchFn *= sf;
                            
                            // record info should this be a gradient quadrature point
                            // remember, there is no self-contribution here!
                            if ( atomI != atomJ ) {
                                
                                sfIdx = gradientAtomCount + sfAtomCount;
                                
                                // and make sure to add more space as needed
                                if ( sfIdx >= allGradIncriments.size() ) {
                                    allGradIncriments.resize(sfIdx + 3);
                                    localForceIncriments.resize(sfIdx + 3);
                                }
                                
                                thisIncriment.sf     = sf * r2quadpt;
                                thisIncriment.r      = r * r;
                                thisIncriment.radInt = radInt;
                                thisIncriment.angInt = angInt;
                                
                                allGradIncriments[sfIdx] = thisIncriment;
                                
                                thisForce.dx = deltaXYZ[0];
                                thisForce.dy = deltaXYZ[1];
                                thisForce.dz = deltaXYZ[2];
                                thisForce.atomNum = atomJ;
                                
                                localForceIncriments[sfIdx] = thisForce;
                                
                                sfAtomCount ++;
                            }
                        }
                        } // inside atom / in between check
                        
                    } // switching function while loop
                    } // are atoms at this location check
                
                } else { // non-periodic boundary method
                    
                    // bracket quadrature to local space (cubic cutoff)
                    if ( quadPt[0] > minX && quadPt[0] < maxX &&
                         quadPt[1] > minY && quadPt[1] < maxY &&
                         quadPt[2] > minZ && quadPt[2] < maxZ ) {
                        
                        // if gridpoint is local, find location in gridspace
                        i = floor((quadPt[0] + transVec[0]) / deltaR);
                        j = floor((quadPt[1] + transVec[1]) / deltaR);
                        k = floor((quadPt[2] + transVec[2]) / deltaR);
                        
                        gridLoc = i +
                                  j * gridDim[0] +
                                  k * gridDim[0] * gridDim[1];
                        numLocalAtoms = lookupTable[gridLoc][0];
                        
                        // are atoms at this location in gridspace?
                        if ( numLocalAtoms > 0 ) {
                        
                        // if so calculate the atomic density (switching function)
                        atomCounter = 0;
                        while (atomCounter < numLocalAtoms) {
                            atomCounter ++;
                            
                            atomJ     = lookupTable[gridLoc][atomCounter];
                            atomJxyz  = atomCoordinates[atomJ];
                            radiusJ   = atomicRadii[atomJ];
                            if (radiusJ != 0.0) radiusJ = (radiusJ + swLen) * PBradiusMod;
                            
                            deltaXYZ[0] = quadPt[0] - atomJxyz[0];
                            deltaXYZ[1] = quadPt[1] - atomJxyz[1];
                            deltaXYZ[2] = quadPt[2] - atomJxyz[2];
                            deltaXYZ[3] = DOT3( deltaXYZ, deltaXYZ );
                            r = SQRT( deltaXYZ[3] );
                            
                            // is quad pt inside this atom, outside, or in between?
                            if ( r < radiusJ + swLen ) {
                            if ( r < radiusJ - swLen ) {
                                switchFn = zero;             // totally inside atom
                                atomCounter = numLocalAtoms; // exit loop
                                sfAtomCount = 0;
                            } else {
                                
                                r2quadpt = r;
                                r -= radiusJ;
                                sf = half + PRECALC1 * r - PRECALC2 * r * r * r;
                                switchFn *= sf;
                                
                                // record info should this be a gradient quadrature point
                                // remember, there is no self-contribution here!
                                if ( atomI != atomJ ) {
                                    
                                    sfIdx = gradientAtomCount + sfAtomCount;
                                    
                                    // and make sure to add more space as needed
                                    if ( sfIdx >= allGradIncriments.size() ) {
                                        allGradIncriments.resize(sfIdx + 3);
                                        localForceIncriments.resize(sfIdx + 3);
                                    }
                                    
                                    thisIncriment.sf     = sf * r2quadpt;
                                    thisIncriment.r      = r * r;
                                    thisIncriment.radInt = radInt;
                                    thisIncriment.angInt = angInt;
                                    
                                    allGradIncriments[sfIdx] = thisIncriment;
                                    
                                    thisForce.dx = deltaXYZ[0];
                                    thisForce.dy = deltaXYZ[1];
                                    thisForce.dz = deltaXYZ[2];
                                    thisForce.atomNum = atomJ;
                                    
                                    localForceIncriments[sfIdx] = thisForce;
                                    
                                    sfAtomCount ++;
                                }
                            }
                            } // inside atom / in between check
                            
                        } // switching function while loop
                        } // are atoms at this location check
                    } // bracket quadrature to local space check
                } // periodic boundary check
                
                // keep or scrap this point?
                if ( sfAtomCount > 0 ) {
                    gradientAtomCount += sfAtomCount;
                    allSwitchFn[radInt][angInt] = switchFn;
                }
                
                // integrate angular components
                angSum += wtLeb * (one - switchFn);
                
            } // angular loop
            
            rad2 = radGauLeg * radGauLeg;
            rad5 = rad2 * rad2 * radGauLeg;
            
            // integrate radial components
            volSum_AA0 += wtGauLeg * angSum / rad2;
            volSum_AA1 += wtGauLeg * angSum / rad5;
            
        } // radial shell loop
        
        // now convert the integral into a Born radius for this atom
        RealOpenMM AA0term = AA0 * (one / Rmin - volSum_AA0);
        RealOpenMM dG1     = SQRT(SQRT(PRECALC3 - volSum_AA1));
        RealOpenMM AA1term = AA1 * dG1;
        RealOpenMM BornR   = one / (AA0term + AA1term); 
        bornRadii[atomI]   = BornR; 
        
        // if gradient components found, then go for it!
        if ( gradientAtomCount > 0 ) {
            
            nForceIncrimentAtoms[atomI] = gradientAtomCount;
            
            // make sure to add more space as needed
            if ( gradientAtomCount >= gbswChain[atomI].size() )
                gbswChain[atomI].resize(gradientAtomCount + 10);
            
            // a little precalculation
            RealOpenMM dG1sum  = fourth / (dG1*dG1*dG1);
            
            for ( sfIdx = 0; sfIdx < gradientAtomCount; sfIdx++ ) {
                
                // retrieve data used to calculate force for this 
                // atom - quadrature point pair
                thisIncriment = allGradIncriments[sfIdx];
                sf        = thisIncriment.sf;
                r         = thisIncriment.r;
                switchFn  = allSwitchFn[thisIncriment.radInt][thisIncriment.angInt];
                radGauLeg = QuadR[thisIncriment.radInt];
                wtGauLeg  = QuadWrad[thisIncriment.radInt];
                wtLeb     = QuadWang[thisIncriment.angInt];
                
                // calc slope of the force: delta-BornR / delta-radius
                RealOpenMM aa0Term = wtGauLeg*wtLeb / (radGauLeg * radGauLeg);
                RealOpenMM aa1Term = aa0Term / (radGauLeg * radGauLeg * radGauLeg);
                RealOpenMM aa0aa1Term = AA0*aa0Term + AA1*aa1Term * dG1sum;
                RealOpenMM wtAndSf = BornR*BornR * switchFn * aa0aa1Term;
                RealOpenMM dBornR_dr = wtAndSf * (PRECALC1 - PRECALC4 * r) / sf;
                
                
                
                // now we input it into the global array for later calculations
                thisForce = localForceIncriments[sfIdx];
                thisForce.dx *= dBornR_dr;
                thisForce.dy *= dBornR_dr;
                thisForce.dz *= dBornR_dr;
                
                gbswChain[atomI][sfIdx] = thisForce;
            }
        } // end of gradient componenet
    } // main atom loop
}

/**---------------------------------------------------------------------------------------

    Get nonpolar solvation force constribution via ACE approximation

    @param gbswParameters parameters
    @param bornRadii                 Born radii
    @param energy                    energy (output): value is incremented from input value 
    @param forces                    forces: values are incremented from input values

    --------------------------------------------------------------------------------------- */

void ReferenceGBSW::computeAceNonPolarForce(const GBSWParameters* gbswParameters,
                                      const std::vector<RealOpenMM>& bornRadii,
                                      RealOpenMM* energy,
                                      std::vector<RealOpenMM>& forces) const {

    // ---------------------------------------------------------------------------------------

    static const RealOpenMM zero     = static_cast<RealOpenMM>(0.0);
    static const RealOpenMM minusSix = -6.0;
    static const RealOpenMM six      = static_cast<RealOpenMM>(6.0);

    // ---------------------------------------------------------------------------------------

    // compute the nonpolar solvation via ACE approximation

    const RealOpenMM probeRadius          = gbswParameters->getProbeRadius();
    const RealOpenMM surfaceAreaFactor    = gbswParameters->getPi4Asolv();

    const std::vector<RealOpenMM>& atomicRadii   = gbswParameters->getAtomicRadii();
    int numberOfAtoms                     = gbswParameters->getNumberOfAtoms();

    // the original ACE equation is based on Eq.2 of

    // M. Schaefer, C. Bartels and M. Karplus, "Solution Conformations
    // and Thermodynamics of Structured Peptides: Molecular Dynamics
    // Simulation with an Implicit Solvation Model", J. Mol. Biol.,
    // 284, 835-848 (1998)  (ACE Method)

    // The original equation includes the factor (atomicRadii[atomI]/bornRadii[atomI]) to the first power,
    // whereas here the ratio is raised to the sixth power: (atomicRadii[atomI]/bornRadii[atomI])**6

    // This modification was made by Jay Ponder who observed it gave better correlations w/
    // observed values. He did not think it was important enough to write up, so there is
    // no paper to cite.

    for (int atomI = 0; atomI < numberOfAtoms; atomI++) {
        if (bornRadii[atomI] > zero) {
            RealOpenMM r            = atomicRadii[atomI] + probeRadius;
            RealOpenMM ratio6       = POW(atomicRadii[atomI]/bornRadii[atomI], six);
            RealOpenMM saTerm       = surfaceAreaFactor*r*r*ratio6;
            *energy                += saTerm;
            forces[atomI]          += minusSix*saTerm/bornRadii[atomI]; 
        }
    }
}

/**---------------------------------------------------------------------------------------

    Get GBSW Born energy and forces

    @param atomCoordinates     atomic coordinates
    @param partialCharges      partial charges
    @param forces              forces

    The array bornRadii is also updated and the gbswEnergy

    --------------------------------------------------------------------------------------- */

RealOpenMM ReferenceGBSW::computeBornEnergyForces( const std::vector<RealVec>& atomCoordinates,
                                            const std::vector<RealOpenMM>& partialCharges, 
                                            std::vector<RealVec>& inputForces ){

    // ---------------------------------------------------------------------------------------

    static const RealOpenMM zero    = static_cast<RealOpenMM>( 0.0 );
    static const RealOpenMM one     = static_cast<RealOpenMM>( 1.0 );
    static const RealOpenMM two     = static_cast<RealOpenMM>( 2.0 );
    static const RealOpenMM four    = static_cast<RealOpenMM>( 4.0 );
    static const RealOpenMM half    = static_cast<RealOpenMM>( 0.5 );
    static const RealOpenMM fourth  = static_cast<RealOpenMM>( 0.25 );

    // constants
    const int numberOfAtoms = _gbswParameters->getNumberOfAtoms();
    const RealOpenMM cutoffDistance = _gbswParameters->getCutoffDistance();
    const RealOpenMM soluteDielectric = _gbswParameters->getSoluteDielectric();
    const RealOpenMM solventDielectric = _gbswParameters->getSolventDielectric();
    RealOpenMM preFactor;
    if (soluteDielectric != zero && solventDielectric != zero)
        preFactor = two*_gbswParameters->getElectricConstant()*((one/soluteDielectric) - (one/solventDielectric));
    else
        preFactor = zero;
    
    // ---------------------------------------------------------------------------------------
    // compute Born radii
    
    std::vector<RealOpenMM> bornRadii( numberOfAtoms );
    std::vector< std::vector<int> > lookupTable            = getLookupTable();
    std::vector< std::vector<localForce> > gbswChain       = getGBSWChain();
    std::vector<int> nForceIncrimentAtoms             = getnForceIncrimentAtoms();
    
    // generate lookup table and Born Radii
    computeLookupTable( atomCoordinates, lookupTable );
    computeBornRadii( atomCoordinates, lookupTable, bornRadii, nForceIncrimentAtoms, gbswChain );
    
    // initialize and set energy/forces to zero
    RealOpenMM gbswEnergy                 = zero;
    std::vector<RealOpenMM> bornForces( numberOfAtoms, 0.0 );
    
    // ---------------------------------------------------------------------------------------
    // first main loop: neighbor-atom forces
    
    for( int atomI = 0; atomI < numberOfAtoms; atomI++ ){
        
        // triangular 2D atom-atom interaction calculation
        RealOpenMM partialChargeI = preFactor*partialCharges[atomI];
        for( int atomJ = atomI; atomJ < numberOfAtoms; atomJ++ ){
            
            // cal atom-atom distance, check for cutoffs
            RealOpenMM deltaR[ReferenceForce::LastDeltaRIndex];
            if (_gbswParameters->getPeriodic())
                ReferenceForce::getDeltaRPeriodic( 
                    atomCoordinates[atomI], atomCoordinates[atomJ],
                    _gbswParameters->getPeriodicBox(), deltaR );
            else
                ReferenceForce::getDeltaR( 
                    atomCoordinates[atomI], atomCoordinates[atomJ], deltaR );
            if (_gbswParameters->getUseCutoff() && deltaR[ReferenceForce::RIndex] > cutoffDistance)
                continue;
            
            // parse atom-atom distance components
            RealOpenMM r2                 = deltaR[ReferenceForce::R2Index];
            RealOpenMM deltaX             = deltaR[ReferenceForce::XIndex];
            RealOpenMM deltaY             = deltaR[ReferenceForce::YIndex];
            RealOpenMM deltaZ             = deltaR[ReferenceForce::ZIndex];
            
            // use the Still equation to calc the polar solvation free 
            // energy as an electrostatic term from the Born radii of atoms I and J
            RealOpenMM alpha2_ij          = bornRadii[atomI]*bornRadii[atomJ];
            RealOpenMM D_ij               = r2/(four*alpha2_ij);
            
            RealOpenMM expTerm            = EXP( -D_ij );
            RealOpenMM denominator2       = r2 + alpha2_ij*expTerm; 
            RealOpenMM denominator        = SQRT( denominator2 ); 
            
            RealOpenMM Gpol               = (partialChargeI*partialCharges[atomJ])/denominator; 
            RealOpenMM dGpol_dr           = -Gpol*( one - fourth*expTerm )/denominator2;  
            
            RealOpenMM dGpol_dalpha2_ij   = -half*Gpol*expTerm*( one + D_ij )/denominator2;
            
            RealOpenMM energy = Gpol;
            
            if( atomI != atomJ ){
                
                bornForces[atomJ]        += dGpol_dalpha2_ij*bornRadii[atomI];
        
                deltaX                   *= dGpol_dr;
                deltaY                   *= dGpol_dr;
                deltaZ                   *= dGpol_dr;
                
                inputForces[atomI][0]    += deltaX;
                inputForces[atomI][1]    += deltaY;
                inputForces[atomI][2]    += deltaZ;
                
                inputForces[atomJ][0]    -= deltaX;
                inputForces[atomJ][1]    -= deltaY;
                inputForces[atomJ][2]    -= deltaZ;
                
            } else { // self-contribution to energy
                energy *= half;
            }
            
            gbswEnergy        += energy;
            bornForces[atomI] += dGpol_dalpha2_ij*bornRadii[atomJ];
       }
    }
    
    // ---------------------------------------------------------------------------------------
    // second main loop: Born Radius gradient component
    
    for( int atomI = 0; atomI < numberOfAtoms; atomI++ ) {
        
        // self-term (preFactor * charges_I^2 / BornR_I^2) is already included!
        int gradientAtomCount = nForceIncrimentAtoms[atomI];
        RealOpenMM dGdRGB_I  = bornForces[atomI];
        
        RealOpenMM GBdx_I = zero;
        RealOpenMM GBdy_I = zero;
        RealOpenMM GBdz_I = zero;
        
        for ( int sfIdx = 0; sfIdx < gradientAtomCount; sfIdx++ ) {
            
            localForce thisForce = gbswChain[atomI][sfIdx];
            int atomJ  = thisForce.atomNum;
            
            // quadrature point's influence on atom J
            RealOpenMM GBdx_J = thisForce.dx * dGdRGB_I;
            RealOpenMM GBdy_J = thisForce.dy * dGdRGB_I;
            RealOpenMM GBdz_J = thisForce.dz * dGdRGB_I;
            
            // quadrature point's influence on atom I
            GBdx_I += GBdx_J;
            GBdy_I += GBdy_J;
            GBdz_I += GBdz_J;
            
            // save forces to atom J
            inputForces[atomJ][0] -= GBdx_J;
            inputForces[atomJ][1] -= GBdy_J;
            inputForces[atomJ][2] -= GBdz_J;
        }
        
        // save forces to atom I
        inputForces[atomI][0] += GBdx_I;
        inputForces[atomI][1] += GBdy_I;
        inputForces[atomI][2] += GBdz_I;
    }

    return gbswEnergy;
}


