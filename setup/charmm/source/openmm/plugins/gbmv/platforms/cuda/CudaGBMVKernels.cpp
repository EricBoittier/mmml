/* -------------------------------------------------------------------------- *
 *                                 OpenMMGBMV                                 *
 * -------------------------------------------------------------------------- */

#include "CudaGBMVKernels.h"
#include "CudaGBMVKernelSources.h"

#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/cuda/CudaNonbondedUtilities.h"
#include "openmm/internal/ContextImpl.h"

#include <cmath>
#include <cstdio>
#include <cstring>


using namespace OpenMM;
using namespace OpenMMGBMV;
using namespace std;

class CudaGBMVForceInfo : public CudaForceInfo {
public:
    CudaGBMVForceInfo(const GBMVForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, radius1, radius2;
        force.getParticleParameters(particle1, charge1, radius1);
        force.getParticleParameters(particle2, charge2, radius2);
        return (charge1 == charge2 && radius1 == radius2);
    }
private:
    const GBMVForce& force;
};

CudaCalcGBMVForceKernel::~CudaCalcGBMVForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
    if (bornRadii != NULL)
        delete bornRadii;
    if (bornForce != NULL)
        delete bornForce;
    if (gbmvChain != NULL)
        delete gbmvChain;
    if (GridDimXYZ != NULL)
        delete GridDimXYZ;
    if (lookupTable != NULL)
        delete lookupTable;
    if (QuadPts != NULL)
        delete QuadPts;
    if (QuadPtWeights_gbmv != NULL)
        delete QuadPtWeights_gbmv;
    if (lastWtrLE != NULL)
        delete lastWtrLE;
  //if (Energy_SASA != NULL)
  //    delete Energy_SASA;
  //if (Forces_SASA != NULL)
  //    delete Forces_SASA;
    if (QuadXYZW != NULL)
        delete QuadXYZW;
  //if (nGBMVchainAtoms != NULL)
  //    delete nGBMVchainAtoms;
}

void CudaCalcGBMVForceKernel::initialize(const System& system, const GBMVForce& force) {
    
    //--------------------------------------------------------------------------
    // obtain system
    
    cu.setAsCurrent();
    if (cu.getPlatformData().contexts.size() > 1)
        throw OpenMMException("GBMVForce does not support using multiple CUDA devices");
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    
    // check that the quadrature requested is one supported
  //if (force.getNumGauLegRad() != 24) // this is numRadii
  //    throw OpenMMException("GBMVForce requires a 24-radii Gaussian-Legendre quadrature");
  //if (force.getNumLebAng() != 50 && force.getNumLebAng() != 38) // this is numAngles
  //    throw OpenMMException("GBMVForce requires either the 38-angle or the 50-angle Lebedev quadrature");
    
    //--------------------------------------------------------------------------
    // quadrature information
    
    // radial quadrature data (gaussian-legendre)
    const double QuadR [] = {
        0.010000000,  0.020000000,   0.030000000,     
        0.040000000,  0.050000000,   0.075000000,     
        0.100000000,  0.125000000,   0.150000000,     
        0.175000000,  0.200000000,   0.250000000,     
        0.300000000,  0.350000000,   0.400000000,     
        0.500000000,  0.600000000,   0.700000000,     
        0.800000000,  1.000000000,   1.200000000,     
        1.600000000,  2.000000000,   2.400000000};
    const double QuadWrad [] = {
        0.00592317212640454, 0.0119657167624842,  0.0142222222222222, 
        0.0119657167624842,  0.00592317212640454, 0.0184886988182386,
        0.0425735154274146,  0.0655923156007592,  0.0869155205413276,
        0.105988313269961,   0.122316264412369,   0.135476367064926,
        0.145127739962567,   0.151020401224257,   0.153001727356345,
        0.151020401224257,   0.145127739962567,   0.135476367064926,
        0.122316264412369,   0.105988313269961,   0.0869155205413276,
        0.0655923156007592,  0.0425735154274146,  0.0184886988182386 };
    // integrate some inner radii of QuadR and QuadWrad before calculation to make GBMV faster
    // sum(QuadR / QuadWrad^2)
    const double PreIntrgrate_aa0 [] = { 0.0,
        2.16170080571196,    5.32141784018352,    7.84981290191191,
        9.37888454653048,    9.99999646953608,   11.6084459083209,
       13.8504007512081,    15.6296711288455,    16.835995518868,
       17.6311963507724,    18.1632306902153,    18.5289241256755,
       18.7874360481663,    18.974854593768,     19.1136316707352,
       19.2180990050755,    19.2976433940108,    19.3585562447867,
       19.4051337763975,    19.4403465424016,    19.4662552685592,
       19.4842728744621,    19.4953302985971,    19.4999859966073 };
    // sum(QuadR / QuadWrad^5)
    const double PreIntrgrate_aa1 [] = { 0.0,
       15071.5330779857,    28630.0566652263,    34623.289404138,
       36832.1154210189,    37499.065203623,     38804.2109492056,
       39660.968386426,     39912.3453213338,    39974.7208541481,
       39991.0628498993,    39995.8892447257,    39997.5110458528,
       39998.1256221342,    39998.3847284378,    39998.5046092947,
       39998.5647125304,    39998.5969899068,    39998.6153543176,
       39998.6262992948,    39998.6330424535,    39998.637259119,
       39998.6398530789,    39998.6413166898,    39998.6419049948 };
    
    // angular quadrature data (lebadev)
    vector<double> QuadX, QuadY, QuadZ, QuadWang;

    if (force.getNumLebAng() == 6) { // 6 angles
    const double qx [] = {
        1.000000000000000, -1.000000000000000,  0.000000000000000,
        0.000000000000000,  0.000000000000000,  0.000000000000000 };
    const double qy [] = {
        0.000000000000000,  0.000000000000000,  1.000000000000000,
       -1.000000000000000,  0.000000000000000,  0.000000000000000 };
    const double qz [] = {
        0.000000000000000,  0.000000000000000,  0.000000000000000,
        0.000000000000000,  1.000000000000000, -1.000000000000000 };
    const double qw [] = {
        0.166666666666667,  0.166666666666667,  0.166666666666667,
        0.166666666666667,  0.166666666666667,  0.166666666666667 };
    QuadX.insert(QuadX.end(), qx, qx+6);
    QuadY.insert(QuadY.end(), qy, qy+6);
    QuadZ.insert(QuadZ.end(), qz, qz+6);
    QuadWang.insert(QuadWang.end(), qw, qw+6);
    }
    
    if (force.getNumLebAng() == 14) { // 14 angles
    const double qx [] = {
        1.000000000000000, -1.000000000000000,  0.000000000000000,
        0.000000000000000,  0.000000000000000,  0.000000000000000,
        0.577350269189626, -0.577350269189626,  0.577350269189626,
       -0.577350269189626,  0.577350269189626, -0.577350269189626,
        0.577350269189626, -0.577350269189626 };
    const double qy [] = {
        0.000000000000000,  0.000000000000000,  1.000000000000000,
       -1.000000000000000,  0.000000000000000,  0.000000000000000,
        0.577350269189626,  0.577350269189626, -0.577350269189626,
       -0.577350269189626,  0.577350269189626,  0.577350269189626,
       -0.577350269189626, -0.577350269189626 };
    const double qz [] = {
        0.000000000000000,  0.000000000000000,  0.000000000000000,
        0.000000000000000,  1.000000000000000, -1.000000000000000,
        0.577350269189626,  0.577350269189626,  0.577350269189626,
        0.577350269189626, -0.577350269189626, -0.577350269189626,
       -0.577350269189626, -0.577350269189626 };
    const double qw [] = {
        0.066666666666667,  0.066666666666667,  0.066666666666667,
        0.066666666666667,  0.066666666666667,  0.066666666666667,
        0.075000000000000,  0.075000000000000,  0.075000000000000,
        0.075000000000000,  0.075000000000000,  0.075000000000000,
        0.075000000000000,  0.075000000000000 };
    QuadX.insert(QuadX.end(), qx, qx+14);
    QuadY.insert(QuadY.end(), qy, qy+14);
    QuadZ.insert(QuadZ.end(), qz, qz+14);
    QuadWang.insert(QuadWang.end(), qw, qw+14);
    }
    
    if (force.getNumLebAng() == 26) { // 26 angles
    const double qx [] = {
        1.000000000000000, -1.000000000000000,  0.000000000000000,
        0.000000000000000,  0.000000000000000,  0.000000000000000,
        0.000000000000000,  0.000000000000000,  0.000000000000000,
        0.000000000000000,  0.707106781186548, -0.707106781186548,
        0.707106781186548, -0.707106781186548,  0.707106781186548,
       -0.707106781186548,  0.707106781186548, -0.707106781186548,
        0.577350269189626, -0.577350269189626,  0.577350269189626,
       -0.577350269189626,  0.577350269189626, -0.577350269189626,
        0.577350269189626, -0.577350269189626 };
    const double qy [] = {
        0.000000000000000,  0.000000000000000,  1.000000000000000,
       -1.000000000000000,  0.000000000000000,  0.000000000000000,
        0.707106781186548, -0.707106781186548,  0.707106781186548,
       -0.707106781186548,  0.000000000000000,  0.000000000000000,
        0.000000000000000,  0.000000000000000,  0.707106781186548,
        0.707106781186548, -0.707106781186548, -0.707106781186548,
        0.577350269189626,  0.577350269189626, -0.577350269189626,
       -0.577350269189626,  0.577350269189626,  0.577350269189626,
       -0.577350269189626, -0.577350269189626 };
    const double qz [] = {
        0.000000000000000,  0.000000000000000,  0.000000000000000, 
        0.000000000000000,  1.000000000000000, -1.000000000000000,
        0.707106781186548,  0.707106781186548, -0.707106781186548,
       -0.707106781186548,  0.707106781186548,  0.707106781186548,
       -0.707106781186548, -0.707106781186548,  0.000000000000000,  
        0.000000000000000,  0.000000000000000,  0.000000000000000,
        0.577350269189626,  0.577350269189626,  0.577350269189626,
        0.577350269189626, -0.577350269189626, -0.577350269189626,
       -0.577350269189626, -0.577350269189626 };
    const double qw [] = {
        0.047619047619048,  0.047619047619048,  0.047619047619048,
        0.047619047619048,  0.047619047619048,  0.047619047619048,
        0.038095238095238,  0.038095238095238,  0.038095238095238,
        0.038095238095238,  0.038095238095238,  0.038095238095238,
        0.038095238095238,  0.038095238095238,  0.038095238095238,
        0.038095238095238,  0.038095238095238,  0.038095238095238,
        0.032142857142857,  0.032142857142857,  0.032142857142857,
        0.032142857142857,  0.032142857142857,  0.032142857142857,
        0.032142857142857,  0.032142857142857 };
    QuadX.insert(QuadX.end(), qx, qx+26);
    QuadY.insert(QuadY.end(), qy, qy+26);
    QuadZ.insert(QuadZ.end(), qz, qz+26);
    QuadWang.insert(QuadWang.end(), qw, qw+26);
    }
     
    if (force.getNumLebAng() == 38) { // 38 angles
    const double qx [] = {
        1.00000000000000000,  -1.00000000000000000,  0.00000000000000000,
        0.00000000000000000,   0.00000000000000000,  0.00000000000000000,
        0.57735026918962573,   0.57735026918962573,  0.57735026918962573,
        0.57735026918962573,  -0.57735026918962573, -0.57735026918962573,
       -0.57735026918962573,  -0.57735026918962573,  0.88807383397711526,
        0.88807383397711526,  -0.88807383397711526, -0.88807383397711526,
        0.00000000000000000,   0.00000000000000000,  0.00000000000000000,
        0.00000000000000000,   0.45970084338098310, -0.45970084338098310,
        0.45970084338098310,  -0.45970084338098310,  0.45970084338098310,
       -0.45970084338098310,   0.45970084338098310, -0.45970084338098310,
        0.88807383397711526,   0.88807383397711526, -0.88807383397711526,
       -0.88807383397711526,   0.00000000000000000,  0.00000000000000000,
        0.00000000000000000,   0.00000000000000000};
    const double qy [] = {
        0.0000000000000000,  0.0000000000000000,  1.0000000000000000,
       -1.0000000000000000,  0.0000000000000000,  0.0000000000000000,
        0.5773502691896257,  0.5773502691896257, -0.5773502691896257,
       -0.5773502691896257,  0.5773502691896257,  0.5773502691896257,
       -0.5773502691896257, -0.5773502691896257,  0.4597008433809831,
       -0.4597008433809831,  0.4597008433809831, -0.4597008433809831,
        0.8880738339771153,  0.8880738339771153, -0.8880738339771153,
       -0.8880738339771153,  0.0000000000000000,  0.0000000000000000,
        0.0000000000000000,  0.0000000000000000,  0.8880738339771153,
        0.8880738339771153, -0.8880738339771153, -0.8880738339771153,
        0.0000000000000000,  0.0000000000000000,  0.0000000000000000,
        0.0000000000000000,  0.4597008433809831, -0.4597008433809831,
        0.4597008433809831, -0.4597008433809831};
    const double qz [] = {
        0.0000000000000000,   0.0000000000000000,  0.0000000000000000,
        0.0000000000000000,   1.0000000000000000, -1.0000000000000000,
        0.5773502691896257,  -0.5773502691896257,  0.5773502691896257,
       -0.5773502691896257,   0.5773502691896257, -0.5773502691896257,
        0.5773502691896257,  -0.5773502691896257,  0.0000000000000000,
        0.0000000000000000,   0.0000000000000000,  0.0000000000000000,
        0.4597008433809831,  -0.4597008433809831,  0.4597008433809831,
       -0.4597008433809831,   0.8880738339771153,  0.8880738339771153,
       -0.8880738339771153,  -0.8880738339771153,  0.0000000000000000,
        0.0000000000000000,   0.0000000000000000,  0.0000000000000000,
        0.4597008433809831,  -0.4597008433809831,  0.4597008433809831,
       -0.4597008433809831,   0.8880738339771153,  0.8880738339771153,
       -0.8880738339771153,  -0.8880738339771153};

    const double qw [] = {
        0.00952380952380952,  0.00952380952380952,  0.00952380952380952,
        0.00952380952380952,  0.00952380952380952,  0.00952380952380952,
        0.0321428571428571,   0.0321428571428571,   0.0321428571428571,
        0.0321428571428571,   0.0321428571428571,   0.0321428571428571,
        0.0321428571428571,   0.0321428571428571,   0.0285714285714286,
        0.0285714285714286,   0.0285714285714286,   0.0285714285714286,
        0.0285714285714286,   0.0285714285714286,   0.0285714285714286,
        0.0285714285714286,   0.0285714285714286,   0.0285714285714286,
        0.0285714285714286,   0.0285714285714286,   0.0285714285714286,
        0.0285714285714286,   0.0285714285714286,   0.0285714285714286,
        0.0285714285714286,   0.0285714285714286,   0.0285714285714286,
        0.0285714285714286,   0.0285714285714286,   0.0285714285714286,
        0.0285714285714286,   0.0285714285714286 };
    QuadX.insert(QuadX.end(), qx, qx+38);
    QuadY.insert(QuadY.end(), qy, qy+38);
    QuadZ.insert(QuadZ.end(), qz, qz+38);
    QuadWang.insert(QuadWang.end(), qw, qw+38);
    } // 38 angles
    
    if (force.getNumLebAng() == 50) { // 50 angles
    const double qx [] = {
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
    const double qy [] = {
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
    const double qz [] = {
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
    const double qw [] = {
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
    QuadX.insert(QuadX.end(), qx, qx+50);
    QuadY.insert(QuadY.end(), qy, qy+50);
    QuadZ.insert(QuadZ.end(), qz, qz+50);
    QuadWang.insert(QuadWang.end(), qw, qw+50);
    }
    
 /*
  *    this energy offset is not ready yet. while adjusts system energy for OpenMM's
  *    exclusions properly, it doesn't yet accomodate exceptions (1-4 interactions).
  *    ignoring the offset doesn't affect the forces of a simulation, and only 
  *    affects the total energy of the system. see "USE_COULOMBIC_CUTOFF_OFFSET"
  *
  *  // add an offset to nonbonded electric forces
  *  if ((force.getNonbondedMethod() != GBMVForce::NoCutoff) && 
  *      (force.getReactionFieldDielectric() == 1.0)) {
  *      usingCutoffOffset = true;
  *      rfDielectric = force.getReactionFieldDielectric();
  *      cutoffOffset = -1.0 / force.getCutoffDistance();
  *  } else
  */    usingCutoffOffset = false;
    
    // is CPHMD active?
    numTitratingGroups = 0;
    if (force.usingCPHMD() && force.getNumTitratingGroups() > 0) {
        numTitratingGroups  = force.getNumTitratingGroups();
        usingCPHMD          = true;
        if (force.getNonbondedMethod() != GBMVForce::NoCutoff)
            usingCutoff = true;
        else
            usingCutoff = false;
    } else {
        usingCPHMD = false;
        usingCutoff = false;
    }
    
    // is there a membrane in this system?
    if ( force.getMembraneThickness() > 0.0 ) {
        membThickness = force.getMembraneThickness();
        membSwLen     = force.getMembraneSwLen();
        if (membSwLen <= 0.0)
            membSwLen = force.getSwitchingLength();
        usingMembrane = true;
    } else 
        usingMembrane = false;

		// nb.getCutoffDistance() does not exist in OpenMM 7.0.0
		// do not switch this back
		// vvv is compatible with OpenMM 6.3.0 and 7.0.0
		cutoff = force.getCutoffDistance();
		cuton = force.getCutonDistance();
    
    // length of switching function around atoms (nm)
    // swLen = force.getSwitchingLength(); // 0.03 default
    swLen = 0.21f;

    // The parameters in the GBMV2 model
    BETA_GBMV2   = force.getBETA_GBMV2(); // -12.0 default
    LAMBDA_GBMV2 = force.getLAMBDA_GBMV2(); // 0.5f
    A0_GBMV2     =  1.0f-sqrt(0.5f); 
    A1_GBMV2     =  1.0f;
    ALPHA_GBMV2  = force.getALPHA_GBMV2(); // -19.8f
    SLOPE_GBMV2  = force.getSLOPE_GBMV2(); // 0.9085f
    SHIFT_GBMV2  = force.getSHIFT_GBMV2(); //-0.0102f
    HSX1_GBMV2   = force.getHSX1_GBMV2(); //
    HSX2_GBMV2   = force.getHSX2_GBMV2(); //
    ONX_GBMV2    = force.getONX_GBMV2(); //
    OFFX_GBMV2   = force.getOFFX_GBMV2(); //
    P1_GBMV2     = force.getP1_GBMV2(); //
    P2_GBMV2     = force.getP2_GBMV2(); //
    P3_GBMV2     = force.getP3_GBMV2(); // 0.65f 
    P6_GBMV2     = force.getP6_GBMV2(); //

    // AA0 and AA1 phenomenological coefficients to fit the Born R to PB calculations
    AA0 = force.getAA0(); // -0.1801 default
    AA1 = force.getAA1(); // 1.81745 default
    
  //// width of lookup table voxel (GBMV deltaR=0.15nm and GBMV is deltaR=0.10nm )
    deltaR = 0.15f;
    
    // Debye-Huckel length used for salt representation (nm)
    kappa = force.getDebyeHuckelLength(); // 0.0 default
    saltFactorA = -ONE_4PI_EPS0 / force.getSolventDielectric();
    saltFactorB = -ONE_4PI_EPS0 / force.getSoluteDielectric();
    
    // this is the "surfaceAreaFactor" used for the ACE surface area approximation
    sgamma = force.getSurfaceAreaEnergy(); // 0.0 default
    
  //// PB radius modifying factor, {PBradiusMod = 0.9520;} if {swLen = 0.03;}
  //float swLenRadiusMod [] = { 0.979, 0.965, 0.952, 0.939, 0.927,
  //                            0.914, 0.901, 0.888, 0.875, 0.861 };
  //PBradiusMod = swLenRadiusMod[(int)floor(swLen * 100.0000001) - 1];
    
    // minimum integration radius
    Rmin = 0.05f;
    
    // this is the default number of threads used to find the extrema of the system
    numThreads = 64;
    
    // diagonal length of lookup table voxel (_switchLen + sqrtf(3.0f) * _deltaR / 2.0f;)
    // RBuffer = swLen + sqrt(3.0) * deltaR / 2.0;
       //BufR  = 0.05f; //0.5A !I have checked it is the same of gbmv
    float BufR  = 0.00f; //0.5A !I have checked it is the same of gbmv
    CUTNUM_GBMV2 = force.getCUTNUM_GBMV2(); //
  //printf("CUTNUM_GBMV2 = %d \n", CUTNUM_GBMV2);
    
    // number of atoms allowed to fit in a voxel
    // maxAtomsPerVoxel = 27; // Original
    maxAtomsPerVoxel = 150;
    
    // number of integration radii
    numAngles      = force.getNumLebAng(); // number of angles integrated (38 default)
    numRadii       = force.getNumGauLegRad(); // #  
  //printf("numAngles = %d \n", numAngles);
  //printf("numRadii = %d \n", numRadii);
    if (usingCPHMD) {
        numRadii_H      = 18; // number of integration radii (hydrogen atoms)
        numRadii_HEAVY  = 16; // number of integration radii (heavy, non-hydrogen atoms)
        skipRadii_H     = 0;  // number of inner radii to skip (hydrogen atoms)
        skipRadii_HEAVY = 2;  // number of inner radii to skip (heavy atoms)
        maxSfNeighbors  = 506; // num of contributions saved in the Born Force derivative
        
        // preintegrate the Gaussian-Legengre quadrature for so many radii
        preintegrate1 = PreIntrgrate_aa0[skipRadii_H]; // AA0 term - hydrogen
        preintegrate2 = PreIntrgrate_aa1[skipRadii_H]; // AA1 term - hydrogen
        preintegrate3 = PreIntrgrate_aa0[skipRadii_HEAVY]; // AA0 term - heavy
        preintegrate4 = PreIntrgrate_aa1[skipRadii_HEAVY]; // AA1 term - heavy
    } else {
        numRadii_H      = 23; // number of integration radii (hydrogen atoms)
        numRadii_HEAVY  = 23;  // number of integration radii (heavy, non-hydrogen atoms)
        skipRadii_H     = 0;  // number of inner radii to skip (hydrogen atoms)
        skipRadii_HEAVY = 0;  // number of inner radii to skip (heavy atoms)
        maxSfNeighbors  = 6*numAngles*numRadii; // num of contributions saved in the Born Force derivative
        
        // preintegrate the Gaussian-Legengre quadrature for so many radii
        preintegrate1 = PreIntrgrate_aa0[skipRadii_H]; // AA0 term - hydrogen
        preintegrate2 = PreIntrgrate_aa1[skipRadii_H]; // AA1 term - hydrogen
        preintegrate3 = PreIntrgrate_aa0[skipRadii_HEAVY]; // AA0 term - heavy
        preintegrate4 = PreIntrgrate_aa1[skipRadii_HEAVY]; // AA1 term - heavy
    }
    
    // "tau" or the factor to convert units to kJ / mol
    prefactor = -ONE_4PI_EPS0*((1.0/force.getSoluteDielectric())-(1.0/force.getSolventDielectric()));
    
    // calc whether this simulation is periodic
    bool usePeriodic = (force.getNonbondedMethod() != GBMVForce::NoCutoff && 
        force.getNonbondedMethod() != GBMVForce::CutoffNonPeriodic);
    
    //--------------------------------------------------------------------------
    // allocate memory on GPU. doing this first improves code stability
    
    GridDimXYZ = CudaArray::create<int>  (cu, 4, "GridDimXYZ");
    params = CudaArray::create<float2>(cu, cu.getPaddedNumAtoms(), "GBMVParams");
    bornRadii = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "bornRadii");
    bornForce = CudaArray::create<unsigned long long>(cu, cu.getPaddedNumAtoms(), "bornForce");
    gbmvChain = CudaArray::create<float>(cu, cu.getNumAtoms()*numAngles*numRadii*6, "gbmvChain");
    QuadPts = CudaArray::create<float4>(cu, numRadii * numAngles, "QuadPts");
    lastWtrLE = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "lastWtrLE");
  //Energy_SASA = CudaArray::create<float>(cu, cu.getPaddedNumAtoms(), "Energy_SASA");
  //Forces_SASA = CudaArray::create<float4>(cu, cu.getPaddedNumAtoms(), "Forces_SASA");
    QuadXYZW = CudaArray::create<float4>(cu, numAngles, "QuadXYZW");
    QuadPtWeights_gbmv = CudaArray::create<float2>(cu, numRadii * numAngles, "QuadPtWeights_gbmv");
  //nGBMVchainAtoms = CudaArray::create<int>(cu, cu.getNumAtoms(), "nGBMVchainAtoms");
    if (usingCPHMD) {
        cphmdRandSeed   = CudaArray::create<uint4>(cu, numTitratingGroups, "cphmdRandSeed");
        cphmdRandNum    = CudaArray::create<float4>(cu, numTitratingGroups, "cphmdRandNum");
        cphmdLambdaXtheta = CudaArray::create<float4>(cu, numTitratingGroups, "cphmdLambdaXtheta");
        cphmdForce       = CudaArray::create<float>(cu, cu.getPaddedNumAtoms()*2, "cphmdForce");
        cphmdLambdaXvf  = CudaArray::create<float4>(cu, numTitratingGroups, "cphmdLambdaXvf");
        cphmdAtomRanges = CudaArray::create<uint4>(cu, numTitratingGroups, "cphmdAtomRanges");
        cphmdUphUbarr   = CudaArray::create<float4>(cu, numTitratingGroups, "cphmdUphUbarr");
        cphmdUmod0123   = CudaArray::create<float4>(cu, numTitratingGroups, "cphmdUmod0123");
        cphmdUmod4567   = CudaArray::create<float4>(cu, numTitratingGroups, "cphmdUmod4567");
        cphmdAtomQfac = CudaArray::create<float2>(cu, cu.getPaddedNumAtoms(), "cphmdAtomQfac");
        cphmdChargeStates = CudaArray::create<float4>(cu, cu.getPaddedNumAtoms(), "cphmdChargeStates");
        cphmdLambdaXvOld = CudaArray::create<float4>(cu, numTitratingGroups, "cphmdLambdaXvOld");
    }
    
    // lookup table memory: if there is a periodic box, then the lookup table
    // simply is the size of the box. otherwise, make an initial guess of the 
    // lookup table size based on the maximum dimensions of the molecule. in the
    // latter case allocate memory for the lookup table during the first "execute"
    if (usePeriodic) {
        double4 box = cu.getPeriodicBoxSize();
        gridDimlocal[0] = ceil( box.x / deltaR );
        gridDimlocal[1] = ceil( box.y / deltaR );
        gridDimlocal[2] = ceil( box.z / deltaR );
        lookupMemory = (gridDimlocal[0]*gridDimlocal[1]*gridDimlocal[2] + 100) * maxAtomsPerVoxel;
        lookupTable = CudaArray::create<int> (cu, lookupMemory, "lookupTable"); // ***
    }
    
  //cu.addAutoclearBuffer(*nGBMVchainAtoms);
    cu.addAutoclearBuffer(*bornForce);
    
    //--------------------------------------------------------------------------
    // format and initialize inputs for GBMV calculations
    
    // quadrature point formatting
    vector<float4> QuadPtsVector(numRadii * numAngles);
    vector<float2> QuadPtWeightsVector_gbmv(numRadii * numAngles);
    
    double SkipRtmp1, SkipRtmp2, R2=0.0f;
    for ( int i = 0; i < numRadii; i++ ) {
        for ( int j = 0; j < numAngles; j++ ) {
            R2= 0.5f * (QuadR[i] + QuadR[i+1]);
            // Rotation of quadrature points for each quad. Radius
            if ( ((i+1) % 2) ==0 ){
              QuadPtsVector[ i*numAngles + j ] = 
                make_float4( QuadX[j] * R2,
                    QuadY[j] * R2,
                    QuadZ[j] * R2,
                    QuadR[i] );
            }else{

            QuadPtsVector[ i*numAngles + j ] = 
              make_float4(  QuadY[j]  * R2,
                  (- QuadX[j]) * R2,
                  QuadZ[j]  * R2,
                  QuadR[i] );
            }
      QuadPtWeightsVector_gbmv[ i*numAngles + j ] = 
        make_float2( QuadWang[j]*( (1.0 / QuadR[i])-(1.0/QuadR[i+1])) ,
            QuadWang[j]* 0.25f*(1.0 / ( QuadR[i]* QuadR[i]* QuadR[i]* QuadR[i] )
              -(1.0 / ( QuadR[i+1] * QuadR[i+1] * QuadR[i+1] * QuadR[i+1])
               )));
    //QuadPtWeightsVector_gbmv[ i*numAngles + j ] = 
    //  make_float2( QuadWang[j]*( (QuadR[i+1]-QuadR[i]) / (QuadR[i]*QuadR[i+1])) ,
    //0.25f*(1.0f/QuadR[i]+1.0f/QuadR[i+1])*(1.0f/(QuadR[i]*QuadR[i])+1.0f/(QuadR[i+1]*QuadR[i+1])));
    }
  }
    QuadPts->upload(QuadPtsVector);
    QuadPtWeights_gbmv->upload(QuadPtWeightsVector_gbmv);
    
  // format SASA quadrature integral
    vector<float4> QuadXYZWVector(numAngles);
    for ( int j = 0; j < numAngles; j++) {
        QuadXYZWVector[j] = make_float4(QuadX[j], QuadY[j], QuadZ[j], QuadWang[j]);
    }
    QuadXYZW->upload(QuadXYZWVector);

    //---------------------------mara
    vector<float> lastWtrLE_Vector(cu.getPaddedNumAtoms());
    int cnt,
        numatom= cu.getNumAtoms();
    float RvdWM; RvdWM=0.f;
    double radius, charge;
    for (int i = 0; i < numatom; i++) {
      cnt=0;
      force.getParticleParameters(i, charge, radius);
      if (RvdWM<radius) RvdWM=radius;
      lastWtrLE_Vector[i] = QuadR[0];
      for ( int j = numRadii-1; j >= 0 ; j-- ) {
        if (QuadR[j] < radius &&  cnt==0){
          lastWtrLE_Vector[i] = QuadR[j];
          cnt ++; 
          break;
        } 
 
      }
    }
    lastWtrLE-> upload(lastWtrLE_Vector);

    RBuffer = RvdWM + OFFX_GBMV2 + sqrt(3.f) * deltaR / 2.f + BufR;
  //-------------------------------

    // format atomic radii, and posq (atom position coordinates with charge Q)
    CudaArray& posq = cu.getPosq();
    vector<double4> temp(posq.getSize());
    float4* posqf = (float4*) &temp[0];
    double4* posqd = (double4*) &temp[0];
    vector<float2> paramsVector(cu.getPaddedNumAtoms(), make_float2(1, 1));
    vector<float4> chargeStates(cu.getPaddedNumAtoms(), make_float4(0, 0, 0, 0));
    vector<uint4> atomRanges(numTitratingGroups, make_uint4(0, 0, 0, 0));
    vector<int> allTitrateResIDs(numTitratingGroups, 0);
    
    int beginRes = 0, endRes = 0, resCounter = 0, titrateResID, oldRes = 0;
    for (int i = 0; i < force.getNumParticles(); i++) {
        
        double charge, radius, refChargeState1, refChargeState2, chargeState1, chargeState2;
        force.getParticleParameters(i, charge, radius);
        
        if (usingCPHMD) {
            force.getCphmdParameters(i, titrateResID, 
                refChargeState1, refChargeState2, chargeState1, chargeState2);
            
            // track atom ranges for each titrating residue
            chargeStates[i] = make_float4(charge, charge, charge, charge);
            if ( titrateResID != 0 ) {
                chargeStates[i] = make_float4(
                    (float)refChargeState1, (float)refChargeState2,
                    (float)chargeState1,    (float)chargeState2 );
                if ( titrateResID != oldRes ) {
                    if ( resCounter-1 >= 0 ) {
                        atomRanges[resCounter-1] = make_uint4(beginRes, endRes, 0, 0);
                        allTitrateResIDs[resCounter-1] = oldRes;
                    }
                    oldRes = titrateResID;
                    resCounter ++;
                    beginRes = i;
                    endRes = i;
                } else
                    endRes = i;
            }
        }
        
      //// modulate atomic radii by the switching length and a PB scaling factor
      //if (radius != 0.0)
      //    radius = (radius + swLen) * PBradiusMod;
        
        paramsVector[i] = make_float2((float) radius, (float) ((radius + swLen) * (radius + swLen)));
        if (cu.getUseDoublePrecision())
            posqd[i] = make_double4(0, 0, 0, charge);
        else
            posqf[i] = make_float4(0, 0, 0, (float) charge);
    }
    if ( usingCPHMD && resCounter-1 >= 0 ) { // record atom ranges for final residue
        atomRanges[resCounter-1] = make_uint4(beginRes, endRes, 0, 0);
        allTitrateResIDs[resCounter-1] = oldRes;
    }
    
    if ( usingCPHMD && numTitratingGroups > 0 ) {
        
        vector<uint4> randomSeed(numTitratingGroups, make_uint4(0, 0, 0, 0));
        vector<float> initialForce(cu.getPaddedNumAtoms()*2, 0);
        vector<float4> initialVelocityForce(numTitratingGroups, make_float4(0, 0, 0, 0));
        vector<float4> UphUbarr(numTitratingGroups, make_float4(0, 0, 0, 0));
        vector<float4> Umod0123(numTitratingGroups, make_float4(0, 0, 0, 0));
        vector<float4> Umod4567(numTitratingGroups, make_float4(0, 0, 0, 0));
        vector<float2> initialAtomQfac(cu.getPaddedNumAtoms(), make_float2(0, 0));
        LambdaXtheta = (float4*) malloc(numTitratingGroups * sizeof(float4));
        writeGroup   = (bool*) malloc(numTitratingGroups * sizeof(bool));
        
        double resPKA1, resPKA2, UpH1, UpH2, barrier1, barrier2,
            a0, a1, a2, a3, a4, a5, a6, a7, a8;
        double pH = force.getSystemPH();
        double T = force.getThetaTemp();
        double M = force.getThetaMass();
        double beta = force.getPHbeta();
        if (!( beta > 0.0 ))
            throw OpenMMException("CPHMD requires a positive beta value to run Langivan dynamics");
        
        // CPHMD was based on the CHARMM forcefield, so we use AKMA units
        // "332.0716" converts units to kcal/mol. since the forces on theta 
        // move in a fictitious dimension, the actual units are not that important
        cphmdGBMVFac = (1.0/force.getSoluteDielectric())-(1.0/force.getSolventDielectric());
        cphmdCoulombicFac = 332.0716*(-0.1);
        
        double time = force.getThetaTimestep(); // picoseconds
        double BOLTZ_AKMA = 0.001987191;
        cphmdGamma = beta * time;
        ntimesteps = 0;
        outputFrequency = force.getLambdaOutputFrequency();
        timestepCounter = 0;
        
        cphmdFileID = fopen( force.getLambdaOutputFile(), "w+" );
        
        timeFactor = time / 0.0488882129; // convert to CHARMM's "AKMA" time units
        randomForceScale = sqrt(2.0 * M * BOLTZ_AKMA * T * cphmdGamma) / timeFactor;
        massTimeFactor = (time / 0.0488882129) / (2.0 * M);
        
        unsigned int r = (unsigned int) rand();
        for ( int i = 0; i < numTitratingGroups; i++ ) {
            
            // get titrating parameters
            force.getTitratingGroupParameters( i, 
                resPKA1, resPKA2, barrier1, barrier2, 
                a0, a1, a2, a3, a4, a5, a6, a7, a8);
            
            // do a little pre-calculation for the pH-pKa relationship
            UpH1 = log(10.0) * BOLTZ_AKMA * T * (resPKA1 - pH);
            UpH2 = log(10.0) * BOLTZ_AKMA * T * (resPKA2 - pH);
            UphUbarr[i] = make_float4(UpH1, UpH2, barrier1, barrier2);
            Umod0123[i] = make_float4(a0, a1, a2, a3);
            Umod4567[i] = make_float4(a4, a5, a6, a7);
            
            // cram the a8 term into atomRanges
            float float_a8 = (float)a8;
            std::memcpy(&atomRanges[i].z, &float_a8, 4);
            
            // uncouple the random number seeds for each residue
            randomSeed[i].x = r = (1664525*r + 1013904223) & 0xFFFFFFFF;
            randomSeed[i].y = r = (1664525*r + 1013904223) & 0xFFFFFFFF;
            randomSeed[i].z = r = (1664525*r + 1013904223) & 0xFFFFFFFF;
            randomSeed[i].w = r = (1664525*r + 1013904223) & 0xFFFFFFFF;
            
            initialVelocityForce[i] = make_float4(0, 0, 0, 0);
            LambdaXtheta[i] = make_float4(0.25*M_PI, 0.25*M_PI, 0.5, 0.5);
            if ( a0 == 0.0 && a1 == 0.0 && a2 == 0.0 && a3 == 0.0 )
                writeGroup[i] = false;
            else
                writeGroup[i] = true;
        }
        
        // initialize lambda/x output file, make sure it follows CHARMM's format
        int tmpInt = 0;
        nchar = sprintf( lineBuffer, "# ititr ");
        fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
        for ( int i = 0; i < numTitratingGroups; i++ ) {
            if (writeGroup[i])
                nchar = sprintf( lineBuffer, "%5d%5d", tmpInt++, ++tmpInt);
            else
                nchar = sprintf( lineBuffer, "%5d", ++tmpInt);
            fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
        }
        nchar = sprintf( lineBuffer, "\n#  ires ");
        fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
        for ( int i = 0; i < numTitratingGroups; i++ ) {
            if (writeGroup[i])
                nchar = sprintf( lineBuffer, "%5d%5d", allTitrateResIDs[i], allTitrateResIDs[i]);
            else
                nchar = sprintf( lineBuffer, "%5d", allTitrateResIDs[i]);
            fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
        }
        nchar = sprintf( lineBuffer, "\n# itauto" );
        fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
        for ( int i = 0; i < numTitratingGroups; i++ ) {
            if (writeGroup[i]) {
                if (UphUbarr[i].x != UphUbarr[i].y)
                    nchar = sprintf( lineBuffer, "%5d%5d", 1, 2);
                else
                    nchar = sprintf( lineBuffer, "%5d%5d", 3, 4);
            } else
                nchar = sprintf( lineBuffer, "%5d", 0);
            fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
        }
        nchar = sprintf( lineBuffer, "\n# ParK" );
        fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
        for ( int i = 0; i < numTitratingGroups; i++ ) {
            if (writeGroup[i])
                nchar = sprintf( lineBuffer, "%8.3f%8.3f", UphUbarr[i].x, UphUbarr[i].y);
            else
                nchar = sprintf( lineBuffer, "%8.3f", UphUbarr[i].x);
            fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
        }
        nchar = sprintf( lineBuffer, "\n" );
        fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
        
        // atom exclusion fixes
        int atom1, atom2;
        exceptionFixes.resize(force.getNumNonbondedExceptions());
        for ( int i = 0; i < force.getNumNonbondedExceptions(); i++ ) {
            force.getNonbondedException(i, atom1, atom2);
            exceptionFixes[i].x = atom1;
            exceptionFixes[i].y = atom2;
        }
        
        // initial charge factors
        for ( int i = 0; i < force.getNumParticles(); i++ ) {
            float4 qstate = chargeStates[i];
            float L = 0.5, X = 0.5;
            float newdQdX = L * (qstate.z - qstate.w) +
                    (1.0f - L)* (qstate.x - qstate.y);
            float newdQdLambda =  X*qstate.z + (1.0f-X)*qstate.w -
                                  X*qstate.x - (1.0f-X)*qstate.y;
            float newQ = (1.0f - L)*(X*qstate.x + (1.0f-X)*qstate.y) +
                                 L *(X*qstate.z + (1.0f-X)*qstate.w);
                                  
            initialAtomQfac[i] = make_float2( newdQdLambda, newdQdX );
            if (cu.getUseDoublePrecision())
                posqd[i] = make_double4(0, 0, 0, newQ);
            else
                posqf[i] = make_float4(0, 0, 0, (float) newQ);
        }
        
        cphmdRandSeed->upload(randomSeed);
        cphmdLambdaXtheta->upload(LambdaXtheta);
        cphmdForce->upload(initialForce);
        cphmdLambdaXvf->upload(initialVelocityForce);
        cphmdAtomRanges->upload(atomRanges);
        cphmdAtomQfac->upload(initialAtomQfac);
        cphmdUphUbarr->upload(UphUbarr);
        cphmdUmod0123->upload(Umod0123);
        cphmdUmod4567->upload(Umod4567);
        cphmdChargeStates->upload(chargeStates);
    }
    posq.upload(&temp[0]);
    params->upload(paramsVector);
}

double CudaCalcGBMVForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    if (!hasCreatedKernels) {
        // These Kernels cannot be created in initialize(), because the 
        // CudaNonbondedUtilities has not been initialized yet then.
        
        // to make an initial guess of the system's size, we needed atom 
        // positions to be set. now that they are, we can assess the system's size
        if (!nb.getUsePeriodic()) {
            
          //double buffer = 1.0;
            float buffer = 2.4f;
            int maxDim;
            int maxDimX, maxDimY, maxDimZ;
            
            // grab access to the XYZ data of the system
            CudaArray& posqTmp = cu.getPosq();
            vector<double4> tempPos(posqTmp.getSize());
            float4* posqftmp = (float4*) &tempPos[0];
            double4* posqdtmp = (double4*) &tempPos[0];
            posqTmp.download(&tempPos[0]);
            
            // find the maxima and minima of all dimensions
            if (cu.getUseDoublePrecision()) {
                double maxX = posqdtmp[0].x+buffer, maxY = posqdtmp[0].y+buffer, maxZ = posqdtmp[0].z+buffer,
                       minX = posqdtmp[0].x-buffer, minY = posqdtmp[0].y-buffer, minZ = posqdtmp[0].z-buffer;
                for (int i = 0; i < cu.getNumAtoms(); i++) {
                    if (posqdtmp[i].x+buffer > maxX) maxX = posqdtmp[i].x+buffer;
                    if (posqdtmp[i].y+buffer > maxY) maxY = posqdtmp[i].y+buffer;
                    if (posqdtmp[i].z+buffer > maxZ) maxZ = posqdtmp[i].z+buffer;
                    if (posqdtmp[i].x-buffer < minX) minX = posqdtmp[i].x-buffer;
                    if (posqdtmp[i].y-buffer < minY) minY = posqdtmp[i].y-buffer;
                    if (posqdtmp[i].z-buffer < minZ) minZ = posqdtmp[i].z-buffer;
                }
                maxDim = max(max(ceil((maxX - minX)/deltaR), ceil((maxY - minY)/deltaR)), ceil((maxZ - minZ)/deltaR));
                maxDimX = ceil((maxX - minX)/deltaR);
                maxDimY = ceil((maxY - minY)/deltaR);
                maxDimZ = ceil((maxZ - minZ)/deltaR);
            } else {
                float maxX = posqftmp[0].x+buffer, maxY = posqftmp[0].y+buffer, maxZ = posqftmp[0].z+buffer,
                      minX = posqftmp[0].x-buffer, minY = posqftmp[0].y-buffer, minZ = posqftmp[0].z-buffer;
                for (int i = 0; i < cu.getNumAtoms(); i++) {
                    if (posqftmp[i].x+buffer > maxX) maxX = posqftmp[i].x+buffer;
                    if (posqftmp[i].y+buffer > maxY) maxY = posqftmp[i].y+buffer;
                    if (posqftmp[i].z+buffer > maxZ) maxZ = posqftmp[i].z+buffer;
                    if (posqftmp[i].x-buffer < minX) minX = posqftmp[i].x-buffer;
                    if (posqftmp[i].y-buffer < minY) minY = posqftmp[i].y-buffer;
                    if (posqftmp[i].z-buffer < minZ) minZ = posqftmp[i].z-buffer;
                }
                maxDim = max(max(ceil((maxX - minX)/deltaR), ceil((maxY - minY)/deltaR)), ceil((maxZ - minZ)/deltaR));
                maxDimX = ceil((maxX - minX)/deltaR);
                maxDimY = ceil((maxY - minY)/deltaR);
                maxDimZ = ceil((maxZ - minZ)/deltaR);
            }
            
            // allocate memory
            if (maxDim < 10) maxDim = 10;
            if (maxDimX < 10) maxDimX = 10;
            if (maxDimY < 10) maxDimY = 10;
            if (maxDimZ < 10) maxDimZ = 10;
          //lookupMemory = (maxDim*maxDim*maxDim + 100) * maxAtomsPerVoxel;
            lookupMemory = (maxDimX*maxDimY*maxDimZ + 100) * maxAtomsPerVoxel;
            lookupTable = CudaArray::create<int> (cu, lookupMemory, "lookupTable");
        }
        
        // CHARMM uses 1-4 interactions in its CPHMD exclusion list, and 
        // OpenMM doesn't. here we download the OpenMM exclusions and edit them 
        // from a CHARMM 1-4 nonbonded input.
        if (usingCPHMD) {
            
            int atom1, atom2, x, y, offset1, offset2, index, nTiles, tileSize;
            bool foundExcl;
            
            nTiles = nb.getExclusionTiles().getSize();
            tileSize = CudaContext::TileSize;
            
            // exclusion booleans
            tileflags allFlags = (tileflags) -1;
            vector<tileflags> exclusionVec(nTiles*tileSize, allFlags);
            nb.getExclusions().download(exclusionVec);
            
            // exclusion tile pairs
#if OMM_VER >= 75
            vector<int2> exclusionTilesVec(nTiles);
#else
            vector<ushort2> exclusionTilesVec(nTiles);
#endif /* OMM_VER */

            nb.getExclusionTiles().download(exclusionTilesVec);
            
            // make a map for finding exclusion tiles
            map<pair<int, int>, int> exclusionTileMap;
            for (int i = 0; i < (int) exclusionTilesVec.size(); i++) {
#if OMM_VER >= 75
              int2 tile = exclusionTilesVec[i];
#else
              ushort2 tile = exclusionTilesVec[i];
#endif /* OMM_VER */

                exclusionTileMap[make_pair(tile.x, tile.y)] = i;
            }
            
            for ( int i = 0; i < exceptionFixes.size(); i++ ) {
                
                // download exception
                atom1 = exceptionFixes[i].x;
                atom2 = exceptionFixes[i].y;
                
                // locate exception tile
                x = atom1/tileSize;
                offset1 = atom1 - x*tileSize;
                y = atom2/tileSize;
                offset2 = atom2 - y*tileSize;
                
                // remove exception
                if (x > y) {
                    index = exclusionTileMap[make_pair(x, y)] * tileSize;
                    foundExcl = !((exclusionVec[index+offset1]>>offset2) & 0x1);
                    if (foundExcl)
                        exclusionVec[index+offset1] += (1<<offset2);
                } else {
                    index = exclusionTileMap[make_pair(y, x)] * tileSize;
                    foundExcl = !((exclusionVec[index+offset2]>>offset1) & 0x1);
                    if (foundExcl)
                        exclusionVec[index+offset2] += (1<<offset1);
                }
                
                // download symmetric exception
                atom1 = exceptionFixes[i].y;
                atom2 = exceptionFixes[i].x;
                
                // locate exception tile
                x = atom1/tileSize;
                offset1 = atom1 - x*tileSize;
                y = atom2/tileSize;
                offset2 = atom2 - y*tileSize;
                
                // remove exception
                if (x > y) {
                    index = exclusionTileMap[make_pair(x, y)] * tileSize;
                    foundExcl = !((exclusionVec[index+offset1]>>offset2) & 0x1);
                    if (foundExcl)
                        exclusionVec[index+offset1] += (1<<offset2);
                } else {
                    index = exclusionTileMap[make_pair(y, x)] * tileSize;
                    foundExcl = !((exclusionVec[index+offset2]>>offset1) & 0x1);
                    if (foundExcl)
                        exclusionVec[index+offset2] += (1<<offset1);
                }
            }
            
            cphmdExclusions = CudaArray::create<tileflags> (cu, nTiles*tileSize, "cphmdExclusions");
            cphmdExclusions->upload(exclusionVec);
        }
        
        hasCreatedKernels = true;
        maxTiles = (nb.getUseCutoff() ? nb.getInteractingTiles().getSize() : 
            cu.getNumAtomBlocks()*(cu.getNumAtomBlocks()+1)/2);
        
        // define internal CUDA variables
        map<string, string> defines;
        if (nb.getUseCutoff())
            defines["USE_CUTOFF"] = "1";
        if (nb.getUsePeriodic())
            defines["USE_PERIODIC"] = "1";
        if (cu.getComputeCapability() >= 3.0 && !cu.getUseDoublePrecision())
            defines["ENABLE_SHUFFLE"] = "1"; // not used in GBMV, keep for nonbonded force parts
        
        if ( kappa > 0.00001 ) {
            defines["USE_SALT"] = "1";
            defines["ONE_OVER_KAPPA"] = cu.doubleToString(1.0 / kappa);
            defines["SALT_FAC_A"] = cu.doubleToString(saltFactorA);
            defines["SALT_FAC_B"] = cu.doubleToString(saltFactorB);
        }
        if ( sgamma > 0.00001 ) {
            defines["USE_SURFACE_AREA"] = "1";
        }
            defines["SURFACE_AREA_FACTOR"] = cu.doubleToString(sgamma);
        if (usingCPHMD) {
            defines["USE_CPHMD"] = "1";
            defines["SCALE_RANDOM_FORCE"] = cu.doubleToString(randomForceScale);
            defines["TIME_FACTOR"] = cu.doubleToString(timeFactor);
            defines["MASS_TIME_FACTOR"] = cu.doubleToString(massTimeFactor);
            defines["ONE_MINUS_GAMMA"] = cu.doubleToString(1.0 - cphmdGamma);
            defines["GBMV_FAC"] = cu.doubleToString(cphmdGBMVFac);
            defines["COULOMBIC_FAC"] = cu.doubleToString(cphmdCoulombicFac);
            defines["CPHMD_SALT_FAC"] = cu.doubleToString( -1.0 / ONE_4PI_EPS0 );
            if (usingCutoff)
                defines["USE_CPHMD_CUTOFF"] = "1";
        }
        if (usingCutoffOffset) {
            defines["USE_COULOMBIC_CUTOFF_OFFSET"] = "1";
            defines["COULOMBIC_CUTOFF_OFFSET"] = cu.doubleToString(cutoffOffset);
        }
        if (usingMembrane) {
            defines["USE_MEMBRANE"] = "1";
            defines["MEMBRANE_INNER_R"] = cu.doubleToString((membThickness/2.0) - membSwLen);
            defines["MEMBRANE_OUTER_R"] = cu.doubleToString((membThickness/2.0) + membSwLen);
            defines["MEMBRANE_R"] = cu.doubleToString(membThickness/2.0);
            defines["MEMBRANE_PRECALC1"] = cu.doubleToString( 0.75 / membSwLen );
            defines["MEMBRANE_PRECALC2"] = cu.doubleToString( 0.25 / (membSwLen*membSwLen*membSwLen) );
        }
        defines["CUTOFF_SQUARED"] = cu.doubleToString(cutoff * cutoff);
        defines["CUTON_SQUARED"] = cu.doubleToString(cuton * cuton);
        defines["CUTOFF"] = cu.doubleToString(cutoff);
        defines["PREFACTOR"] = cu.doubleToString(prefactor);
        defines["DELTA_R"] = cu.doubleToString(deltaR);
        defines["R_BUFFER"] = cu.doubleToString(RBuffer);
        defines["INVERSE_DELTA_R"] = cu.doubleToString(1.0 / deltaR);
        defines["MAX_ATOMS_IN_VOXEL"] = cu.intToString(maxAtomsPerVoxel);
        defines["NUM_RADII_H"] = cu.intToString(numRadii_H);
        defines["NUM_RADII_HEAVY"] = cu.intToString(numRadii_HEAVY);
        defines["NUM_RADII"] = cu.intToString(numRadii); // ***
        defines["NUM_ANGLES"] = cu.intToString(numAngles);
        defines["SKIP_RADII_H"] = cu.intToString(skipRadii_H);
        defines["SKIP_RADII_HEAVY"] = cu.intToString(skipRadii_HEAVY);
        defines["MAX_SF_NEIGHBORS"] = cu.intToString(maxSfNeighbors);
        defines["PREINTEGRATE1"] = cu.doubleToString(preintegrate1);
        defines["PREINTEGRATE2"] = cu.doubleToString(preintegrate2);
        defines["PREINTEGRATE3"] = cu.doubleToString(preintegrate3);
        defines["PREINTEGRATE4"] = cu.doubleToString(preintegrate4);
        defines["SWICH_LEN"] = cu.doubleToString(swLen);
        defines["CONST_RMIN"] = cu.doubleToString(Rmin);
        defines["PRECALC1"] = cu.doubleToString( 0.75 / swLen );
        defines["PRECALC2"] = cu.doubleToString( 0.25 / (swLen*swLen*swLen) );
        defines["PRECALC3"] = cu.doubleToString( 0.25 / (Rmin*Rmin*Rmin*Rmin) );
        defines["PRECALC4"] = cu.doubleToString( 0.75 / (swLen*swLen*swLen) );
        defines["BETA_GBMV2"] = cu.doubleToString(BETA_GBMV2);
        defines["LAMBDA_GBMV2"] = cu.doubleToString(LAMBDA_GBMV2);
        defines["A0_GBMV2"] = cu.doubleToString(A0_GBMV2);
        defines["A1_GBMV2"] = cu.doubleToString(A1_GBMV2);
        defines["ALPHA_GBMV2"] = cu.doubleToString(ALPHA_GBMV2);
        defines["SLOPE_GBMV2"] = cu.doubleToString(SLOPE_GBMV2);
        defines["SHIFT_GBMV2"] = cu.doubleToString(SHIFT_GBMV2);
        defines["HSX1_GBMV2"] = cu.doubleToString(HSX1_GBMV2);
        defines["HSX2_GBMV2"] = cu.doubleToString(HSX2_GBMV2);
        defines["ONX_GBMV2"] = cu.doubleToString(ONX_GBMV2);
        defines["OFFX_GBMV2"] = cu.doubleToString(OFFX_GBMV2);
        defines["P1_GBMV2"] = cu.doubleToString(P1_GBMV2);
        defines["P2_GBMV2"] = cu.doubleToString(P2_GBMV2);
        defines["P3_GBMV2"] = cu.doubleToString(P3_GBMV2);
        defines["P6_GBMV2"] = cu.doubleToString(P6_GBMV2);
        defines["CUTNUM_GBMV2"] = cu.intToString(CUTNUM_GBMV2);
        defines["CONST_AA0"] = cu.doubleToString(AA0);
        defines["CONST_AA1"] = cu.doubleToString(AA1);
        defines["NUM_THREADS"] = cu.intToString(numThreads);
        defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        defines["FORCE_WORK_GROUP_SIZE"] = cu.intToString(nb.getForceThreadBlockSize());
        defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        int numExclusionTiles = nb.getExclusionTiles().getSize();
        defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
        int numContexts = cu.getPlatformData().contexts.size();
        int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
        int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
        defines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
        defines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);
        defines["OPENMM_VERSION"] = cu.intToString(OMM_VER);
        map<string, string> replacements;
        CUmodule module = cu.createModule(CudaGBMVKernelSources::vectorOps + 
            cu.replaceStrings(CudaGBMVKernelSources::gbmv1, replacements), defines);
        
        // now assemble the arguments for the CUDA kernels
        if (!nb.getUsePeriodic()) {
            calcSysExtremaKernel = cu.getKernel(module, "calcSysExtrema");
            sysExtremaArgs.push_back(&cu.getPosq().getDevicePointer());
            sysExtremaArgs.push_back(&params->getDevicePointer());
            sysExtremaArgs.push_back(&GridDimXYZ->getDevicePointer());
        }
        resetLookupTableKernel = cu.getKernel(module, "resetLookupTable");
        resetLookupTableArgs.push_back(&lookupTable->getDevicePointer());
        fillLookupTableKernel = cu.getKernel(module, "fillLookupTable");
        fillLookupTableArgs.push_back(&cu.getPosq().getDevicePointer());
        fillLookupTableArgs.push_back(&params->getDevicePointer());
        if (nb.getUsePeriodic())
            fillLookupTableArgs.push_back(cu.getPeriodicBoxSizePointer());
        fillLookupTableArgs.push_back(&lookupTable->getDevicePointer());
        sortLookupTableKernel = cu.getKernel(module, "sortLookupTable");
        sortLookupTableArgs.push_back(&cu.getPosq().getDevicePointer());
        if (nb.getUsePeriodic()) {
            sortLookupTableArgs.push_back(cu.getPeriodicBoxSizePointer());
            sortLookupTableArgs.push_back(cu.getInvPeriodicBoxSizePointer());
        }
        sortLookupTableArgs.push_back(&lookupTable->getDevicePointer());
        
        //calcBornRKernel_hydrogen = cu.getKernel(module, "calcBornR_hydrogen");
        calcBornRKernel_heavy = cu.getKernel(module, "calcBornR");
        calcBornRArgs.push_back(&cu.getPosq().getDevicePointer());
        calcBornRArgs.push_back(&params->getDevicePointer());
        calcBornRArgs.push_back(&QuadPts->getDevicePointer());
        //calcBornRArgs.push_back(&QuadPtWeights->getDevicePointer());
        calcBornRArgs.push_back(&QuadPtWeights_gbmv->getDevicePointer());
        calcBornRArgs.push_back(&lastWtrLE->getDevicePointer());
        calcBornRArgs.push_back(&lookupTable->getDevicePointer());
        if (nb.getUsePeriodic()) {
            calcBornRArgs.push_back(cu.getPeriodicBoxSizePointer());
            calcBornRArgs.push_back(cu.getInvPeriodicBoxSizePointer());
        }
        calcBornRArgs.push_back(&bornRadii->getDevicePointer());
        calcBornRArgs.push_back(&gbmvChain->getDevicePointer());
      //calcBornRArgs.push_back(&nGBMVchainAtoms->getDevicePointer());
        computeGBMVForceKernel = cu.getKernel(module, "computeGBMVForce");
        computeGBMVForceArgs.push_back(&cu.getForce().getDevicePointer());
        computeGBMVForceArgs.push_back(&bornForce->getDevicePointer());
        computeGBMVForceArgs.push_back(&cu.getEnergyBuffer().getDevicePointer());
        computeGBMVForceArgs.push_back(&cu.getPosq().getDevicePointer());
        computeGBMVForceArgs.push_back(&bornRadii->getDevicePointer());
        if (nb.getUseCutoff()) {
            computeGBMVForceArgs.push_back(&nb.getInteractingTiles().getDevicePointer());
            computeGBMVForceArgs.push_back(&nb.getInteractionCount().getDevicePointer());
            computeGBMVForceArgs.push_back(cu.getPeriodicBoxSizePointer());
            computeGBMVForceArgs.push_back(cu.getInvPeriodicBoxSizePointer());
            computeGBMVForceArgs.push_back(&maxTiles);
            computeGBMVForceArgs.push_back(&nb.getBlockCenters().getDevicePointer());
            computeGBMVForceArgs.push_back(&nb.getBlockBoundingBoxes().getDevicePointer());
            computeGBMVForceArgs.push_back(&nb.getInteractingAtoms().getDevicePointer());
        }
        else
            computeGBMVForceArgs.push_back(&maxTiles); 
        if (usingCPHMD) {
            computeGBMVForceArgs.push_back(&cphmdAtomQfac->getDevicePointer());
            computeGBMVForceArgs.push_back(&cphmdForce->getDevicePointer());
            computeGBMVForceArgs.push_back(&cphmdExclusions->getDevicePointer());
        }
        if (usingCutoffOffset)
            computeGBMVForceArgs.push_back(&nb.getExclusions().getDevicePointer());
        computeGBMVForceArgs.push_back(&nb.getExclusionTiles().getDevicePointer());

        // Nonpolar term based on the SASA approximation
        calcSASAKernel = cu.getKernel(module, "calcSASA");
        calcSASAArgs.push_back(&cu.getPosq().getDevicePointer());
        calcSASAArgs.push_back(&params->getDevicePointer());
        calcSASAArgs.push_back(&QuadXYZW->getDevicePointer());
        calcSASAArgs.push_back(&lookupTable->getDevicePointer());
        if (nb.getUsePeriodic()) {
            calcSASAArgs.push_back(cu.getPeriodicBoxSizePointer());
            calcSASAArgs.push_back(cu.getInvPeriodicBoxSizePointer());
        }
      //calcSASAArgs.push_back(&Energy_SASA->getDevicePointer());
      //calcSASAArgs.push_back(&Forces_SASA->getDevicePointer());
        calcSASAArgs.push_back(&cu.getEnergyBuffer().getDevicePointer());
        calcSASAArgs.push_back(&cu.getForce().getDevicePointer());

        reduceGBMVForceKernel = cu.getKernel(module, "reduceGBMVForce");
        reduceGBMVForceArgs.push_back(&cu.getPosq().getDevicePointer());
        reduceGBMVForceArgs.push_back(&params->getDevicePointer());
        reduceGBMVForceArgs.push_back(&QuadPts->getDevicePointer());
        reduceGBMVForceArgs.push_back(&lookupTable->getDevicePointer());
        reduceGBMVForceArgs.push_back(&gbmvChain->getDevicePointer());
        reduceGBMVForceArgs.push_back(&bornForce->getDevicePointer());
      //reduceGBMVForceArgs.push_back(&Forces_SASA->getDevicePointer());
        reduceGBMVForceArgs.push_back(&cu.getForce().getDevicePointer());

        if (usingCPHMD) {
            cphmdLambdaDynamicsKernel = cu.getKernel(module, "cphmdLambdaDynamics");
            cphmdLambdaDynamicsArgs.push_back(&cu.getPosq().getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdRandSeed->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdRandNum->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdLambdaXtheta->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdForce->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdLambdaXvf->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdLambdaXvOld->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdAtomRanges->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdUphUbarr->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdUmod0123->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdUmod4567->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdAtomQfac->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cphmdChargeStates->getDevicePointer());
            cphmdLambdaDynamicsArgs.push_back(&cu.getEnergyBuffer().getDevicePointer());
            if (usingCutoff) {
                cphmdApplyAtomIndexChargesKernel = cu.getKernel(module, "cphmdApplyAtomIndex_Charges");
                cphmdApplyAtomIndexChargesArgs.push_back(&cu.getAtomIndexArray().getDevicePointer());
                cphmdApplyAtomIndexChargesArgs.push_back(&cu.getPosq().getDevicePointer());
                cphmdApplyAtomIndexChargesArgs.push_back(&cphmdAtomQfac->getDevicePointer());
                cphmdApplyAtomIndexForcesKernel = cu.getKernel(module, "cphmdApplyAtomIndex_Forces");
                cphmdApplyAtomIndexForcesArgs.push_back(&cu.getAtomIndexArray().getDevicePointer());
                cphmdApplyAtomIndexForcesArgs.push_back(&cphmdForce->getDevicePointer());
            }
        }
    }
    if (nb.getUseCutoff()) {
        if (maxTiles < nb.getInteractingTiles().getSize()) {
            maxTiles = nb.getInteractingTiles().getSize();
            computeGBMVForceArgs[5] = &nb.getInteractingTiles().getDevicePointer();
            computeGBMVForceArgs[12] = &nb.getInteractingAtoms().getDevicePointer();
        }
    }
    
    // extrema kernel, find system max / min, translation vector
    if (!nb.getUsePeriodic()) {
        
        // GBMV requires block and thread control to function properly
        // for the remainder of this forcefield, kernels are launched manually
        cuLaunchKernel( calcSysExtremaKernel, // kernel
            1, 1, 1,                          // blocks X, Y, Z
            64, 1, 1,                         // threads X, Y, Z
            0, 0, &sysExtremaArgs[0], NULL ); // options and arguments
        
        // ensure lookup table is big enough
        GridDimXYZ->download(gridDimlocal);
        if ( lookupMemory < gridDimlocal[0]*gridDimlocal[1]*gridDimlocal[2] * maxAtomsPerVoxel ) {
            lookupMemory = (gridDimlocal[0]*gridDimlocal[1]*gridDimlocal[2] + 100) * maxAtomsPerVoxel;
            
            // if old lookup table is too small, resize the grid
            delete lookupTable;
            lookupTable = NULL; // Avoid an error in the destructor if the following allocation fails
            lookupTable = CudaArray::create<int> (cu, lookupMemory, "lookupTable");
        }
    }
    
    // generate and prepare atom lookup table
    cuLaunchKernel( resetLookupTableKernel, gridDimlocal[0], gridDimlocal[1], gridDimlocal[2], 1, 1, 1,
        0, 0, &resetLookupTableArgs[0], NULL );
    //cuLaunchKernel( fillLookupTableKernel, cu.getNumAtoms(), 1, 1, 256, 1, 1, 
    //    0, 0, &fillLookupTableArgs[0], NULL ); //mara: changed vector grid= numThreads (we can decrease) 
    cuLaunchKernel( fillLookupTableKernel, cu.getNumAtoms(), 1, 1, 1021, 1, 1, 
        0, 0, &fillLookupTableArgs[0], NULL ); //mara: changed vector grid= numThreads (we can decrease) 
    cuLaunchKernel( sortLookupTableKernel, gridDimlocal[0], gridDimlocal[1], gridDimlocal[2], maxAtomsPerVoxel, 1, 1,
        0, 0, &sortLookupTableArgs[0], NULL );
    
    // calculate Nonpolar term
    if ( sgamma > 0.00001 ) 
    cuLaunchKernel( calcSASAKernel, cu.getNumAtoms(), 1, 1, numAngles, 1, 1,
        0, 0, &calcSASAArgs[0], NULL );
    
    // calculate electrostatic term
    //cuLaunchKernel( calcBornRKernel_hydrogen, cu.getNumAtoms(), 1, 1, numAngles, numRadii_H, 1,
    //    0, 0, &calcBornRArgs[0], NULL );
    //cuLaunchKernel( calcBornRKernel_heavy, cu.getNumAtoms(), 1, 1, numAngles, numRadii_HEAVY, 1,
    //    0, 0, &calcBornRArgs[0], NULL );
    int nthread=256;
    cuLaunchKernel( calcBornRKernel_heavy, cu.getNumAtoms(), 1, 1, nthread, 1, 1,
        0, 0, &calcBornRArgs[0], NULL );
    cu.executeKernel( computeGBMVForceKernel, &computeGBMVForceArgs[0], 
        nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
    cuLaunchKernel( reduceGBMVForceKernel, cu.getNumAtoms(), 1, 1, nthread, 1, 1,
        0, 0, &reduceGBMVForceArgs[0], NULL );
    

    // titrate system using langevin dynamics (if CPHMD active)
    //if (usingCPHMD && doingDynamics) {
    if (usingCPHMD) {
        
        if (usingCutoff)
            cuLaunchKernel( cphmdApplyAtomIndexForcesKernel, cu.getPaddedNumAtoms(), 1, 1, 1, 1, 1,
                0, 0, &cphmdApplyAtomIndexForcesArgs[0], NULL );
        
        timestepCounter ++;
        cuLaunchKernel( cphmdLambdaDynamicsKernel, numTitratingGroups, 1, 1, 1, 1, 1, 
                0, 0, &cphmdLambdaDynamicsArgs[0], NULL );
        
        if (usingCutoff)
            cuLaunchKernel( cphmdApplyAtomIndexChargesKernel, cu.getNumAtoms(), 1, 1, 1, 1, 1,
                0, 0, &cphmdApplyAtomIndexChargesArgs[0], NULL );
        
        
        // record the output if required
        if ( timestepCounter >= outputFrequency ) {
            ntimesteps += timestepCounter;
            timestepCounter = 0;
            cphmdLambdaXtheta->download(LambdaXtheta);
            nchar = sprintf( lineBuffer, "%8u", ntimesteps); // line number
            fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
            for ( int i = 0; i < numTitratingGroups; i++ ) {
                if (writeGroup[i])
                    nchar = sprintf( lineBuffer, "%5.2f%5.2f", LambdaXtheta[i].z, LambdaXtheta[i].w );
                else
                    nchar = sprintf( lineBuffer, "%5.2f", LambdaXtheta[i].z );
                fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
            }
            nchar = sprintf( lineBuffer, "\n" );
            fwrite( lineBuffer, sizeof(char), nchar, cphmdFileID );
        }
    }
//Mara print lookupTable
//FILE *file_lookup_table;
//file_lookup_table = fopen("LookupTable_cuda.dat","w");
//int *lookupTableCPU;
//lookupTableCPU= (int*) malloc(lookupMemory*sizeof(int)); 
//lookupTable -> download(lookupTableCPU);
//
//for ( int k=0;k<gridDimlocal[2];k++ ){
//  for (int i=0;i< gridDimlocal[0];i++ ){
//    for (int j=0;j< gridDimlocal[1];j++ ){
//      int gridLoc = ( i + 
//          j * gridDimlocal[0]+ 
//          k * gridDimlocal[0] * gridDimlocal[1]) * maxAtomsPerVoxel;
//      int numAtomsInGrid = lookupTableCPU[gridLoc];
//
//      if (lookupTableCPU[gridLoc] >= 1) {
//        fprintf(file_lookup_table,"%12d%12d%12d%12d", i, j, k, lookupTableCPU[gridLoc]);
//        for (int n = 1; n <= lookupTableCPU[gridLoc]; n++) {
//          fprintf(file_lookup_table, "%12d", lookupTableCPU[gridLoc+n]+1 );
//
//        }
//        fprintf(file_lookup_table, "\n");
//      }
//    } //i
//    } //j
//  }  //k
//
//  fclose(file_lookup_table);

////++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
////-------------mara
//FILE *file_gb;
//file_gb = fopen("RGB_cuda.dat","w");
//float *bornR_CPU;
//bornR_CPU = (float*) malloc( cu.getPaddedNumAtoms() *sizeof(float)); 
//bornRadii -> download(bornR_CPU);
//for (int i=0; i < cu.getNumAtoms(); i++){
//  fprintf(file_gb, "%f \n", bornR_CPU[i]*10.0f );
//}
//fclose(file_gb);

////-------------mara
//FILE *file_gb;
//file_gb = fopen("Energy_SASA_cuda.dat","w");
//float *Energy_SASA_CPU;
//Energy_SASA_CPU = (float*) malloc( cu.getPaddedNumAtoms() *sizeof(float)); 
//Energy_SASA -> download(Energy_SASA_CPU);
//for (int i=0; i < cu.getNumAtoms(); i++){
//  fprintf(file_gb, "%f \n", Energy_SASA_CPU[i]/4.1840f );
//}
//fclose(file_gb);

//FILE *file_gb;
//file_gb = fopen("Forces_SASA_cuda.dat","w");
//float4 *Forces_SASA_CPU;
//Forces_SASA_CPU = (float4*) malloc( cu.getPaddedNumAtoms() *sizeof(float4)); 
//Forces_SASA -> download(Forces_SASA_CPU);
//for (int i=0; i < cu.getNumAtoms(); i++){
//  fprintf(file_gb, "%f\n %f\n %f\n", Forces_SASA_CPU[i].x*0.0239f, Forces_SASA_CPU[i].y*0.0239f, Forces_SASA_CPU[i].z*0.0239f );
//}
//fclose(file_gb);

////++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
////-----------nGBMVchainAtoms
////FILE *file_gb;
//file_gb = fopen("nGBMVchainAtoms_cuda.dat","w");
//int *nGBMVchainAtoms_CPU;
//nGBMVchainAtoms_CPU = (int*) malloc( cu.getNumAtoms() *sizeof(int)); 
//nGBMVchainAtoms -> download(nGBMVchainAtoms_CPU);
//for (int i=0; i < cu.getNumAtoms(); i++){
//  fprintf(file_gb, "%d \n", nGBMVchainAtoms_CPU[i] );
//}
//fclose(file_gb);

////------------gbmvChain
////FILE *file_gb;
//file_gb = fopen("gbmvChain_cuda.dat","w");
//float4 *gbmvChain_CPU;
//gbmvChain_CPU = (float4*) malloc( cu.getNumAtoms()*(maxSfNeighbors+2) *sizeof(float4));
//gbmvChain -> download(gbmvChain_CPU);
//for (int i=0; i<cu.getNumAtoms()*(maxSfNeighbors+2); i++){
//  fprintf(file_gb,"%f %f %f \n", gbmvChain_CPU[i].x,gbmvChain_CPU[i].y,gbmvChain_CPU[i].z);
//}
//fclose(file_gb);

////------------params
////FILE *file_gb
//file_gb = fopen("params_cuda.dat","w");
//float2 *params_CPU;
//params_CPU = (float2*) malloc(cu.getPaddedNumAtoms()*sizeof(float2));
//params -> download(params_CPU);
//for (int i=0; i<cu.getPaddedNumAtoms(); i++){
//   fprintf(file_gb,"%f %f\n",params_CPU[i].x,params_CPU[i].y);
//}
//fclose(file_gb);  

////------------posq
////FILE *file_gb
//file_gb = fopen("posq_cuda.dat","w");
//      CudaArray& posqTmp = cu.getPosq();
//      vector<double4> tempPos(posqTmp.getSize());
//      float4* posqftmp = (float4*) &tempPos[0];
//      double4* posqdtmp = (double4*) &tempPos[0];
//      posqTmp.download(&tempPos[0]);
//for (int i=0; i<cu.getNumAtoms(); i++){
//   fprintf(file_gb,"%f %f %f\n",posqftmp[i].x*10.f,posqftmp[i].y*10.f,posqftmp[i].z*10.f);
//}
//fclose(file_gb);  

////------------QuadPts
////FILE *file_gb
//file_gb = fopen("QuadPts_cuda.dat","w");
//float4 *QuadPts_CPU;
//QuadPts_CPU = (float4*) malloc(numRadii * numAngles *sizeof(float4));
//QuadPts -> download(QuadPts_CPU);
//for (int i=0; i<numRadii * numAngles; i++){
//   fprintf(file_gb,"%f %f %f %f\n",QuadPts_CPU[i].x,QuadPts_CPU[i].y,QuadPts_CPU[i].z,QuadPts_CPU[i].w);
//}
//fclose(file_gb);  

////------------QuadXYZW
////FILE *file_gb
//file_gb = fopen("QuadXYZW_cuda.dat","w");
//float4 *QuadXYZW_CPU;
//QuadXYZW_CPU = (float4*) malloc(numAngles *sizeof(float4));
//QuadXYZW -> download(QuadXYZW_CPU);
//for (int i=0; i<numAngles; i++){
//   fprintf(file_gb,"%f %f %f %f\n",QuadXYZW_CPU[i].x,QuadXYZW_CPU[i].y,QuadXYZW_CPU[i].z,QuadXYZW_CPU[i].w);
//}
//fclose(file_gb);  

////------------QuadPtWeights_gbmv
////FILE *file_gb
//file_gb = fopen("QuadPtWeights_gbmv_cuda.dat","w");
//float2 *QuadPtWeights_CPU;
//QuadPtWeights_CPU = (float2*) malloc(numRadii * numAngles *sizeof(float2));
//QuadPtWeights_gbmv -> download(QuadPtWeights_CPU);
//for (int i=0; i<numRadii * numAngles; i++){
//   fprintf(file_gb,"%f %f\n",QuadPtWeights_CPU[i].x,QuadPtWeights_CPU[i].y);
//}
//fclose(file_gb);  

////------------lastWtrLE
////FILE *file_gb
//file_gb = fopen("lastWtrLE_cuda.dat","w");
//float *lastWtrLE_CPU;
//lastWtrLE_CPU = (float*) malloc(cu.getPaddedNumAtoms() *sizeof(float));
//lastWtrLE -> download(lastWtrLE_CPU);
//for (int i=0; i<cu.getPaddedNumAtoms(); i++){
//   fprintf(file_gb,"%f\n",lastWtrLE_CPU[i]*10.f);
//}
//fclose(file_gb);  

////
  return 0.0;
}

// return lambda dynamics state
void CudaCalcGBMVForceKernel::getLambdaInfo(ContextImpl& context, std::vector<double>& LambdaPosVelForce) {
    cu.setAsCurrent();
    int numCharmmTitratingGroups = LambdaPosVelForce.size() / 5;
    
    // theta (current)
    cphmdLambdaXtheta->download(LambdaXtheta);
    int loc = 0;
    for ( int i = 0; i < numTitratingGroups; i++ ){
        if (writeGroup[i]) {
            LambdaPosVelForce[loc]   = (double)LambdaXtheta[i].x;
            LambdaPosVelForce[loc+1] = (double)LambdaXtheta[i].y;
            loc += 2;
        } else {
            LambdaPosVelForce[loc]   = (double)LambdaXtheta[i].x;
            loc ++;
        }
    }
    
    // velocity (current)
    cphmdLambdaXvf->download(LambdaXtheta);
    loc = numCharmmTitratingGroups*2;
    for ( int i = 0; i < numTitratingGroups; i++ ){
        if (writeGroup[i]) {
            LambdaPosVelForce[loc]   = (double)LambdaXtheta[i].x;
            LambdaPosVelForce[loc+1] = (double)LambdaXtheta[i].y;
            loc += 2;
        } else {
            LambdaPosVelForce[loc]   = (double)LambdaXtheta[i].x;
            loc ++;
        }
    }
    
    // force (current)
    loc = numCharmmTitratingGroups*4;
    for ( int i = 0; i < numTitratingGroups; i++ ){
        if (writeGroup[i]) {
            LambdaPosVelForce[loc]   = (double)LambdaXtheta[i].z;
            LambdaPosVelForce[loc+1] = (double)LambdaXtheta[i].w;
            loc += 2;
        } else {
            LambdaPosVelForce[loc]   = (double)LambdaXtheta[i].z;
            loc ++;
        }
    }
    
    // theta (previous timestep)
    cphmdLambdaXvOld->download(LambdaXtheta);
    loc = numCharmmTitratingGroups;
    for ( int i = 0; i < numTitratingGroups; i++ ){
        if (writeGroup[i]) {
            LambdaPosVelForce[loc]   = (double)LambdaXtheta[i].x;
            LambdaPosVelForce[loc+1] = (double)LambdaXtheta[i].y;
            loc += 2;
        } else {
            LambdaPosVelForce[loc]   = (double)LambdaXtheta[i].x;
            loc ++;
        }
    }
    
    // velocity (previous timestep)
    loc = numCharmmTitratingGroups*3;
    for ( int i = 0; i < numTitratingGroups; i++ ){
        if (writeGroup[i]) {
            LambdaPosVelForce[loc]   = (double)LambdaXtheta[i].z;
            LambdaPosVelForce[loc+1] = (double)LambdaXtheta[i].w;
            loc += 2;
        } else {
            LambdaPosVelForce[loc]   = (double)LambdaXtheta[i].z;
            loc ++;
        }
    }
}

// set lambda dynamics state
void CudaCalcGBMVForceKernel::setLambdaInfo(ContextImpl& context, std::vector<double>& LambdaPosVelForce) {
    cu.setAsCurrent();
    int numCharmmTitratingGroups = LambdaPosVelForce.size() / 5;
    int loc, i;
    float x, y, z, w;
    // theta (current)
    loc = 0; i = 0;
    while ( i < numCharmmTitratingGroups ){
        if (writeGroup[loc]) {
            x = (float)LambdaPosVelForce[i];
            y = (float)LambdaPosVelForce[i+1];
            z = sin(x);
            w = sin(y);
            if (loc > numTitratingGroups)
                throw OpenMMException("GBMVForce CPHMD input doesn't match current setup");
            LambdaXtheta[loc] = make_float4(x, y, z*z, w*w);
            loc ++; i += 2;
        } else {
            x = (float)LambdaPosVelForce[i];
            z = sin(x);
            if (loc > numTitratingGroups)
                throw OpenMMException("GBMVForce CPHMD input doesn't match current setup");
            LambdaXtheta[loc] = make_float4(x, 0.25*M_PI, z*z, 0.5);
            loc ++; i ++;
        }
    }
    cphmdLambdaXtheta->upload(LambdaXtheta);
    
    
    // velocity (current) and force (current)
    loc = 0; i = numCharmmTitratingGroups*2;
    while ( i < numCharmmTitratingGroups*3 ){
        if (writeGroup[loc]) {
            x = (float)LambdaPosVelForce[i];
            y = (float)LambdaPosVelForce[i+1];
            z = (float)LambdaPosVelForce[i + numCharmmTitratingGroups*2];
            w = (float)LambdaPosVelForce[i+1 + numCharmmTitratingGroups*2];
            LambdaXtheta[loc] = make_float4(x, y, z, w);
            loc ++; i += 2;
        } else {
            x = (float)LambdaPosVelForce[i];
            z = (float)LambdaPosVelForce[i + numCharmmTitratingGroups*2];
            LambdaXtheta[loc] = make_float4(x, 0.0f, z*z, 0.0f);
            loc ++; i ++;
        }
    }
    cphmdLambdaXvf->upload(LambdaXtheta);
    
    // theta (previous timestep) and velocity (previous timestep)
    loc = 0; i = numCharmmTitratingGroups;
    while ( i < numCharmmTitratingGroups*2 ){
        if (writeGroup[loc]) {
            x = (float)LambdaPosVelForce[i];
            y = (float)LambdaPosVelForce[i+1];
            z = (float)LambdaPosVelForce[i + numCharmmTitratingGroups*2];
            w = (float)LambdaPosVelForce[i+1 + numCharmmTitratingGroups*2];
            LambdaXtheta[loc] = make_float4(x, y, z, w);
            loc ++; i += 2;
        } else {
            x = (float)LambdaPosVelForce[i];
            z = (float)LambdaPosVelForce[i + numCharmmTitratingGroups*2];
            LambdaXtheta[loc] = make_float4(x, 0.0f, z*z, 0.0f);
            loc ++; i ++;
        }
    }
    cphmdLambdaXvOld->upload(LambdaXtheta);
}

void CudaCalcGBMVForceKernel::copyParametersToContext(ContextImpl& context, const GBMVForce& force) {
    // Make sure the new parameters are acceptable.
    
    cu.setAsCurrent();
    int numParticles = force.getNumParticles();
    if (numParticles != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    
    // Record the per-particle parameters.
    
    CudaArray& posq = cu.getPosq();
    float4* posqf = (float4*) cu.getPinnedBuffer();
    double4* posqd = (double4*) cu.getPinnedBuffer();
    posq.download(cu.getPinnedBuffer());
    vector<float2> paramsVector(cu.getPaddedNumAtoms(), make_float2(1, 1));
    const double swLenTmp = 0.03;
    const double PBradiusModTmp = 0.9520;
    for (int i = 0; i < numParticles; i++) {
        
        double charge, radius;
        force.getParticleParameters(i, charge, radius);
        
      //// modulate atomic radii by the switching length and a PB scaling factor
      //if (radius != 0.0)
      //    radius = (radius + swLenTmp) * PBradiusModTmp;
        
        paramsVector[i] = make_float2((float) radius, (float) ((radius + swLen) * (radius + swLen)));
        if (cu.getUseDoublePrecision())
            posqd[i].w = charge;
        else
            posqf[i].w = (float) charge;
    }
    posq.upload(cu.getPinnedBuffer());
    params->upload(paramsVector);
    
    // Mark that the current reordering may be invalid.
    
    cu.invalidateMolecules();
}

