#define PROBE_RADIUS 0.14f

__device__ float4 minXYZ, maxXYZ;
__device__ real4  TranslateXYZ;
__device__ int4 GridDimXYZ; // we can't touch "gridDim". this is a CUDA kernel variable
__device__ int randomCounter = 2;


#ifdef USE_CPHMD

/**
 * Generate random numbers (code from nonbonded utilities). these random numbers
 * follow a unit-width Gaussian distribution
 */
__device__ void cphmdGenerateRandomNumbers(float4& __restrict__ random, uint4& __restrict__ seed) {
    
    // download seed locally
    uint4 state = seed;
    
    // initialize random number calculation
    unsigned int carry = 0;
    float4 value;
    
    // Generate first two values.
    state.x = state.x * 69069 + 1;
    state.y ^= state.y << 13;
    state.y ^= state.y >> 17;
    state.y ^= state.y << 5;
    unsigned int k = (state.z >> 2) + (state.w >> 3) + (carry >> 2);
    unsigned int m = state.w + state.w + state.z + carry;
    state.z = state.w;
    state.w = m;
    carry = k >> 30;
    float x1 = (float)max(state.x + state.y + state.w, 0x00000001u) / (float)0xffffffff;
    state.x = state.x * 69069 + 1;
    state.y ^= state.y << 13;
    state.y ^= state.y >> 17;
    state.y ^= state.y << 5;
    x1 = SQRT(-2.0f * LOG(x1));
    k = (state.z >> 2) + (state.w >> 3) + (carry >> 2);
    m = state.w + state.w + state.z + carry;
    state.z = state.w;
    state.w = m;
    carry = k >> 30;
    float x2 = (float)(state.x + state.y + state.w) / (float)0xffffffff;
    value.x = x1 * COS(2.0f * 3.14159265f * x2);
    value.y = x1 * SIN(2.0f * 3.14159265f * x2);
    
    // Generate next two values.
    state.x = state.x * 69069 + 1;
    state.y ^= state.y << 13;
    state.y ^= state.y >> 17;
    state.y ^= state.y << 5;
    k = (state.z >> 2) + (state.w >> 3) + (carry >> 2);
    m = state.w + state.w + state.z + carry;
    state.z = state.w;
    state.w = m;
    carry = k >> 30;
    float x3 = (float)max(state.x + state.y + state.w, 0x00000001u) / (float)0xffffffff;
    state.x = state.x * 69069 + 1;
    state.y ^= state.y << 13;
    state.y ^= state.y >> 17;
    state.y ^= state.y << 5;
    x3 = SQRT(-2.0f * LOG(x3));
    k = (state.z >> 2) + (state.w >> 3) + (carry >> 2);
    m = state.w + state.w + state.z + carry;
    state.z = state.w;
    state.w = m;
    carry = k >> 30;
    float x4 = (float)(state.x + state.y + state.w) / (float)0xffffffff;
    value.z = x3 * COS(2.0f * 3.14159265f * x4);
    value.w = x3 * SIN(2.0f * 3.14159265f * x4);
    
    // output the random number, and seed the next one
    seed = state;
    random = value;
}
#endif

/*
 * update arrays according to atom index to account for atom reordering
 */
#ifdef USE_CPHMD
#ifdef USE_CPHMD_CUTOFF

__device__ float4 cphmdAtomQfac_tmp[NUM_ATOMS];
__device__ float2 cphmdForce_tmp[PADDED_NUM_ATOMS];

extern "C" __global__ void cphmdApplyAtomIndex_Charges( 
    int* __restrict__ atomIndex, real4* __restrict__ posq, float2* __restrict__ cphmdAtomQfac ) {
    
    int i = blockIdx.x;
    int j = atomIndex[blockIdx.x];
    
    // check if this is a titrating atom
    float4 tmp = cphmdAtomQfac_tmp[j];
    if ( tmp.w == -10.0f ) {
        
        // charge
        real4 pos = posq[i];
        posq[i] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) tmp.z);
        
        // charge derivatives with respect to lambda
        cphmdAtomQfac[i] = make_float2( tmp.x, tmp.y );
    }
    cphmdAtomQfac_tmp[j] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
}


extern "C" __global__ void cphmdApplyAtomIndex_Forces( 
    int* __restrict__ atomIndex, float* __restrict__ cphmdForce ){
    
    int i = blockIdx.x;
    int j = atomIndex[blockIdx.x];
    
    cphmdForce_tmp[j] = make_float2(cphmdForce[i], cphmdForce[i+PADDED_NUM_ATOMS]);
    cphmdForce[i] = 0.0f;
    cphmdForce[i+PADDED_NUM_ATOMS] = 0.0f;
}
#endif
#endif

/*
 * calculate Langevin dynamics (CHARMM style) to propagate pH-dependent lambda
 */

#ifdef USE_CPHMD
extern "C" __global__ void cphmdLambdaDynamics( 
    real4* __restrict__ posq, uint4* __restrict__ cphmdRandSeed, 
    float4* __restrict__ randomNumber, float4* __restrict__ lambdaXtheta, 
    float* __restrict__ cphmdForces, float4* __restrict__ lambdaXvelForce, 
    float4* __restrict__ lambdaXvelOld, int4* __restrict__ atomRanges, 
    float4* __restrict__ cphmdUphUbarr, float4* __restrict__ cphmdUmod0123, 
    float4* __restrict__ cphmdUmod4567, float2* __restrict__ cphmdAtomQfac, 
    const float4* __restrict__ chargeStates, mixed* __restrict__ energyBuffer ) {
    
    int randCount = randomCounter;
    float randForceL, randForceX;
    float4 RandNumSet;
    float4 oldVelForce = lambdaXvelForce[blockIdx.x];
    float4 oldTheta = lambdaXtheta[blockIdx.x];
    
    //--------------------------------------------------------------------------
    // random numbers
    
    // these random numbers come in batches of 4. calculate them as needed
    if (randCount < 2) {
        RandNumSet = randomNumber[blockIdx.x];
        randForceL = SCALE_RANDOM_FORCE * RandNumSet.z;
        randForceX = SCALE_RANDOM_FORCE * RandNumSet.w;
        randomCounter = 2;
    } else {
        uint4 seed = cphmdRandSeed[blockIdx.x];
        cphmdGenerateRandomNumbers(RandNumSet, seed);
        randForceL = SCALE_RANDOM_FORCE * RandNumSet.x;
        randForceX = SCALE_RANDOM_FORCE * RandNumSet.y;
        randomNumber[blockIdx.x] = RandNumSet;
        cphmdRandSeed[blockIdx.x] = seed;
        randomCounter = 1;
    }
    
    //--------------------------------------------------------------------------
    // force contributions
    
    // sum up all forces on atom range for theta
    int4 atomrange = atomRanges[blockIdx.x];
    float thetaForceLambda = 0.0f, thetaForceX = 0.0f;
    
#ifdef USE_CPHMD_CUTOFF
    float2 tmp;
    for ( int i = atomrange.x; i <= atomrange.y; i++ ) {
        tmp = cphmdForce_tmp[i];
        thetaForceLambda += tmp.x;
        thetaForceX      += tmp.y;
        cphmdForce_tmp[i] = make_float2( 0.0f, 0.0f );
    }
#else
    for ( int i = atomrange.x; i <= atomrange.y; i++ ) {
        thetaForceLambda += cphmdForces[i];
        thetaForceX      += cphmdForces[i+PADDED_NUM_ATOMS];
        cphmdForces[i] = 0.0f;
        cphmdForces[i+PADDED_NUM_ATOMS] = 0.0f;
    }
#endif
    
    //  L = titration lambda-coodrinate, X = tautomeric x-coordinate
    //  model potential Umod here uses the general-case eq. 17 from the paper
    //  'Constant pH Molecular Dynamics with Proton Tautomerism'
    //  This equation follows the following form:  
    //  
    //   Umod = a0(L^2)(X^2) + a1(L^2)(X) + a2(L)(X^2) + a3(L)(X) +
    //          a4(L^2)      + a5(X^2)    + a6(L)      + a7(X)    + a8
    //    
    //  and the forces are
    //  
    //   dUmod / dL = a0(2*L)(X^2) + a1(2*L)(X) + a2(X^2) + a3(X) + a4(2*L) + a6
    //   dUmod / dX = a0(L^2)(2*X) + a1(L^2) + a2(L)(2*X) + a3(L) + a5(2*X) + a7
    
    float4 UphUbarr = cphmdUphUbarr[blockIdx.x]; // pH potentials and barriers
    float4 Umod0123 = cphmdUmod0123[blockIdx.x]; // a0, a1, a2, a3
    float4 Umod4567 = cphmdUmod4567[blockIdx.x]; // a4, a5, a6, a7
    float a8; memcpy(&a8, &atomrange.z, 4);      // a8
    
    float L = oldTheta.z;
    float X = oldTheta.w;
    float LX = L*X;
    float X2 = X*X;
    float L2 = L*L;
    
    float dUmodL = Umod0123.x*2.0f*LX*X + Umod0123.y*2.0f*LX + 
                   Umod0123.z*X2 + Umod0123.w*X + Umod4567.x*2.0f*L + Umod4567.z;
    float dUmodX = Umod0123.x*2.0f*L*LX + Umod0123.y*L2 + 
                   Umod0123.z*2.0f*LX + Umod0123.w*L + Umod4567.y*2.0f*X + Umod4567.w;
    
    float dUbarrL = 8.0f * UphUbarr.w * (L - 0.5); // 8 * barrX * (L - 1/2)
    float dUbarrX = 8.0f * UphUbarr.z * (X - 0.5); // 8 * barrL * (X - 1/2)
    
    float dUphL = UphUbarr.y*X + UphUbarr.x*(1.0f - X); // pkL*X + pkX*(1.0f - X)
    float dUphX = (UphUbarr.y - UphUbarr.x) * L;       // (pkL - pkX) * L
    
    // pull all the forces together
    thetaForceLambda *= COULOMBIC_FAC;                  // scale to kcal/mol
    thetaForceLambda +=  -dUmodL - dUbarrL + dUphL;     // barrier functions
    thetaForceLambda *= sin(oldTheta.x * 2.0f);  // partial-derivative lambda by theta
    thetaForceLambda += randForceL;                     // random forces
    
    if (UphUbarr.z != 0.0f) {
        thetaForceX *= COULOMBIC_FAC;
        thetaForceX += -dUmodX - dUbarrX + dUphX;
        thetaForceX *= sin(oldTheta.y * 2.0f);
        thetaForceX += randForceX;
    } else
        thetaForceX = 0.0f;
    
    //--------------------------------------------------------------------------
    // pH energy contribution
    
    real UpH = L * (X * UphUbarr.y + (1.0f - X) * UphUbarr.x);
    real Ubarrier = 4.0f * UphUbarr.w * (L - 0.5) * (L - 0.5) +
                    4.0f * UphUbarr.z * (X - 0.5) * (X - 0.5);
    real Umodel = Umod0123.x*L2*X2 + Umod0123.y*L2*X + Umod0123.z*L*X2 + Umod0123.w*L*X + 
                  Umod4567.x*L2 +    Umod4567.y*X2 +   Umod4567.z*L +    Umod4567.w*X + a8;
    
    energyBuffer[blockIdx.x] += (UpH - Ubarrier - Umodel) * 4.184f;
    
    //--------------------------------------------------------------------------
    // lambda dynamics
    
    // lambda
    float newVelLambda = ONE_MINUS_GAMMA * (oldVelForce.x - MASS_TIME_FACTOR*(thetaForceLambda + oldVelForce.z));
    float newThetaLambda = oldTheta.x + newVelLambda*TIME_FACTOR - MASS_TIME_FACTOR*TIME_FACTOR*thetaForceLambda;
    float newLambda = sin(newThetaLambda);
    newLambda *= newLambda;
    
    // X
    float newVelX = 0.0f, newThetaX = 0.0f, newX = 0.0f;
    if (UphUbarr.z != 0.0f) {
        newVelX = ONE_MINUS_GAMMA * (oldVelForce.y - MASS_TIME_FACTOR*(thetaForceX + oldVelForce.w));
        newThetaX = oldTheta.y + newVelX*TIME_FACTOR - MASS_TIME_FACTOR*TIME_FACTOR*thetaForceX;
        newX = sin(newThetaX);
        newX *= newX; 
    }
    
    // save old forces
    lambdaXtheta[blockIdx.x] = make_float4(newThetaLambda, newThetaX, newLambda, newX);
    lambdaXvelForce[blockIdx.x] = make_float4(newVelLambda, newVelX, thetaForceLambda, thetaForceX);
    lambdaXvelOld[blockIdx.x] = make_float4(oldTheta.x, oldTheta.y, oldVelForce.x, oldVelForce.y);
    
    //--------------------------------------------------------------------------
    // generate new set of charges, calculate dQ / dLambda, and dQ / dX
    
    
    // input all charge states, atom ranges, posq of atoms
    float newdQdLambda, newdQdX, newQ;
    float4 qstate;
    
    for ( int i = atomrange.x; i <= atomrange.y; i++ ) {
        qstate = chargeStates[i];
        
        newdQdX =       newLambda *(qstate.z - qstate.w) +
                (1.0f - newLambda)*(qstate.x - qstate.y);
        
        newdQdLambda =  newX*qstate.z + (1.0f-newX)*qstate.w -
                        newX*qstate.x - (1.0f-newX)*qstate.y;
        
        newQ = (1.0f-newLambda)*(newX*qstate.x + (1.0f-newX)*qstate.y) +
                     newLambda *(newX*qstate.z + (1.0f-newX)*qstate.w);
        
#ifdef USE_CPHMD_CUTOFF
        cphmdAtomQfac_tmp[i] = make_float4(newdQdLambda, newdQdX, newQ, -10.0f);
#else
        // save new charge
        real4 pos = posq[i];
        posq[i] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) newQ);
        
        // save charge derivatives with respect to lambda
        cphmdAtomQfac[i] = make_float2(newdQdLambda, newdQdX);
#endif
    }
}
#endif


/*
 * calculate the dimensions of the system (only used for non-periodic boundaries)
 */

extern "C" __global__ void calcSysExtrema(
    // inputs
    const real4* __restrict__ posq, const float2* __restrict__ params,
    // outputs
    int* GridDimXYZ_external
    ) {
    
    // global matrix to hold all extrema
    __shared__ float allExtrema[NUM_THREADS * 7], tmpExtrema[7];
    
    int atomInt, numGridsX, numGridsY, numGridsZ, baseLocation;
    real maxX, maxY, maxZ, minX, minY, minZ,
        atomRadius, atomX, atomY, atomZ,Rmax,
        thisExtreme;
    real4 posq_XYZQ;
    
    // initialize gridspace extrema variables
    maxX = -99999.0f; maxY = -99999.0f; maxZ = -99999.0f; // max floats
    minX =  99999.0f; minY =  99999.0f; minZ =  99999.0f; // min floats
    Rmax = -99999.0f;
    // each thread takes a few atoms and check amongst those few the maxima
    for (atomInt = threadIdx.x; atomInt < NUM_ATOMS; atomInt += NUM_THREADS) {
        
        // this atom's info
        atomRadius = params[atomInt].x;
        posq_XYZQ = posq[atomInt];
        
        atomX = posq_XYZQ.x;
        atomY = posq_XYZQ.y;
        atomZ = posq_XYZQ.z;
        
        // check for maximum (but only if no periodic bounds)
        if (atomX > maxX)  maxX = atomX;
        if (atomY > maxY)  maxY = atomY;
        if (atomZ > maxZ)  maxZ = atomZ;
        
        // check for minimum
        if (atomX  < minX)  minX = atomX;
        if (atomY  < minY)  minY = atomY;
        if (atomZ  < minZ)  minZ = atomZ;

        // mara: check for max input atom radius
        if (atomRadius > Rmax) {
          Rmax = atomRadius;
        }
    }
    
    // funnel results into global matrix
    allExtrema[threadIdx.x]                   = minX;
    allExtrema[threadIdx.x + 1 * NUM_THREADS] = minY;
    allExtrema[threadIdx.x + 2 * NUM_THREADS] = minZ;
    allExtrema[threadIdx.x + 3 * NUM_THREADS] = maxX;
    allExtrema[threadIdx.x + 4 * NUM_THREADS] = maxY;
    allExtrema[threadIdx.x + 5 * NUM_THREADS] = maxZ;
    allExtrema[threadIdx.x + 6 * NUM_THREADS] = Rmax;
    
    __syncthreads();
    
    // now run through the results of each thread
    if (threadIdx.x < 3) { // minima
        
        // initialize extreme
        thisExtreme = 99999.0f; 
        baseLocation = threadIdx.x * NUM_THREADS;
        
        // find the extreme for this thread
        for (atomInt = 0; atomInt < NUM_THREADS; atomInt++)
            if (thisExtreme > allExtrema[baseLocation + atomInt])
                thisExtreme = allExtrema[baseLocation + atomInt];
        
        tmpExtrema[threadIdx.x] = thisExtreme;
        
    } else if (threadIdx.x < 6) { // maxima
        
        // initialize extreme
        thisExtreme = -99999.0f; 
        baseLocation = threadIdx.x * NUM_THREADS;
        
        // find the extreme for this thread
        for (atomInt = 0; atomInt < NUM_THREADS; atomInt++)
            if (thisExtreme < allExtrema[baseLocation + atomInt])
                thisExtreme = allExtrema[baseLocation + atomInt];
        
        tmpExtrema[threadIdx.x] = thisExtreme;
 
    }  else if (threadIdx.x < 7) { // Rmax
        // initialize extreme
        thisExtreme = -99999.0f; 
        baseLocation = threadIdx.x * NUM_THREADS;
        for (atomInt = 0; atomInt < NUM_THREADS; atomInt++){
          if (thisExtreme < allExtrema[baseLocation + atomInt])
            thisExtreme = allExtrema[baseLocation + atomInt];
        }
        tmpExtrema[threadIdx.x] = thisExtreme;
    }
    
    // be sure all threads have finished before continuing, then calc grid
    // dimensions and the translation vector for each atom
    __syncthreads();
    
   if (threadIdx.x == 0) { // x dim
        Rmax = tmpExtrema[6]+R_BUFFER+DELTA_R;
        minX = tmpExtrema[0]-Rmax;
        maxX = tmpExtrema[3]+Rmax;
        numGridsX = floor((maxX - minX ) * INVERSE_DELTA_R)+1;
        GridDimXYZ_external[0] = numGridsX;
        GridDimXYZ.x = numGridsX;
        TranslateXYZ.x = 0.5f * ((((maxX - minX) * INVERSE_DELTA_R) +1.0f) -1.0f) * DELTA_R - (maxX + minX) * 0.5f;
        
    } else if (threadIdx.x == 1) { // y dim
        Rmax = tmpExtrema[6]+ R_BUFFER+DELTA_R;
        minY = tmpExtrema[1]-(Rmax);
        maxY = tmpExtrema[4]+(Rmax);
        numGridsY = floor((maxY - minY ) * INVERSE_DELTA_R)+1;
        GridDimXYZ_external[1] = numGridsY;
        GridDimXYZ.y = numGridsY;
        TranslateXYZ.y = 0.5f * ((((maxY - minY ) * INVERSE_DELTA_R)+1.0f)-1.0f) * DELTA_R - (maxY + minY) * 0.5f;
        
    } else if (threadIdx.x == 2) { // z dim
        Rmax = tmpExtrema[6]+R_BUFFER+DELTA_R;
        minZ = tmpExtrema[2]-(Rmax);
        maxZ = tmpExtrema[5]+(Rmax);
        numGridsZ = floor((maxZ - minZ ) * INVERSE_DELTA_R)+1;
        GridDimXYZ_external[2] = numGridsZ;
        GridDimXYZ.z = numGridsZ;
        TranslateXYZ.z = 0.5f * ((((maxZ - minZ ) * INVERSE_DELTA_R)+1.0f)-1.0f) * DELTA_R - (maxZ + minZ) * 0.5f;
        
    } else if (threadIdx.x == 3) { 
        Rmax = tmpExtrema[6]+R_BUFFER+DELTA_R;
        minXYZ = make_float4( tmpExtrema[0]-Rmax, tmpExtrema[1]-Rmax, tmpExtrema[2]-Rmax, 0.0f );
    } else if (threadIdx.x == 4) { 
        Rmax = tmpExtrema[6]+R_BUFFER+DELTA_R;
        maxXYZ = make_float4( tmpExtrema[3]+Rmax, tmpExtrema[4]+Rmax, tmpExtrema[5]+Rmax, 0.0f );
    }
}

/*
 * resetting the lookup needs its own kernel, since we only need to reset one element
 */


// reset the lookup table!
extern "C" __global__ void resetLookupTable( int *lookupTable ) {
    
    // location of this voxel in the lookup array
    const int gridLoc = ( blockIdx.x + 
                                   blockIdx.y * gridDim.x + 
                                   blockIdx.z * gridDim.x * gridDim.y ) * MAX_ATOMS_IN_VOXEL;
    
    // set the number of atoms to zero!
    lookupTable[gridLoc] = 0;
}


/*
 * to speed things up we pre-calculate all grids that need to be checked around
 * any give atom. these are voxel-vectors with a rounded radius
 */

typedef signed char int8_t;



//typedef struct {
//    int8_t  x, y, z, r;
//} gridVector;
//__device__ gridVector gridVects [123] = {
//        { -1, -1, -1,  1 },  { -1, -1,  0,  1 },  { -1, -1,  1,  1 },  {  0, -1, -1,  1 },
//        {  0, -1,  0,  1 },  {  0, -1,  1,  1 },  {  1, -1, -1,  1 },  {  1, -1,  0,  1 },
//        {  1, -1,  1,  1 },  { -1,  0, -1,  1 },  { -1,  0,  0,  1 },  { -1,  0,  1,  1 },
//        {  0,  0, -1,  1 },  {  0,  0,  0,  1 },  {  0,  0,  1,  1 },  {  1,  0, -1,  1 },
//        {  1,  0,  0,  1 },  {  1,  0,  1,  1 },  { -1,  1, -1,  1 },  { -1,  1,  0,  1 },
//        { -1,  1,  1,  1 },  {  0,  1, -1,  1 },  {  0,  1,  0,  1 },  {  0,  1,  1,  1 },
//        {  1,  1, -1,  1 },  {  1,  1,  0,  1 },  {  1,  1,  1,  1 },  { -2, -2,  0,  2 },
//        { -1, -2, -1,  2 },  { -1, -2,  0,  2 },  { -1, -2,  1,  2 },  {  0, -2, -2,  2 },
//        {  0, -2, -1,  2 },  {  0, -2,  0,  2 },  {  0, -2,  1,  2 },  {  0, -2,  2,  2 },
//        {  1, -2, -1,  2 },  {  1, -2,  0,  2 },  {  1, -2,  1,  2 },  {  2, -2,  0,  2 },
//        { -2, -1, -1,  2 },  { -2, -1,  0,  2 },  { -2, -1,  1,  2 },  { -1, -1, -2,  2 },
//        { -1, -1,  2,  2 },  {  0, -1, -2,  2 },  {  0, -1,  2,  2 },  {  1, -1, -2,  2 },
//        {  1, -1,  2,  2 },  {  2, -1, -1,  2 },  {  2, -1,  0,  2 },  {  2, -1,  1,  2 },
//        { -2,  0, -2,  2 },  { -2,  0, -1,  2 },  { -2,  0,  0,  2 },  { -2,  0,  1,  2 },
//        { -2,  0,  2,  2 },  { -1,  0, -2,  2 },  { -1,  0,  2,  2 },  {  0,  0, -2,  2 },
//        {  0,  0,  2,  2 },  {  1,  0, -2,  2 },  {  1,  0,  2,  2 },  {  2,  0, -2,  2 },
//        {  2,  0, -1,  2 },  {  2,  0,  0,  2 },  {  2,  0,  1,  2 },  {  2,  0,  2,  2 },
//        { -2,  1, -1,  2 },  { -2,  1,  0,  2 },  { -2,  1,  1,  2 },  { -1,  1, -2,  2 },
//        { -1,  1,  2,  2 },  {  0,  1, -2,  2 },  {  0,  1,  2,  2 },  {  1,  1, -2,  2 },
//        {  1,  1,  2,  2 },  {  2,  1, -1,  2 },  {  2,  1,  0,  2 },  {  2,  1,  1,  2 },
//        { -2,  2,  0,  2 },  { -1,  2, -1,  2 },  { -1,  2,  0,  2 },  { -1,  2,  1,  2 },
//        {  0,  2, -2,  2 },  {  0,  2, -1,  2 },  {  0,  2,  0,  2 },  {  0,  2,  1,  2 },
//        {  0,  2,  2,  2 },  {  1,  2, -1,  2 },  {  1,  2,  0,  2 },  {  1,  2,  1,  2 },
//        {  2,  2,  0,  2 },  {  0, -3,  0,  3 },  { -2, -2, -1,  3 },  { -2, -2,  1,  3 },
//        { -1, -2, -2,  3 },  { -1, -2,  2,  3 },  {  1, -2, -2,  3 },  {  1, -2,  2,  3 },
//        {  2, -2, -1,  3 },  {  2, -2,  1,  3 },  { -2, -1, -2,  3 },  { -2, -1,  2,  3 },
//        {  2, -1, -2,  3 },  {  2, -1,  2,  3 },  { -3,  0,  0,  3 },  {  0,  0, -3,  3 },
//        {  0,  0,  3,  3 },  {  3,  0,  0,  3 },  { -2,  1, -2,  3 },  { -2,  1,  2,  3 },
//        {  2,  1, -2,  3 },  {  2,  1,  2,  3 },  { -2,  2, -1,  3 },  { -2,  2,  1,  3 },
//        { -1,  2, -2,  3 },  { -1,  2,  2,  3 },  {  1,  2, -2,  3 },  {  1,  2,  2,  3 },
//        {  2,  2, -1,  3 },  {  2,  2,  1,  3 },  {  0,  3,  0,  3 } };

// this list of grid vectors is more accurate, but slows down things. the lower-
// resolution vectors (above) deviate from the reference energy by less than 0.5%
/*
typedef struct {
    int8_t  x, y, z, r;
} gridVector;
__device__ gridVector gridVects [251] = {
        { -2, -3, -1,  3 }, { -2, -3,  0,  3 }, { -2, -3,  1,  3 }, { -1, -3, -2,  3 }, 
        { -1, -3, -1,  3 }, { -1, -3,  0,  3 }, { -1, -3,  1,  3 }, { -1, -3,  2,  3 }, 
        {  0, -3, -2,  3 }, {  0, -3, -1,  3 }, {  0, -3,  0,  3 }, {  0, -3,  1,  3 }, 
        {  0, -3,  2,  3 }, {  1, -3, -2,  3 }, {  1, -3, -1,  3 }, {  1, -3,  0,  3 }, 
        {  1, -3,  1,  3 }, {  1, -3,  2,  3 }, {  2, -3, -1,  3 }, {  2, -3,  0,  3 }, 
        {  2, -3,  1,  3 }, { -3, -2, -1,  3 }, { -3, -2,  0,  3 }, { -3, -2,  1,  3 }, 
        { -2, -2, -2,  3 }, { -2, -2, -1,  3 }, { -2, -2,  0,  2 }, { -2, -2,  1,  3 }, 
        { -2, -2,  2,  3 }, { -1, -2, -3,  3 }, { -1, -2, -2,  3 }, { -1, -2, -1,  2 }, 
        { -1, -2,  0,  2 }, { -1, -2,  1,  2 }, { -1, -2,  2,  3 }, { -1, -2,  3,  3 }, 
        {  0, -2, -3,  3 }, {  0, -2, -2,  2 }, {  0, -2, -1,  2 }, {  0, -2,  0,  2 }, 
        {  0, -2,  1,  2 }, {  0, -2,  2,  2 }, {  0, -2,  3,  3 }, {  1, -2, -3,  3 }, 
        {  1, -2, -2,  3 }, {  1, -2, -1,  2 }, {  1, -2,  0,  2 }, {  1, -2,  1,  2 }, 
        {  1, -2,  2,  3 }, {  1, -2,  3,  3 }, {  2, -2, -2,  3 }, {  2, -2, -1,  3 }, 
        {  2, -2,  0,  2 }, {  2, -2,  1,  3 }, {  2, -2,  2,  3 }, {  3, -2, -1,  3 }, 
        {  3, -2,  0,  3 }, {  3, -2,  1,  3 }, { -3, -1, -2,  3 }, { -3, -1, -1,  3 }, 
        { -3, -1,  0,  3 }, { -3, -1,  1,  3 }, { -3, -1,  2,  3 }, { -2, -1, -3,  3 }, 
        { -2, -1, -2,  3 }, { -2, -1, -1,  2 }, { -2, -1,  0,  2 }, { -2, -1,  1,  2 }, 
        { -2, -1,  2,  3 }, { -2, -1,  3,  3 }, { -1, -1, -3,  3 }, { -1, -1, -2,  2 }, 
        { -1, -1, -1,  1 }, { -1, -1,  0,  1 }, { -1, -1,  1,  1 }, { -1, -1,  2,  2 }, 
        { -1, -1,  3,  3 }, {  0, -1, -3,  3 }, {  0, -1, -2,  2 }, {  0, -1, -1,  1 }, 
        {  0, -1,  0,  1 }, {  0, -1,  1,  1 }, {  0, -1,  2,  2 }, {  0, -1,  3,  3 }, 
        {  1, -1, -3,  3 }, {  1, -1, -2,  2 }, {  1, -1, -1,  1 }, {  1, -1,  0,  1 }, 
        {  1, -1,  1,  1 }, {  1, -1,  2,  2 }, {  1, -1,  3,  3 }, {  2, -1, -3,  3 }, 
        {  2, -1, -2,  3 }, {  2, -1, -1,  2 }, {  2, -1,  0,  2 }, {  2, -1,  1,  2 }, 
        {  2, -1,  2,  3 }, {  2, -1,  3,  3 }, {  3, -1, -2,  3 }, {  3, -1, -1,  3 }, 
        {  3, -1,  0,  3 }, {  3, -1,  1,  3 }, {  3, -1,  2,  3 }, { -3,  0, -2,  3 }, 
        { -3,  0, -1,  3 }, { -3,  0,  0,  3 }, { -3,  0,  1,  3 }, { -3,  0,  2,  3 }, 
        { -2,  0, -3,  3 }, { -2,  0, -2,  2 }, { -2,  0, -1,  2 }, { -2,  0,  0,  2 }, 
        { -2,  0,  1,  2 }, { -2,  0,  2,  2 }, { -2,  0,  3,  3 }, { -1,  0, -3,  3 }, 
        { -1,  0, -2,  2 }, { -1,  0, -1,  1 }, { -1,  0,  0,  1 }, { -1,  0,  1,  1 }, 
        { -1,  0,  2,  2 }, { -1,  0,  3,  3 }, {  0,  0, -3,  3 }, {  0,  0, -2,  2 }, 
        {  0,  0, -1,  1 }, {  0,  0,  0,  1 }, {  0,  0,  1,  1 }, {  0,  0,  2,  2 }, 
        {  0,  0,  3,  3 }, {  1,  0, -3,  3 }, {  1,  0, -2,  2 }, {  1,  0, -1,  1 }, 
        {  1,  0,  0,  1 }, {  1,  0,  1,  1 }, {  1,  0,  2,  2 }, {  1,  0,  3,  3 }, 
        {  2,  0, -3,  3 }, {  2,  0, -2,  2 }, {  2,  0, -1,  2 }, {  2,  0,  0,  2 }, 
        {  2,  0,  1,  2 }, {  2,  0,  2,  2 }, {  2,  0,  3,  3 }, {  3,  0, -2,  3 }, 
        {  3,  0, -1,  3 }, {  3,  0,  0,  3 }, {  3,  0,  1,  3 }, {  3,  0,  2,  3 }, 
        { -3,  1, -2,  3 }, { -3,  1, -1,  3 }, { -3,  1,  0,  3 }, { -3,  1,  1,  3 }, 
        { -3,  1,  2,  3 }, { -2,  1, -3,  3 }, { -2,  1, -2,  3 }, { -2,  1, -1,  2 }, 
        { -2,  1,  0,  2 }, { -2,  1,  1,  2 }, { -2,  1,  2,  3 }, { -2,  1,  3,  3 }, 
        { -1,  1, -3,  3 }, { -1,  1, -2,  2 }, { -1,  1, -1,  1 }, { -1,  1,  0,  1 }, 
        { -1,  1,  1,  1 }, { -1,  1,  2,  2 }, { -1,  1,  3,  3 }, {  0,  1, -3,  3 }, 
        {  0,  1, -2,  2 }, {  0,  1, -1,  1 }, {  0,  1,  0,  1 }, {  0,  1,  1,  1 }, 
        {  0,  1,  2,  2 }, {  0,  1,  3,  3 }, {  1,  1, -3,  3 }, {  1,  1, -2,  2 }, 
        {  1,  1, -1,  1 }, {  1,  1,  0,  1 }, {  1,  1,  1,  1 }, {  1,  1,  2,  2 }, 
        {  1,  1,  3,  3 }, {  2,  1, -3,  3 }, {  2,  1, -2,  3 }, {  2,  1, -1,  2 }, 
        {  2,  1,  0,  2 }, {  2,  1,  1,  2 }, {  2,  1,  2,  3 }, {  2,  1,  3,  3 }, 
        {  3,  1, -2,  3 }, {  3,  1, -1,  3 }, {  3,  1,  0,  3 }, {  3,  1,  1,  3 }, 
        {  3,  1,  2,  3 }, { -3,  2, -1,  3 }, { -3,  2,  0,  3 }, { -3,  2,  1,  3 }, 
        { -2,  2, -2,  3 }, { -2,  2, -1,  3 }, { -2,  2,  0,  2 }, { -2,  2,  1,  3 }, 
        { -2,  2,  2,  3 }, { -1,  2, -3,  3 }, { -1,  2, -2,  3 }, { -1,  2, -1,  2 }, 
        { -1,  2,  0,  2 }, { -1,  2,  1,  2 }, { -1,  2,  2,  3 }, { -1,  2,  3,  3 }, 
        {  0,  2, -3,  3 }, {  0,  2, -2,  2 }, {  0,  2, -1,  2 }, {  0,  2,  0,  2 }, 
        {  0,  2,  1,  2 }, {  0,  2,  2,  2 }, {  0,  2,  3,  3 }, {  1,  2, -3,  3 }, 
        {  1,  2, -2,  3 }, {  1,  2, -1,  2 }, {  1,  2,  0,  2 }, {  1,  2,  1,  2 }, 
        {  1,  2,  2,  3 }, {  1,  2,  3,  3 }, {  2,  2, -2,  3 }, {  2,  2, -1,  3 }, 
        {  2,  2,  0,  2 }, {  2,  2,  1,  3 }, {  2,  2,  2,  3 }, {  3,  2, -1,  3 }, 
        {  3,  2,  0,  3 }, {  3,  2,  1,  3 }, { -2,  3, -1,  3 }, { -2,  3,  0,  3 }, 
        { -2,  3,  1,  3 }, { -1,  3, -2,  3 }, { -1,  3, -1,  3 }, { -1,  3,  0,  3 }, 
        { -1,  3,  1,  3 }, { -1,  3,  2,  3 }, {  0,  3, -2,  3 }, {  0,  3, -1,  3 }, 
        {  0,  3,  0,  3 }, {  0,  3,  1,  3 }, {  0,  3,  2,  3 }, {  1,  3, -2,  3 }, 
        {  1,  3, -1,  3 }, {  1,  3,  0,  3 }, {  1,  3,  1,  3 }, {  1,  3,  2,  3 }, 
        {  2,  3, -1,  3 }, {  2,  3,  0,  3 }, {  2,  3,  1,  3 } };
*/

//
//Mara  
typedef struct {
    int8_t  x, y, z;
} gridVector;
__device__ gridVector gridVects [1021] = {
  { -6, -1, -1}, { -6, -1,  0}, { -6, -1,  1}, { -6,  0, -1}, 
  { -6,  0,  0}, { -6,  0,  1}, { -6,  1, -1}, { -6,  1,  0}, 
  { -6,  1,  1}, { -5, -3, -2}, { -5, -3, -1}, { -5, -3,  0}, 
  { -5, -3,  1}, { -5, -3,  2}, { -5, -2, -3}, { -5, -2, -2}, 
  { -5, -2, -1}, { -5, -2,  0}, { -5, -2,  1}, { -5, -2,  2}, 
  { -5, -2,  3}, { -5, -1, -3}, { -5, -1, -2}, { -5, -1, -1}, 
  { -5, -1,  0}, { -5, -1,  1}, { -5, -1,  2}, { -5, -1,  3}, 
  { -5,  0, -3}, { -5,  0, -2}, { -5,  0, -1}, { -5,  0,  0}, 
  { -5,  0,  1}, { -5,  0,  2}, { -5,  0,  3}, { -5,  1, -3}, 
  { -5,  1, -2}, { -5,  1, -1}, { -5,  1,  0}, { -5,  1,  1}, 
  { -5,  1,  2}, { -5,  1,  3}, { -5,  2, -3}, { -5,  2, -2}, 
  { -5,  2, -1}, { -5,  2,  0}, { -5,  2,  1}, { -5,  2,  2}, 
  { -5,  2,  3}, { -5,  3, -2}, { -5,  3, -1}, { -5,  3,  0}, 
  { -5,  3,  1}, { -5,  3,  2}, { -4, -4, -2}, { -4, -4, -1}, 
  { -4, -4,  0}, { -4, -4,  1}, { -4, -4,  2}, { -4, -3, -3}, 
  { -4, -3, -2}, { -4, -3, -1}, { -4, -3,  0}, { -4, -3,  1}, 
  { -4, -3,  2}, { -4, -3,  3}, { -4, -2, -4}, { -4, -2, -3}, 
  { -4, -2, -2}, { -4, -2, -1}, { -4, -2,  0}, { -4, -2,  1}, 
  { -4, -2,  2}, { -4, -2,  3}, { -4, -2,  4}, { -4, -1, -4}, 
  { -4, -1, -3}, { -4, -1, -2}, { -4, -1, -1}, { -4, -1,  0}, 
  { -4, -1,  1}, { -4, -1,  2}, { -4, -1,  3}, { -4, -1,  4}, 
  { -4,  0, -4}, { -4,  0, -3}, { -4,  0, -2}, { -4,  0, -1}, 
  { -4,  0,  0}, { -4,  0,  1}, { -4,  0,  2}, { -4,  0,  3}, 
  { -4,  0,  4}, { -4,  1, -4}, { -4,  1, -3}, { -4,  1, -2}, 
  { -4,  1, -1}, { -4,  1,  0}, { -4,  1,  1}, { -4,  1,  2}, 
  { -4,  1,  3}, { -4,  1,  4}, { -4,  2, -4}, { -4,  2, -3}, 
  { -4,  2, -2}, { -4,  2, -1}, { -4,  2,  0}, { -4,  2,  1}, 
  { -4,  2,  2}, { -4,  2,  3}, { -4,  2,  4}, { -4,  3, -3}, 
  { -4,  3, -2}, { -4,  3, -1}, { -4,  3,  0}, { -4,  3,  1}, 
  { -4,  3,  2}, { -4,  3,  3}, { -4,  4, -2}, { -4,  4, -1}, 
  { -4,  4,  0}, { -4,  4,  1}, { -4,  4,  2}, { -3, -5, -2}, 
  { -3, -5, -1}, { -3, -5,  0}, { -3, -5,  1}, { -3, -5,  2}, 
  { -3, -4, -3}, { -3, -4, -2}, { -3, -4, -1}, { -3, -4,  0}, 
  { -3, -4,  1}, { -3, -4,  2}, { -3, -4,  3}, { -3, -3, -4}, 
  { -3, -3, -3}, { -3, -3, -2}, { -3, -3, -1}, { -3, -3,  0}, 
  { -3, -3,  1}, { -3, -3,  2}, { -3, -3,  3}, { -3, -3,  4}, 
  { -3, -2, -5}, { -3, -2, -4}, { -3, -2, -3}, { -3, -2, -2}, 
  { -3, -2, -1}, { -3, -2,  0}, { -3, -2,  1}, { -3, -2,  2}, 
  { -3, -2,  3}, { -3, -2,  4}, { -3, -2,  5}, { -3, -1, -5}, 
  { -3, -1, -4}, { -3, -1, -3}, { -3, -1, -2}, { -3, -1, -1}, 
  { -3, -1,  0}, { -3, -1,  1}, { -3, -1,  2}, { -3, -1,  3}, 
  { -3, -1,  4}, { -3, -1,  5}, { -3,  0, -5}, { -3,  0, -4}, 
  { -3,  0, -3}, { -3,  0, -2}, { -3,  0, -1}, { -3,  0,  0}, 
  { -3,  0,  1}, { -3,  0,  2}, { -3,  0,  3}, { -3,  0,  4}, 
  { -3,  0,  5}, { -3,  1, -5}, { -3,  1, -4}, { -3,  1, -3}, 
  { -3,  1, -2}, { -3,  1, -1}, { -3,  1,  0}, { -3,  1,  1}, 
  { -3,  1,  2}, { -3,  1,  3}, { -3,  1,  4}, { -3,  1,  5}, 
  { -3,  2, -5}, { -3,  2, -4}, { -3,  2, -3}, { -3,  2, -2}, 
  { -3,  2, -1}, { -3,  2,  0}, { -3,  2,  1}, { -3,  2,  2}, 
  { -3,  2,  3}, { -3,  2,  4}, { -3,  2,  5}, { -3,  3, -4}, 
  { -3,  3, -3}, { -3,  3, -2}, { -3,  3, -1}, { -3,  3,  0}, 
  { -3,  3,  1}, { -3,  3,  2}, { -3,  3,  3}, { -3,  3,  4}, 
  { -3,  4, -3}, { -3,  4, -2}, { -3,  4, -1}, { -3,  4,  0}, 
  { -3,  4,  1}, { -3,  4,  2}, { -3,  4,  3}, { -3,  5, -2}, 
  { -3,  5, -1}, { -3,  5,  0}, { -3,  5,  1}, { -3,  5,  2}, 
  { -2, -5, -3}, { -2, -5, -2}, { -2, -5, -1}, { -2, -5,  0}, 
  { -2, -5,  1}, { -2, -5,  2}, { -2, -5,  3}, { -2, -4, -4}, 
  { -2, -4, -3}, { -2, -4, -2}, { -2, -4, -1}, { -2, -4,  0}, 
  { -2, -4,  1}, { -2, -4,  2}, { -2, -4,  3}, { -2, -4,  4}, 
  { -2, -3, -5}, { -2, -3, -4}, { -2, -3, -3}, { -2, -3, -2}, 
  { -2, -3, -1}, { -2, -3,  0}, { -2, -3,  1}, { -2, -3,  2}, 
  { -2, -3,  3}, { -2, -3,  4}, { -2, -3,  5}, { -2, -2, -5}, 
  { -2, -2, -4}, { -2, -2, -3}, { -2, -2, -2}, { -2, -2, -1}, 
  { -2, -2,  0}, { -2, -2,  1}, { -2, -2,  2}, { -2, -2,  3}, 
  { -2, -2,  4}, { -2, -2,  5}, { -2, -1, -5}, { -2, -1, -4}, 
  { -2, -1, -3}, { -2, -1, -2}, { -2, -1, -1}, { -2, -1,  0}, 
  { -2, -1,  1}, { -2, -1,  2}, { -2, -1,  3}, { -2, -1,  4}, 
  { -2, -1,  5}, { -2,  0, -5}, { -2,  0, -4}, { -2,  0, -3}, 
  { -2,  0, -2}, { -2,  0, -1}, { -2,  0,  0}, { -2,  0,  1}, 
  { -2,  0,  2}, { -2,  0,  3}, { -2,  0,  4}, { -2,  0,  5}, 
  { -2,  1, -5}, { -2,  1, -4}, { -2,  1, -3}, { -2,  1, -2}, 
  { -2,  1, -1}, { -2,  1,  0}, { -2,  1,  1}, { -2,  1,  2}, 
  { -2,  1,  3}, { -2,  1,  4}, { -2,  1,  5}, { -2,  2, -5}, 
  { -2,  2, -4}, { -2,  2, -3}, { -2,  2, -2}, { -2,  2, -1}, 
  { -2,  2,  0}, { -2,  2,  1}, { -2,  2,  2}, { -2,  2,  3}, 
  { -2,  2,  4}, { -2,  2,  5}, { -2,  3, -5}, { -2,  3, -4}, 
  { -2,  3, -3}, { -2,  3, -2}, { -2,  3, -1}, { -2,  3,  0}, 
  { -2,  3,  1}, { -2,  3,  2}, { -2,  3,  3}, { -2,  3,  4}, 
  { -2,  3,  5}, { -2,  4, -4}, { -2,  4, -3}, { -2,  4, -2}, 
  { -2,  4, -1}, { -2,  4,  0}, { -2,  4,  1}, { -2,  4,  2}, 
  { -2,  4,  3}, { -2,  4,  4}, { -2,  5, -3}, { -2,  5, -2}, 
  { -2,  5, -1}, { -2,  5,  0}, { -2,  5,  1}, { -2,  5,  2}, 
  { -2,  5,  3}, { -1, -6, -1}, { -1, -6,  0}, { -1, -6,  1}, 
  { -1, -5, -3}, { -1, -5, -2}, { -1, -5, -1}, { -1, -5,  0}, 
  { -1, -5,  1}, { -1, -5,  2}, { -1, -5,  3}, { -1, -4, -4}, 
  { -1, -4, -3}, { -1, -4, -2}, { -1, -4, -1}, { -1, -4,  0}, 
  { -1, -4,  1}, { -1, -4,  2}, { -1, -4,  3}, { -1, -4,  4}, 
  { -1, -3, -5}, { -1, -3, -4}, { -1, -3, -3}, { -1, -3, -2}, 
  { -1, -3, -1}, { -1, -3,  0}, { -1, -3,  1}, { -1, -3,  2}, 
  { -1, -3,  3}, { -1, -3,  4}, { -1, -3,  5}, { -1, -2, -5}, 
  { -1, -2, -4}, { -1, -2, -3}, { -1, -2, -2}, { -1, -2, -1}, 
  { -1, -2,  0}, { -1, -2,  1}, { -1, -2,  2}, { -1, -2,  3}, 
  { -1, -2,  4}, { -1, -2,  5}, { -1, -1, -6}, { -1, -1, -5}, 
  { -1, -1, -4}, { -1, -1, -3}, { -1, -1, -2}, { -1, -1, -1}, 
  { -1, -1,  0}, { -1, -1,  1}, { -1, -1,  2}, { -1, -1,  3}, 
  { -1, -1,  4}, { -1, -1,  5}, { -1, -1,  6}, { -1,  0, -6}, 
  { -1,  0, -5}, { -1,  0, -4}, { -1,  0, -3}, { -1,  0, -2}, 
  { -1,  0, -1}, { -1,  0,  0}, { -1,  0,  1}, { -1,  0,  2}, 
  { -1,  0,  3}, { -1,  0,  4}, { -1,  0,  5}, { -1,  0,  6}, 
  { -1,  1, -6}, { -1,  1, -5}, { -1,  1, -4}, { -1,  1, -3}, 
  { -1,  1, -2}, { -1,  1, -1}, { -1,  1,  0}, { -1,  1,  1}, 
  { -1,  1,  2}, { -1,  1,  3}, { -1,  1,  4}, { -1,  1,  5}, 
  { -1,  1,  6}, { -1,  2, -5}, { -1,  2, -4}, { -1,  2, -3}, 
  { -1,  2, -2}, { -1,  2, -1}, { -1,  2,  0}, { -1,  2,  1}, 
  { -1,  2,  2}, { -1,  2,  3}, { -1,  2,  4}, { -1,  2,  5}, 
  { -1,  3, -5}, { -1,  3, -4}, { -1,  3, -3}, { -1,  3, -2}, 
  { -1,  3, -1}, { -1,  3,  0}, { -1,  3,  1}, { -1,  3,  2}, 
  { -1,  3,  3}, { -1,  3,  4}, { -1,  3,  5}, { -1,  4, -4}, 
  { -1,  4, -3}, { -1,  4, -2}, { -1,  4, -1}, { -1,  4,  0}, 
  { -1,  4,  1}, { -1,  4,  2}, { -1,  4,  3}, { -1,  4,  4}, 
  { -1,  5, -3}, { -1,  5, -2}, { -1,  5, -1}, { -1,  5,  0}, 
  { -1,  5,  1}, { -1,  5,  2}, { -1,  5,  3}, { -1,  6, -1}, 
  { -1,  6,  0}, { -1,  6,  1}, {  0, -6, -1}, {  0, -6,  0}, 
  {  0, -6,  1}, {  0, -5, -3}, {  0, -5, -2}, {  0, -5, -1}, 
  {  0, -5,  0}, {  0, -5,  1}, {  0, -5,  2}, {  0, -5,  3}, 
  {  0, -4, -4}, {  0, -4, -3}, {  0, -4, -2}, {  0, -4, -1}, 
  {  0, -4,  0}, {  0, -4,  1}, {  0, -4,  2}, {  0, -4,  3}, 
  {  0, -4,  4}, {  0, -3, -5}, {  0, -3, -4}, {  0, -3, -3}, 
  {  0, -3, -2}, {  0, -3, -1}, {  0, -3,  0}, {  0, -3,  1}, 
  {  0, -3,  2}, {  0, -3,  3}, {  0, -3,  4}, {  0, -3,  5}, 
  {  0, -2, -5}, {  0, -2, -4}, {  0, -2, -3}, {  0, -2, -2}, 
  {  0, -2, -1}, {  0, -2,  0}, {  0, -2,  1}, {  0, -2,  2}, 
  {  0, -2,  3}, {  0, -2,  4}, {  0, -2,  5}, {  0, -1, -6}, 
  {  0, -1, -5}, {  0, -1, -4}, {  0, -1, -3}, {  0, -1, -2}, 
  {  0, -1, -1}, {  0, -1,  0}, {  0, -1,  1}, {  0, -1,  2}, 
  {  0, -1,  3}, {  0, -1,  4}, {  0, -1,  5}, {  0, -1,  6}, 
  {  0,  0, -6}, {  0,  0, -5}, {  0,  0, -4}, {  0,  0, -3}, 
  {  0,  0, -2}, {  0,  0, -1}, {  0,  0,  0}, {  0,  0,  1}, 
  {  0,  0,  2}, {  0,  0,  3}, {  0,  0,  4}, {  0,  0,  5}, 
  {  0,  0,  6}, {  0,  1, -6}, {  0,  1, -5}, {  0,  1, -4}, 
  {  0,  1, -3}, {  0,  1, -2}, {  0,  1, -1}, {  0,  1,  0}, 
  {  0,  1,  1}, {  0,  1,  2}, {  0,  1,  3}, {  0,  1,  4}, 
  {  0,  1,  5}, {  0,  1,  6}, {  0,  2, -5}, {  0,  2, -4}, 
  {  0,  2, -3}, {  0,  2, -2}, {  0,  2, -1}, {  0,  2,  0}, 
  {  0,  2,  1}, {  0,  2,  2}, {  0,  2,  3}, {  0,  2,  4}, 
  {  0,  2,  5}, {  0,  3, -5}, {  0,  3, -4}, {  0,  3, -3}, 
  {  0,  3, -2}, {  0,  3, -1}, {  0,  3,  0}, {  0,  3,  1}, 
  {  0,  3,  2}, {  0,  3,  3}, {  0,  3,  4}, {  0,  3,  5}, 
  {  0,  4, -4}, {  0,  4, -3}, {  0,  4, -2}, {  0,  4, -1}, 
  {  0,  4,  0}, {  0,  4,  1}, {  0,  4,  2}, {  0,  4,  3}, 
  {  0,  4,  4}, {  0,  5, -3}, {  0,  5, -2}, {  0,  5, -1}, 
  {  0,  5,  0}, {  0,  5,  1}, {  0,  5,  2}, {  0,  5,  3}, 
  {  0,  6, -1}, {  0,  6,  0}, {  0,  6,  1}, {  1, -6, -1}, 
  {  1, -6,  0}, {  1, -6,  1}, {  1, -5, -3}, {  1, -5, -2}, 
  {  1, -5, -1}, {  1, -5,  0}, {  1, -5,  1}, {  1, -5,  2}, 
  {  1, -5,  3}, {  1, -4, -4}, {  1, -4, -3}, {  1, -4, -2}, 
  {  1, -4, -1}, {  1, -4,  0}, {  1, -4,  1}, {  1, -4,  2}, 
  {  1, -4,  3}, {  1, -4,  4}, {  1, -3, -5}, {  1, -3, -4}, 
  {  1, -3, -3}, {  1, -3, -2}, {  1, -3, -1}, {  1, -3,  0}, 
  {  1, -3,  1}, {  1, -3,  2}, {  1, -3,  3}, {  1, -3,  4}, 
  {  1, -3,  5}, {  1, -2, -5}, {  1, -2, -4}, {  1, -2, -3}, 
  {  1, -2, -2}, {  1, -2, -1}, {  1, -2,  0}, {  1, -2,  1}, 
  {  1, -2,  2}, {  1, -2,  3}, {  1, -2,  4}, {  1, -2,  5}, 
  {  1, -1, -6}, {  1, -1, -5}, {  1, -1, -4}, {  1, -1, -3}, 
  {  1, -1, -2}, {  1, -1, -1}, {  1, -1,  0}, {  1, -1,  1}, 
  {  1, -1,  2}, {  1, -1,  3}, {  1, -1,  4}, {  1, -1,  5}, 
  {  1, -1,  6}, {  1,  0, -6}, {  1,  0, -5}, {  1,  0, -4}, 
  {  1,  0, -3}, {  1,  0, -2}, {  1,  0, -1}, {  1,  0,  0}, 
  {  1,  0,  1}, {  1,  0,  2}, {  1,  0,  3}, {  1,  0,  4}, 
  {  1,  0,  5}, {  1,  0,  6}, {  1,  1, -6}, {  1,  1, -5}, 
  {  1,  1, -4}, {  1,  1, -3}, {  1,  1, -2}, {  1,  1, -1}, 
  {  1,  1,  0}, {  1,  1,  1}, {  1,  1,  2}, {  1,  1,  3}, 
  {  1,  1,  4}, {  1,  1,  5}, {  1,  1,  6}, {  1,  2, -5}, 
  {  1,  2, -4}, {  1,  2, -3}, {  1,  2, -2}, {  1,  2, -1}, 
  {  1,  2,  0}, {  1,  2,  1}, {  1,  2,  2}, {  1,  2,  3}, 
  {  1,  2,  4}, {  1,  2,  5}, {  1,  3, -5}, {  1,  3, -4}, 
  {  1,  3, -3}, {  1,  3, -2}, {  1,  3, -1}, {  1,  3,  0}, 
  {  1,  3,  1}, {  1,  3,  2}, {  1,  3,  3}, {  1,  3,  4}, 
  {  1,  3,  5}, {  1,  4, -4}, {  1,  4, -3}, {  1,  4, -2}, 
  {  1,  4, -1}, {  1,  4,  0}, {  1,  4,  1}, {  1,  4,  2}, 
  {  1,  4,  3}, {  1,  4,  4}, {  1,  5, -3}, {  1,  5, -2}, 
  {  1,  5, -1}, {  1,  5,  0}, {  1,  5,  1}, {  1,  5,  2}, 
  {  1,  5,  3}, {  1,  6, -1}, {  1,  6,  0}, {  1,  6,  1}, 
  {  2, -5, -3}, {  2, -5, -2}, {  2, -5, -1}, {  2, -5,  0}, 
  {  2, -5,  1}, {  2, -5,  2}, {  2, -5,  3}, {  2, -4, -4}, 
  {  2, -4, -3}, {  2, -4, -2}, {  2, -4, -1}, {  2, -4,  0}, 
  {  2, -4,  1}, {  2, -4,  2}, {  2, -4,  3}, {  2, -4,  4}, 
  {  2, -3, -5}, {  2, -3, -4}, {  2, -3, -3}, {  2, -3, -2}, 
  {  2, -3, -1}, {  2, -3,  0}, {  2, -3,  1}, {  2, -3,  2}, 
  {  2, -3,  3}, {  2, -3,  4}, {  2, -3,  5}, {  2, -2, -5}, 
  {  2, -2, -4}, {  2, -2, -3}, {  2, -2, -2}, {  2, -2, -1}, 
  {  2, -2,  0}, {  2, -2,  1}, {  2, -2,  2}, {  2, -2,  3}, 
  {  2, -2,  4}, {  2, -2,  5}, {  2, -1, -5}, {  2, -1, -4}, 
  {  2, -1, -3}, {  2, -1, -2}, {  2, -1, -1}, {  2, -1,  0}, 
  {  2, -1,  1}, {  2, -1,  2}, {  2, -1,  3}, {  2, -1,  4}, 
  {  2, -1,  5}, {  2,  0, -5}, {  2,  0, -4}, {  2,  0, -3}, 
  {  2,  0, -2}, {  2,  0, -1}, {  2,  0,  0}, {  2,  0,  1}, 
  {  2,  0,  2}, {  2,  0,  3}, {  2,  0,  4}, {  2,  0,  5}, 
  {  2,  1, -5}, {  2,  1, -4}, {  2,  1, -3}, {  2,  1, -2}, 
  {  2,  1, -1}, {  2,  1,  0}, {  2,  1,  1}, {  2,  1,  2}, 
  {  2,  1,  3}, {  2,  1,  4}, {  2,  1,  5}, {  2,  2, -5}, 
  {  2,  2, -4}, {  2,  2, -3}, {  2,  2, -2}, {  2,  2, -1}, 
  {  2,  2,  0}, {  2,  2,  1}, {  2,  2,  2}, {  2,  2,  3}, 
  {  2,  2,  4}, {  2,  2,  5}, {  2,  3, -5}, {  2,  3, -4}, 
  {  2,  3, -3}, {  2,  3, -2}, {  2,  3, -1}, {  2,  3,  0}, 
  {  2,  3,  1}, {  2,  3,  2}, {  2,  3,  3}, {  2,  3,  4}, 
  {  2,  3,  5}, {  2,  4, -4}, {  2,  4, -3}, {  2,  4, -2}, 
  {  2,  4, -1}, {  2,  4,  0}, {  2,  4,  1}, {  2,  4,  2}, 
  {  2,  4,  3}, {  2,  4,  4}, {  2,  5, -3}, {  2,  5, -2}, 
  {  2,  5, -1}, {  2,  5,  0}, {  2,  5,  1}, {  2,  5,  2}, 
  {  2,  5,  3}, {  3, -5, -2}, {  3, -5, -1}, {  3, -5,  0}, 
  {  3, -5,  1}, {  3, -5,  2}, {  3, -4, -3}, {  3, -4, -2}, 
  {  3, -4, -1}, {  3, -4,  0}, {  3, -4,  1}, {  3, -4,  2}, 
  {  3, -4,  3}, {  3, -3, -4}, {  3, -3, -3}, {  3, -3, -2}, 
  {  3, -3, -1}, {  3, -3,  0}, {  3, -3,  1}, {  3, -3,  2}, 
  {  3, -3,  3}, {  3, -3,  4}, {  3, -2, -5}, {  3, -2, -4}, 
  {  3, -2, -3}, {  3, -2, -2}, {  3, -2, -1}, {  3, -2,  0}, 
  {  3, -2,  1}, {  3, -2,  2}, {  3, -2,  3}, {  3, -2,  4}, 
  {  3, -2,  5}, {  3, -1, -5}, {  3, -1, -4}, {  3, -1, -3}, 
  {  3, -1, -2}, {  3, -1, -1}, {  3, -1,  0}, {  3, -1,  1}, 
  {  3, -1,  2}, {  3, -1,  3}, {  3, -1,  4}, {  3, -1,  5}, 
  {  3,  0, -5}, {  3,  0, -4}, {  3,  0, -3}, {  3,  0, -2}, 
  {  3,  0, -1}, {  3,  0,  0}, {  3,  0,  1}, {  3,  0,  2}, 
  {  3,  0,  3}, {  3,  0,  4}, {  3,  0,  5}, {  3,  1, -5}, 
  {  3,  1, -4}, {  3,  1, -3}, {  3,  1, -2}, {  3,  1, -1}, 
  {  3,  1,  0}, {  3,  1,  1}, {  3,  1,  2}, {  3,  1,  3}, 
  {  3,  1,  4}, {  3,  1,  5}, {  3,  2, -5}, {  3,  2, -4}, 
  {  3,  2, -3}, {  3,  2, -2}, {  3,  2, -1}, {  3,  2,  0}, 
  {  3,  2,  1}, {  3,  2,  2}, {  3,  2,  3}, {  3,  2,  4}, 
  {  3,  2,  5}, {  3,  3, -4}, {  3,  3, -3}, {  3,  3, -2}, 
  {  3,  3, -1}, {  3,  3,  0}, {  3,  3,  1}, {  3,  3,  2}, 
  {  3,  3,  3}, {  3,  3,  4}, {  3,  4, -3}, {  3,  4, -2}, 
  {  3,  4, -1}, {  3,  4,  0}, {  3,  4,  1}, {  3,  4,  2}, 
  {  3,  4,  3}, {  3,  5, -2}, {  3,  5, -1}, {  3,  5,  0}, 
  {  3,  5,  1}, {  3,  5,  2}, {  4, -4, -2}, {  4, -4, -1}, 
  {  4, -4,  0}, {  4, -4,  1}, {  4, -4,  2}, {  4, -3, -3}, 
  {  4, -3, -2}, {  4, -3, -1}, {  4, -3,  0}, {  4, -3,  1}, 
  {  4, -3,  2}, {  4, -3,  3}, {  4, -2, -4}, {  4, -2, -3}, 
  {  4, -2, -2}, {  4, -2, -1}, {  4, -2,  0}, {  4, -2,  1}, 
  {  4, -2,  2}, {  4, -2,  3}, {  4, -2,  4}, {  4, -1, -4}, 
  {  4, -1, -3}, {  4, -1, -2}, {  4, -1, -1}, {  4, -1,  0}, 
  {  4, -1,  1}, {  4, -1,  2}, {  4, -1,  3}, {  4, -1,  4}, 
  {  4,  0, -4}, {  4,  0, -3}, {  4,  0, -2}, {  4,  0, -1}, 
  {  4,  0,  0}, {  4,  0,  1}, {  4,  0,  2}, {  4,  0,  3}, 
  {  4,  0,  4}, {  4,  1, -4}, {  4,  1, -3}, {  4,  1, -2}, 
  {  4,  1, -1}, {  4,  1,  0}, {  4,  1,  1}, {  4,  1,  2}, 
  {  4,  1,  3}, {  4,  1,  4}, {  4,  2, -4}, {  4,  2, -3}, 
  {  4,  2, -2}, {  4,  2, -1}, {  4,  2,  0}, {  4,  2,  1}, 
  {  4,  2,  2}, {  4,  2,  3}, {  4,  2,  4}, {  4,  3, -3}, 
  {  4,  3, -2}, {  4,  3, -1}, {  4,  3,  0}, {  4,  3,  1}, 
  {  4,  3,  2}, {  4,  3,  3}, {  4,  4, -2}, {  4,  4, -1}, 
  {  4,  4,  0}, {  4,  4,  1}, {  4,  4,  2}, {  5, -3, -2}, 
  {  5, -3, -1}, {  5, -3,  0}, {  5, -3,  1}, {  5, -3,  2}, 
  {  5, -2, -3}, {  5, -2, -2}, {  5, -2, -1}, {  5, -2,  0}, 
  {  5, -2,  1}, {  5, -2,  2}, {  5, -2,  3}, {  5, -1, -3}, 
  {  5, -1, -2}, {  5, -1, -1}, {  5, -1,  0}, {  5, -1,  1}, 
  {  5, -1,  2}, {  5, -1,  3}, {  5,  0, -3}, {  5,  0, -2}, 
  {  5,  0, -1}, {  5,  0,  0}, {  5,  0,  1}, {  5,  0,  2}, 
  {  5,  0,  3}, {  5,  1, -3}, {  5,  1, -2}, {  5,  1, -1}, 
  {  5,  1,  0}, {  5,  1,  1}, {  5,  1,  2}, {  5,  1,  3}, 
  {  5,  2, -3}, {  5,  2, -2}, {  5,  2, -1}, {  5,  2,  0}, 
  {  5,  2,  1}, {  5,  2,  2}, {  5,  2,  3}, {  5,  3, -2}, 
  {  5,  3, -1}, {  5,  3,  0}, {  5,  3,  1}, {  5,  3,  2}, 
  {  6, -1, -1}, {  6, -1,  0}, {  6, -1,  1}, {  6,  0, -1}, 
  {  6,  0,  0}, {  6,  0,  1}, {  6,  1, -1}, {  6,  1,  0}, 
  {  6,  1,  1}}; 
//


/*
 * this kernel adds atoms to the lookup table. each block takes an atom, and each
 * thread takes a grid translation vector (gridLoc)
 */


extern "C" __global__ void fillLookupTable(
    // inputs
    const real4* __restrict__ posq, const float2* __restrict__ params,
#ifdef USE_PERIODIC
    real4 periodicBoxSize,
#endif
    // outputs
    int *lookupTable) {
    
    int i, j, k, gridLoc, index;
    float R2grid_2, atomRBuffer_2, atomX, atomY, atomZ, atomR, dx, dy, dz;
    gridVector thisGrid;
    
    real4 posq_XYZQ, tran;
    int4   gdim;
    float bufferR = R_BUFFER; 
    atomR = params[blockIdx.x].x; 
    
#ifdef USE_PERIODIC
    
    real4 box;
    
    // only input atom if radius is not zero
    if ( atomR != 0.0f ) {
        
        thisGrid  = gridVects[threadIdx.x];
        posq_XYZQ = posq[blockIdx.x];
        tran      = posq[0];
        
        // find thread's location in grid
        atomX = posq_XYZQ.x - tran.x;
        i = floor((atomX + DELTA_R * 0.5f) * INVERSE_DELTA_R) + thisGrid.x;
        atomY = posq_XYZQ.y - tran.y;
        j = floor((atomY + DELTA_R * 0.5f) * INVERSE_DELTA_R) + thisGrid.y;
        atomZ = posq_XYZQ.z - tran.z;
        k = floor((atomZ + DELTA_R * 0.5f) * INVERSE_DELTA_R) + thisGrid.z;
        
        // atom's radius
      //atomR += R_BUFFER;
        atomRBuffer_2 = bufferR * bufferR;
        
        // grid radius to atom
        dx = atomX - (float)i * DELTA_R;
        dy = atomY - (float)j * DELTA_R;
        dz = atomZ - (float)k * DELTA_R;
        R2grid_2 = dx*dx + dy*dy + dz*dz;
        
        // final spherical distance check
        if ( R2grid_2 <= atomRBuffer_2 ) {
            
            box = periodicBoxSize;
            gdim = make_int4(ceil(box.x / DELTA_R), ceil(box.y / DELTA_R), ceil(box.z / DELTA_R), 0);
            
            // enforce periodic bounds
            if (i < 0) i += gdim.x; else if (i >= gdim.x) i -= gdim.x;
            if (j < 0) j += gdim.y; else if (j >= gdim.y) j -= gdim.y;
            if (k < 0) k += gdim.z; else if (k >= gdim.z) k -= gdim.z;
            
            gridLoc = ( i + 
                        j*gdim.x + 
                        k*gdim.x*gdim.y ) * MAX_ATOMS_IN_VOXEL;
            
            // input this atom into the grid!
            index = atomicAdd( &lookupTable[gridLoc], (int)1 );
            if ( index < MAX_ATOMS_IN_VOXEL - 1 ) {
                lookupTable[gridLoc + index + 1] = blockIdx.x;
            } else {
                atomicAdd( &lookupTable[gridLoc], -1 );
            }
        }
    } // radius != 0.0 check
    
#else 
    // only input atom if radius is not zero
    if ( atomR != 0.0f ) {
        
        thisGrid = gridVects[threadIdx.x];
        posq_XYZQ = posq[blockIdx.x];
        tran = TranslateXYZ;
        
        // are we looking at a voxel in the grid? check / calc X/i Y/j Z/k
        atomX = posq_XYZQ.x + tran.x;
        i = floor(atomX / DELTA_R) + thisGrid.x;
        if ( i >= 0 && i < GridDimXYZ.x ) {
        
        atomY = posq_XYZQ.y + tran.y;
        j = floor(atomY / DELTA_R) + thisGrid.y;
        if ( j >= 0 && j < GridDimXYZ.y ) {
        
        atomZ = posq_XYZQ.z + tran.z;
        k = floor(atomZ / DELTA_R) + thisGrid.z;
        if ( k >= 0 && k < GridDimXYZ.z ) {
            
          //atomR += R_BUFFER;
            atomRBuffer_2 = bufferR * bufferR;
            
            dx = atomX - ((float)i * DELTA_R + 0.5f * DELTA_R);
            dy = atomY - ((float)j * DELTA_R + 0.5f * DELTA_R);
            dz = atomZ - ((float)k * DELTA_R + 0.5f * DELTA_R);
            R2grid_2 = dx*dx + dy*dy + dz*dz;
            
            // final spherical distance check
            if ( R2grid_2 <= atomRBuffer_2 ) {
                
                gdim = GridDimXYZ;
                
                gridLoc = ( i + 
                            j*gdim.x + 
                            k*gdim.x*gdim.y ) * MAX_ATOMS_IN_VOXEL;
                
                // input this atom into the grid!
                index = atomicAdd( &lookupTable[gridLoc], (int)1 );
                if ( index < MAX_ATOMS_IN_VOXEL - 1 ) {
                    lookupTable[gridLoc + index + 1] = blockIdx.x;
                } else {
                    atomicAdd( &lookupTable[gridLoc], -1 );
                }
            }
        } // Z/k grid check
        } // Y/j grid check
        } // X/i grid check
    } // radius != 0.0 check
#endif
}

/*
 * if we have a built lookup table, sort it! this cuts the Born Radius calculation 
 * time roughly in half
 */

extern "C" __global__ void sortLookupTable(
    // inputs
    const real4* __restrict__ posq,
#ifdef USE_PERIODIC
    real4 periodicBoxSize, real4 invPeriodicBoxSize,
#endif
    // outputs
    int *lookupTable) {
    
    __shared__ float allR2grid_2[MAX_ATOMS_IN_VOXEL];
    
    float dx, dy, dz, R2grid_2;
    int numAtomsInGrid, gridLoc, atomInt, numPriorAtoms, n;
    
    real4 posq_XYZQ, tran, box, invbox;
    
#ifdef USE_PERIODIC
    box = periodicBoxSize;
    int4 gdim = make_int4(ceil(box.x / DELTA_R), 
                          ceil(box.y / DELTA_R), 0, 0);
    
    // location of this voxel in the lookup array
    gridLoc = ( blockIdx.x + 
                blockIdx.y * gdim.x + 
                blockIdx.z * gdim.x * gdim.y ) * MAX_ATOMS_IN_VOXEL;
#else
    // location of this voxel in the lookup array
    gridLoc = ( blockIdx.x + 
                blockIdx.y * gridDim.x + 
                blockIdx.z * gridDim.x * gridDim.y ) * MAX_ATOMS_IN_VOXEL;
#endif
    
    // check number of atoms in the grid
    // 0 atoms = nothing to sort, 1 atom = sorting is pointless
    if ( lookupTable[gridLoc] > 1 ) {
        
        // assign one thread per resident atom
        numAtomsInGrid = lookupTable[gridLoc];
        if ( threadIdx.x < numAtomsInGrid ) {
            
            atomInt = lookupTable[ gridLoc + threadIdx.x + 1 ];
            posq_XYZQ = posq[atomInt];
            
#ifdef USE_PERIODIC
            // calc and save distance to grid center
            tran = posq[0];
            invbox = invPeriodicBoxSize;
            dx = posq_XYZQ.x + tran.x - ((float)blockIdx.x * DELTA_R + 0.5f * DELTA_R);
            dy = posq_XYZQ.y + tran.y - ((float)blockIdx.y * DELTA_R + 0.5f * DELTA_R);
            dz = posq_XYZQ.z + tran.z - ((float)blockIdx.z * DELTA_R + 0.5f * DELTA_R);
            dx -= floor( dx*invbox.x + 0.5f ) * box.x;
            dy -= floor( dy*invbox.y + 0.5f ) * box.y;
            dz -= floor( dz*invbox.z + 0.5f ) * box.z;
            
#else
            // calc and save distance to grid center
            tran = TranslateXYZ;
            dx = ((float)blockIdx.x * DELTA_R + 0.5f * DELTA_R) - (posq_XYZQ.x + tran.x);
            dy = ((float)blockIdx.y * DELTA_R + 0.5f * DELTA_R) - (posq_XYZQ.y + tran.y);
            dz = ((float)blockIdx.z * DELTA_R + 0.5f * DELTA_R) - (posq_XYZQ.z + tran.z);
#endif
            
            R2grid_2 = dx*dx + dy*dy + dz*dz;
            allR2grid_2[threadIdx.x] = R2grid_2;
            
            // 
            numPriorAtoms = 0;
            __syncthreads();
            for ( n = 0; n < numAtomsInGrid; n++ )
                if ( allR2grid_2[n] < R2grid_2 )
                    numPriorAtoms++;
            
            // record sorted-index of atoms
            lookupTable[ gridLoc + numPriorAtoms + 1 ] = atomInt;
        }
    }
}


/*
 * Neighboring-atom portion of the force calculation
 */

typedef struct {
    real x, y, z;
    real q;
    real fx, fy, fz, fw;
    real bornRadius;
#ifdef USE_CPHMD
    float lambdaQfac, XQfac, lambdaForce, XForce;
#endif
} AtomData2;

extern "C" __global__ void computeGBMVForce(unsigned long long* __restrict__ forceBuffers, 
    unsigned long long* __restrict__ global_bornForce, mixed* __restrict__ energyBuffer, 
    const real4* __restrict__ posq, const float* __restrict__ global_bornRadii,
#ifdef USE_CUTOFF
        const int* __restrict__ tiles, const int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize, 
        unsigned int maxTiles, const real4* __restrict__ blockCenter, const real4* __restrict__ blockSize, const unsigned int* __restrict__ interactingAtoms,
#else
        unsigned int numTiles,
#endif
#ifdef USE_CPHMD
        const float2* __restrict__ cphmdAtomQfac,
        float* __restrict__ cphmdForce, 
        const tileflags* __restrict__ exclusions,
#endif
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
        const tileflags* __restrict__ openmmExclusions,
#endif
#if OPENMM_VERSION < 75
        const ushort2* __restrict__ exclusionTiles ) {
#else
        const int2* __restrict__ exclusionTiles ) {
#endif
    const unsigned int totalWarps = (blockDim.x*gridDim.x)/TILE_SIZE;
    const unsigned int warp = (blockIdx.x*blockDim.x+threadIdx.x)/TILE_SIZE;
    const unsigned int tgx = threadIdx.x & (TILE_SIZE-1);
    const unsigned int tbx = threadIdx.x - tgx;
    mixed energy = 0;
    real swi = 1.f; 
    real dSdR = 0.f;
    __shared__ AtomData2 localData[FORCE_WORK_GROUP_SIZE];

    // First loop: process tiles that contain exclusions.
    
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
#if OPENMM_VERSION < 75
        const ushort2 tileIndices = exclusionTiles[pos];
#else
        const int2 tileIndices = exclusionTiles[pos];
#endif
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        real4 force = make_real4(0);
        unsigned int atom1 = x*TILE_SIZE + tgx;
        real4 posq1 = posq[atom1];
        real bornRadius1 = global_bornRadii[atom1];
#ifdef USE_CPHMD
        float atom1LambdaForce = 0.0f, atom1XForce = 0.0f;
        float2 atom1Qfac = cphmdAtomQfac[atom1];
        tileflags excl = exclusions[pos*TILE_SIZE+tgx];
#endif
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
        tileflags openmmExcl = openmmExclusions[pos*TILE_SIZE+tgx];
#endif
        if (x == y) {
            // This tile is on the diagonal.

            localData[threadIdx.x].x = posq1.x;
            localData[threadIdx.x].y = posq1.y;
            localData[threadIdx.x].z = posq1.z;
            localData[threadIdx.x].q = posq1.w;
            localData[threadIdx.x].bornRadius = bornRadius1;
#ifdef USE_CPHMD
            localData[threadIdx.x].lambdaQfac = atom1Qfac.x;
            localData[threadIdx.x].XQfac      = atom1Qfac.y;
#endif
            
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                if (atom1 < NUM_ATOMS && y*TILE_SIZE+j < NUM_ATOMS) {
                    real4 posq2 = make_real4(localData[tbx+j].x, localData[tbx+j].y, localData[tbx+j].z, localData[tbx+j].q);
                    real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                    delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                    delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                    delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                    if (r2 < CUTOFF_SQUARED) {

                        real x1 = CUTOFF_SQUARED;
                        real x2 = CUTON_SQUARED;
                        x2 = 1.f/(x1-x2);
                        x1 = (x1-r2)*x2;
                        swi = (x1>1.f)? 1.f : x1*x1*(3.f-2.f*x1);
                        dSdR = (x1>1.f)? 0.f : 12.f*x1*(1.f-x1)*x2;
#endif
                        real bornRadius2 = localData[tbx+j].bornRadius;
                        real alpha2_ij = bornRadius1*bornRadius2;
                        real D_ij = r2*RECIP(P6_GBMV2*alpha2_ij);
                        real expTerm = EXP(-D_ij);
                        real denominator2 = r2 + alpha2_ij*expTerm;
                        real denominator = SQRT(denominator2);
                        
#ifdef USE_SALT  // if salt used
                        real saltEffect = SALT_FAC_A * EXP(-denominator * ONE_OVER_KAPPA);
                        real saltPrefactor = SALT_FAC_B - saltEffect; 
                        real scaledChargeProduct = saltPrefactor*posq1.w*posq2.w;
                        real recipDem = RECIP(denominator);
                        real saltForce = saltPrefactor*recipDem - ONE_OVER_KAPPA*saltEffect;
                        real tempEnergy = scaledChargeProduct*recipDem;
                        real Gpol = posq1.w*posq2.w*recipDem*recipDem * saltForce;  
#else // if no salt used
                        real scaledChargeProduct = PREFACTOR*posq1.w*posq2.w;
                        real tempEnergy = scaledChargeProduct*RECIP(denominator);
                        real Gpol = tempEnergy*RECIP(denominator2);
#endif
                        
                        real dGpol_dalpha2_ij = -0.5f*Gpol*expTerm*(1.0f+D_ij);
                        real dEdR = Gpol*(1.0f - expTerm / P6_GBMV2);
                        force.w += dGpol_dalpha2_ij*bornRadius2*swi;
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
                        bool noOffsetNeeded = !(openmmExcl & 0x1);
                        if ( !noOffsetNeeded )
                            energy += 0.5f*scaledChargeProduct*COULOMBIC_CUTOFF_OFFSET;
#endif
                        energy += 0.5f*tempEnergy*swi;
                        dEdR = dEdR*swi + 0.5f*tempEnergy*dSdR;
                        delta *= dEdR;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
                        
#ifdef USE_CPHMD
                        real invR = 0.0f;
                        bool isExcluded = !(excl & 0x1);
                        if ( !isExcluded )
                            invR = RSQRT(r2);
                        
#ifdef USE_SALT
                        float tmpThetaForce = (CPHMD_SALT_FAC*saltPrefactor*recipDem - invR) * posq2.w;
#else
                        float tmpThetaForce = (GBMV_FAC*RECIP(denominator) - invR) * posq2.w;
#endif
                        atom1LambdaForce += tmpThetaForce * atom1Qfac.x;
                        atom1XForce += tmpThetaForce * atom1Qfac.y;
#endif
#ifdef USE_CUTOFF
                    }
#endif
                }
#ifdef USE_CPHMD
                excl >>= 1;
#endif
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
                openmmExcl >>= 1;
#endif
            }
        } else {
            // This is an off-diagonal tile.

            unsigned int j = y*TILE_SIZE + tgx;
            real4 tempPosq = posq[j];
            localData[threadIdx.x].x = tempPosq.x;
            localData[threadIdx.x].y = tempPosq.y;
            localData[threadIdx.x].z = tempPosq.z;
            localData[threadIdx.x].q = tempPosq.w;
            localData[threadIdx.x].bornRadius = global_bornRadii[j];
            localData[threadIdx.x].fx = 0.0f;
            localData[threadIdx.x].fy = 0.0f;
            localData[threadIdx.x].fz = 0.0f;
            localData[threadIdx.x].fw = 0.0f;
            
#ifdef USE_CPHMD
            float2 tempQfac = cphmdAtomQfac[j];
            localData[threadIdx.x].lambdaQfac = tempQfac.x;
            localData[threadIdx.x].XQfac      = tempQfac.y;
            localData[threadIdx.x].lambdaForce = 0.0f;
            localData[threadIdx.x].XForce      = 0.0f;
            excl = (excl >> tgx) | (excl << (TILE_SIZE - tgx));
#endif
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
            openmmExcl = (openmmExcl >> tgx) | (openmmExcl << (TILE_SIZE - tgx));
#endif
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                if (atom1 < NUM_ATOMS && y*TILE_SIZE+tj < NUM_ATOMS) {
                    real4 posq2 = make_real4(localData[tbx+tj].x, localData[tbx+tj].y, localData[tbx+tj].z, localData[tbx+tj].q);
                    real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                    delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                    delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                    delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                    if (r2 < CUTOFF_SQUARED) {

                        real x1 = CUTOFF_SQUARED;
                        real x2 = CUTON_SQUARED;
                        x2 = 1.f/(x1-x2);
                        x1 = (x1-r2)*x2;
                        swi = (x1>1.f)? 1.f : x1*x1*(3.f-2.f*x1);
                        dSdR = (x1>1.f)? 0.f : 12.f*x1*(1.f-x1)*x2;
#endif
                        real bornRadius2 = localData[tbx+tj].bornRadius;
                        real alpha2_ij = bornRadius1*bornRadius2;
                        real D_ij = r2*RECIP(P6_GBMV2*alpha2_ij);
                        real expTerm = EXP(-D_ij);
                        real denominator2 = r2 + alpha2_ij*expTerm;
                        real denominator = SQRT(denominator2);
                        
#ifdef USE_SALT  // if salt used
                        real saltEffect = SALT_FAC_A * EXP(-denominator * ONE_OVER_KAPPA);
                        real saltPrefactor = SALT_FAC_B - saltEffect;
                        real scaledChargeProduct = saltPrefactor*posq1.w*posq2.w;
                        real recipDem = RECIP(denominator);
                        real saltForce = saltPrefactor*recipDem - ONE_OVER_KAPPA*saltEffect;
                        real tempEnergy = scaledChargeProduct*recipDem;
                        real Gpol = posq1.w*posq2.w*recipDem*recipDem * saltForce; 
#else // if no salt used
                        real scaledChargeProduct = PREFACTOR*posq1.w*posq2.w;
                        real tempEnergy = scaledChargeProduct*RECIP(denominator);
                        real Gpol = tempEnergy*RECIP(denominator2);
#endif
                        
                        real dGpol_dalpha2_ij = -0.5f*Gpol*expTerm*(1.0f+D_ij);
                        real dEdR = Gpol*(1.0f - expTerm / P6_GBMV2);
                        force.w += dGpol_dalpha2_ij*bornRadius2*swi;
                        
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
                        bool noOffsetNeeded = !(openmmExcl & 0x1);
                        if ( !noOffsetNeeded )
                            energy += scaledChargeProduct*COULOMBIC_CUTOFF_OFFSET;
#endif
                        energy += tempEnergy*swi;
                        dEdR = dEdR*swi + tempEnergy*dSdR;
                        delta *= dEdR;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
                        localData[tbx+tj].fx += delta.x;
                        localData[tbx+tj].fy += delta.y;
                        localData[tbx+tj].fz += delta.z;
                        localData[tbx+tj].fw += dGpol_dalpha2_ij*bornRadius1*swi;
                        
#ifdef USE_CPHMD
                        real invR = 0.0f;
                        bool isExcluded = !(excl & 0x1);
                        if ( !isExcluded )
                            invR = RSQRT(r2);
#ifdef USE_SALT
                        float tmpForceFac = CPHMD_SALT_FAC*saltPrefactor*recipDem - invR;
#else
                        float tmpForceFac = GBMV_FAC*RECIP(denominator) - invR;
#endif
                        float tmpThetaForce = tmpForceFac * posq1.w;
                        localData[tbx+tj].lambdaForce += tmpThetaForce * localData[tbx+tj].lambdaQfac;
                        localData[tbx+tj].XForce += tmpThetaForce * localData[tbx+tj].XQfac;
                        
                        tmpThetaForce = tmpForceFac * posq2.w;
                        atom1LambdaForce += tmpThetaForce * atom1Qfac.x;
                        atom1XForce += tmpThetaForce * atom1Qfac.y;
#endif     
#ifdef USE_CUTOFF
                    }
#endif
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
#ifdef USE_CPHMD
                excl >>= 1;
#endif
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
                openmmExcl >>= 1;
#endif
            }
        }
        
        // Write results.
        
        unsigned int offset = x*TILE_SIZE + tgx;
        atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
        atomicAdd(&global_bornForce[offset], static_cast<unsigned long long>((long long) (force.w*0x100000000)));
#ifdef USE_CPHMD
        atomicAdd(&cphmdForce[offset], atom1LambdaForce);
        atomicAdd(&cphmdForce[offset+PADDED_NUM_ATOMS], atom1XForce);
#endif 
        if (x != y) {
            offset = y*TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fx*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fy*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fz*0x100000000)));
            atomicAdd(&global_bornForce[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fw*0x100000000)));
#ifdef USE_CPHMD
            atomicAdd(&cphmdForce[offset], localData[threadIdx.x].lambdaForce);
            atomicAdd(&cphmdForce[offset+PADDED_NUM_ATOMS], localData[threadIdx.x].XForce);
#endif 
        }
    }

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

#ifdef USE_CUTOFF
    unsigned int numTiles = interactionCount[0];
    int pos = warp*(numTiles > maxTiles ? NUM_BLOCKS*(NUM_BLOCKS+1)/2 : numTiles)/totalWarps;
    int end = (warp+1)*(numTiles > maxTiles ? NUM_BLOCKS*(NUM_BLOCKS+1)/2 : numTiles)/totalWarps;
#else
    int pos = warp*numTiles/totalWarps;
    int end = (warp+1)*numTiles/totalWarps;
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
    __shared__ int atomIndices[FORCE_WORK_GROUP_SIZE];
    __shared__ volatile int skipTiles[FORCE_WORK_GROUP_SIZE];
    skipTiles[threadIdx.x] = -1;

    while (pos < end) {
        real4 force = make_real4(0);
        bool includeTile = true;

#ifdef USE_CPHMD
        float atom1LambdaForce = 0.0f, atom1XForce = 0.0f;
#endif
        // Extract the coordinates of this tile.
        
        unsigned int x, y;
        bool singlePeriodicCopy = false;
#ifdef USE_CUTOFF
        if (numTiles <= maxTiles) {
            x = tiles[pos];
            real4 blockSizeX = blockSize[x];
            singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= CUTOFF &&
                                  0.5f*periodicBoxSize.y-blockSizeX.y >= CUTOFF &&
                                  0.5f*periodicBoxSize.z-blockSizeX.z >= CUTOFF);
        }
        else
#endif
        {
            y = (unsigned int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
            if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
                y += (x < y ? -1 : 1);
                x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
            }

            // Skip over tiles that have exclusions, since they were already processed.

            while (skipTiles[tbx+TILE_SIZE-1] < pos) {
                if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
#if OPENMM_VERSION < 75
                    ushort2 tile = exclusionTiles[skipBase+tgx];
#else
                    int2 tile = exclusionTiles[skipBase+tgx];
#endif
                    skipTiles[threadIdx.x] = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
                }
                else
                    skipTiles[threadIdx.x] = end;
                skipBase += TILE_SIZE;            
                currentSkipIndex = tbx;
            }
            while (skipTiles[currentSkipIndex] < pos)
                currentSkipIndex++;
            includeTile = (skipTiles[currentSkipIndex] != pos);
        }
        if (includeTile) {
            unsigned int atom1 = x*TILE_SIZE + tgx;

            // Load atom data for this tile.
            
            real4 posq1 = posq[atom1];
            real bornRadius1 = global_bornRadii[atom1];
#ifdef USE_CPHMD
            float2 atom1Qfac = cphmdAtomQfac[atom1];
#endif
#ifdef USE_CUTOFF
            unsigned int j = (numTiles <= maxTiles ? interactingAtoms[pos*TILE_SIZE+tgx] : y*TILE_SIZE + tgx);
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[threadIdx.x] = j;
            if (j < PADDED_NUM_ATOMS) {
                real4 tempPosq = posq[j];
                localData[threadIdx.x].x = tempPosq.x;
                localData[threadIdx.x].y = tempPosq.y;
                localData[threadIdx.x].z = tempPosq.z;
                localData[threadIdx.x].q = tempPosq.w;
                localData[threadIdx.x].bornRadius = global_bornRadii[j];
                localData[threadIdx.x].fx = 0.0f;
                localData[threadIdx.x].fy = 0.0f;
                localData[threadIdx.x].fz = 0.0f;
                localData[threadIdx.x].fw = 0.0f;
#ifdef USE_CPHMD
                float2 tempQfac = cphmdAtomQfac[j];
                localData[threadIdx.x].lambdaQfac = tempQfac.x;
                localData[threadIdx.x].XQfac      = tempQfac.y;
                localData[threadIdx.x].lambdaForce = 0.0f;
                localData[threadIdx.x].XForce      = 0.0f;
#endif
            }
#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                // The box is small enough that we can just translate all the atoms into a single periodic
                // box, then skip having to apply periodic boundary conditions later.

                real4 blockCenterX = blockCenter[x];
                posq1.x -= floor((posq1.x-blockCenterX.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                posq1.y -= floor((posq1.y-blockCenterX.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                posq1.z -= floor((posq1.z-blockCenterX.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
                localData[threadIdx.x].x -= floor((localData[threadIdx.x].x-blockCenterX.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                localData[threadIdx.x].y -= floor((localData[threadIdx.x].y-blockCenterX.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                localData[threadIdx.x].z -= floor((localData[threadIdx.x].z-blockCenterX.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = atomIndices[tbx+tj];
                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                        real4 posq2 = make_real4(localData[tbx+tj].x, localData[tbx+tj].y, localData[tbx+tj].z, localData[tbx+tj].q);
                        real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
                        real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                        if (r2 < CUTOFF_SQUARED) {

                            real x1 = CUTOFF_SQUARED;
                            real x2 = CUTON_SQUARED;
                            x2 = 1.f/(x1-x2);
                            x1 = (x1-r2)*x2;
                            swi = (x1>1.f)? 1.f : x1*x1*(3.f-2.f*x1);
                            dSdR = (x1>1.f)? 0.f : 12.f*x1*(1.f-x1)*x2;

                            real bornRadius2 = localData[tbx+tj].bornRadius;
                            real alpha2_ij = bornRadius1*bornRadius2;
                            real D_ij = r2*RECIP(P6_GBMV2*alpha2_ij);
                            real expTerm = EXP(-D_ij);
                            real denominator2 = r2 + alpha2_ij*expTerm;
                            real denominator = SQRT(denominator2);
                            
#ifdef USE_SALT  // if salt used
                            real saltEffect = SALT_FAC_A * EXP(-denominator * ONE_OVER_KAPPA);
                            real saltPrefactor = SALT_FAC_B - saltEffect;
                            real scaledChargeProduct = saltPrefactor*posq1.w*posq2.w;
                            real recipDem = RECIP(denominator);
                            real saltForce = saltPrefactor*recipDem - ONE_OVER_KAPPA*saltEffect;
                            real tempEnergy = scaledChargeProduct*recipDem;
                            real Gpol = posq1.w*posq2.w*recipDem*recipDem * saltForce;
#else // if no salt used
                            real scaledChargeProduct = PREFACTOR*posq1.w*posq2.w;
                            real tempEnergy = scaledChargeProduct*RECIP(denominator);
                            real Gpol = tempEnergy*RECIP(denominator2);
#endif
                            real dGpol_dalpha2_ij = -0.5f*Gpol*expTerm*(1.0f+D_ij);
                            real dEdR = Gpol*(1.0f - expTerm / P6_GBMV2);
                            force.w += dGpol_dalpha2_ij*bornRadius2*swi;
                            
                            
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
                            energy += tempEnergy + scaledChargeProduct*COULOMBIC_CUTOFF_OFFSET;
#else
                            energy += tempEnergy*swi;
#endif
                            dEdR = dEdR*swi + tempEnergy*dSdR;
                            delta *= dEdR;
                            force.x -= delta.x;
                            force.y -= delta.y;
                            force.z -= delta.z;
                            localData[tbx+tj].fx += delta.x;
                            localData[tbx+tj].fy += delta.y;
                            localData[tbx+tj].fz += delta.z;
                            localData[tbx+tj].fw += dGpol_dalpha2_ij*bornRadius1*swi;
#ifdef USE_CPHMD
                            real invR = RSQRT(r2);
#ifdef USE_SALT
                            float tmpForceFac = CPHMD_SALT_FAC*saltPrefactor*recipDem - invR;
#else
                            float tmpForceFac = GBMV_FAC*RECIP(denominator) - invR;
#endif
                            float tmpThetaForce = tmpForceFac * posq1.w;
                            localData[tbx+tj].lambdaForce += tmpThetaForce * localData[tbx+tj].lambdaQfac;
                            localData[tbx+tj].XForce += tmpThetaForce * localData[tbx+tj].XQfac;
                            
                            tmpThetaForce = tmpForceFac * posq2.w;
                            atom1LambdaForce += tmpThetaForce * atom1Qfac.x;
                            atom1XForce += tmpThetaForce * atom1Qfac.y;
#endif  
                        }
                    }
                    tj = (tj + 1) & (TILE_SIZE - 1);
                }
            }
            else
#endif
            {
                // We need to apply periodic boundary conditions separately for each interaction.

                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = atomIndices[tbx+tj];
                    if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                        real4 posq2 = make_real4(localData[tbx+tj].x, localData[tbx+tj].y, localData[tbx+tj].z, localData[tbx+tj].q);
                        real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                        delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                        delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                        delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                        real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                        if (r2 < CUTOFF_SQUARED) {

                            real x1 = CUTOFF_SQUARED;
                            real x2 = CUTON_SQUARED;
                            x2 = 1.f/(x1-x2);
                            x1 = (x1-r2)*x2;
                            swi = (x1>1.f)? 1.f : x1*x1*(3.f-2.f*x1);
                            dSdR = (x1>1.f)? 0.f : 12.f*x1*(1.f-x1)*x2;

#endif
                            real bornRadius2 = localData[tbx+tj].bornRadius;
                            real alpha2_ij = bornRadius1*bornRadius2;
                            real D_ij = r2*RECIP(P6_GBMV2*alpha2_ij);
                            real expTerm = EXP(-D_ij);
                            real denominator2 = r2 + alpha2_ij*expTerm;
                            real denominator = SQRT(denominator2);
                            
#ifdef USE_SALT  // if salt used
                            real saltEffect = SALT_FAC_A * EXP(-denominator * ONE_OVER_KAPPA);
                            real saltPrefactor = SALT_FAC_B - saltEffect;
                            real scaledChargeProduct = saltPrefactor*posq1.w*posq2.w;
                            real recipDem = RECIP(denominator);
                            real saltForce = saltPrefactor*recipDem - ONE_OVER_KAPPA*saltEffect;
                            real tempEnergy = scaledChargeProduct*recipDem;
                            real Gpol = posq1.w*posq2.w*recipDem*recipDem * saltForce;
#else // if no salt used
                            real scaledChargeProduct = PREFACTOR*posq1.w*posq2.w;
                            real tempEnergy = scaledChargeProduct*RECIP(denominator);
                            real Gpol = tempEnergy*RECIP(denominator2);
#endif
                            
                            real dGpol_dalpha2_ij = -0.5f*Gpol*expTerm*(1.0f+D_ij);
                            real dEdR = Gpol*(1.0f - expTerm / P6_GBMV2);
                            force.w += dGpol_dalpha2_ij*bornRadius2*swi;
                            
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
                            energy += tempEnergy + scaledChargeProduct*COULOMBIC_CUTOFF_OFFSET;
#else
                            energy += tempEnergy*swi;
#endif
                            dEdR = dEdR*swi + tempEnergy*dSdR;
                            delta *= dEdR;
                            force.x -= delta.x;
                            force.y -= delta.y;
                            force.z -= delta.z;
                            localData[tbx+tj].fx += delta.x;
                            localData[tbx+tj].fy += delta.y;
                            localData[tbx+tj].fz += delta.z;
                            localData[tbx+tj].fw += dGpol_dalpha2_ij*bornRadius1*swi;
#ifdef USE_CPHMD
                            real invR = RSQRT(r2);
#ifdef USE_SALT
                            float tmpForceFac = CPHMD_SALT_FAC*saltPrefactor*recipDem - invR;
#else
                            float tmpForceFac = GBMV_FAC*RECIP(denominator) - invR;
#endif
                            float tmpThetaForce = tmpForceFac * posq1.w;
                            localData[tbx+tj].lambdaForce += tmpThetaForce * localData[tbx+tj].lambdaQfac;
                            localData[tbx+tj].XForce += tmpThetaForce * localData[tbx+tj].XQfac;
                            
                            tmpThetaForce = tmpForceFac * posq2.w;
                            atom1LambdaForce += tmpThetaForce * atom1Qfac.x;
                            atom1XForce += tmpThetaForce * atom1Qfac.y;
#endif  
#ifdef USE_CUTOFF
                        }
#endif
                    }
                    tj = (tj + 1) & (TILE_SIZE - 1);
                }
            }

            // Write results.

            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
            atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
            atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
            atomicAdd(&global_bornForce[atom1], static_cast<unsigned long long>((long long) (force.w*0x100000000)));
#ifdef USE_CPHMD
            atomicAdd(&cphmdForce[atom1], atom1LambdaForce);
            atomicAdd(&cphmdForce[atom1+PADDED_NUM_ATOMS], atom1XForce);
#endif
#ifdef USE_CUTOFF
            unsigned int atom2 = atomIndices[threadIdx.x];
#else
            unsigned int atom2 = y*TILE_SIZE + tgx;
#endif
            if (atom2 < PADDED_NUM_ATOMS) {
                atomicAdd(&forceBuffers[atom2], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fx*0x100000000)));
                atomicAdd(&forceBuffers[atom2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fy*0x100000000)));
                atomicAdd(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fz*0x100000000)));
                atomicAdd(&global_bornForce[atom2], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fw*0x100000000)));
#ifdef USE_CPHMD
                atomicAdd(&cphmdForce[atom2], localData[threadIdx.x].lambdaForce);
                atomicAdd(&cphmdForce[atom2+PADDED_NUM_ATOMS], localData[threadIdx.x].XForce);
#endif 
            }
        }
        pos++;
    }
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
}

/*
 * second part of the Born radius calculations for all atoms.  
 */


extern "C" __global__ void calcBornR( 
    // inputs
// posq[IAtom].{x,y,z,w}: x, y, z and q of Ith atom
// params[IAtom].{x,y}: Vdw radius, unknown of Ith atom
// QuadPts[NGrid].{x,y,z,w}: x, y, z, radius of Nth grid point
// QuadPtWeights_gbmv[NGrid].{x,y}: weights (w0 and w1) of two types of integrals in the GBMV2 model.
// lastWtrLE[IAtom]: the effective radius of each atom for the integrals of Born radii.
// lookupTable[AllGrids * (1+Neighbor atoms)]
    const real4* __restrict__ posq, const float2* __restrict__ params,
    const float4* QuadPts, const float2* QuadPtWeights_gbmv,const float* lastWtrLE,
    int *lookupTable,
#ifdef USE_PERIODIC
    real4 periodicBoxSize, real4 invPeriodicBoxSize,
#endif
    // outputs
// BornRadii[IAtom]: the born radius of each atom.
// gbmvChain[NGrid*IAtom]: record the temporary data for electrostatic force, to save time.
  //float* __restrict__ BornRadii, float* gbmvChain, int *nGBMVchainAtoms ) {
    float* __restrict__ BornRadii, float* gbmvChain ) {
//
  //float* __restrict__ BornRadii ) {

    // these arrays hold values to allow threads of this block to communicate
    __shared__ float G0_Sums[256],G1_Sums[256];
    __shared__ float dG1,BornR;

    // separate these variables from all other threads
    float quadPtX, quadPtY, quadPtZ, // the grid positions
          RvdWJ, atomI_inner, // RvdW at each atom
          ddx, ddy, ddz, R2;  

    float SMV,Smv2,Svdw,Fvdw,Fmv2,VMV,EXPSMV,XX;

    float  Y1,Y2,Y3,Y4;
    float  X1,X2,X3_X,X3_Y,X3_Z,X3X3;
    double G0,G1;

    int   numAtomsInGrid, atomCounter, atomJ, atomI, 
          i, j, k, gridLoc, gridLoci, Idx, NGrids;
          Idx=0;

          atomI  = blockIdx.x;  // each block governs an atom

    real4   posqI, posqJ; 


#ifdef USE_PERIODIC
    real4 tran = posq[0], 
          box = periodicBoxSize,
          invbox = invPeriodicBoxSize;
    int4  gdim = make_int4( ceil(periodicBoxSize.x / DELTA_R), 
                            ceil(periodicBoxSize.y / DELTA_R), 
                            ceil(periodicBoxSize.z / DELTA_R), 0);
#else
    // system extrema, translation vector
    const float MinX = minXYZ.x, MaxX = maxXYZ.x,
                MinY = minXYZ.y, MaxY = maxXYZ.y,
                MinZ = minXYZ.z, MaxZ = maxXYZ.z,
                tranX = TranslateXYZ.x,
                tranY = TranslateXYZ.y,
                tranZ = TranslateXYZ.z;

    // read in this block's dimesions and translation vector
    const int dimX = GridDimXYZ.x,
              dimY = GridDimXYZ.y,
              dimZ = GridDimXYZ.z;
#endif


  //// GBMV2 parameters; 
  //// The nm unit was used in the OpenMM, instead of the A unit for the distance
  //const float    
  //      BETA_GBMV2   = -12.0f,
  //      LAMBDA_GBMV2 = 0.5f, 
  //      A0_GBMV2     = 1.0f - sqrtf(0.5f), 
  //      A1_GBMV2     = 1.0f,
  //      ALPHA_GBMV2  = -19.8f,
  //      SLOPE_GBMV2     = 0.9085f,  // slope
  //      SHIFT_GBMV2     = -0.0102f, // shift
  //      P3_GBMV2     = 0.65f;    // S0     
    float VDWON_GBMV2,VDWOFF_GBMV2,MV2ON_GBMV2,MV2OFF_GBMV2,P1,P2;
          VDWON_GBMV2=HSX1_GBMV2; VDWOFF_GBMV2=HSX2_GBMV2;
          MV2OFF_GBMV2=OFFX_GBMV2;  MV2ON_GBMV2 = ONX_GBMV2;
          P1=P1_GBMV2; P2=P2_GBMV2;
    float4 quadPt;
    float2 quadWt;

//--------------------------------------------------------------------------
// COMPUTING THE MOLECULAR VOLUME AT EACH GRID POINT (VMV).
//
BornRadii[atomI] = 0.1f;
if (params[atomI].x == 0.f) return;
atomI_inner = lastWtrLE[atomI] ;
posqI = posq[atomI]; // Ith atomic position // ***

G0=0.f;G1=0.f;
NGrids=NUM_RADII*NUM_ANGLES;

  for ( Idx = threadIdx.x; Idx < NGrids; Idx += blockDim.x ) {
//Idx=threadIdx.x;
if(Idx<NGrids){
  quadPt = QuadPts[Idx];// grid positions
  SMV=0.f; 
  gridLoci = atomI*NGrids*6+Idx*6; 
  gbmvChain[gridLoci] = 0.f; 
if ( quadPt.w >=  atomI_inner ) { // effective points
  quadPtZ = quadPt.z + posqI.z;
  k = floor((quadPtZ-MinZ) * INVERSE_DELTA_R);
  if ( k >= 0  &&  k < dimZ ) {
    quadPtX = quadPt.x + posqI.x;
    i = floor((quadPtX-MinX) * INVERSE_DELTA_R);
    if ( i >= 0  &&  i < dimX ) {      
      quadPtY = quadPt.y + posqI.y;
      j = floor((quadPtY-MinY) * INVERSE_DELTA_R);
      if ( j >= 0  &&  j < dimY  ) {       

        // locate voxel in lookup table
        gridLoc = (i + 
            (j*dimX) +
            (k*dimX*dimY)) * MAX_ATOMS_IN_VOXEL;

        // calculate the #atoms near to the gridLoc voxel
        numAtomsInGrid = lookupTable[gridLoc]; // ***

        // check if gridspace is empty. If not, calculate atomic occupancy
        if (numAtomsInGrid > 0 ) {

          // calculate the X1, X2, X3_X, X3_Y, X3_Z
          X1=0.f;X2=0.f;X3_X=0.f;X3_Y=0.f;X3_Z=0.f;
          Smv2=0.f;Svdw=0.f;SMV=0.f;         

          atomCounter = 1;
          if (numAtomsInGrid > CUTNUM_GBMV2) numAtomsInGrid=CUTNUM_GBMV2;
          while ( Svdw<2.0f && atomCounter <= numAtomsInGrid ) {

            // gather atomic info for these atoms near to the grid
            atomJ = lookupTable[gridLoc + atomCounter]; // the Jth atomic label // ***
            posqJ = posq[atomJ]; // the Jth atomic positions // ***
            RvdWJ = params[atomJ].x; // the Jth vdW radii // ***
            atomCounter++;

            // calculate the necessary varibales involving the |r_n + R_I - R_J|  
            ddx   = quadPtX - posqJ.x; 
            ddy   = quadPtY - posqJ.y;
            ddz   = quadPtZ - posqJ.z;
            R2    = ddx * ddx + ddy * ddy + ddz * ddz; 
            Y1    = RvdWJ + VDWON_GBMV2;  Y1 *= Y1;
            Y2    = RvdWJ + VDWOFF_GBMV2; Y2 *= Y2;
            Y3    = RvdWJ + MV2ON_GBMV2;  Y3 *= Y3;
            Y4    = RvdWJ + MV2OFF_GBMV2; Y4 *= Y4;
            
            // calculate X1,X2,X3_X,X3_Y,X3_Y,Svdw,Smv2,SMV
            if (R2 <= Y4) {
               if (R2 <= Y1) {
                //Fvdw  = 1.f;Svdw += 2.f * Fvdw;Fmv2  = 0.f;
                //X1   += Fmv2;X2   += R2  * Fmv2 * Fmv2;
                //X3_X += ddx * Fmv2;X3_Y += ddy * Fmv2;X3_Z += ddz * Fmv2;
                //Smv2  = P3_GBMV2 * X1 / numAtomsInGrid;SMV   = Svdw + Smv2;
                  Svdw += 2.f;
                //SMV   = Svdw; // EXIT
               } else if (R2 <= Y2) { 
                  Fvdw  = (R2-Y1) / (Y2-Y1);
                  Fvdw  = 1.f + Fvdw*Fvdw*Fvdw*(Fvdw*(15.f-6.f*Fvdw)-10.f);
                  Svdw += 2.f * Fvdw; // VDW 
                  //Fmv2  = (1.f - Fvdw) * expf(ALPHA_GBMV2*(sqrtf(R2)-RvdWJ));
                  XX  = P1 + P2*RvdWJ; Fmv2=XX+R2-RvdWJ*RvdWJ;Fmv2*=Fmv2;Fmv2=XX*XX/Fmv2;Fmv2=(1.f-Fvdw)*Fmv2;
                  X1   += Fmv2;
                  X2   += R2  * Fmv2 * Fmv2;
                  X3_X += ddx * Fmv2;
                  X3_Y += ddy * Fmv2;
                  X3_Z += ddz * Fmv2;
                //Smv2  = P3_GBMV2 * X1 / numAtomsInGrid; // estimated MV2
                //SMV   = Svdw + Smv2;
               } else if (R2 <= Y3) {
                //Fvdw  = 0.f; Svdw += 2.f * Fvdw; // VDW
                //Fmv2  = expf(ALPHA_GBMV2*(sqrtf(R2)-RvdWJ));
                  XX  = P1 + P2*RvdWJ; Fmv2=XX+R2-RvdWJ*RvdWJ;Fmv2*=Fmv2;Fmv2=XX*XX/Fmv2;
                  X1   += Fmv2;
                  X2   += R2  * Fmv2 * Fmv2;
                  X3_X += ddx * Fmv2;
                  X3_Y += ddy * Fmv2;
                  X3_Z += ddz * Fmv2;
                //Smv2  = P3_GBMV2 * X1 / numAtomsInGrid; // estimated MV2
                //SMV   = Svdw + Smv2;
               } else {
                //Fvdw  = 0.f; Svdw += 2.f * Fvdw; // VDW 
                  Fvdw  = (R2-Y3) / (Y4-Y3);
                  Fvdw  = 1.f + Fvdw*Fvdw*Fvdw*(Fvdw*(15.f-6.f*Fvdw)-10.f);
                //Fmv2  = Fvdw * expf(ALPHA_GBMV2*(sqrtf(R2)-RvdWJ));
                  XX  = P1 + P2*RvdWJ; Fmv2=XX+R2-RvdWJ*RvdWJ;Fmv2*=Fmv2;Fmv2=XX*XX/Fmv2;Fmv2=Fvdw*Fmv2;
                  X1   += Fmv2;
                  X2   += R2  * Fmv2 * Fmv2;
                  X3_X += ddx * Fmv2;
                  X3_Y += ddy * Fmv2;
                  X3_Z += ddz * Fmv2;
                //Smv2  = P3_GBMV2 * X1 / numAtomsInGrid; // estimated MV2
                //SMV   = Svdw + Smv2;
               }
            } // if (R2 <= Y4)
            
          } // while loop neighbor atoms 

          X3X3 = X3_X*X3_X + X3_Y*X3_Y + X3_Z*X3_Z;if(X3X3<1.0e-15) X3X3=1.0e-15;
          Smv2 = P3_GBMV2 * X1 * X2 / X3X3;
          SMV  = Svdw + Smv2;
          if (SMV<2.f) {
            EXPSMV = expf(BETA_GBMV2*(SMV - LAMBDA_GBMV2));
            VMV = 1.0f/(1.0f+ EXPSMV);
            quadWt = QuadPtWeights_gbmv[Idx]; // grid weights
            G0 += quadWt.x*VMV; 
            G1 += quadWt.y*VMV; 
            // record info for the forces calculations
            if (SMV>0.f) {
              gridLoci = atomI*NGrids*6+Idx*6;
              gbmvChain[gridLoci]   = BETA_GBMV2*EXPSMV*VMV*VMV;
              gbmvChain[gridLoci+1] = P3_GBMV2*X2/X3X3;
              gbmvChain[gridLoci+2] = P3_GBMV2*X1/X3X3;
              gbmvChain[gridLoci+3] = 2.f*Smv2*X3_X/X3X3;
              gbmvChain[gridLoci+4] = 2.f*Smv2*X3_Y/X3X3;
              gbmvChain[gridLoci+5] = 2.f*Smv2*X3_Z/X3X3;
            }
          } else {
            EXPSMV = 0.f;
            VMV = 1.f;
            quadWt = QuadPtWeights_gbmv[Idx]; // grid weights
            G0 += quadWt.x; 
            G1 += quadWt.y; 
          } // 

        } // if (numAtomsInGrid>0)

      } // bracket Y realspace
    } // bracket X realspace
  } // bracket Z realspace

} // if ( quadPt.w >=  atomI_inner )

} // if
} // for


// Do the summation 
  G0_Sums[threadIdx.x]=G0;G1_Sums[threadIdx.x]=G1;__syncthreads();
  if(threadIdx.x== 0) {
    G0=0.f;G1=0.f;for(i=0;i<blockDim.x;i++) {G0+=G0_Sums[i];G1+=G1_Sums[i];}
    XX  = 1.0f / atomI_inner; G0  = XX - G0;
    XX *= XX; G1 = 0.25f*XX*XX - G1;
    dG1 = sqrtf(sqrtf(G1)); // ***
    BornR = 1.0f/(A0_GBMV2*G0 + A1_GBMV2*dG1);
    BornRadii[atomI] = SLOPE_GBMV2 * BornR + SHIFT_GBMV2;  // ***
  }

__syncthreads();
// record info for the forces calculations
  for ( Idx = threadIdx.x; Idx < NGrids; Idx += blockDim.x ) {
    if(Idx<NGrids){
      quadWt = QuadPtWeights_gbmv[Idx]; // grid weights
      XX = 0.25f / (dG1*dG1*dG1);
      XX = A0_GBMV2*quadWt.x + A1_GBMV2*quadWt.y*XX;
      XX *= SLOPE_GBMV2*BornR*BornR;
      gridLoci = atomI*NGrids*6+Idx*6;
      gbmvChain[gridLoci] *= XX; 
    }
  } 
   


} // END SUBROUTINE 













/*
 * Atomic SASA calculations for each atom. 
 */


extern "C" __global__ void calcSASA( 
    // inputs
    const real4* __restrict__ posq, const float2* __restrict__ params,
    const float4* QuadXYZW, int *lookupTable,
#ifdef USE_PERIODIC
    real4 periodicBoxSize, real4 invPeriodicBoxSize,
#endif
    // outputs
  //float* __restrict__ Energy_SASA,
  //float4* __restrict__ Forces_SASA,
    mixed* __restrict__ energyBuffer, 
    unsigned long long* __restrict__ forceBuffers ) {

    // shared data
    __shared__ float Energy, ForcesX, ForcesY, ForcesZ; Energy=0.f;ForcesX=0.f;ForcesY=0.f;ForcesZ=0.f;

    // separate these variables from all other threads
    float quadPtX, quadPtY, quadPtZ, // the grid positions
          RvdWI, RvdWJ, RvdWK; // RvdW at each atom

    float SA_ON=PROBE_RADIUS-0.02f, SA_OFF=PROBE_RADIUS+0.01f;
    float ddx, ddy, ddz, R2, Rv_ON, Rv_OFF, RA, Fx, Fy, Fz; Fx=0.f;Fy=0.f;Fz=0.f;
    float umij, f_umij, Factor, VEmi; 

    int   numAtomsInGrid, atomCounter, numAtomsInGridi, atomCounteri, atomJ,  
          angInt, atomI, i, j, k, gridLoc, gridLoci; 

          atomI  = blockIdx.x;  // each block governs an atom
          angInt = threadIdx.x;  // each block's x thread handles an angular integration

#ifdef USE_PERIODIC
    real4 tran = posq[0], 
          box = periodicBoxSize,
          invbox = invPeriodicBoxSize;
    int4  gdim = make_int4( ceil(periodicBoxSize.x / DELTA_R), 
                            ceil(periodicBoxSize.y / DELTA_R), 
                            ceil(periodicBoxSize.z / DELTA_R), 0);
#else
    // system extrema, translation vector
    const float MinX = minXYZ.x, MaxX = maxXYZ.x,
                MinY = minXYZ.y, MaxY = maxXYZ.y,
                MinZ = minXYZ.z, MaxZ = maxXYZ.z,
                tranX = TranslateXYZ.x,
                tranY = TranslateXYZ.y,
                tranZ = TranslateXYZ.z;

    // read in this block's dimesions and translation vector
    const int dimX = GridDimXYZ.x,
              dimY = GridDimXYZ.y,
              dimZ = GridDimXYZ.z;
#endif

    // read atomic positions and grids' positions and weights
    float4 quadPt = QuadXYZW[ angInt ]; // grid positions
    real4  posqI, posqJ, posqK;
    float4 thisForce;

if (params[atomI].x == 0.f) return;
//--------------------------------------------------------------------------
// COMPUTING THE ENERGY & FORCES(1)
//
posqI   = posq[atomI];
RvdWI   = params[atomI].x;
RA      = RvdWI + PROBE_RADIUS;
quadPtX = posqI.x + RA*quadPt.x;
quadPtY = posqI.y + RA*quadPt.y;
quadPtZ = posqI.z + RA*quadPt.z;
numAtomsInGrid = 0;
k = floor((quadPtZ-MinZ) * INVERSE_DELTA_R);
if ( k >= 0  &&  k < dimZ ) {
  i = floor((quadPtX-MinX) * INVERSE_DELTA_R);
  if ( i >= 0  &&  i < dimX ) {      
    j = floor((quadPtY-MinY) * INVERSE_DELTA_R);
    if ( j >= 0  &&  j < dimY  ) {       
        gridLoc = (i + 
           (j*dimX) +
           (k*dimX*dimY)) * MAX_ATOMS_IN_VOXEL;
        numAtomsInGrid = lookupTable[gridLoc]; 
    }
  }
}
VEmi = 0.0f;
thisForce.x=0.f; thisForce.y=0.f; thisForce.z=0.f;
atomCounter = 1;
while(atomCounter <= numAtomsInGrid && VEmi<1.0f) {
   atomJ = lookupTable[gridLoc + atomCounter];
   if(atomJ != atomI) {
     posqJ = posq[atomJ];
     RvdWJ = params[atomJ].x;
     ddx   = quadPtX - posqJ.x;
     ddy   = quadPtY - posqJ.y;
     ddz   = quadPtZ - posqJ.z;
     Rv_ON = (RvdWJ+SA_ON);  Rv_ON  *= Rv_ON;
     Rv_OFF= (RvdWJ+SA_OFF); Rv_OFF *= Rv_OFF;
     umij  = (ddx*ddx+ddy*ddy+ddz*ddz - Rv_ON) / (Rv_OFF - Rv_ON);
     if (umij<0.f) { 
       VEmi = 2.0f;}
     else if (umij<1.f) {
       f_umij  = 1.0f + umij*umij*umij*(umij*(15.0f-6.0f*umij)-10.0f);
       VEmi   += 2.0f*f_umij;
       f_umij = (4.f*umij*umij*(umij*(60.f-30.f*umij)-30.f))/(Rv_OFF - Rv_ON);
       thisForce.x += f_umij * ddx;
       thisForce.y += f_umij * ddy;
       thisForce.z += f_umij * ddz;
     }
   } 
   atomCounter++;
} // while
Factor=SURFACE_AREA_FACTOR*12.566371f*RA*RA*quadPt.w;
if (VEmi < 1.0f) {
  R2 = Factor*(1.0f + VEmi*VEmi*VEmi*(VEmi*(15.0f-6.0f*VEmi)-10.0f));
  atomicAdd(&Energy,R2); // Energy calculations
  R2 = Factor*VEmi*VEmi*(VEmi*(60.f-30.f*VEmi)-30.f);
  Fx = thisForce.x * R2; 
  Fy = thisForce.y * R2; 
  Fz = thisForce.z * R2;
}

//
    

//--------------------------------------------------------------------------
// COMPUTING THE FORCES(2)
//
//posqI   = posq[atomI];
//RvdWI   = params[atomI].x;
//RA      = RvdWI + PROBE_RADIUS;
Rv_ON   = RvdWI+SA_ON;  Rv_ON  *= Rv_ON; 
Rv_OFF  = RvdWI+SA_OFF; Rv_OFF *= Rv_OFF;
Factor  = 4.0f*quadPt.w/(Rv_OFF - Rv_ON);
R2      = 0.1025f+PROBE_RADIUS; // Mean of RvdW [0.0225, 0.2275]
quadPtX = posqI.x - R2*quadPt.x;
quadPtY = posqI.y - R2*quadPt.y;
quadPtZ = posqI.z - R2*quadPt.z;
numAtomsInGrid = 0;
k = floor((quadPtZ-MinZ) * INVERSE_DELTA_R);
if ( k >= 0  &&  k < dimZ ) {
  i = floor((quadPtX-MinX) * INVERSE_DELTA_R);
  if ( i >= 0  &&  i < dimX ) {      
    j = floor((quadPtY-MinY) * INVERSE_DELTA_R);
    if ( j >= 0  &&  j < dimY  ) {       
        gridLoc = (i + 
           (j*dimX) +
           (k*dimX*dimY)) * MAX_ATOMS_IN_VOXEL;
        numAtomsInGrid = lookupTable[gridLoc]; 
    }
  }
}
thisForce.x=0.f; thisForce.y=0.f; thisForce.z=0.f;
atomCounter = 1;
while(atomCounter <= numAtomsInGrid) {
   atomJ = lookupTable[gridLoc + atomCounter];
   if(atomJ != atomI) {
     posqJ = posq[atomJ];
     RvdWJ = params[atomJ].x;
     // RA
     RA    = RvdWJ + PROBE_RADIUS;
     // [ddx, ddy, ddz]
     ddx   = posqI.x - RA*quadPt.x - posqJ.x;
     ddy   = posqI.y - RA*quadPt.y - posqJ.y;
     ddz   = posqI.z - RA*quadPt.z - posqJ.z;
     // f_umij & VEmi
     f_umij=0.0f;
     VEmi = 0.0f;
     Rv_ON = (RvdWI+SA_ON);  Rv_ON  *= Rv_ON;
     Rv_OFF= (RvdWI+SA_OFF); Rv_OFF *= Rv_OFF;
     umij  = (ddx*ddx+ddy*ddy+ddz*ddz - Rv_ON) / (Rv_OFF - Rv_ON);
     if (0.f<umij && umij<1.f) { 
       f_umij = umij*umij*(umij*(60.f-30.f*umij)-30.f);
       quadPtX = posqJ.x + RA*quadPt.x;
       quadPtY = posqJ.y + RA*quadPt.y;
       quadPtZ = posqJ.z + RA*quadPt.z; 
       numAtomsInGridi = 0;
       k = floor((quadPtZ-MinZ) * INVERSE_DELTA_R);
       if ( k >= 0  &&  k < dimZ ) {
         i = floor((quadPtX-MinX) * INVERSE_DELTA_R);
         if ( i >= 0  &&  i < dimX ) {
           j = floor((quadPtY-MinY) * INVERSE_DELTA_R);
           if ( j >= 0  &&  j < dimY  ) {
               gridLoci = (i +
                  (j*dimX) +
                  (k*dimX*dimY)) * MAX_ATOMS_IN_VOXEL;
               numAtomsInGridi = lookupTable[gridLoci];
           }
         }
       }
       umij = 0.0f;
       atomCounteri = 1;
       while(atomCounteri <= numAtomsInGridi && umij<1.0f) {
          k = lookupTable[gridLoci + atomCounteri];
          if(k != atomJ) {
            posqK    = posq[k];
            RvdWK    = params[k].x;
            posqK.x  = quadPtX - posqK.x;
            posqK.y  = quadPtY - posqK.y;
            posqK.z  = quadPtZ - posqK.z;
            Rv_ON   = (RvdWK+SA_ON);  Rv_ON  *= Rv_ON;
            Rv_OFF  = (RvdWK+SA_OFF); Rv_OFF *= Rv_OFF;
            R2  = (posqK.x*posqK.x+posqK.y*posqK.y+posqK.z*posqK.z - Rv_ON) / (Rv_OFF - Rv_ON);
            if (R2<0.0f) {
              umij = 2.0f;}
            else if (R2<1.f) {
              R2 = 1.0f + R2*R2*R2*(R2*(15.0f-6.0f*R2)-10.0f);
              umij += 2.0f*R2;
            }
          }
          atomCounteri++;
       }
       if (umij < 1.0f) VEmi  = umij*umij*(umij*(60.f-30.f*umij)-30.f); 
     } // VEmi
     // 
     R2 = SURFACE_AREA_FACTOR*12.566371f*RA*RA*f_umij*VEmi;
     thisForce.x += R2 * ddx;
     thisForce.y += R2 * ddy;
     thisForce.z += R2 * ddz;
   } // if(atomJ != atomI) 
   atomCounter++;
} // while
Fx += thisForce.x * Factor; 
Fy += thisForce.y * Factor; 
Fz += thisForce.z * Factor;
//R2 = Fx*Fx + Fy*Fy + Fz*Fz;
//if (R2>1.0e-4f) {
  atomicAdd( &ForcesX, Fx);
  atomicAdd( &ForcesY, Fy);
  atomicAdd( &ForcesZ, Fz);
//}
  
__syncthreads();
if (threadIdx.x==0 ) {  
  energyBuffer[atomI] += Energy; //Problem: CUDA(700), because the size(energyBuffer) < #NAtom 
  forceBuffers[atomI] += static_cast<unsigned long long>((long long) (-ForcesX*0x100000000));
  forceBuffers[atomI + PADDED_NUM_ATOMS] += static_cast<unsigned long long>((long long) (-ForcesY*0x100000000));
  forceBuffers[atomI + 2*PADDED_NUM_ATOMS] += static_cast<unsigned long long>((long long) (-ForcesZ*0x100000000));
//Energy_SASA[ atomI ]   = Energy;
//Forces_SASA[ atomI ].x = ForcesX;
//Forces_SASA[ atomI ].y = ForcesY;
//Forces_SASA[ atomI ].z = ForcesZ;
}
// ------------------------------------------------
//

} // END SUBROUTINE 



/*
 * here we finish the force calculation by summing up the gradient components
 *  and apply the forces on each atom
 */

extern "C" __global__ void reduceGBMVForce(
    // inputs
// bornForce: the GB force in terms of atomic positions
// forceBuffers: the atomic forces
    const real4* __restrict__ posq, const float2* __restrict__ params,
    const float4* QuadPts, int *lookupTable,
    float* gbmvChain, unsigned long long* __restrict__ bornForce,
    // outputs
  //float4* __restrict__ Forces_SASA,
    unsigned long long* __restrict__ forceBuffers ) {

    // these arrays hold values to allow threads of this block to communicate
    __shared__ float Fx_Sums[256],Fy_Sums[256],Fz_Sums[256];

    // separate these variables from all other threads
    float quadPtX, quadPtY, quadPtZ, // the grid positions
          RvdWI, RvdWJ, // RvdW at each atom
          ddx, ddy, ddz, R2, ForcesX=0.f,ForcesY=0.f,ForcesZ=0.f;
    float Fx,Fy,Fz;  

    float unij,Fvdw,Fmv2,DFvdw,DFmv2,EXPSMV,XX; 

    float  Y1,Y2,Y3,Y4,Y5,dGdRGB_I,dGdRGB_J;
    float  X1,X2,X3_X,X3_Y,X3_Z,X4;

    int   numAtomsInGrid, atomCounter, atomJ, atomI,
          i, j, k, gridLoc, gridLoci,  Idx, NGrids;

          atomI  = blockIdx.x;  // each block governs an atom

    real4   posqI, posqJ; 


#ifdef USE_PERIODIC
    real4 tran = posq[0], 
          box = periodicBoxSize,
          invbox = invPeriodicBoxSize;
    int4  gdim = make_int4( ceil(periodicBoxSize.x / DELTA_R), 
                            ceil(periodicBoxSize.y / DELTA_R), 
                            ceil(periodicBoxSize.z / DELTA_R), 0);
#else
    // system extrema, translation vector
    const float MinX = minXYZ.x, MaxX = maxXYZ.x,
                MinY = minXYZ.y, MaxY = maxXYZ.y,
                MinZ = minXYZ.z, MaxZ = maxXYZ.z,
                tranX = TranslateXYZ.x,
                tranY = TranslateXYZ.y,
                tranZ = TranslateXYZ.z;

    // read in this block's dimesions and translation vector
    const int dimX = GridDimXYZ.x,
              dimY = GridDimXYZ.y,
              dimZ = GridDimXYZ.z;
#endif


  //// GBMV2 parameters; 
  //// The nm unit was used in the OpenMM, instead of the A unit for the distance
  //const float    
  //      BETA_GBMV2   = -12.0f,
  //      LAMBDA_GBMV2 = 0.5f, 
  //      A0_GBMV2     = 1.0f - sqrtf(0.5f), 
  //      A1_GBMV2     = 1.0f,
  //      ALPHA_GBMV2  = -19.8f,
  //      SLOPE_GBMV2     = 0.9085f,  // slope
  //      SHIFT_GBMV2     = -0.0102f, // shift
  //      P3_GBMV2     = 0.65f;    // S0     
    float VDWON_GBMV2,VDWOFF_GBMV2,MV2ON_GBMV2,MV2OFF_GBMV2,P1,P2;
          VDWON_GBMV2=HSX1_GBMV2; VDWOFF_GBMV2=HSX2_GBMV2;
          MV2OFF_GBMV2=OFFX_GBMV2;  MV2ON_GBMV2 = ONX_GBMV2;
          P1=P1_GBMV2; P2=P2_GBMV2;
    float4 quadPt;

if (params[atomI].x == 0.f) return;
// Common variables
posqI = posq[atomI]; // Ith atomic position // ***
NGrids=NUM_RADII*NUM_ANGLES;

//--------------------------------------------------------------------------
// PART I: 
//
Fx=0.f; Fy=0.f; Fz=0.f;
for ( Idx = threadIdx.x; Idx < NGrids; Idx += blockDim.x ) {
  //Idx=threadIdx.x;
  gridLoci = atomI*NGrids*6 + Idx*6;
  EXPSMV = gbmvChain[gridLoci];  
  if( EXPSMV < 0.f && Idx<NGrids){

    // look for the neighbor atoms
    quadPt = QuadPts[Idx];// grid positions
    quadPtX = quadPt.x + posqI.x;
    quadPtY = quadPt.y + posqI.y;
    quadPtZ = quadPt.z + posqI.z;
    k = floor((quadPtZ-MinZ) * INVERSE_DELTA_R);
    if ( k >= 0  &&  k < dimZ ) {
      i = floor((quadPtX-MinX) * INVERSE_DELTA_R);
      if ( i >= 0  &&  i < dimX ) {      
        j = floor((quadPtY-MinY) * INVERSE_DELTA_R);
        if ( j >= 0  &&  j < dimY  ) {       
 
          gridLoc = (i + 
              (j*dimX) +
              (k*dimX*dimY)) * MAX_ATOMS_IN_VOXEL;
          numAtomsInGrid = lookupTable[gridLoc]; // ***
 
        } // bracket Y realspace
      } // bracket X realspace
    } // bracket Z realspace
 
    // calculate Fx, Fy, Fz
    X1  =gbmvChain[gridLoci+1];
    X2  =gbmvChain[gridLoci+2];
    X3_X=gbmvChain[gridLoci+3];
    X3_Y=gbmvChain[gridLoci+4];
    X3_Z=gbmvChain[gridLoci+5];
    X4  =gbmvChain[gridLoci+6];
    atomCounter = 1;
    if (numAtomsInGrid > CUTNUM_GBMV2) numAtomsInGrid=CUTNUM_GBMV2;
    while ( atomCounter <= numAtomsInGrid ) {
 
      // gather atomic info for these atoms near to the grid
      atomJ = lookupTable[gridLoc + atomCounter]; // the Jth atomic label // ***
      posqJ = posq[atomJ]; // the Jth atomic positions // ***
      RvdWJ = params[atomJ].x; // the Jth vdW radii // ***
      atomCounter++;
 
      // calculate the necessary varibales involving the |r_n + R_I - R_J|  
      ddx   = quadPtX - posqJ.x; 
      ddy   = quadPtY - posqJ.y;
      ddz   = quadPtZ - posqJ.z;
      R2    = ddx * ddx + ddy * ddy + ddz * ddz; 
      Y1    = RvdWJ + VDWON_GBMV2;  Y1 *= Y1;
      Y2    = RvdWJ + VDWOFF_GBMV2; Y2 *= Y2;
      Y3    = RvdWJ + MV2ON_GBMV2;  Y3 *= Y3;
      Y4    = RvdWJ + MV2OFF_GBMV2; Y4 *= Y4;
      
      // calculate Fx, Fy, Fz
      //if (atomI != atomJ && R2 < Y4 && R2 > Y1) {
      if (R2 < Y4 && R2 > Y1) {
         if (R2 <= Y2) { 
            unij  = (R2-Y1) / (Y2-Y1);
            Fvdw  = 1.f + unij*unij*unij*(unij*(15.f-6.f*unij)-10.f);
            DFvdw = 60.f*unij*unij*(unij*(2.f-unij)-1.f)/(Y2-Y1);
            XX    = P1 + P2*RvdWJ; unij  = XX+R2-RvdWJ*RvdWJ;
            Fmv2  = XX*XX/(unij*unij);DFmv2=-4.f*XX*XX/(unij*unij*unij); 
            DFmv2 = DFmv2*(1.f-Fvdw)-Fmv2*DFvdw;Fmv2=(1.f-Fvdw)*Fmv2;
            //Fmv2  = (1.f - Fvdw) * expf(ALPHA_GBMV2*(sqrtf(R2)-RvdWJ));
            //DFmv2 = sqrtf(R2); DFmv2=ALPHA_GBMV2*expf(ALPHA_GBMV2*(DFmv2-RvdWJ))/DFmv2;
            XX  = X1*DFmv2;
            XX += X2*2.f*Fmv2*(Fmv2+DFmv2*R2);
            XX -= DFmv2*(ddx*X3_X+ddy*X3_Y+ddz*X3_Z);
            XX += 2.f*DFvdw;
            Fx += EXPSMV*(XX*ddx - Fmv2*X3_X);
            Fy += EXPSMV*(XX*ddy - Fmv2*X3_Y);
            Fz += EXPSMV*(XX*ddz - Fmv2*X3_Z);
         } else if (R2 <= Y3) {
          //Fvdw  = 0.f; Svdw += 2.f * Fvdw; // VDW
          //Fmv2  = expf(ALPHA_GBMV2*(sqrtf(R2)-RvdWJ));
            XX    = P1 + P2*RvdWJ; unij  = XX+R2-RvdWJ*RvdWJ;
            Fmv2  = XX*XX/(unij*unij);
            DFmv2 = -4.f*XX*XX/(unij*unij*unij);
            XX  = X1*DFmv2;
            XX += X2*2.f*Fmv2*(Fmv2+DFmv2*R2);
            XX -= DFmv2*(ddx*X3_X+ddy*X3_Y+ddz*X3_Z);
            Fx += EXPSMV*(XX*ddx - Fmv2*X3_X);
            Fy += EXPSMV*(XX*ddy - Fmv2*X3_Y);
            Fz += EXPSMV*(XX*ddz - Fmv2*X3_Z);
         } else {
            unij  = (R2-Y3) / (Y4-Y3);
            Fvdw  = 1.f + unij*unij*unij*(unij*(15.f-6.f*unij)-10.f);
            DFvdw = 60.f*unij*unij*(unij*(2.f-unij)-1.f)/(Y4-Y3);
            XX    = P1 + P2*RvdWJ; unij  = XX+R2-RvdWJ*RvdWJ;
            Fmv2  = XX*XX/(unij*unij);DFmv2=-4.f*XX*XX/(unij*unij*unij); 
            DFmv2 = DFmv2*Fvdw+Fmv2*DFvdw;Fmv2=Fvdw*Fmv2;
            XX  = X1*DFmv2;
            XX += X2*2.f*Fmv2*(Fmv2+DFmv2*R2);
            XX -= DFmv2*(ddx*X3_X+ddy*X3_Y+ddz*X3_Z);
            Fx += EXPSMV*(XX*ddx - Fmv2*X3_X);
            Fy += EXPSMV*(XX*ddy - Fmv2*X3_Y);
            Fz += EXPSMV*(XX*ddz - Fmv2*X3_Z);
         }
      } // if (R2 <= Y4 && R2 > Y1)  
      
    } // while loop neighbor atoms 

  } // if(EXPSMV)
} // for (loop grids)

// Do the summation 
  Fx_Sums[threadIdx.x]=Fx;Fy_Sums[threadIdx.x]=Fy;Fz_Sums[threadIdx.x]=Fz;__syncthreads();
  if(threadIdx.x== 0) {
    Fx=0.f;Fy=0.f;Fz=0.f;
    for(i=0;i<blockDim.x;i++) {Fx+=Fx_Sums[i];Fy+=Fy_Sums[i];Fz+=Fz_Sums[i];}
    dGdRGB_I = RECIP(0x100000000)*static_cast<long long>(bornForce[atomI]);
    ForcesX = dGdRGB_I*Fx;
    ForcesY = dGdRGB_I*Fy;
    ForcesZ = dGdRGB_I*Fz;
  }

// Wait for the second part
__syncthreads();
//

//--------------------------------------------------------------------------
// PART II: 
//
RvdWI = params[atomI].x; // the Ith vdW radii // ***
Y1    = RvdWI + VDWON_GBMV2;  Y1 *= Y1;
Y2    = RvdWI + VDWOFF_GBMV2; Y2 *= Y2;
Y3    = RvdWI + MV2ON_GBMV2;  Y3 *= Y3;
Y4    = RvdWI + MV2OFF_GBMV2; Y4 *= Y4;
Y5    = RvdWI + MV2OFF_GBMV2 + sqrt(3.f)*DELTA_R/2.f+0.1f; Y5 *= Y5;
Fx=0.f; Fy=0.f; Fz=0.f;
for ( Idx = threadIdx.x; Idx < NGrids; Idx += blockDim.x ) {
  //Idx=threadIdx.x;
  if(Idx<NGrids){

    // look for the neighbor atoms
    quadPt = QuadPts[Idx];// grid positions
    quadPtX = posqI.x - quadPt.x;
    quadPtY = posqI.y - quadPt.y;
    quadPtZ = posqI.z - quadPt.z;
    k = floor((quadPtZ-MinZ) * INVERSE_DELTA_R);
    if ( k >= 0  &&  k < dimZ ) {
      i = floor((quadPtX-MinX) * INVERSE_DELTA_R);
      if ( i >= 0  &&  i < dimX ) {      
        j = floor((quadPtY-MinY) * INVERSE_DELTA_R);
        if ( j >= 0  &&  j < dimY  ) {       
 
          gridLoc = (i + 
              (j*dimX) +
              (k*dimX*dimY)) * MAX_ATOMS_IN_VOXEL;
          numAtomsInGrid = lookupTable[gridLoc]; // ***
 
        } // bracket Y realspace
      } // bracket X realspace
    } // bracket Z realspace
 
    // calculate Fx, Fy, Fz
    atomCounter = 1;
    while ( atomCounter <= numAtomsInGrid ) {
 
      // gather atomic info for these atoms near to the grid
      atomJ = lookupTable[gridLoc + atomCounter]; // the Jth atomic label // ***
      posqJ = posq[atomJ]; // the Jth atomic positions // ***
      atomCounter++;
 
      // calculate the necessary varibales involving the |r_n + R_I - R_J|  
      ddx   = quadPt.x + posqJ.x - posqI.x; 
      ddy   = quadPt.y + posqJ.y - posqI.y;
      ddz   = quadPt.z + posqJ.z - posqI.z;
      R2    = ddx * ddx + ddy * ddy + ddz * ddz; 
      
      if ( R2 > Y5 ) atomCounter = numAtomsInGrid + 1;

      // calculate Fx, Fy, Fz
      if ( R2 <= Y4 && R2 > Y1) {
         gridLoci = atomJ*NGrids*6+Idx*6;
         EXPSMV = gbmvChain[gridLoci];  
         if (EXPSMV < 0.f) {
            X1  =gbmvChain[gridLoci+1];
            X2  =gbmvChain[gridLoci+2];
            X3_X=gbmvChain[gridLoci+3];
            X3_Y=gbmvChain[gridLoci+4];
            X3_Z=gbmvChain[gridLoci+5];
            X4  =gbmvChain[gridLoci+6];
            dGdRGB_J = RECIP(0x100000000)*static_cast<long long>(bornForce[atomJ]);
            if (R2 <= Y2) { 
               unij  = (R2-Y1) / (Y2-Y1);
               Fvdw  = 1.f + unij*unij*unij*(unij*(15.f-6.f*unij)-10.f);
               DFvdw = 60.f*unij*unij*(unij*(2.f-unij)-1.f)/(Y2-Y1);
               XX    = P1 + P2*RvdWI; unij  = XX+R2-RvdWI*RvdWI;
               Fmv2  = XX*XX/(unij*unij);DFmv2=-4.f*XX*XX/(unij*unij*unij); 
               DFmv2=DFmv2*(1.f-Fvdw)-Fmv2*DFvdw;Fmv2=(1.f-Fvdw)*Fmv2;
               //Fmv2  = (1.f - Fvdw) * expf(ALPHA_GBMV2*(sqrtf(R2)-RvdWI));
               //DFmv2 = sqrtf(R2); DFmv2=ALPHA_GBMV2*expf(ALPHA_GBMV2*(DFmv2-RvdWI))/DFmv2;
               XX  = X1*DFmv2;
               XX += X2*2.f*Fmv2*(Fmv2+DFmv2*R2);
               XX += 2.f*DFvdw;
               XX -= DFmv2*(ddx*X3_X+ddy*X3_Y+ddz*X3_Z);
               Fx += dGdRGB_J*EXPSMV*(XX*ddx - Fmv2*X3_X); 
               Fy += dGdRGB_J*EXPSMV*(XX*ddy - Fmv2*X3_Y);
               Fz += dGdRGB_J*EXPSMV*(XX*ddz - Fmv2*X3_Z);
            } else if (R2 <= Y3) {
             //Fvdw  =  0.f; Svdw += 2.f * Fvdw; // VDW
             //Fmv2  =  expf(ALPHA_GBMV2*(sqrtf(R2)-RvdWI));
               XX    = P1 + P2*RvdWI; unij  = XX+R2-RvdWI*RvdWI;
               Fmv2  = XX*XX/(unij*unij);
               DFmv2 = -4.f*XX*XX/(unij*unij*unij);
               XX  = X1*DFmv2;
               XX += X2*2.f*Fmv2*(Fmv2+DFmv2*R2);
               XX -= DFmv2*(ddx*X3_X+ddy*X3_Y+ddz*X3_Z);
               Fx += dGdRGB_J*EXPSMV*(XX*ddx - Fmv2*X3_X); 
               Fy += dGdRGB_J*EXPSMV*(XX*ddy - Fmv2*X3_Y);
               Fz += dGdRGB_J*EXPSMV*(XX*ddz - Fmv2*X3_Z);
            } else {
               unij  = (R2-Y3) / (Y4-Y3);
               Fvdw  = 1.f + unij*unij*unij*(unij*(15.f-6.f*unij)-10.f);
               DFvdw = 60.f*unij*unij*(unij*(2.f-unij)-1.f)/(Y4-Y3);
               XX    = P1 + P2*RvdWI; unij  = XX+R2-RvdWI*RvdWI;
               Fmv2  = XX*XX/(unij*unij);DFmv2=-4.f*XX*XX/(unij*unij*unij); 
               DFmv2 = DFmv2*Fvdw+Fmv2*DFvdw;Fmv2=Fvdw*Fmv2;
               XX  = X1*DFmv2;
               XX += X2*2.f*Fmv2*(Fmv2+DFmv2*R2);
               XX -= DFmv2*(ddx*X3_X+ddy*X3_Y+ddz*X3_Z);
               Fx += dGdRGB_J*EXPSMV*(XX*ddx - Fmv2*X3_X); 
               Fy += dGdRGB_J*EXPSMV*(XX*ddy - Fmv2*X3_Y);
               Fz += dGdRGB_J*EXPSMV*(XX*ddz - Fmv2*X3_Z);
            }
         } // if (EXPSMV)
      } // if (R2 <= Y4 && R2 > Y1) //  
      
    } // while loop neighbor atoms 

  } // if(Idx < NGrids)
} // for (loop grids)
//

// Do the summation 
  Fx_Sums[threadIdx.x]=Fx;Fy_Sums[threadIdx.x]=Fy;Fz_Sums[threadIdx.x]=Fz;__syncthreads();
  if(threadIdx.x== 0) {
    Fx=0.f;Fy=0.f;Fz=0.f;
    for(i=0;i<blockDim.x;i++) {Fx+=Fx_Sums[i];Fy+=Fy_Sums[i];Fz+=Fz_Sums[i];}
    ForcesX -= Fx; ForcesY -= Fy; ForcesZ -= Fz;
    forceBuffers[atomI] += static_cast<unsigned long long>((long long) ( ForcesX*0x100000000));
    forceBuffers[atomI + PADDED_NUM_ATOMS] += static_cast<unsigned long long>((long long) ( ForcesY*0x100000000));
    forceBuffers[atomI + 2*PADDED_NUM_ATOMS] += static_cast<unsigned long long>((long long) ( ForcesZ*0x100000000));
  //Forces_SASA[ atomI ].x = ForcesX;
  //Forces_SASA[ atomI ].y = ForcesY;
  //Forces_SASA[ atomI ].z = ForcesZ;
  }
//

} // END SUBROUTINE 


