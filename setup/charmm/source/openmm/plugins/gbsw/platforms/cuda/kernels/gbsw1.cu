#define PROBE_RADIUS 0.14f

__device__ float4 minXYZ, maxXYZ, TranslateXYZ;
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
    const float4* __restrict__ chargeStates, real* __restrict__ energyBuffer ) {
    
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
    __shared__ float allExtrema[NUM_THREADS * 6], tmpExtrema[6];
    
    int atomInt, numGridsX, numGridsY, numGridsZ, baseLocation;
    real maxX, maxY, maxZ, minX, minY, minZ,
        atomRadius, atomX, atomY, atomZ,
        thisExtreme;
    real4 posq_XYZQ;
    
    // initialize gridspace extrema variables
    maxX = -99999.0f; maxY = -99999.0f; maxZ = -99999.0f; // max floats
    minX =  99999.0f; minY =  99999.0f; minZ =  99999.0f; // min floats
    
    // each thread takes a few atoms and check amongst those few the maxima
    for (atomInt = threadIdx.x; atomInt < NUM_ATOMS; atomInt += NUM_THREADS) {
        
        // this atom's info
        atomRadius = params[atomInt].x;
        posq_XYZQ = posq[atomInt];
        
        atomX = posq_XYZQ.x;
        atomY = posq_XYZQ.y;
        atomZ = posq_XYZQ.z;
        
        // check for maximum (but only if no periodic bounds)
        if (atomX + atomRadius > maxX)  maxX = atomX + atomRadius;
        if (atomY + atomRadius > maxY)  maxY = atomY + atomRadius;
        if (atomZ + atomRadius > maxZ)  maxZ = atomZ + atomRadius;
        
        // check for minimum
        if (atomX - atomRadius < minX)  minX = atomX - atomRadius;
        if (atomY - atomRadius < minY)  minY = atomY - atomRadius;
        if (atomZ - atomRadius < minZ)  minZ = atomZ - atomRadius;
    }
    
    // funnel results into global matrix
    allExtrema[threadIdx.x]                   = minX;
    allExtrema[threadIdx.x + 1 * NUM_THREADS] = minY;
    allExtrema[threadIdx.x + 2 * NUM_THREADS] = minZ;
    allExtrema[threadIdx.x + 3 * NUM_THREADS] = maxX;
    allExtrema[threadIdx.x + 4 * NUM_THREADS] = maxY;
    allExtrema[threadIdx.x + 5 * NUM_THREADS] = maxZ;
    
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
    }
    
    // be sure all threads have finished before continuing, then calc grid
    // dimensions and the translation vector for each atom
    __syncthreads();
    
   if (threadIdx.x == 0) { // x dim
        minX = tmpExtrema[0];
        maxX = tmpExtrema[3];
        numGridsX = floor((maxX - minX + 2.0f*R_BUFFER) * INVERSE_DELTA_R);
        GridDimXYZ_external[0] = numGridsX;
        GridDimXYZ.x = numGridsX;
        TranslateXYZ.x = 0.5f * ((float)numGridsX-1.0f) * DELTA_R - (maxX + minX) * 0.5f;
        
    } else if (threadIdx.x == 1) { // y dim
        minY = tmpExtrema[1];
        maxY = tmpExtrema[4];
        numGridsY = floor((maxY - minY + 2.0f*R_BUFFER) * INVERSE_DELTA_R);
        GridDimXYZ_external[1] = numGridsY;
        GridDimXYZ.y = numGridsY;
        TranslateXYZ.y = 0.5f * ((float)numGridsY-1.0f) * DELTA_R - (maxY + minY) * 0.5f;
        
    } else if (threadIdx.x == 2) { // z dim
        minZ = tmpExtrema[2];
        maxZ = tmpExtrema[5];
        numGridsZ = floor((maxZ - minZ + 2.0f*R_BUFFER) * INVERSE_DELTA_R);
        GridDimXYZ_external[2] = numGridsZ;
        GridDimXYZ.z = numGridsZ;
        TranslateXYZ.z = 0.5f * ((float)numGridsZ-1.0f) * DELTA_R - (maxZ + minZ) * 0.5f;
        
    } else if (threadIdx.x == 3) { 
        minXYZ = make_float4( tmpExtrema[0], tmpExtrema[1], tmpExtrema[2], 0.0f );
    } else if (threadIdx.x == 4) { 
        maxXYZ = make_float4( tmpExtrema[3], tmpExtrema[4], tmpExtrema[5], 0.0f );
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

typedef struct {
    int8_t  x, y, z, r;
} gridVector;


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
        atomR += R_BUFFER;
        atomRBuffer_2 = atomR * atomR;
        
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
            
            atomR += R_BUFFER;
            atomRBuffer_2 = atomR * atomR;
            
            dx = atomX - (float)i * DELTA_R;
            dy = atomY - (float)j * DELTA_R;
            dz = atomZ - (float)k * DELTA_R;
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
            dx = posq_XYZQ.x - tran.x - (float)blockIdx.x * DELTA_R;
            dy = posq_XYZQ.y - tran.y - (float)blockIdx.y * DELTA_R;
            dz = posq_XYZQ.z - tran.z - (float)blockIdx.z * DELTA_R;
            dx -= floor( dx*invbox.x + 0.5f ) * box.x;
            dy -= floor( dy*invbox.y + 0.5f ) * box.y;
            dz -= floor( dz*invbox.z + 0.5f ) * box.z;
            
#else
            // calc and save distance to grid center
            tran = TranslateXYZ;
            dx = posq_XYZQ.x + tran.x - (float)blockIdx.x * DELTA_R;
            dy = posq_XYZQ.y + tran.y - (float)blockIdx.y * DELTA_R;
            dz = posq_XYZQ.z + tran.z - (float)blockIdx.z * DELTA_R;
#endif
            
            R2grid_2 = dx*dx + dy*dy + dz*dz;
            allR2grid_2[threadIdx.x] = R2grid_2;
            
            // since this is one warp, no need to sync the threads
            numPriorAtoms = 0;
            for ( n = 0; n < numAtomsInGrid; n++ )
                if ( allR2grid_2[n] < R2grid_2 )
                    numPriorAtoms++;
            
            // record sorted-index of atoms
            lookupTable[ gridLoc + numPriorAtoms + 1 ] = atomInt;
        }
    }
}

/*
 * the Born Radius calculation is split by hydrogens (radius == 0.0) and heavy
 * atoms (radius > 0.0). these use a different number of radii, so this split 
 * speeds things up a little. these kernels also calculate the first derivative
 * of the Born R, which is finished up in the reduceGBSWForceKernel
 */

extern "C" __global__ void calcBornR_hydrogen( 
    // inputs
    const real4* __restrict__ posq, const float2* __restrict__ params,
    const float4* QuadPts, const float2* QuadPtWeights,
    int *lookupTable,
#ifdef USE_PERIODIC
    real4 periodicBoxSize, real4 invPeriodicBoxSize,
#endif
    // outputs
    real* __restrict__ BornRadii, float4* gbswChain, int *nGBSWchainAtoms ) {
    
    if ( params[blockIdx.x].x - SWICH_LEN < 0.0f ) {
    
    //--------------------------------------------------------------------------
    // initialize variables
    
    // these arrays hold values to allow threads of this block to communicate
    __shared__ float allRadSums_AA0[NUM_RADII_H], allRadSums_AA1[NUM_RADII_H],
        allAngularSums[NUM_RADII_H][NUM_ANGLES], dG1, BornR;
    
    // separate these variables from all other threads
    float // quadrature calculation floats
        atomBufferedR_2, R2quadPt_2,
        quadPtX, quadPtY, quadPtZ, atomR,
        atomInnerR_2, angularSum, dx, dy, dz,
        // switching function floats
        r, r2, switchFn, sf, wtAndSf, aa0aa1Term,
        // born radius calculation floats
        volumetricSum_AA0, volumetricSum_AA1, dG1sum, dBornR_dr, r2point,
        AA0term, AA1term, sfList[MAX_ATOMS_IN_VOXEL], rList[MAX_ATOMS_IN_VOXEL];
    
    int numAtomsInGrid, atomCounter, atomJ, 
        angInt, radInt, atomI, sfAtomCount,
        i, j, k, gridLoc;
    real4 localForces[MAX_ATOMS_IN_VOXEL], thisForce, posq_XYZQ;
    
    //--------------------------------------------------------------------------
    // -- import parameters from system for this Block (atom) --
    
    // main parallel loop variables
    atomI = blockIdx.x;  // each block governs an atom
    radInt = threadIdx.y;  // each block's y thread handles a radial integration
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
    const float MinX = minXYZ.x - SWICH_LEN, MaxX = maxXYZ.x + SWICH_LEN,
                MinY = minXYZ.y - SWICH_LEN, MaxY = maxXYZ.y + SWICH_LEN,
                MinZ = minXYZ.z - SWICH_LEN, MaxZ = maxXYZ.z + SWICH_LEN,
                tranX = TranslateXYZ.x,
                tranY = TranslateXYZ.y,
                tranZ = TranslateXYZ.z;
    
    // read in this block's dimesions and translation vector
    const int dimX = GridDimXYZ.x,
              dimY = GridDimXYZ.y;
#endif
    
    float4 quadPt = QuadPts[ (threadIdx.y + SKIP_RADII_H)*NUM_ANGLES + threadIdx.x ];
    float2 quadWt = QuadPtWeights[ (threadIdx.y + SKIP_RADII_H)*NUM_ANGLES + threadIdx.x ];
    
    //--------------------------------------------------------------------------
    // calculate XYZ of each quadrature point, evaluate atomic occupancy at that
    // point, sum it all up together (integrate)
    
    // record atomic occupancy (switchFn is 0, inside atom; switchFn is 1, outside atoms)
    switchFn = 1.0f;
    sfAtomCount = 0;
    posq_XYZQ = posq[atomI];
    
    
#ifdef USE_PERIODIC
    // generate quadrature point (quadX, quadY, quadZ)
    quadPtX = quadPt.x + posq_XYZQ.x - tran.x;
    quadPtY = quadPt.y + posq_XYZQ.y - tran.y;
    quadPtZ = quadPt.z + posq_XYZQ.z - tran.z;
    
#ifdef USE_MEMBRANE
    if ( abs(quadPtZ) > MEMBRANE_INNER_R ) {
        
    // is point is inside membrane's switching function?
    if ( abs(quadPtZ) < MEMBRANE_OUTER_R ) {
        
        // above or below the XY plane?
        if ( quadPtZ > 0.0f ) {
            r = quadPtZ - MEMBRANE_R;
            sf = 0.5f + MEMBRANE_PRECALC1 * r - MEMBRANE_PRECALC2 * r * r * r;
        } else {
            r = quadPtZ + MEMBRANE_R;
            sf = 0.5f - MEMBRANE_PRECALC1 * r + MEMBRANE_PRECALC2 * r * r * r;
        }
        switchFn *= sf;
    }
#endif
    
    // convert to gridspace
    i = floor((quadPtX + DELTA_R * 0.5f) * INVERSE_DELTA_R);
    j = floor((quadPtY + DELTA_R * 0.5f) * INVERSE_DELTA_R);
    k = floor((quadPtZ + DELTA_R * 0.5f) * INVERSE_DELTA_R);
    
    // enforce periodic bounds
    if (i < 0) i += gdim.x; else if (i >= gdim.x) i -= gdim.x;
    if (j < 0) j += gdim.y; else if (j >= gdim.y) j -= gdim.y;
    if (k < 0) k += gdim.z; else if (k >= gdim.z) k -= gdim.z;
    
    // locate voxel in lookup table
    gridLoc = (i + 
              (j*gdim.x) +
              (k*gdim.x*gdim.y)) * MAX_ATOMS_IN_VOXEL;
    
    
    // check if gridspace is empty. If not, calculate atomic occupancy
    if ( lookupTable[gridLoc] > 0 ) {
        
        // location on lookup table
        numAtomsInGrid = lookupTable[gridLoc];
        
        // begin occupancy calculation
        atomCounter = 1;
        while ( atomCounter <= numAtomsInGrid ) {
            
            // gather information of this atom in grid
            atomJ = lookupTable[gridLoc + atomCounter];
            posq_XYZQ = posq[atomJ];
            atomR = params[atomJ].x;
            atomCounter++;
            
            // calculate distance to quadrature point
            dx = quadPtX - posq_XYZQ.x + tran.x;
            dy = quadPtY - posq_XYZQ.y + tran.y;
            dz = quadPtZ - posq_XYZQ.z + tran.z;
            
            // apply periodic bounds
            dx -= floor( dx*invbox.x + 0.5f ) * box.x;
            dy -= floor( dy*invbox.y + 0.5f ) * box.y;
            dz -= floor( dz*invbox.z + 0.5f ) * box.z;
            R2quadPt_2 = dx * dx + dy * dy + dz * dz;
        
            // atom radius^2
            atomBufferedR_2 = atomR + SWICH_LEN;
            atomBufferedR_2 *= atomBufferedR_2;
            
            // check if point is inside range of neighboring atom
            if ( R2quadPt_2 < atomBufferedR_2 ) {
            
            // inner radius (below which switching function is 0)
            atomInnerR_2 = atomR - SWICH_LEN;
            atomInnerR_2 *= atomInnerR_2;
        
            // is point within inner radius of neighbor atom?
            if ( R2quadPt_2 <= atomInnerR_2 ) {
                
                // we are inside neighbor atom, so swiching function is zero
                switchFn = 0.0f;
                atomCounter = numAtomsInGrid + 1; // exit loop!
                sfAtomCount = 0;
                
            } else {
                
                // otherwise evaluate switching function
                r2point = sqrtf(R2quadPt_2);
                r = r2point - atomR;
                sf = 0.5f + PRECALC1 * r - PRECALC2 * r * r * r;
                switchFn *= sf;
                
                // record info should this be a gradient quadrature point
                // remember, there is no self-contribution here!
                if ( atomJ != atomI && sfAtomCount < MAX_ATOMS_IN_VOXEL ) {
                    
                    sfList[sfAtomCount] = sf * r2point;
                    rList[sfAtomCount] = r * r;
                    
                    thisForce.x = dx;
                    thisForce.y = dy;
                    thisForce.z = dz;
                    
                    // a float4 is easier to deal with than an arbitrary struct
                    memcpy(&thisForce.w, &atomJ, 4);
                    localForces[sfAtomCount] = thisForce;
                    
                    sfAtomCount ++;
                }
                
            } // decide within inner radius of atom or at smooth switching
            } // inside range of neighbor atom check
        } // calculate occupancy loop
    } // empty grid check
    
#ifdef USE_MEMBRANE
    } else
        switchFn = 0.0f; // inside membrane
#endif
    
#else // periodic bounds check
    
    // generate quadrature point (quadX, quadY, quadZ)
    // bracket quadrature to local space
    quadPtZ = quadPt.z + posq_XYZQ.z;
    
#ifdef USE_MEMBRANE
    if ( abs(quadPtZ) > MEMBRANE_INNER_R ) {
        
    // is point is inside membrane's switching function?
    if ( abs(quadPtZ) < MEMBRANE_OUTER_R ) {
        
        // above or below the XY plane?
        if ( quadPtZ > 0.0f ) {
            r = quadPtZ - MEMBRANE_R;
            sf = 0.5f + MEMBRANE_PRECALC1 * r - MEMBRANE_PRECALC2 * r * r * r;
        } else {
            r = quadPtZ + MEMBRANE_R;
            sf = 0.5f - MEMBRANE_PRECALC1 * r + MEMBRANE_PRECALC2 * r * r * r;
        }
        switchFn *= sf;
    }
#endif
    
    if ( quadPtZ < MaxZ  &&  quadPtZ > MinZ ) {
    quadPtX = quadPt.x + posq_XYZQ.x;
    if ( quadPtX < MaxX  &&  quadPtX > MinX ) {
    quadPtY = quadPt.y + posq_XYZQ.y;
    if ( quadPtY < MaxY  &&  quadPtY > MinY ) {
        
        // if gridpoint is local, convert to gridspace
        i = floor((quadPtX + tranX + DELTA_R * 0.5f) * INVERSE_DELTA_R);
        j = floor((quadPtY + tranY + DELTA_R * 0.5f) * INVERSE_DELTA_R);
        k = floor((quadPtZ + tranZ + DELTA_R * 0.5f) * INVERSE_DELTA_R);
        
        // locate voxel in lookup table
        gridLoc = (i + 
                  (j*dimX) +
                  (k*dimX*dimY)) * MAX_ATOMS_IN_VOXEL;
        
        // check if gridspace is empty. If not, calculate atomic occupancy
        if ( lookupTable[gridLoc] > 0 ) {
        
        // location on lookup table
        numAtomsInGrid = lookupTable[gridLoc];
        
        // begin occupancy calculation
        atomCounter = 1;
        while ( atomCounter <= numAtomsInGrid ) {
            
            // gather information of this atom in grid
            atomJ = lookupTable[gridLoc + atomCounter];
            posq_XYZQ = posq[atomJ];
            atomR = params[atomJ].x;
            atomCounter++;
            
            // calculate distance to quadrature point
            dx = quadPtX - posq_XYZQ.x;
            dy = quadPtY - posq_XYZQ.y;
            dz = quadPtZ - posq_XYZQ.z;
            R2quadPt_2 = dx * dx + dy * dy + dz * dz;
            
            // atom radius^2
            atomBufferedR_2 = atomR + SWICH_LEN;
            atomBufferedR_2 *= atomBufferedR_2;
            
            // check if point is inside range of neighboring atom
            if ( R2quadPt_2 < atomBufferedR_2 ) {
            
            // inner radius (below which switching function is 0)
            atomInnerR_2 = atomR - SWICH_LEN;
            atomInnerR_2 *= atomInnerR_2;
            
            // is point within inner radius of neighbor atom?
            if ( R2quadPt_2 <= atomInnerR_2 ) {
                
                // we are inside neighbor atom, so swiching function is zero
                switchFn = 0.0f;
                atomCounter = numAtomsInGrid + 1; // exit loop!
                sfAtomCount = 0;
                
            } else {
                
                // otherwise evaluate switching function
                r2point = sqrtf(R2quadPt_2);
                r = r2point - atomR;
                sf = 0.5f + PRECALC1 * r - PRECALC2 * r * r * r;
                switchFn *= sf;
                
                // record info should this be a gradient quadrature point
                // remember, there is no self-contribution here!
                if ( atomJ != atomI && sfAtomCount < MAX_ATOMS_IN_VOXEL ) {
                    
                    sfList[sfAtomCount] = sf * r2point;
                    rList[sfAtomCount] = r * r;
                    
                    thisForce.x = dx;
                    thisForce.y = dy;
                    thisForce.z = dz;
                    
                    // a float4 is easier to deal with than an arbitrary struct
                    memcpy(&thisForce.w, &atomJ, 4);
                    localForces[sfAtomCount] = thisForce;
                    
                    sfAtomCount ++;
                }
                
            } // decide within inner radius of atom or at smooth switching
            } // inside range of neighbor atom check
        } // calculate occupancy loop
        } // empty grid check
        
    
    } // bracket Y realspace
    } // bracket X realspace
    } // bracket Z realspace
#ifdef USE_MEMBRANE
    } else
        switchFn = 0.0f; // inside membrane
#endif
    
#endif
    
    // save angular quadrature sum, convert curly-H (switching function) to "nu"
    allAngularSums[radInt][angInt] = quadWt.x * (1.0f - switchFn);
    
    // ensure all threads are done before summation of the quadrature
    __syncthreads();
    
    //--------------------------------------------------------------------------
    // -- integrate angular components of atomic occupancy --
    
    // only finish job with y threads (radInt is variable, angInt not needed)
    if ( threadIdx.x == 0 ) {
        
        // add up (integrate) the angular quadratures for current radius
        angularSum = 0.0f;
        for (angInt = 0; angInt < NUM_ANGLES; angInt++)
            angularSum += allAngularSums[radInt][angInt];
        
        // save radial sum until other threads have finished
        allRadSums_AA0[radInt] = angularSum;
        allRadSums_AA1[radInt] = angularSum * quadWt.y;

    }
    __syncthreads();
    
    //--------------------------------------------------------------------------
    // -- convert atomic occupancy into Born Radius --
    
    // only finish job with one thread
    if ( threadIdx.y == 0 && threadIdx.x == 0 ) {
        
        // calculate total volumetric sums
        volumetricSum_AA0 = PREINTEGRATE1;
        volumetricSum_AA1 = PREINTEGRATE2;
        for (radInt = 0; radInt < NUM_RADII_H; radInt++) {
            volumetricSum_AA0 += allRadSums_AA0[radInt];
            volumetricSum_AA1 += allRadSums_AA1[radInt];
        }
        
        // do final Born radius calculation
        AA0term = CONST_AA0 * (1.0f / CONST_RMIN - volumetricSum_AA0);
        dG1 = sqrtf(sqrtf(PRECALC3 - volumetricSum_AA1));
        AA1term = CONST_AA1 * dG1;
        BornR = 1.0f / (AA0term + AA1term);
        BornRadii[atomI] = BornR;
    }
    
    __syncthreads();
    
    // if this is a gradient-lying quadrature point, we need to calculate
    // the d(Born R) / d(r atom) gradient component. since only the slow threads
    // get this far, they self-select to fit within a warp or two
    if ( sfAtomCount > 0 ) {
        
        // locate where to fit the info into the global matricies
        atomCounter = atomicAdd( &nGBSWchainAtoms[atomI], sfAtomCount );
        
        // case 1: all atoms can be input
        if ( atomCounter + sfAtomCount < MAX_SF_NEIGHBORS ) {
            gridLoc = atomCounter + atomI * (MAX_SF_NEIGHBORS+1);
            
            // precalc of the term
            dG1sum = 0.25f / (dG1*dG1*dG1);
            aa0aa1Term = CONST_AA0 * quadWt.x + dG1sum*CONST_AA1 * quadWt.x * quadWt.y;
            wtAndSf = BornR*BornR * switchFn * aa0aa1Term;
            
            for ( int n = 0; n < sfAtomCount; n++ ) {
                
                r = rList[n];
                sf = sfList[n];
                
                dBornR_dr = wtAndSf * (PRECALC1 - PRECALC4 * r) / sf;
                thisForce = localForces[n];
                thisForce.x *= dBornR_dr;
                thisForce.y *= dBornR_dr;
                thisForce.z *= dBornR_dr;
                
                gbswChain[gridLoc + n] = thisForce;
            }
        // case 2: fill up the last remnant of free space
        } else if ( atomCounter < MAX_SF_NEIGHBORS ) {
            gridLoc = atomCounter + atomI * (MAX_SF_NEIGHBORS+1);
            sfAtomCount = MAX_SF_NEIGHBORS - atomCounter;
            
            // precalc of the term
            dG1sum = 0.25f / (dG1*dG1*dG1);
            aa0aa1Term = CONST_AA0 * quadWt.x + dG1sum*CONST_AA1 * quadWt.x * quadWt.y;
            wtAndSf = BornR*BornR * switchFn * aa0aa1Term;
            
            for ( int n = 0; n < sfAtomCount; n++ ) {
                r = rList[n];
                sf = sfList[n];
                
                dBornR_dr = wtAndSf * (PRECALC1 - PRECALC4 * r) / sf;
                thisForce = localForces[n];
                thisForce.x *= dBornR_dr;
                thisForce.y *= dBornR_dr;
                thisForce.z *= dBornR_dr;
                
                gbswChain[gridLoc + n] = thisForce;
            }
        } // end force gradient calculation
    } // quad pt on gradient check
    
    } // check if the atom of this block is a hydrogen
}


/*
 * second part of the Born radius calculations for heavy atoms. 
 */


extern "C" __global__ void calcBornR_heavy( 
    // inputs
    const real4* __restrict__ posq, const float2* __restrict__ params,
    const float4* QuadPts, const float2* QuadPtWeights,
    int *lookupTable,
#ifdef USE_PERIODIC
    real4 periodicBoxSize, real4 invPeriodicBoxSize,
#endif
    // outputs
    real* __restrict__ BornRadii, float4* gbswChain, int *nGBSWchainAtoms ) {
    
    float atomR_inner = params[blockIdx.x].x - SWICH_LEN;
    if ( atomR_inner > 0.0f ) {
    
    //--------------------------------------------------------------------------
    // initialize variables
    
    // these arrays hold values to allow threads of this block to communicate
    __shared__ float allRadSums_AA0[NUM_RADII_HEAVY], allRadSums_AA1[NUM_RADII_HEAVY],
        allAngularSums[NUM_RADII_HEAVY][NUM_ANGLES], dG1, BornR;
    
    // separate these variables from all other threads
    float // quadrature calculation floats
        atomBufferedR_2, R2quadPt_2,
        quadPtX, quadPtY, quadPtZ, atomR,
        atomInnerR_2, angularSum, dx, dy, dz,
        // switching function floats
        r, r2, switchFn, sf, wtAndSf, aa0aa1Term,
        // born radius calculation floats
        volumetricSum_AA0, volumetricSum_AA1, dG1sum, dBornR_dr, r2point,
        AA0term, AA1term, sfList[MAX_ATOMS_IN_VOXEL], rList[MAX_ATOMS_IN_VOXEL];
    
    int numAtomsInGrid, atomCounter, atomJ, 
        angInt, radInt, atomI, sfAtomCount,
        i, j, k, gridLoc;
    real4 localForces[MAX_ATOMS_IN_VOXEL], thisForce, posq_XYZQ;
    
    //--------------------------------------------------------------------------
    // -- import parameters from system for this Block (atom) --
    
    // main parallel loop variables
    atomI = blockIdx.x;  // each block governs an atom
    radInt = threadIdx.y;  // each block's y thread handles a radial integration
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
    const float MinX = minXYZ.x - SWICH_LEN, MaxX = maxXYZ.x + SWICH_LEN,
                MinY = minXYZ.y - SWICH_LEN, MaxY = maxXYZ.y + SWICH_LEN,
                MinZ = minXYZ.z - SWICH_LEN, MaxZ = maxXYZ.z + SWICH_LEN,
                tranX = TranslateXYZ.x,
                tranY = TranslateXYZ.y,
                tranZ = TranslateXYZ.z;
    
    // read in this block's dimesions and translation vector
    const int dimX = GridDimXYZ.x,
              dimY = GridDimXYZ.y;
#endif
    
    float4 quadPt = QuadPts[ (threadIdx.y + SKIP_RADII_HEAVY)*NUM_ANGLES + threadIdx.x ];
    float2 quadWt = QuadPtWeights[ (threadIdx.y + SKIP_RADII_HEAVY)*NUM_ANGLES + threadIdx.x ];
    
    //--------------------------------------------------------------------------
    // calculate XYZ of each quadrature point, evaluate atomic occupancy at that
    // point, sum it all up together (integrate)
    
    // record atomic occupancy (switchFn is 0, inside atom; switchFn is 1, outside atoms)
    switchFn = 0.0f;
    sfAtomCount = 0;
    
#ifdef USE_PERIODIC
    
    // check if atom is outside inner radius of central atom
    if ( quadPt.w >= atomR_inner ) {
        
        switchFn = 1.0f;
        posq_XYZQ = posq[atomI];
        
        // generate quadrature point (quadX, quadY, quadZ)
        quadPtX = quadPt.x + posq_XYZQ.x - tran.x;
        quadPtY = quadPt.y + posq_XYZQ.y - tran.y;
        quadPtZ = quadPt.z + posq_XYZQ.z - tran.z;
        
#ifdef USE_MEMBRANE
    if ( abs(quadPtZ) > MEMBRANE_INNER_R ) {
        
    // is point is inside membrane's switching function?
    if ( abs(quadPtZ) < MEMBRANE_OUTER_R ) {
        
        // above or below the XY plane?
        if ( quadPtZ > 0.0f ) {
            r = quadPtZ - MEMBRANE_R;
            sf = 0.5f + MEMBRANE_PRECALC1 * r - MEMBRANE_PRECALC2 * r * r * r;
        } else {
            r = quadPtZ + MEMBRANE_R;
            sf = 0.5f - MEMBRANE_PRECALC1 * r + MEMBRANE_PRECALC2 * r * r * r;
        }
        switchFn *= sf;
    }
#endif
        
        // convert to gridspace
        i = floor((quadPtX + DELTA_R * 0.5f) * INVERSE_DELTA_R);
        j = floor((quadPtY + DELTA_R * 0.5f) * INVERSE_DELTA_R);
        k = floor((quadPtZ + DELTA_R * 0.5f) * INVERSE_DELTA_R);
        
        // enforce periodic bounds
        if (i < 0) i += gdim.x; else if (i >= gdim.x) i -= gdim.x;
        if (j < 0) j += gdim.y; else if (j >= gdim.y) j -= gdim.y;
        if (k < 0) k += gdim.z; else if (k >= gdim.z) k -= gdim.z;
        
        // locate voxel in lookup table
        gridLoc = (i + 
                      (j*gdim.x) +
                      (k*gdim.x*gdim.y)) * MAX_ATOMS_IN_VOXEL;
        
        // check if gridspace is empty. If not, calculate atomic occupancy
        if ( lookupTable[gridLoc] > 0 ) {
            
            // location on lookup table
            numAtomsInGrid = lookupTable[gridLoc];
            
            // begin occupancy calculation
            atomCounter = 1;
            while ( atomCounter <= numAtomsInGrid ) {
                
                // gather information of this atom in grid
                atomJ = lookupTable[gridLoc + atomCounter];
                posq_XYZQ = posq[atomJ];
                atomR = params[atomJ].x;
                atomCounter++;
                
                // calculate distance to quadrature point
                dx = quadPtX - posq_XYZQ.x + tran.x;
                dy = quadPtY - posq_XYZQ.y + tran.y;
                dz = quadPtZ - posq_XYZQ.z + tran.z;
                
                // apply periodic bounds
                dx -= floor( dx*invbox.x + 0.5f ) * box.x;
                dy -= floor( dy*invbox.y + 0.5f ) * box.y;
                dz -= floor( dz*invbox.z + 0.5f ) * box.z;
                R2quadPt_2 = dx * dx + dy * dy + dz * dz;
                
                // atom radius^2
                atomBufferedR_2 = atomR + SWICH_LEN;
                atomBufferedR_2 *= atomBufferedR_2;
                
                // check if point is inside range of neighboring atom
                if ( R2quadPt_2 < atomBufferedR_2 ) {
                
                // inner radius (below which switching function is 0)
                atomInnerR_2 = atomR - SWICH_LEN;
                atomInnerR_2 *= atomInnerR_2;
                
                // is point within inner radius of neighbor atom?
                if ( R2quadPt_2 <= atomInnerR_2 ) {
                    
                    // we are inside neighbor atom, so swiching function is zero
                    switchFn = 0.0f;
                    atomCounter = numAtomsInGrid + 1; // exit loop!
                    sfAtomCount = 0;
                    
                } else {
                    
                    // otherwise evaluate switching function
                    r2point = sqrtf(R2quadPt_2);
                    r = r2point - atomR;
                    sf = 0.5f + PRECALC1 * r - PRECALC2 * r * r * r;
                    switchFn *= sf;
                    
                    // record info should this be a gradient quadrature point
                    // remember, there is no self-contribution here!
                    if ( atomJ != atomI && sfAtomCount < MAX_ATOMS_IN_VOXEL ) {
                        
                        sfList[sfAtomCount] = sf * r2point;
                        rList[sfAtomCount] = r * r;
                        
                        thisForce.x = dx;
                        thisForce.y = dy;
                        thisForce.z = dz;
                        
                        // a float4 is easier to deal with than an arbitrary struct
                        memcpy(&thisForce.w, &atomJ, 4);
                        localForces[sfAtomCount] = thisForce;
                        
                        sfAtomCount ++;
                    }
                    
                } // decide within inner radius of atom or at smooth switching
                } // inside range of neighbor atom check
            } // calculate occupancy loop
        } // empty grid check
        
#ifdef USE_MEMBRANE
    } else
        switchFn = 0.0f; // inside membrane
#endif
        
    } // check if atom is outside inner radius
    
#else
    // check if atom is outside inner radius of central atom
    if ( quadPt.w >= atomR_inner ) {
        
        switchFn = 1.0f;
        posq_XYZQ = posq[atomI];
        
        // generate quadrature point (quadX, quadY, quadZ)
        // bracket quadrature to local space
        quadPtZ = quadPt.z + posq_XYZQ.z;
        
#ifdef USE_MEMBRANE
        if ( abs(quadPtZ) > MEMBRANE_INNER_R ) {
        
        // is point is inside membrane's switching function?
        if ( abs(quadPtZ) < MEMBRANE_OUTER_R ) {
            
            // above or below the XY plane?
            if ( quadPtZ > 0.0f ) {
                r = quadPtZ - MEMBRANE_R;
                sf = 0.5f + MEMBRANE_PRECALC1 * r - MEMBRANE_PRECALC2 * r * r * r;
            } else {
                r = quadPtZ + MEMBRANE_R;
                sf = 0.5f - MEMBRANE_PRECALC1 * r + MEMBRANE_PRECALC2 * r * r * r;
            }
            switchFn *= sf;
        }
#endif
        // generate quadrature point (quadX, quadY, quadZ)
        // bracket quadrature to local space
        if ( quadPtZ < MaxZ  &&  quadPtZ > MinZ ) {
        quadPtX = quadPt.x + posq_XYZQ.x;
        if ( quadPtX < MaxX  &&  quadPtX > MinX ) {
        quadPtY = quadPt.y + posq_XYZQ.y;
        if ( quadPtY < MaxY  &&  quadPtY > MinY ) {
            
            // if gridpoint is local, convert to gridspace
            i = floor((quadPtX + tranX + DELTA_R * 0.5f) * INVERSE_DELTA_R);
            j = floor((quadPtY + tranY + DELTA_R * 0.5f) * INVERSE_DELTA_R);
            k = floor((quadPtZ + tranZ + DELTA_R * 0.5f) * INVERSE_DELTA_R);
            
            // locate voxel in lookup table
            gridLoc = (i + 
                      (j*dimX) +
                      (k*dimX*dimY)) * MAX_ATOMS_IN_VOXEL;
            
            // check if gridspace is empty. If not, calculate atomic occupancy
            if ( lookupTable[gridLoc] > 0 ) {
            
            // location on lookup table
            numAtomsInGrid = lookupTable[gridLoc];
            
            // begin occupancy calculation
            atomCounter = 1;
            while ( atomCounter <= numAtomsInGrid ) {
                
                // gather information of this atom in grid
                atomJ = lookupTable[gridLoc + atomCounter];
                posq_XYZQ = posq[atomJ];
                atomR = params[atomJ].x;
                atomCounter++;
                
                // calculate distance to quadrature point
                dx = quadPtX - posq_XYZQ.x;
                dy = quadPtY - posq_XYZQ.y;
                dz = quadPtZ - posq_XYZQ.z;
                R2quadPt_2 = dx * dx + dy * dy + dz * dz;
                
                // atom radius^2
                atomBufferedR_2 = atomR + SWICH_LEN;
                atomBufferedR_2 *= atomBufferedR_2;
                
                // check if point is inside range of neighboring atom
                if ( R2quadPt_2 < atomBufferedR_2 ) {
                
                // inner radius (below which switching function is 0)
                atomInnerR_2 = atomR - SWICH_LEN;
                atomInnerR_2 *= atomInnerR_2;
                
                // is point within inner radius of neighbor atom?
                if ( R2quadPt_2 <= atomInnerR_2 ) {
                    
                    // we are inside neighbor atom, so swiching function is zero
                    switchFn = 0.0f;
                    atomCounter = numAtomsInGrid + 1; // exit loop!
                    sfAtomCount = 0;
                    
                } else {
                    
                    // otherwise evaluate switching function
                    r2point = sqrtf(R2quadPt_2);
                    r = r2point - atomR;
                    sf = 0.5f + PRECALC1 * r - PRECALC2 * r * r * r;
                    switchFn *= sf;
                    
                    // record info should this be a gradient quadrature point
                    // remember, there is no self-contribution here!
                    if ( atomJ != atomI && sfAtomCount < MAX_ATOMS_IN_VOXEL ) {
                        
                        sfList[sfAtomCount] = sf * r2point;
                        rList[sfAtomCount] = r * r;
                        
                        thisForce.x = dx;
                        thisForce.y = dy;
                        thisForce.z = dz;
                        
                        // a float4 is easier to deal with than an arbitrary struct
                        memcpy(&thisForce.w, &atomJ, 4);
                        localForces[sfAtomCount] = thisForce;
                        
                        sfAtomCount ++;
                    }
                    
                } // decide within inner radius of atom or at smooth switching
                } // inside range of neighbor atom check
            } // calculate occupancy loop
            } // empty grid check
            
        } // bracket Y realspace
        } // bracket X realspace
        } // bracket Z realspace
#ifdef USE_MEMBRANE
        } else
            switchFn = 0.0f; // inside membrane
#endif
    } // check if atom is outside inner radius
#endif
    
    // save angular quadrature sum, convert curly-H (switching function) to "nu"
    allAngularSums[radInt][angInt] = quadWt.x * (1.0f - switchFn);
    
    // ensure all threads are done before summation of the quadrature
    __syncthreads();
    
    //--------------------------------------------------------------------------
    // -- integrate angular components of atomic occupancy --
    
    // only finish job with y threads (radInt is variable, angInt not needed)
    if ( threadIdx.x == 0 ) {
        
        // add up (integrate) the angular quadratures for current radius
        angularSum = 0.0f;
        for (angInt = 0; angInt < NUM_ANGLES; angInt++)
            angularSum += allAngularSums[radInt][angInt];
        
        // add up radial quadrature sum
        // calculate for AA0 (coulombic) and AA1 (corrective) terms
        allRadSums_AA0[radInt] = angularSum;
        allRadSums_AA1[radInt] = angularSum * quadWt.y;
    }
    __syncthreads();
    
    //--------------------------------------------------------------------------
    // -- convert atomic occupancy into Born Radius --
    
    // only finish job with one thread
    if ( threadIdx.y == 0 && threadIdx.x == 0 ) {
        
        // calculate total volumetric sums
        volumetricSum_AA0 = PREINTEGRATE3;
        volumetricSum_AA1 = PREINTEGRATE4;
        for (radInt = 0; radInt < NUM_RADII_HEAVY; radInt++) {
            volumetricSum_AA0 += allRadSums_AA0[radInt];
            volumetricSum_AA1 += allRadSums_AA1[radInt];
        }
        
        // do final Born radius calculation
        AA0term = CONST_AA0 * (1.0f / CONST_RMIN - volumetricSum_AA0);
        dG1 = sqrtf(sqrtf(PRECALC3 - volumetricSum_AA1));
        AA1term = CONST_AA1 * dG1;
        BornR = 1.0f / (AA0term + AA1term);
        BornRadii[atomI] = BornR;
    }
    
    __syncthreads();
    
    // if this is a gradient-lying quadrature point, we need to calculate
    // the d(Born R) / d(r atom) gradient component. since only the slow threads
    // get this far, they self-select to fit within a warp or two
    if ( sfAtomCount > 0 ) {
        
        // locate where to fit the info into the global matricies
        atomCounter = atomicAdd( &nGBSWchainAtoms[atomI], sfAtomCount );
        
        // case 1: all atoms can be input
        if ( atomCounter + sfAtomCount < MAX_SF_NEIGHBORS ) {
            gridLoc = atomCounter + atomI * (MAX_SF_NEIGHBORS+1);
            
            // precalc of the term
            dG1sum = 0.25f / (dG1*dG1*dG1);
            aa0aa1Term = CONST_AA0 * quadWt.x + dG1sum*CONST_AA1 * quadWt.x * quadWt.y;
            wtAndSf = BornR*BornR * switchFn * aa0aa1Term;
            
            for ( int n = 0; n < sfAtomCount; n++ ) {
                
                r = rList[n];
                sf = sfList[n];
                
                dBornR_dr = wtAndSf * (PRECALC1 - PRECALC4 * r) / sf;
                thisForce = localForces[n];
                thisForce.x *= dBornR_dr;
                thisForce.y *= dBornR_dr;
                thisForce.z *= dBornR_dr;
                
                gbswChain[gridLoc + n] = thisForce;
                
            }
        // case 2: fill up the last remnant of free space
        } else if ( atomCounter < MAX_SF_NEIGHBORS ) {
            gridLoc = atomCounter + atomI * (MAX_SF_NEIGHBORS+1);
            sfAtomCount = MAX_SF_NEIGHBORS - atomCounter;
            
            // precalc of the term
            dG1sum = 0.25f / (dG1*dG1*dG1);
            aa0aa1Term = CONST_AA0 * quadWt.x + dG1sum*CONST_AA1 * quadWt.x * quadWt.y;
            wtAndSf = BornR*BornR * switchFn * aa0aa1Term;
            
            for ( int n = 0; n < sfAtomCount; n++ ) {
                r = rList[n];
                sf = sfList[n];
                
                dBornR_dr = wtAndSf * (PRECALC1 - PRECALC4 * r) / sf;
                thisForce = localForces[n];
                thisForce.x *= dBornR_dr;
                thisForce.y *= dBornR_dr;
                thisForce.z *= dBornR_dr;
                
                gbswChain[gridLoc + n] = thisForce;
                
            }
        } // end force gradient calculation
    } // quad pt on gradient check
    
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

extern "C" __global__ void computeGBSWForce(unsigned long long* __restrict__ forceBuffers, 
    unsigned long long* __restrict__ global_bornForce, real* __restrict__ energyBuffer, 
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
    real energy = 0;
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
#endif
                        real bornRadius2 = localData[tbx+j].bornRadius;
                        real alpha2_ij = bornRadius1*bornRadius2;
                        real D_ij = r2*RECIP(4.0f*alpha2_ij);
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
                        real dEdR = Gpol*(1.0f - 0.25f*expTerm);
                        force.w += dGpol_dalpha2_ij*bornRadius2;
                        
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
                        bool noOffsetNeeded = !(openmmExcl & 0x1);
                        if ( !noOffsetNeeded )
                            energy += 0.5f*scaledChargeProduct*COULOMBIC_CUTOFF_OFFSET;
#endif
                        energy += 0.5f*tempEnergy;
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
                        float tmpThetaForce = (GBSW_FAC*RECIP(denominator) - invR) * posq2.w;
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
#endif
                        real bornRadius2 = localData[tbx+tj].bornRadius;
                        real alpha2_ij = bornRadius1*bornRadius2;
                        real D_ij = r2*RECIP(4.0f*alpha2_ij);
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
                        real dEdR = Gpol*(1.0f - 0.25f*expTerm);
                        force.w += dGpol_dalpha2_ij*bornRadius2;
                        
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
                        bool noOffsetNeeded = !(openmmExcl & 0x1);
                        if ( !noOffsetNeeded )
                            energy += scaledChargeProduct*COULOMBIC_CUTOFF_OFFSET;
#endif
                        energy += tempEnergy;
                        delta *= dEdR;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
                        localData[tbx+tj].fx += delta.x;
                        localData[tbx+tj].fy += delta.y;
                        localData[tbx+tj].fz += delta.z;
                        localData[tbx+tj].fw += dGpol_dalpha2_ij*bornRadius1;
                        
#ifdef USE_CPHMD
                        real invR = 0.0f;
                        bool isExcluded = !(excl & 0x1);
                        if ( !isExcluded )
                            invR = RSQRT(r2);
#ifdef USE_SALT
                        float tmpForceFac = CPHMD_SALT_FAC*saltPrefactor*recipDem - invR;
#else
                        float tmpForceFac = GBSW_FAC*RECIP(denominator) - invR;
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
                            real bornRadius2 = localData[tbx+tj].bornRadius;
                            real alpha2_ij = bornRadius1*bornRadius2;
                            real D_ij = r2*RECIP(4.0f*alpha2_ij);
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
                            real dEdR = Gpol*(1.0f - 0.25f*expTerm);
                            force.w += dGpol_dalpha2_ij*bornRadius2;
                            
                            
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
                            energy += tempEnergy + scaledChargeProduct*COULOMBIC_CUTOFF_OFFSET;
#else
                            energy += tempEnergy;
#endif
                            delta *= dEdR;
                            force.x -= delta.x;
                            force.y -= delta.y;
                            force.z -= delta.z;
                            localData[tbx+tj].fx += delta.x;
                            localData[tbx+tj].fy += delta.y;
                            localData[tbx+tj].fz += delta.z;
                            localData[tbx+tj].fw += dGpol_dalpha2_ij*bornRadius1;
#ifdef USE_CPHMD
                            real invR = RSQRT(r2);
#ifdef USE_SALT
                            float tmpForceFac = CPHMD_SALT_FAC*saltPrefactor*recipDem - invR;
#else
                            float tmpForceFac = GBSW_FAC*RECIP(denominator) - invR;
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
#endif
                            real bornRadius2 = localData[tbx+tj].bornRadius;
                            real alpha2_ij = bornRadius1*bornRadius2;
                            real D_ij = r2*RECIP(4.0f*alpha2_ij);
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
                            real dEdR = Gpol*(1.0f - 0.25f*expTerm);
                            force.w += dGpol_dalpha2_ij*bornRadius2;
                            
#ifdef USE_COULOMBIC_CUTOFF_OFFSET
                            energy += tempEnergy + scaledChargeProduct*COULOMBIC_CUTOFF_OFFSET;
#else
                            energy += tempEnergy;
#endif
                            delta *= dEdR;
                            force.x -= delta.x;
                            force.y -= delta.y;
                            force.z -= delta.z;
                            localData[tbx+tj].fx += delta.x;
                            localData[tbx+tj].fy += delta.y;
                            localData[tbx+tj].fz += delta.z;
                            localData[tbx+tj].fw += dGpol_dalpha2_ij*bornRadius1;
#ifdef USE_CPHMD
                            real invR = RSQRT(r2);
#ifdef USE_SALT
                            float tmpForceFac = CPHMD_SALT_FAC*saltPrefactor*recipDem - invR;
#else
                            float tmpForceFac = GBSW_FAC*RECIP(denominator) - invR;
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
 * here we finish the force calculation by summing up the gradient components
 * (the first derivative of the Born Radius) and apply the forces on each atom
 */

extern "C" __global__ void reduceGBSWForce(
    // inputs
    float4* gbswChain, int *nGBSWchainAtoms, unsigned long long* __restrict__ bornForce,
    // outputs
#ifdef USE_SURFACE_AREA
    real* __restrict__ energyBuffer, const float2* __restrict__ params,
    real* __restrict__ BornRadii,
#endif
    unsigned long long* __restrict__ forceBuffers ) {
    
    
    int atomI = blockIdx.x, atomJ, lookupIndex = blockIdx.x * (MAX_SF_NEIGHBORS+1), 
        sfAtomCount, sfIdx;
    float4 thisForce;
    
#ifdef USE_SURFACE_AREA
    float GBdx_I = 0.0f, GBdy_I = 0.0f, GBdz_I = 0.0f, GBdx_J, GBdy_J, GBdz_J;
    __shared__ float dGdRGB_I;
#else
    float GBdx_I = 0.0f, GBdy_I = 0.0f, GBdz_I = 0.0f, GBdx_J, GBdy_J, GBdz_J, dGdRGB_I;
#endif
    
    sfAtomCount = nGBSWchainAtoms[atomI];
    if ( sfAtomCount > MAX_SF_NEIGHBORS )
        sfAtomCount = MAX_SF_NEIGHBORS;
    
    // use a stride-for-loop to run through each neighbor
    if ( sfAtomCount > 0 ){ // prevents crashes
    
#ifdef USE_SURFACE_AREA
    // ACE surface area approximation
    
    if (threadIdx.x == 0) {
        // import from global memory
        dGdRGB_I = RECIP(0x100000000)*static_cast<long long>(bornForce[atomI]);
        float offsetRadius = params[atomI].x;
        real bornRadius = BornRadii[atomI];
        
        // modulate force contribution
        real r = offsetRadius+PROBE_RADIUS;
        real ratio6 = POW(offsetRadius/bornRadius, 6);
        real saTerm = SURFACE_AREA_FACTOR*r*r*ratio6;
        dGdRGB_I += saTerm/bornRadius;
        
        //// modify energy -> removed! (contributes < 1% energy, unstable in systems >7,000 atoms)
        //energyBuffer[atomI] += saTerm/-6.0f;
    }
    
    // share modulated force contribution with all other threads
    __syncthreads();
    
#else
    // download force for this atom
    dGdRGB_I = RECIP(0x100000000)*static_cast<long long>(bornForce[atomI]);
#endif
    
    for ( sfIdx = lookupIndex+threadIdx.x; sfIdx < lookupIndex+sfAtomCount; sfIdx += blockDim.x ) {
        
        thisForce = gbswChain[sfIdx];
        memcpy(&atomJ, &thisForce.w, 4);
        
        // calculate the final (dG / dRGB) * (dRGB / dr)
        GBdx_J = thisForce.x * dGdRGB_I;
        GBdy_J = thisForce.y * dGdRGB_I;
        GBdz_J = thisForce.z * dGdRGB_I;
        
        GBdx_I += GBdx_J;
        GBdy_I += GBdy_J;
        GBdz_I += GBdz_J;
        
        // subtract force contribution to atom J
        // the force buffers are a flattened 2D matrix. assign X, then Y, then Z
        atomicAdd( &forceBuffers[ atomJ ], 
            static_cast<unsigned long long>((long long) (-GBdx_J*0x100000000)) );
        atomicAdd( &forceBuffers[ atomJ + PADDED_NUM_ATOMS ], 
            static_cast<unsigned long long>((long long) (-GBdy_J*0x100000000)) );
        atomicAdd( &forceBuffers[ atomJ + 2*PADDED_NUM_ATOMS ], 
            static_cast<unsigned long long>((long long) (-GBdz_J*0x100000000)) );
    }
    
    // add force contribution to atom I
    // the force buffers are a flattened 2D matrix. assign X, then Y, then Z
    atomicAdd( &forceBuffers[ atomI ], 
        static_cast<unsigned long long>((long long) (GBdx_I*0x100000000)) );
    atomicAdd( &forceBuffers[ atomI + PADDED_NUM_ATOMS ], 
        static_cast<unsigned long long>((long long) (GBdy_I*0x100000000)) );
    atomicAdd( &forceBuffers[ atomI + 2*PADDED_NUM_ATOMS ], 
        static_cast<unsigned long long>((long long) (GBdz_I*0x100000000)) );
    }
}

