float4 distPairParams = PARAMS[index];
real kc = distPairParams.x;
real fmax = distPairParams.y;
real dcut = distPairParams.z;
real sexp = distPairParams.w;
real3 dv12 = make_real3(pos1.x-pos2.x, pos1.y-pos2.y, pos1.z-pos2.z);
real3 dv34 = make_real3(pos3.x-pos4.x, pos3.y-pos4.y, pos3.z-pos4.z);
#if APPLY_PERIODIC
APPLY_PERIODIC_TO_DELTA(dv12)
APPLY_PERIODIC_TO_DELTA(dv34)
#endif
real dr12 = SQRT(dv12.x*dv12.x + dv12.y*dv12.y + dv12.z*dv12.z);
real dr34 = SQRT(dv34.x*dv34.x + dv34.y*dv34.y + dv34.z*dv34.z);
real dist = dr12 - dr34;
real absDist = (dist>=0.f) ? dist : -dist;
real invAbsDist = RECIP(absDist); 
real invPowAbsDistSexp = 1.0f/POW(absDist,sexp); 

real x1 = dcut*dcut;
real x2 = 1.0f/sexp;
real softA = x1*(0.5f+x2) - fmax*dcut*(1.0f+x2);
real softB = POW(dcut,sexp+1.0f)*(fmax-dcut)*x2;

x1 = 0.5f*kc*absDist*absDist;
x2 = kc*(softA + softB*invPowAbsDistSexp + fmax*absDist);
x1 = (absDist<=dcut) ? x1 : x2;
energy += x1;

x1 = kc*dist;
x2 = kc*(fmax - softB*sexp*invPowAbsDistSexp*invAbsDist)*dist*invAbsDist;
x1 = (absDist<=dcut) ? x1 : x2;
real3 force2 = dv12/dr12 * x1;
real3 force1 = -force2;
real3 force3 = dv34/dr34 * x1;
real3 force4 = -force3;

