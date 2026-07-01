#ifndef NOCUDAC
#ifndef CUDADIRECTFORCETYPES_H
#define CUDADIRECTFORCETYPES_H

struct DirectEnergyVirial_t {
  // Energies
  double energy_vdw;
  double energy_elec;
  double energy_excl;

  // Finished virial
  double vir[9];

  // DP Shift forces for virial calculation
  double sforce[27*3];

  // FP Shift forces for virial calculation
  long long int sforce_fp[27*3];
};

struct DirectSettings_t {
  float kappa;
  float kappa2;

  float boxx;
  float boxy;
  float boxz;

  float roff, roff2, roff3, roff5;
  float ron, ron2;

  float roffinv;
  float roffinv2;
  float roffinv3;
  float roffinv4;
  float roffinv5;
  float roffinv6;
  float roffinv12;
  float roffinv18;

  float inv_roff2_ron2_3;
  
  float k6, k12, dv6, dv12;

  float ga6, gb6, gc6;
  float ga12, gb12, gc12;
  float GAconst, GBcoef;

  float Aconst, Bconst, Cconst, Dconst;
  float dvc;

  float Acoef, Bcoef, Ccoef;
  float Denom, Eaddr, Constr;
  
  float rips,rips2, ripsr,rips2r,rips6r,rips12r;

  float aipse0, aipse1, aipse2, aipse3, aipse4, aipse5, aipse6, pipsec ;
  float bipse1, bipse2, bipse3, bipse4, bipse5, bipse6 ;

  float aipsvc0, aipsvc1, aipsvc2, aipsvc3, aipsvc4, aipsvc5, aipsvc6, pipsvcc ;
  float bipsvc1, bipsvc2, bipsvc3, bipsvc4, bipsvc5, bipsvc6 ;

  float aipsva0, aipsva1, aipsva2, aipsva3, aipsva4, aipsva5, aipsva6, pipsvac ;
  float bipsva1, bipsva2, bipsva3, bipsva4, bipsva5, bipsva6 ;

  float e14fac;

  float hinv;
  float *ewald_force;

};

// Enum for VdW and electrostatic models
enum {NONE=0, 
      VDW_VSH=1, VDW_VSW=2, VDW_VFSW=3, VDW_VGSH=4, VDW_CUT=5, VDW_IPS=6,
      EWALD=101, CSHIFT=102, CFSWIT=103, CSHFT=104, CSWIT=105, RSWIT=106,
      RSHFT=107, RSHIFT=108, RFSWIT=109, GSHFT=110, EWALD_LOOKUP=111, ELE_IPS=121};

// Enum for vdwparam
enum {VDW_MAIN, VDW_IN14};

#endif // CUDADIRECTFORCETYPES_H
#endif //NOCUDAC
