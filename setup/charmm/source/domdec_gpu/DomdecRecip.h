#ifndef NOCUDAC
#ifndef DOMDECRECIP_H
#define DOMDECRECIP_H

class DomdecRecip {
  
 protected:
  // Settings
  int nfftx, nffty, nfftz;
  int order;
  double kappa;

 public:
 DomdecRecip(const int nfftx, const int nffty, const int nfftz, const int order, const double kappa) : 
  nfftx(nfftx), nffty(nffty), nfftz(nfftz), order(order), kappa(kappa) {}
  ~DomdecRecip() {}

  //virtual void clear_energy_virial() = 0;

  //virtual void get_energy_virial(const bool calc_energy, const bool calc_virial,
  //				 double& energy, double& energy_self, double *virial) = 0;


};

#endif // DOMDECRECIP_H
#endif //NOCUDAC
