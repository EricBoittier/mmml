#ifndef NOCUDAC
#ifndef ENERGYVIRIAL_H
#define ENERGYVIRIAL_H
//
// Storage class for energy names
// (c) Antti-Pekka Hynninen, Feb 2015
// aphynninen@hotmail.com
//
#include <map>
#include <string>

class EnergyVirial {
private:
  // Number of energy terms
  int n;

  // Energy term indices
  std::map<std::string, int> energyIndex;

protected:
  EnergyVirial();
  ~EnergyVirial() {}
  int getEnergyIndex(std::string& name);
  int getN() {return n;}
  
public:
  void insert(std::string& name);
  void insert(const char *name);
};

#endif //ENERGYVIRIAL_H
#endif //NOCUDAC
