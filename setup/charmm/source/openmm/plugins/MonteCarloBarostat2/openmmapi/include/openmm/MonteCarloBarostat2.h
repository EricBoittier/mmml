#ifndef OPENMM_MONTECARLOBAROSTAT2_H_
#define OPENMM_MONTECARLOBAROSTAT2_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include <string>

namespace OpenMM {

/**
 * This class uses a Monte Carlo algorithm to adjust the size of the periodic box, simulating the
 * effect of constant pressure.
 *
 * This class assumes the simulation is also being run at constant temperature, and requires you
 * to specify the system temperature (since it affects the acceptance probability for Monte Carlo
 * moves).  It does not actually perform temperature regulation, however.  You must use another
 * mechanism along with it to maintain the temperature, such as LangevinIntegrator or AndersenThermostat.
 */
class OPENMM_EXPORT MonteCarloBarostat2 : public Force {
public:
    /**
     * This is the name of the parameter which stores the current pressure acting on
     * the system (in bar).
     */
    static const std::string& Pressure() {
        static const std::string key = "MonteCarloPressure2";
        return key;
    }

    /**
     * Create a MonteCarloBarostat.
     *
     * @param defaultPressure   the default pressure acting on the system (in bar)
     * @param temperature       the temperature at which the system is being maintained (in Kelvin)
     * @param frequency         the frequency at which Monte Carlo pressure changes should be attempted (in time steps)
     */
    MonteCarloBarostat2(double defaultPressure, double temperature, Vec3& pressure3D, double surfaceTension=0.0, int frequency = 25);

    /**
     * Get the default pressure acting on the system (in bar).
     *
     * @return the default pressure acting on the system, measured in bar.
     */
    double getDefaultPressure() const {
        return defaultPressure;
    }

    /**
     * Get the frequency (in time steps) at which Monte Carlo pressure changes should be attempted.  If this is set to
     * 0, the barostat is disabled.
     */
    int getFrequency() const {
        return frequency;
    }

    /**
     * Set the frequency (in time steps) at which Monte Carlo pressure changes should be attempted.  If this is set to
     * 0, the barostat is disabled.
     */
    void setFrequency(int freq) {
        frequency = freq;
    }

    /**
     * Get the temperature at which the system is being maintained, measured in Kelvin.
     */
    double getTemperature() const {
        return temperature;
    }

    /**
     * Set the temperature at which the system is being maintained.
     *
     * @param temp     the system temperature, measured in Kelvin.
     */
    void setTemperature(double temp) {
        temperature = temp;
    }

    /**
     * Get the random number seed.  See setRandomNumberSeed() for details.
     */
    int getRandomNumberSeed() const {
        return randomNumberSeed;
    }

    /**
     * Set refence pressures in 3 dimentions
     */
    void setPressureIn3Dimensions(Vec3& pressures) {
        pressure3D = pressures;
    }

    /**
     * Get refence pressures in 3 dimentions
     */
    Vec3& getPressureIn3Dimensions() const{
        return pressure3D;
    }

    /**
     * Set surface tension
     */
    void setSurfaceTension(double tension){
        surfaceTension = tension;
    }

    /**
     * Get surface tension
     */
    double getSurfaceTension() const{
        return surfaceTension;
    }

    /**
     * Set the random number seed.  It is guaranteed that if two simulations are run
     * with different random number seeds, the sequence of Monte Carlo steps will be different.  On
     * the other hand, no guarantees are made about the behavior of simulations that use the same seed.
     * In particular, Platforms are permitted to use non-deterministic algorithms which produce different
     * results on successive runs, even if those runs were initialized identically.
     */
    void setRandomNumberSeed(int seed) {
        randomNumberSeed = seed;
    }

protected:
    virtual ForceImpl* createImpl() const;

private:
    double defaultPressure, temperature, surfaceTension;
    int frequency, randomNumberSeed;
    Vec3& pressure3D;
    
};

} // namespace OpenMM

#endif

