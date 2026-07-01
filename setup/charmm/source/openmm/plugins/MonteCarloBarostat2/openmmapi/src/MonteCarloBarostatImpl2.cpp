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

#include "openmm/internal/MonteCarloBarostatImpl2.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/Context.h"
#include "openmm/CharmmKernels.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdlib>

using namespace OpenMM;
using namespace OpenMM_SFMT;
using namespace std;

const float BOLTZMANN = 1.380658e-23f; // (J/K)
const float AVOGADRO = 6.0221367e23f;
const float RGAS = BOLTZMANN*AVOGADRO; // (J/(mol K))
const float BOLTZ = RGAS/1000;         // (kJ/(mol K))
const float TOLERANCE = 0.00001;
int flag = 0;


MonteCarloBarostatImpl2::MonteCarloBarostatImpl2(const MonteCarloBarostat2& owner) : owner(owner), step(0) {
}

void MonteCarloBarostatImpl2::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(ApplyMonteCarloBarostatKernel2::Name(), context);
    kernel.getAs<ApplyMonteCarloBarostatKernel2>().initialize(context.getSystem(), owner);
    Vec3 box[3];
    context.getPeriodicBoxVectors(box[0], box[1], box[2]);
    double volume = box[0][0]*box[1][1]*box[2][2];
    // lengthScales in 3 Dimensions, which could be different with respect
    // to barostat methods
    // Get reference pressure and surface tension from owner
    double tension = owner.getSurfaceTension();
    Vec3 pressure3D(owner.getPressureIn3Dimensions());

    // Four different barostats
    if (tension != 0) {
        cout << "Constant surface tension and ";
        pressure3D[0] = pressure3D[1] = 0.0;
        if (pressure3D[2] != 0.0) {
            flag = 4;
            cout << "Constant normal pressure (in z)." << endl;
            lengthScale[0] = 0.0033*box[0][0];
            lengthScale[1] = 0.0033*box[1][1];
            lengthScale[2] = 0.0033*box[2][2];
        } else if (pressure3D[2] == 0.0) {
            flag = 3;
            cout << " Constant volume." << endl;
            lengthScale[0] = 0.005*box[0][0];
            lengthScale[1] = 0.005*box[1][1];
        }
    }
    else {
        if (pressure3D[2] == 0) {
            flag = 2;
            cout << "Constant tangential pressure and height." << endl;
            // Check if pressures are the same in x and y (PS: they need to be)
            if (pressure3D[0] != pressure3D[1])
                cout << "Tangential pressures in x and y need to be the same." << endl;
            lengthScale[0] = 0.005*box[0][0];
            lengthScale[1] = 0.005*box[1][1];
            lengthScale[2] = 0.0;
        } else {
            flag = 1;
            cout << "Constant normal pressure (in z) and surface area. (Pressures in x and y are 0.)" << endl;
            lengthScale[0] = 0.0;
            lengthScale[1] = 0.0;
            lengthScale[2] = 0.01*box[2][2];
            cout << "lengthScale[2] " << lengthScale[2] << endl;
        }
    }
    if (flag != 3) {
        for (int i = 0; i < 3; i++)
            cout << "lengthScale[" << i << "] = " << lengthScale[i] << endl;
    } else {
        for (int i = 0; i < 2; i++)
            cout << "lengthScale[" << i << "] = " << lengthScale[i] << endl;
        cout << "lengthScale[2] needs to couple with h_x and h_y to keep constant Volume." << endl;
    }

    // Check if the program choose any one of previous setups
    if (flag == 0)
        cout << "Please check your inputs for desired barostat." << endl;
    numAttempted = 0;
    numAccepted = 0;
    totalAccepted = 0;
    init_gen_rand(owner.getRandomNumberSeed(), random);
}

void MonteCarloBarostatImpl2::updateContextState(ContextImpl& context) {
    if (++step < owner.getFrequency() || owner.getFrequency() == 0)
        return;
    step = 0;

    // Compute the current potential energy.
    double initialEnergy = context.getOwner().getState(State::Energy).getPotentialEnergy();

    // Modify the periodic box size.
    Vec3 box[3];
    context.getPeriodicBoxVectors(box[0], box[1], box[2]);
    double volume = box[0][0]*box[1][1]*box[2][2];
    double surface = box[0][0]*box[1][1];

    Vec3 deltaLength;
    deltaLength[0] = lengthScale[0]*2*(genrand_real2(random)-0.5);
    if (flag == 3 || flag == 4){
        deltaLength[1] = deltaLength[0];
    }else {
	deltaLength[1] = lengthScale[1]*2*(genrand_real2(random)-0.5);
    }

    if (flag != 3) {
        deltaLength[2] = lengthScale[2]*2*(genrand_real2(random)-0.5);
        // For constant volume and surface tension, update h_Z with h_x and h_y
    } else {
	// A minor fix for the equation to get an accurate volume change. 
	// deltaLength[2] = -box[2][2]*(deltaLength[0]/box[0][0]+deltaLength[1]/box[1][1]);
        deltaLength[2] = -box[2][2]*((deltaLength[0]*box[1][1])/((box[0][0]+deltaLength[0])*(box[1][1]+deltaLength[1]))+deltaLength[1]/(box[1][1]+deltaLength[1]));

    }

    double newVolume = (box[0][0]+deltaLength[0])*(box[1][1]+deltaLength[1])*(box[2][2]+deltaLength[2]);
    double deltaVolume = newVolume - volume;
    double newSurface = (box[0][0]+deltaLength[0])*(box[1][1]+deltaLength[1]);
    double deltaSurface = newSurface - surface;
    Vec3 scale;
    for (int i =0; i <3; i++) {
        scale[i] = (box[i][i] + deltaLength[i])/box[i][i];
    }
    kernel.getAs<ApplyMonteCarloBarostatKernel2>().scaleCoordinates(context, scale);
    context.getOwner().setPeriodicBoxVectors(box[0]*scale[0], box[1]*scale[1], box[2]*scale[2]);

    // Compute the energy of the modified system.
    double finalEnergy = context.getOwner().getState(State::Energy).getPotentialEnergy();
    double kT = BOLTZ*owner.getTemperature();
    // Get reference pressure and surface tension from owner
    double tension = owner.getSurfaceTension()*(AVOGADRO*1e-25);
    Vec3 pressure3D(owner.getPressureIn3Dimensions()*(AVOGADRO*1e-25));

    // Choose Hamiltonians
    double w;
    switch (flag) {
    case 1:
        w = finalEnergy - initialEnergy + pressure3D[2]*deltaVolume
                - context.getMolecules().size()*kT*log(newVolume/volume);
        break;
    case 2:
        w = finalEnergy - initialEnergy + pressure3D[0]*deltaVolume
                - context.getMolecules().size()*kT*log(newVolume/volume);
        break;
    case 3:
        w = finalEnergy - initialEnergy - tension*deltaSurface
                - context.getMolecules().size()*kT*log(newVolume/volume);
        break;
    case 4:
        w = finalEnergy - initialEnergy + pressure3D[2]*deltaVolume - tension*deltaSurface
                - context.getMolecules().size()*kT*log(newVolume/volume);
        break;
    }
    if (w > 0 && genrand_real2(random) > exp(-w/kT)) {
        // Reject the step.

        kernel.getAs<ApplyMonteCarloBarostatKernel2>().restoreCoordinates(context);
        context.getOwner().setPeriodicBoxVectors(box[0], box[1], box[2]);
        volume = newVolume;
    }
    else {
        numAccepted++;
        totalAccepted++;
        if (totalAccepted % 100 == 0) {
            switch (flag) {
            // TODO provide names for these magic numbers
            case 1:
                if (abs(deltaSurface/surface) > TOLERANCE) {
                    throw(OpenMMException("MCBarostat2 case 1 tol exceeded"));
                }
                break;
            case 2:
                if (abs(deltaLength[2]/box[2][2]) > TOLERANCE) {
                    throw(OpenMMException("MCBarostat2 case 2 tol exceeded"));
                }
                break;
            case 3:
                if (abs(deltaVolume/volume) > TOLERANCE ) {
                    throw(OpenMMException("MCBarostat2 case 3 tol exceeded"));
                }
                break;
            case 4:
                break;
            }
        }
    }
    numAttempted++;

    if (numAttempted >= 10) {
        if (numAccepted < 0.25*numAttempted) {
            for (int i = 0; i < 3; i++)
                lengthScale[i] /= 1.0322;
            numAttempted = 0;
            numAccepted = 0;
        }
        else if (numAccepted > 0.75*numAttempted) {
            for (int i = 0; i < 3; i++) {
                lengthScale[i] = min(lengthScale[i]*1.0322, pow(volume*0.3, 1.0/3.0));
            }
            numAttempted = 0;
            numAccepted = 0;
        }
    }
}

map<string, double> MonteCarloBarostatImpl2::getDefaultParameters() {
    map<string, double> parameters;
    parameters[MonteCarloBarostat2::Pressure()] = getOwner().getDefaultPressure();
    return parameters;
}

vector<string> MonteCarloBarostatImpl2::getKernelNames() {
    vector<string> names;
    names.push_back(ApplyMonteCarloBarostatKernel2::Name());
    return names;
}

