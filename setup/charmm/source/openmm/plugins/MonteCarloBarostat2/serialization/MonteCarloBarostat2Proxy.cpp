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

#include "MonteCarloBarostat2Proxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include "openmm/MonteCarloBarostat2.h"
#include <sstream>

using namespace OpenMM;
using namespace std;

MonteCarloBarostat2Proxy::MonteCarloBarostat2Proxy() : SerializationProxy("MonteCarloBarostat2") {
}

void MonteCarloBarostat2Proxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);

    const MonteCarloBarostat2& force =
            *reinterpret_cast<const MonteCarloBarostat2*>(object);

    node.setDoubleProperty("pressure", force.getDefaultPressure());
    node.setDoubleProperty("temperature", force.getTemperature());
    node.setIntProperty("frequency", force.getFrequency());
    node.setIntProperty("randomSeed", force.getRandomNumberSeed());

    Vec3 press3D = force.getPressureIn3Dimensions();
    node.setDoubleProperty("pressure3DX", press3D[0]);
    node.setDoubleProperty("pressure3DY", press3D[1]);
    node.setDoubleProperty("pressure3DZ", press3D[2]);

    node.setDoubleProperty("surfaceTension", force.getSurfaceTension());
}

void* MonteCarloBarostat2Proxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    MonteCarloBarostat2* force = NULL;
    try {
      Vec3 press3D(node.getDoubleProperty("pressure3DX"),
                   node.getDoubleProperty("pressure3DY"),
                   node.getDoubleProperty("pressure3DZ"));
      MonteCarloBarostat2* force = new MonteCarloBarostat2(
                                     node.getDoubleProperty("pressure"),
                                     node.getDoubleProperty("temperature"),
                                     press3D,
                                     node.getDoubleProperty("surfaceTension"),
                                     node.getIntProperty("frequency"));
        force->setRandomNumberSeed(node.getIntProperty("randomSeed"));
        return force;
    }
    catch (...) {
        if (force != NULL)
            delete force;
        throw;
    }
}
