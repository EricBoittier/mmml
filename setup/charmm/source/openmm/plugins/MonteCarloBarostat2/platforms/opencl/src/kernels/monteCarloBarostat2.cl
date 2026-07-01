/**
 * Scale the particle positions.
 */

__kernel void scalePositions2(real4 scale, int numMolecules, real4 periodicBoxSize, real4 invPeriodicBoxSize, __global real4* restrict posq,
        __global const int* restrict moleculeAtoms, __global const int* restrict moleculeStartIndex) {
    for (int index = get_global_id(0); index < numMolecules; index += get_global_size(0)) {
        int first = moleculeStartIndex[index];
        int last = moleculeStartIndex[index+1];
        int numAtoms = last-first;

        // Find the center of each molecule.

        real4 center = (real4) 0;
        for (int atom = first; atom < last; atom++)
            center += posq[moleculeAtoms[atom]];
        center /= (real) numAtoms;

        // Move it into the first periodic box.

        int xcell = (int) floor(center.x*invPeriodicBoxSize.x);
        int ycell = (int) floor(center.y*invPeriodicBoxSize.y);
        int zcell = (int) floor(center.z*invPeriodicBoxSize.z);
        real4 delta = (real4) (xcell*periodicBoxSize.x, ycell*periodicBoxSize.y, zcell*periodicBoxSize.z, 0);
        center -= delta;

        // Now scale the position of the molecule center.

        delta.x = center.x*(scale.x-1)-delta.x;
        delta.y = center.y*(scale.y-1)-delta.y;
        delta.z = center.z*(scale.z-1)-delta.z;
        delta.w = center.w*(scale.w-1)-delta.w;
        for (int atom = first; atom < last; atom++) {
            real4 pos = posq[moleculeAtoms[atom]];
            pos.xyz += delta.xyz;
            posq[moleculeAtoms[atom]] = pos;
        }
    }
}

