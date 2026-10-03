/* gpu_compat.h is prepended at compile time */

// generate ligand grid
KERNEL void generateLigGrid(int numRotamers, int NAtoms,
                            int numGrids, GLOBAL int * d_GridNum, float DGrid,
                            GLOBAL float * d_rotamersCoor,
                            GLOBAL float * d_par,
                            GLOBAL float * d_GridMinCoor,
                            GLOBAL float * d_LigGrid) {
  int globalId = THREAD_ID;
  if(globalId<numRotamers){
    float dx,dy,dz;
    int i,j,grid_idx;
    int idx_x,idx_y,idx_z, vdw_grid_idx;
    int xlen,ylen,zlen;
    float xRatio,yRatio,zRatio;
    float eps,vdwr,charge;
    float energyFactor;
    unsigned int NumGridPoints = d_GridNum[0]*d_GridNum[1]*d_GridNum[2];
    unsigned int rotamerOffset,gridTypeOffset;
    xlen = d_GridNum[0];
    ylen = d_GridNum[1];
    zlen = d_GridNum[2];
    rotamerOffset=globalId*numGrids*NumGridPoints;
    for(i=0;i<NAtoms;++i){
      dx = d_rotamersCoor[globalId*3*NAtoms + 3*i + 0]-
        d_GridMinCoor[globalId*3 + 0];
      dy = d_rotamersCoor[globalId*3*NAtoms + 3*i + 1]-
        d_GridMinCoor[globalId*3 + 1];
      dz = d_rotamersCoor[globalId*3*NAtoms + 3*i + 2]-
        d_GridMinCoor[globalId*3 + 2];

      idx_x = floor(dx/DGrid);
      idx_y = floor(dy/DGrid);
      idx_z = floor(dz/DGrid);

      xRatio = (dx-(idx_x)*DGrid)/DGrid;
      yRatio = (dy-(idx_y)*DGrid)/DGrid;
      zRatio = (dz-(idx_z)*DGrid)/DGrid;

      charge = d_par[i*4+0];
      eps    = d_par[i*4+1];
      vdwr   = d_par[i*4+2];
      vdw_grid_idx = d_par[i*4+3];

      energyFactor = 0.0;
      for(j=0; j<numGrids; ++j){
        if(j == vdw_grid_idx){
          energyFactor = sqrt(fabs(eps));
        }
        else if(j == numGrids - 1){
          energyFactor = charge;
        }
        else{
          energyFactor = 0.0;
        }
        gridTypeOffset=j*NumGridPoints;
        //(0,0,0)
        d_LigGrid[rotamerOffset+gridTypeOffset+
                  (idx_x*ylen+idx_y)*zlen+idx_z]+=
          (1-xRatio)*(1-yRatio)*(1-zRatio)*energyFactor;
        //(0,0,1)
        d_LigGrid[rotamerOffset+gridTypeOffset+
                  (idx_x*ylen+idx_y)*zlen+idx_z+1]+=
          (1-xRatio)*(1-yRatio)*(zRatio)*energyFactor;
        //(0,1,0)
        d_LigGrid[rotamerOffset+gridTypeOffset+
                  (idx_x*ylen+idx_y+1)*zlen+idx_z]+=
          (1-xRatio)*yRatio*(1-zRatio)*energyFactor;
        //(0,1,1)
        d_LigGrid[rotamerOffset+gridTypeOffset+
                  (idx_x*ylen+idx_y+1)*zlen+idx_z+1]+=
          (1-xRatio)*yRatio*zRatio*energyFactor;
        //(1,0,0)
        d_LigGrid[rotamerOffset+gridTypeOffset+
                  ((idx_x+1)*ylen+idx_y)*zlen+idx_z]+=
          xRatio*(1-yRatio)*(1-zRatio)*energyFactor;
        //(1,0,1)
        d_LigGrid[rotamerOffset+gridTypeOffset+
                  ((idx_x+1)*ylen+idx_y)*zlen+idx_z+1]+=
          xRatio*(1-yRatio)*zRatio*energyFactor;
        //(1,1,0)
        d_LigGrid[rotamerOffset+gridTypeOffset+
                  ((idx_x+1)*ylen+idx_y+1)*zlen+idx_z]+=
          xRatio*yRatio*(1-zRatio)*energyFactor;
        //(1,1,1)
        d_LigGrid[rotamerOffset+gridTypeOffset+
                  ((idx_x+1)*ylen+idx_y+1)*zlen+idx_z+1]+=
          xRatio*yRatio*zRatio*energyFactor;
      }
    }
  }
}
