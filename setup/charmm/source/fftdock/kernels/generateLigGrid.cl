// generate ligand grid
__kernel void generateLigGrid(int numRotamers, int NAtoms,
                              int numGrids, __global int * d_GridNum, float DGrid,
                              __global float * d_rotamersCoor,
                              __global float * d_par,
                              __global float * d_GridMinCoor,
                              __global float * d_LigGrid) {
  // int globalId=get_group_id(0)*get_local_size(0)+get_local_id(0);
  int globalId = get_global_id(0);
  if(globalId<numRotamers){
    //printf("GlobalId=%d NumRotamers=%d\n",globalId,numRotamers);
    float dx,dy,dz;
    int i,j,grid_idx;
    int idx_x,idx_y,idx_z, vdw_grid_idx;
    int xlen,ylen,zlen;
    float xRatio,yRatio,zRatio;
    float eps,vdwr,charge;
    float energyFactor;
    unsigned int NumGridPoints = d_GridNum[0]*d_GridNum[1]*d_GridNum[2];
    //offset used to locate the right position to write in the LigGrid for each type of grid
    unsigned int rotamerOffset,gridTypeOffset;
    //printf("grid x=%d y=%d z=%d\n",d_GridNum[0],d_GridNum[1],d_GridNum[2]);
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

      //calculate the atomi 3-d index
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

      /*
        if(globalId == 40 or globalId==0){
        printf("idx=%d atom%d dx=%f dy=%f dz=%f idx_x=%d idx_y=%d idx_z=%d\n",globalId,
        i,d_rotamersCoor[globalId*3*NAtoms+3*i+0]
        ,d_rotamersCoor[globalId*3*NAtoms+3*i+1],
        d_rotamersCoor[globalId*3*NAtoms+3*i+2],idx_x,idx_y,idx_z);
        //printf("charge=%f eps=%f vdwr=%f vdw_grid_idx=%d\n",charge,eps,
        //        vdwr,vdw_grid_idx);
        }*/
      //vdw

      //loop through elec,vdwrepl,vdwattr grids
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
        /*
          if(globalId == 38){
          printf("atom%d energyFactor[%d]==%f vdw_grid_idx=%d\n",i,j,energyFactor,
          vdw_grid_idx);
          printf("offset=%d\n",rotamerOffset+gridTypeOffset);
          }*/
        gridTypeOffset=j*NumGridPoints;
        //if (globalId==1){
        //printf("grid%d offset=%d numGridPoints=%d\n",j,rotamerOffset+gridTypeOffset,NumGridPoints);
        //}
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
