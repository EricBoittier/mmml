__kernel void generateProtGrid(__global float * d_probes,
                               __global float * d_parameter,
                               __global float * d_GridPot,
                               int NGrids, int NAtoms,
                               __global int * GridNum,
                               __global float * GridMinCoor,
                               float Fa , float Fb, float Gmax,
                               float DGrid, float VdwEmax,
                               float ElecReplEmax, float ElecAttrEmax,
                               float CCELEC_CHARMM, int ElecMode, float Dielec) {
  /*
    Kernel of generating protein grid
    Each thread handle a grid point
  */
  // int GridGlobalId = get_group_id(0) * get_local_size(0) + get_local_id(0);
  int GridGlobalId = get_global_id(0);
  int NumGridPoints = GridNum[0] * GridNum[1] * GridNum[2];
  /*
    if(GridGlobalId==0){
    printf("I am thread %d\n",GridGlobalId);
    printf("NumGridPoints=%d\n",NumGridPoints);
    printf("NAtoms=%d\n",NAtoms);
    }*/
  if(GridGlobalId<NumGridPoints){
    int Gridx,Gridy,Gridz;
    int gridIdx;
    //calculate the 3-d grid x,y,z index from the 1-d global index
    //GridPotId=(Gridx*GridNum[1]+Gridy)*GridNum[2]+Gridz
    Gridx = GridGlobalId/(GridNum[1]*GridNum[2]);
    Gridy = (GridGlobalId%(GridNum[1]*GridNum[2]))/GridNum[0];
    Gridz = (GridGlobalId%(GridNum[1]*GridNum[2]))%GridNum[0];

    //calculate the grid point coordinate
    //grid point coordinate build from GridPotId
    float x,y,z;
    x = GridMinCoor[0]+Gridx*DGrid;
    y = GridMinCoor[1]+Gridy*DGrid;
    z = GridMinCoor[2]+Gridz*DGrid;
    /*
      if(GridGlobalId==0){
      printf("Grid x=%d y=%d z=%d\n",Gridx,Gridy,Gridz);
      printf("x=%f y=%f z=%f\n",x,y,z);
      }*/
    int n;
    float atomx,atomy,atomz;
    float eps,vdwr,cg;
    float eps_sqrt;
    float r_min;
    float eleconst, vdwconst;
    float r,rc,rh;
    float alpha,beta;
    float dener, aener;
    int hd, ha;
    int NumFeaturePerAtom=8;

    for(n=0; n<NAtoms; ++n){
      //read atom parameters from global memory
      atomx=d_parameter[NumFeaturePerAtom*n+0];
      atomy=d_parameter[NumFeaturePerAtom*n+1];
      atomz=d_parameter[NumFeaturePerAtom*n+2];
      eps=  d_parameter[NumFeaturePerAtom*n+3];
      vdwr= d_parameter[NumFeaturePerAtom*n+4];
      cg=   d_parameter[NumFeaturePerAtom*n+5];
      hd=   d_parameter[NumFeaturePerAtom*n+6];
      ha=   d_parameter[NumFeaturePerAtom*n+7];
      eps_sqrt = sqrt(fabs(eps));
      //calculate distance between grid point and atom n
      r = sqrt((atomx-x)*(atomx-x)+(atomy-y)*(atomy-y)+(atomz-z)*(atomz-z));
      rh = r - Fb;
      //calculate electrostatic energy
      /*
        if(GridGlobalId==0){
        printf("%d x=%f y=%f z=%f eps=%f vdwr=%f cg=%f r=%f\n",n,
        atomx,atomy,atomz,eps,vdwr,cg,r);
        }*/

      /*Electrostatics*/
      //ElecMode == 0 cdie
      //ElecMode == 1 rdie
      eleconst=CCELEC_CHARMM*cg/Dielec;
      //cdie
      if(ElecMode == 0){
        if( cg > 0) { //repulsive
          rc = 2.0f*eleconst/fabs(ElecReplEmax);
          if(r > rc){
            d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] +=
              eleconst/(r);
          }else{
            alpha = eleconst/(rc*rc);
            d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] +=
              (ElecReplEmax-alpha*r);
          }
        }
        else if (cg < 0){ //attractive
          rc = -2.0f*eleconst/fabs(ElecAttrEmax);
          if(r > rc){
            d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId]+=
              eleconst/(r);
          }else{
            alpha = eleconst/(rc*rc);
            d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] +=
              (ElecAttrEmax-alpha*r);
          }
        }
      }
      //rdie
      else if(ElecMode == 1){
        if( cg > 0) { //repulsive
          rc = sqrt(2.0f*fabs(eleconst/ElecReplEmax));
          if(r > rc){
            d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] +=
              eleconst/(r*r);
          }else{
            alpha = fabs(ElecReplEmax / (2.0f*rc*rc));
            d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] +=
              (ElecReplEmax - alpha*(r*r));
          }
        }
        else if (cg < 0){ //attractive
          rc = sqrt(2.0f*fabs(eleconst/ElecAttrEmax));
          if(r > rc){
            d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId]+=
              eleconst/(r*r);
          }else{
            alpha = fabs(ElecAttrEmax / (2.0f*rc*rc));
            d_GridPot[NumGridPoints*(NGrids-1)+GridGlobalId] +=
              (ElecAttrEmax + alpha*(r*r));
          }
        }
      }

      /*Hydrogen donor grid*/
      gridIdx = NGrids - 3;
      dener = ((rh * rh) * Fa + Gmax) * hd;
      if (dener < 0) {
        d_GridPot[NumGridPoints*gridIdx+GridGlobalId] +=
          dener;
      } else{
        d_GridPot[NumGridPoints*gridIdx+GridGlobalId] +=
          0;
      }

      /*Hydrogen acceptor grid*/
      gridIdx = NGrids - 2;
      aener = ((rh * rh) * Fa + Gmax) * ha;
      if (aener < 0) {
        d_GridPot[NumGridPoints*gridIdx+GridGlobalId] +=
          aener;
      } else{
        d_GridPot[NumGridPoints*gridIdx+GridGlobalId] +=
          0;
      }

      /*Van der waals*/
      for(gridIdx=0; gridIdx<NGrids-3; ++gridIdx){
        float radii = d_probes[gridIdx];
        r_min = vdwr + radii;
        vdwconst = 1.0f + sqrt(1.0f + 0.5f*fabs(VdwEmax)/eps_sqrt);
        rc = r_min * pow(vdwconst,-1.0f/6.0f);
        beta = 24.0f * eps_sqrt/VdwEmax * (vdwconst*vdwconst-2.0f*vdwconst);
        alpha = 0.5f*VdwEmax*pow(rc,-1.0f*beta);
        if (r > rc) {
          d_GridPot[NumGridPoints*gridIdx+GridGlobalId] +=
            eps_sqrt*(pow(r_min/r,12.0f) - 2.0f*pow( r_min/r, 6.0f));
        } else{
          d_GridPot[NumGridPoints*gridIdx+GridGlobalId] +=
            VdwEmax * (1.0f - 0.5f * pow(r/rc, beta));
        }
      }

    } //end looping all atoms
  }//enf if (GlobalId <  NumGridPoints)
}
