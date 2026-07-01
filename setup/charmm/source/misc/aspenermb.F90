!
!     Source file for solvation free energy based on surface area.
!     Based on files from Wesson and Eisenberg
!  
!     This module is an  modification  of ASPENER by adding 
!     an Implicit Membrane in molecular model
!
#if KEY_ASPMEMB==1
SUBROUTINE ASPENERMB(EU,X,Y,Z,DX,DY,DZ,QECONT,ECONT)
  !
  ! This is the user energy routine which computes solvation energy forces.
  ! This file and its modifications may be distributed freely.
  !
  !      Arguments passed in:
  !     EU   - User energy to be returned
  !     X,Y,Z - Coordinates for energy evaluation
  !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
  !     QECONT - Flag for energy analysis
  !     ECONT(NATOM) - Analysis array to be filled if QECONT.
  !
  use chm_kinds
  use dimens_fcm
  !  use exfunc
  use psf
  use memory
  !
  implicit none
  !  declaraction of passed arguments:
  real(chm_real),allocatable,dimension(:) :: RX, AREA
  integer,allocatable,dimension(:) :: NHARR

  !      PARAMETER (MARC=201,MOV=200,MPT=300)
  INTEGER, PARAMETER :: MARC=2002,MOV=2000,MPT=3000
  ! For the arrays below I use automatic allocation. clb3
  !MOV
  logical ISKIP(MOV)

  integer INTAG1(MOV)
  integer INTAG(MOV)
  integer ITAG(MOV)
  integer IDER(MOV)
  integer SIGN_YDER(MOV)

  real(chm_real) XC1(MOV),YC1(MOV),ZC1(MOV)
  real(chm_real) BG(MOV),THER(MOV),RI(MOV),RISQ(MOV),B1(MOV),DSQ1(MOV)
  real(chm_real) BSQ1(MOV),GR(MOV)
  real(chm_real) XC(MOV),YC(MOV),ZC(MOV)
  real(chm_real) UX(MOV),UY(MOV),UZ(MOV)
  real(chm_real) DSQ(MOV),BSQ(MOV),B(MOV)
  !MARC
  integer KENT(MARC),KOUT(MARC)
  !MPT
  real(chm_real) ARCI(MPT),ARCF(MPT),EX(MPT)
  integer LT(MPT)

  real(chm_real) EU
  logical QECONT
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) ECONT(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  !
  !
  !
  call chmalloc('aspener.src','ASPENR','RX',NATOM,crl=RX)
  call chmalloc('aspener.src','ASPENR','AREA',NATOM,crl=AREA)
#if KEY_PARALLEL==1
  call chmalloc('aspener.src','ASPENR','NHARR',NATOM,intg=NHARR)
#endif 
  CALL ASPEN1MB(EU,X,Y,Z,DX,DY,DZ,QECONT,ECONT,RX,AREA, &
       MARC,MOV,MPT,ISKIP,INTAG1,INTAG, &
       ITAG,IDER,SIGN_YDER,XC1,YC1, &
       ZC1,BG,THER,RI,RISQ, &
       B1,DSQ1,BSQ1,GR,XC, &
       YC,ZC,UX,UY,UZ,DSQ, &
       BSQ,B,KENT,KOUT,ARCI, &
       ARCF,EX,LT &
#if KEY_PARALLEL==1
       ,NHARR & 
#endif
       )
  call chmdealloc('aspener.src','ASPENR','RX',NATOM,crl=RX)
  call chmdealloc('aspener.src','ASPENR','AREA',NATOM,crl=AREA)
#if KEY_PARALLEL==1
  call chmdealloc('aspener.src','ASPENR','NHARR',NATOM,intg=NHARR)
#endif 
  RETURN
END SUBROUTINE ASPENERMB
!
SUBROUTINE ASPEN1MB(EU,X,Y,Z,DX,DY,DZ,QECONT,ECONT, &
     RX,AREA,MARC,MOV,MPT,ISKIP,INTAG1,INTAG,ITAG,IDER, &
     SIGN_YDER,XC1,YC1,ZC1,BG,THER,RI,RISQ,B1,DSQ1,BSQ1,GR,XC, &
#if KEY_PARALLEL==1
     YC,ZC,UX,UY,UZ,DSQ,BSQ,B,KENT,KOUT,ARCI,ARCF,EX,LT,NHARR) 
#else /**/
     YC,ZC,UX,UY,UZ,DSQ,BSQ,B,KENT,KOUT,ARCI,ARCF,EX,LT)
#endif 
  !
  !  This routine contains the actual adaptation of the Richmond
  !  routine to calculate solvent accessible surface areas.
  !
  !  ANALYTICAL ACCESSIBLE SURFACE AREA AND GRADIENT CALCULATION
  !     QECONT IS A FLAG WHICH WILL BE SET WHEN THE ROUTINE IS
  !      CALLED IN THE ANALYSIS SECTION. INDIVIDUAL ATOM USER ENERGIES ARE
  !      THEN RETURNED IN DATA.
  !      T.J.RICHMOND
  !      Modified 9/86 by Morgan Wesson
  !      Modified 11/03 by Velin Spassov ( added Implicit Membrane )
  !**
  !      for fixed atoms:  the routine calculates derivatives for atom pairs
  !      if one of the atoms is not fixed.
  !      The derivatives of fixed atoms are set to 0 at the end of the routine.
  !      The solvation energy of fixed atoms doesn't contribute to the total
  !      energy that is output.
  !
  !** Notice that PROBE_RADIUS=1.4 set in the surface.f90 common block is the
  !**  solvent probe radius and SIG=.01 is used in the test for significant
  !      sphere overlaps.
  !**
  !      Arguments passed in:
  !     EU   - User energy to be returned
  !     X,Y,Z - Coordinates for energy evaluation
  !     DX,DY,DZ - Forces. Must be modified. Append (dE/dX,...)
  !     QECONT - Flag for analysis
  !     ECONT(NATOM) - Analysis array to be filled if QECONT.
  !** Indices:
  !**      MARC = max. no. of partial arcs for IR sphere (ith sphere)
  !**      MAXA = max. no. of atoms
  !**      MOV  = max. no. of IN overlapping spheres (j & k spheres for the ith)
  !**      MPT  = max. no. of overlap end pts. on a circle of intersection
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
  !
  use psf
  use surface
  use surfmemb
  use consta
  use stream
  use parallel
  implicit none
  !
  ! definitions of passed arguments...
  real(chm_real) EU
  LOGICAL QECONT
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) ECONT(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) RX(*), AREA(*)
  INTEGER MARC,MOV,MPT
  LOGICAL ISKIP(MOV)
  INTEGER INTAG1(MOV),INTAG(MOV),ITAG(MOV),IDER(MOV),SIGN_YDER(MOV)
  real(chm_real)   XC1(MOV),YC1(MOV),ZC1(MOV), &
       BG(MOV),THER(MOV),RI(MOV),RISQ(MOV), &
       B1(MOV),DSQ1(MOV),BSQ1(MOV),GR(MOV), &
       XC(MOV),YC(MOV),ZC(MOV), &
       UX(MOV),UY(MOV),UZ(MOV), &
       DSQ(MOV),BSQ(MOV),B(MOV)
  INTEGER   KENT(MARC),KOUT(MARC),LT(MPT)
  real(chm_real)    ARCI(MPT),ARCF(MPT),EX(MPT)
  !
  !** Variable dimension arrays:
  !
  ! local character variables...
  !C      character*4 NAME_IR, NAME_IN
  ! local logical variables...
  logical QLONE,LTOP,ISKIPS
  ! local integer variables...
  integer IB_LOCAL, JB_LOCAL, I, IR, IO, IN, K, L, IO1, K1, &
       KARC, MI, N, J, M, II, IFAIL
  !
  ! local real variables...
  real(chm_real) SIG, SIGSQ, PIX2, PIX4, PID2, EE, ARCLEN, EXANG, &
       XR, YR, ZR, RR, RRX2, RRSQ, RPLUS, TX, TY, TZ, XYSQ, &
       CCSQ, CC, RMINUS, TXK, TYK, TZK, BSQK, BK, GL, THERK, TD, &
       DK, GK, RISQK, RIK, T1, AXX, AXY, AXZ, AYX, AYY, AZX, AZY, &
       AZZ, TXL, TYL, TZL, UXL, UYL, UZL, DSQL, TB, TXB, TYB, TR, &
       TXR, TYR, TK1, TK2, THE, TF, ARCSUM, T, TT, RCN, BGL, BSQL, &
       RISQL, WXLSQ, WXL, P, V, DEAL, DECL, DTKAL, DTKCL, S, &
       T2, DTLAL, DTLCL, GACA, GACB, FACA, FACB, FACC, DAX, DAY, DAZ, &
       TI, E_DELTA, ASP_IR, ACOS_INPUT, SQRT_INPUT
  !
  real(chm_real) EE1, EU1
  !----------------- Implicit Membrane ----------------------------
  real(chm_real)  abs_dist, L_memb
  real(chm_real)  RR_VIRT, X_VIRT, Y_VIRT, Z_VIRT, up_memb,  &
       down_memb
  Integer  NATOM1

  !----------------------------------------------------------------
#if KEY_PARALLEL==1
  INTEGER NHARR(*)
  !CC      real(chm_real)  DINMOD
  INTEGER IDIM, NHPART(0:MAXNODE)
  INTEGER NHATOM,NFSRF,IX,IQQ,IQ,IRES,NTRY
#endif 
  !----------------- Implicit Membrane ----------------------------
  RR_VIRT =  MEGA 
  L_memb  = L_memb0 +  PROBE_RADIUS
  up_memb = mb_center +  L_memb 
  down_memb = mb_center -  L_memb 
  !..
  !----------------------------------------------------------------
  !
  IF(PRNLEV.GT.8) THEN
     WRITE(OUTU,'(A)')  'Entering ASPEN1'
     WRITE(OUTU,'(2A)') ' Atom TYPE', &
          '     X         Y         Z     VDW_SURF   ASPV  IGNORE'
     do I = 1, NATOM
        WRITE (OUTU,'(I5,1X,A,5F10.5,I5)') I,ATYPE(I)(1:idleng), &
             X(I),Y(I),Z(I),VDW_SURF(I),ASPV(I),IGNORE(I)
     ENDDO
     write (OUTU,'(A,F10.5)') 'probe radius: ', PROBE_RADIUS
  ENDIF
  !
  EU=ZERO
  !**   Overlap significance (also used to test if spheres colinear)
  SIG=.01
  SIGSQ=SIG**2
  PIX2=2.*PI
  PIX4=4.*PI
  PID2=PI/2.
  DO I = 1, NATOM
     RX(I) = VDW_SURF( I )
  ENDDO
  !
  EE=ZERO
  !
  DO I=1,MOV
     IDER(I)=0
     SIGN_YDER(I)=0
  ENDDO
  !
  !**   Process each atom
  !**   Find the IN spheres which overlap the IR sphere
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
  !
  !     Trying to improve load balance. If it does, make it once only procedure.
  !
  !CC      IMERI=IMERI+1
  NHATOM=0
  DO I=1,NATOM
     IF(IGNORE(I).GE.0) THEN
        NHATOM=NHATOM+1
        NHARR(NHATOM)=I
     ENDIF
  ENDDO
  !
  !--------------------------------------------------------------
  !     It turned out that verlet energy is not constant during simulation
  !     if swaping of
  !     equivalent charge atoms takes place. So no need for division
  !     by residue??? - Do we completely discard this code?
  !
#if KEY_NOPARASWAP==0 /*noparaswp*/
  NHPART(0)=0
  DO I = 1, NUMNOD
     NTRY=(NHATOM*I)/NUMNOD
     DO IRES=1,NRES
        IQ=IBASE(IRES+1)
        IF(IQ.GE.NHARR(NTRY)) GOTO 920
     ENDDO
920  NHPART(I)=NTRY
     IF(IQ.GT.NHARR(NTRY)) THEN
        DO IQQ=NHARR(NTRY)+1,IQ
           IF(IGNORE(IQQ).GE.0) NHPART(I)=NHPART(I)+1
        ENDDO
     ENDIF
  ENDDO
  !
  !     Start the writing where the error is...
  !CC      IF((IMERI.GT.3150).AND.(IMERI.LT.-4000))WRITE(OUTU,'(A,4I5)')
  !CC     $     'ENTER ASPENER:ME,STEP,II,IF=',
  !CC     $     MYNOD,IMERI,NHPART(MYNOD)+1,NHPART(MYNODP)
  !
  !
  !CC      IF(MYNOD.EQ.0)WRITE(OUTU,*)'ASP>NATOM,NHATOM',NATOM,NHATOM
  !
  !CC      IF(MYNOD.EQ.0)WRITE(OUTU,*)'NHPART:',(NHPART(I),I=1,NUMNOD)
  !CC      IF(MYNOD.EQ.0)WRITE(OUTU,*)
  !CC     $              'NHPART:',(NHARR(NHPART(I)),I=1,NUMNOD)
  DO IX=NHPART(MYNOD)+1,NHPART(MYNODP)
#else /* (noparaswp)*/
  DO IX=NHATOM*MYNOD/NUMNOD+1,NHATOM*MYNODP/NUMNOD
#endif /* (noparaswp)*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal paralllel compile option'
#endif /* (parstest)*/
     IR=NHARR(IX)
#elif KEY_SPACDEC==1 /*parfmain*/
  DO IR=1,NATOM
#elif KEY_PARASCAL==1 /*parfmain*/
  DO IR=1,NATOM
     IF(JPBLOCK(IR).NE.MYNOD) GOTO 4
#else /* (parfmain)*/
#error  'Illegal paralllel compile option'
#endif /* (parfmain)*/
#else /* (paramain)*/
  DO IR=1,NATOM

#endif /* (paramain)*/
     
     NATOM1 = NATOM
     !
     IF(PRNLEV.GT.8) THEN
        WRITE(OUTU,*)
        WRITE(OUTU,'(I5,1X,A,5F10.5,I5)') IR,ATYPE(IR)(1:idleng), &
             X(IR),Y(IR),Z(IR),VDW_SURF(IR), ASPV(IR), IGNORE(IR)
     ENDIF
     !        otherwise, skip this atom.
     if(IGNORE( IR ).GE.0) then
        ASP_IR = ASPV(IR)

        !-----------------Implicit Membrane  -------------------------------------
        IF(membrane) THEN


           If(memb_dir .eq. 1) then
              abs_dist = abs(X(ir) - mb_center)
              if((L_memb - RX(ir)- abs_dist) .ge. Zero) then 

                 AREA(IR) = zero
                 E_DELTA = zero
                 goto 1001
              elseif((L_memb +RX(ir) - abs_dist) .gt. Zero) then
                 natom1 = natom + 1
                 if(X(IR) .ge. mb_center ) then
                    X_VIRT =  up_memb - RR_VIRT
                 else 
                    X_VIRT =  down_memb + RR_VIRT
                 endif
                 Y_VIRT = Y(IR)
                 Z_VIRT = Z(IR)       
              endif

           elseif(memb_dir .eq. 2) then
              abs_dist = abs(Y(ir) - mb_center)

              if((L_memb - RX(ir)- abs_dist) .ge. Zero) then 

                 AREA(IR) = zero
                 E_DELTA = zero
                 goto 1001
              elseif((L_memb +RX(ir) - abs_dist) .gt. Zero) then
                 natom1 = natom + 1
                 if(Y(IR) .ge. mb_center ) then
                    Y_VIRT =  up_memb - RR_VIRT
                 else 
                    Y_VIRT =  down_memb + RR_VIRT
                 endif
                 X_VIRT = X(IR)
                 Z_VIRT = Z(IR)       
              endif
           else
              abs_dist = abs(Z(ir)- mb_center) 

              if((L_memb - RX(ir)- abs_dist) .ge. Zero) then 

                 AREA(IR) = zero
                 E_DELTA = zero
                 goto 1001
              elseif((L_memb +RX(ir) - abs_dist) .gt. Zero) then
                 natom1 = natom + 1
                 if(Z(IR) .ge. mb_center ) then
                    Z_VIRT =  up_memb - RR_VIRT
                 else 
                    Z_VIRT =  down_memb + RR_VIRT
                 endif
                 X_VIRT = X(IR)
                 Y_VIRT = Y(IR)       
              endif

           endif
           !.......... if the atom is completely inside the membrane

           if((L_memb - RX(ir)- abs_dist) .ge. Zero) then 

              AREA(IR) = zero
              E_DELTA = zero
              goto 1001
           elseif((L_memb +RX(ir) - abs_dist) .gt. Zero) then
              natom1 = natom + 1

           endif

        ENDIF

        !----------------------------------------------------------------
        QLONE=.FALSE.
        IO=1
        JB_LOCAL=0
        IB_LOCAL=0
        ARCLEN=ZERO
        EXANG=ZERO
        AREA (IR ) = ZERO
        XR=X(IR)
        YR=Y(IR)
        ZR=Z(IR)
        RR=RX(IR)
        RRX2=RR*2.
        RRSQ=RR**2
        do IN = 1, NATOM1
           !........   for the real atoms only: 
           IF(IN .le. NATOM ) THEN
              !              if "ignore" is set, this routine ignores the atom entirely.
              if( IGNORE(IN).LT.0) go to 12
              ! both frozen
              IF (IMOVE(IN).GT.0.and.IMOVE(IR).GT.0) go to 12
              !              Is the IN sphere next to the IR sphere
              RPLUS=RR+RX(IN)
              !
              TX=X(IN)-XR
              !              exit IN loop
              IF(ABS(TX).GE.RPLUS)GO TO 12
              !
              TY=Y(IN)-YR
              !              exit IN loop

              IF(ABS(TY).GE.RPLUS)GO TO 12
              !
              TZ=Z(IN)-ZR
              !              exit IN loop
              IF(ABS(TZ).GE.RPLUS)GO TO 12
              !
              !              exclude in = ir

              !              exit IN loop
              if (IN.eq.IR) go to 12

              !........    virtual atom only:
           ELSE
              RPLUS=RR+RR_VIRT
              !                
              TX=X_VIRT - XR
              !              exit IN loop
              IF(ABS(TX).GE.RPLUS)GO TO 12
              !
              TY=Y_VIRT - YR
              !              exit IN loop
              IF(ABS(TY).GE.RPLUS)GO TO 12
              !
              TZ=Z_VIRT - ZR
              !              exit IN loop
              IF(ABS(TZ).GE.RPLUS)GO TO 12
              !
              !              exclude in = ir

              !              exit IN loop
              if (IN.eq.IR) go to 12              
           ENDIF

           !
           !C                IF(PRNLEV.GT.8) THEN
           !C                NAME_IR = RES( GETRES( IR, IBASE, NREST))
           !C                NAME_IN = RES( GETRES( IN, IBASE, NREST))
           !C                write (OUTU,'(2(1X,2A4,I5))') ATYPE(IR), NAME_IR,
           !C                &              imove(ir), ATYPE(IN), NAME_IN, imove(IN)
           !C                write (OUTU,'(5X,A5,4F10.5,I5)') ATYPE(IN),
           !C                &            X(IN),Y(IN),Z(IN),VDW_SURF(IN),IGNORE(IN)
           !C                ENDIF
           !**            Check for overlap of spheres by testing center to center distance
           !**            Against sum and difference of radii
           XYSQ=TX**2+TY**2
           IF (XYSQ.lt.SIGSQ) then
              TX=SIG
              TY=ZERO
              XYSQ=SIGSQ
           ENDIF
           !
           CCSQ=XYSQ+TZ**2
           CC=SQRT(CCSQ)
           !              exit IN loop
           IF(RPLUS-CC.LE.SIG)GO TO 12
           IF(IN .le. NATOM ) THEN 
              RMINUS=RR-RX(IN)
              !............................................
           ELSE
              RMINUS=RR-RR_VIRT                  
           ENDIF
           if (CC-ABS(RMINUS).le.SIG) then
              IF(RMINUS.LE.ZERO) then
                 !                 IR atom is buried, go to next atom in ir loop
                 go to 4
              else
                 !                    IN atom is buried, exit IN loop
                 go to 12
              ENDIF
           ENDIF
           !**            Calc. overlap parameters
           IF(IO.GT.MOV) CALL WRNDIE(-3,'<ANAREA>', &
                'Maximum overlapping spheres limit exceeded.')
           XC1(IO)=TX
           YC1(IO)=TY
           ZC1(IO)=TZ
           !
           DSQ1(IO)=XYSQ
           BSQ1(IO)=CCSQ
           B1(IO)=CC
           ! this is the distance from the IR sphere to the plane containing its
           ! intersection with the IN sphere, divided by the radius of the IR sphere.
           GR(IO)=(CCSQ+RPLUS*RMINUS)/(RRX2*B1(IO))
           INTAG1(IO)=IN
           IO=IO+1
12         continue
        enddo
        IO=IO-1
        !
        if (IO.eq.0) then
           AREA(IR) =PIX4
           GO TO 16
        ENDIF
        !
        if (IO.eq.1) then
           K=1
           QLONE=.TRUE.
           TXK=XC1(1)
           TYK=YC1(1)
           TZK=ZC1(1)
           BSQK=BSQ1(1)
           BK=B1(1)
           INTAG(1)=INTAG1(1)
           ARCSUM=PIX2
           IB_LOCAL=IB_LOCAL+1
           !mbm..27-Dec-94 Fujitsu Port / Bugfix
           !mbm..         GO TO 19
           ARCLEN=ARCLEN+GR(K)*ARCSUM
           IN=INTAG(K)
           !...................................................................
           IF(IN .le. NATOM ) THEN 
              T1=ARCSUM*RRSQ*(BSQK-RRSQ+RX(IN)**2)/(RRX2*BSQK*BK)
           ELSE
              T1=ARCSUM*RRSQ*(BSQK-RRSQ+RR_VIRT**2)/(RRX2*BSQK*BK)
           ENDIF
           !...................................................................
           DX( IR ) = DX( IR ) - ASP_IR * TXK * T1
           DY( IR ) = DY( IR ) - ASP_IR * TYK * T1
           DZ( IR ) = DZ( IR ) - ASP_IR * TZK * T1
           !
           IF(IN .le. NATOM ) THEN 
              DX( IN ) = DX( IN ) + ASP_IR * TXK * T1
              DY( IN ) = DY( IN ) + ASP_IR * TYK * T1
              DZ( IN ) = DZ( IN ) + ASP_IR * TZK * T1
           ENDIF
           !
           GO TO 56
           !mbm..27-Dec-94

        ENDIF
        !**         Sort IN spheres by degree of overlap with IR sphere
        IFAIL=0
        CALL SORTAG(GR,IO,ITAG)
        do L=1,IO
           K=ITAG(L)
           IN=INTAG1(K)
           INTAG(L)=IN
           !
           XC(L)=XC1(K)
           YC(L)=YC1(K)
           ZC(L)=ZC1(K)
           !
           DSQ(L)=DSQ1(K)
           B(L)=B1(K)
           BSQ(L)=BSQ1(K)
           ISKIP(L)=.FALSE.
        ENDDO
        !
        do L=1,IO
           GL=GR(L)*RR
           BG(L)=B(L)*GL
           RISQ(L)=RRSQ-GL**2
           RI(L)=SQRT(RISQ(L))
           !**            Radius of the IN circle on the surface of the sphere
           THER(L)=PID2-ASIN(GR(L))
        ENDDO
        !
        !**         Find boundary of inaccessible area on IR sphere
        IO1=IO-1
        DO K=1,IO1
           if (.not.ISKIP(K)) then
              TXK=XC(K)
              TYK=YC(K)
              TZK=ZC(K)
              BK=B(K)
              THERK=THER(K)
              K1=K+1
              DO L=K1,IO
                 if (.not.ISKIP(L)) then
                    !**               Is L circle intersecting K circle?
                    !**               Distance between circle centers and sum of radii
                    ACOS_INPUT = (TXK*XC(L)+TYK*YC(L)+TZK*ZC(L)) &
                         /(BK*B(L))
                    ACOS_INPUT = MAX( ACOS_INPUT,MINONE+RSMALL )
                    ACOS_INPUT = MIN( ACOS_INPUT, ONE-RSMALL )
                    CC = ACOS( ACOS_INPUT )
                    TD=THERK+THER(L)
                    !**                     Circles enclose separate regions?
                    IF(CC.GE.TD)GO TO 31
                    !**                     Circle L completely inside circle K?
                    if (CC+THER(L).lt.THER(K)) then
                       ISKIP(L)=.TRUE.
                    else
                       !**                        Circles essentially parallel?
                       if (CC.gt.SIG) then
                          !**   IR sphere completely buried?
                          !     IR atom is buried, go to next atom in IR loop.
                          if (PIX2-CC.le.TD) go to 4
                       else
                          ISKIP(L)=.TRUE.
                       ENDIF
                    ENDIF
                 ENDIF
31               continue
              enddo
           ENDIF
        ENDDO
        !**         Find T value of circle intersections
        !C          do K=1,IO
        K=0
17      CONTINUE
        K=K+1
        if (.not.ISKIP(K))then
           ISKIPS=ISKIP(K)
           ISKIP(K)=.TRUE.
           KARC=0
           LTOP=.FALSE.
           TXK=XC(K)
           TYK=YC(K)
           TZK=ZC(K)
           DK=SQRT(DSQ(K))
           BSQK=BSQ(K)
           BK=B(K)
           GK=GR(K)*RR
           RISQK=RISQ(K)
           RIK=RI(K)
           THERK=THER(K)
           !**               Rotation matrix elements
           T1=TZK/(BK*DK)
           AXX=TXK*T1
           AXY=TYK*T1
           AXZ=DK/BK
           AYX=TYK/DK
           AYY=TXK/DK
           AZX=TXK/BK
           AZY=TYK/BK
           AZZ=TZK/BK
           DO L=1,IO
              if (.not.ISKIP(L))then
                 TXL=XC(L)
                 TYL=YC(L)
                 TZL=ZC(L)
                 !**                     Rotate spheres so K vector colinear with z-axis
                 UXL=TXL*AXX+TYL*AXY-TZL*AXZ
                 UYL=TYL*AYY-TXL*AYX
                 UZL=TXL*AZX+TYL*AZY+TZL*AZZ
                 ACOS_INPUT = UZL/B(L)
                 ACOS_INPUT = MAX( ACOS_INPUT, MINONE+RSMALL )
                 ACOS_INPUT = MIN( ACOS_INPUT, ONE-RSMALL )
                 if (ACOS( ACOS_INPUT ).lt.THERK+THER(L)) then
                    GL=GR(L)*RR
                    DSQL=UXL**2+UYL**2
                    TB=UZL*GK-BG(L)
                    TXB=UXL*TB
                    TYB=UYL*TB
                    TD=RIK*DSQL
                    SQRT_INPUT = RISQK * DSQL - TB**2
                    SQRT_INPUT = MAX( SQRT_INPUT, ZERO )
                    TR = SQRT( SQRT_INPUT )
                    TXR=UXL*TR
                    TYR=UYL*TR
                    !**                        T values of intersection for K circle
                    TB=(TXB+TYR)/TD
                    if (ABS(TB).gt.(ONE-RSMALL)) &
                         TB=SIGN(ONE-RSMALL,TB)
                    TK1=ACOS(TB)
                    if (TYB-TXR.lt.ZERO) TK1=PIX2-TK1
                    !
                    !CC      IF((IMERI.EQ.-290).AND.(MYNOD.EQ.7).AND.(IR.EQ.551).AND.(K.EQ.7))
                    !CC     $    THEN
                    !CC            WRITE(OUTU,'(A,5I5,2D20.10)')'TK1:',MYNOD,IMERI,IR,K,KARC,
                    !CC     $                  TB,TK1
                    !CC      ENDIF
                    !
                    TB=(TXB-TYR)/TD
                    if (ABS(TB).gt.(ONE-RSMALL)) &
                         TB=SIGN(ONE-RSMALL,TB)
                    TK2=ACOS(TB)
                    if (TYB+TXR.lt.ZERO) TK2=PIX2-TK2
                    !
                    !CC      IF((IMERI.EQ.-290).AND.(MYNOD.EQ.7).AND.(IR.EQ.551).AND.(K.EQ.7))
                    !CC     $    THEN
                    !CC            WRITE(OUTU,'(A,5I5,2D20.10)')'TK2:',MYNOD,IMERI,IR,K,KARC,
                    !CC     $                  TB,TK2
                    !CC      ENDIF
                    !
                    ACOS_INPUT = (RRSQ*UZL-GK*BG(L))/ &
                         (RIK*RI(L)*B(L))
                    ACOS_INPUT = MAX( ACOS_INPUT, MINONE+RSMALL )
                    ACOS_INPUT = MIN( ACOS_INPUT, ONE-RSMALL )
                    THE = -acos( ACOS_INPUT )
                    !**                        Is TK1 entry or exit point?  check T=0 point.
                    !**                        TI IS EXIT PT., TF IS ENTRY PT.
                    ACOS_INPUT = (UZL*GK-UXL*RIK)/(B(L)*RR)
                    IF(WRNLEV.GE.2) THEN
                       if (ACOS_INPUT.gt.ONE) then
                          WRITE(OUTU,*)'ASPENER: acos input is ' &
                               ,ACOS_INPUT
                       elseif (ACOS_INPUT.lt.MINONE) then
                          WRITE(OUTU,*)'ASPENER: acos input is ' &
                               ,ACOS_INPUT
                       endif
                    ENDIF
                    ACOS_INPUT = MAX( ACOS_INPUT, MINONE+RSMALL )
                    ACOS_INPUT = MIN( ACOS_INPUT, ONE-RSMALL )
                    if ((ACOS(ACOS_INPUT)-THER(L))* &
                         (TK2-TK1).le.0) then
                       TI=TK2
                       TF=TK1
                    else
                       TI=TK1
                       TF=TK2
                    endif
                    KARC=KARC+1
                    if (TF.le.TI) then
                       ARCF(KARC)=TF
                       ARCI(KARC)=ZERO
                       TF=PIX2
                       LT(KARC)=L
                       EX(KARC)=THE
                       LTOP=.TRUE.
                       KARC=KARC+1
                    ENDIF
                    IF(KARC.GT.MPT) CALL WRNDIE(-3,'<ANAREA>', &
                         'Maximum overlap end point limit exceeded')
                    ARCF(KARC)=TF
                    ARCI(KARC)=TI
                    !CC      IF((IMERI.EQ.-290).AND.(MYNOD.EQ.7).AND.(IR.EQ.551).AND.(K.EQ.7))
                    !CC     $    THEN
                    !CC            WRITE(OUTU,'(A,5I5,3F10.5)')'ARC:',MYNOD,IMERI,IR,K,KARC,
                    !CC     $                  ARCI(KARC),ARCF(KARC)
                    !CC      ENDIF
                    LT(KARC)=L
                    EX(KARC)=THE
                    UX(L)=UXL
                    UY(L)=UYL
                    UZ(L)=UZL
                 ENDIF
              ENDIF
           ENDDO
           ISKIP(K)=ISKIPS
           !**               Special case: K circle without intersections?
           !
           !     Start the writing where the error is...
           !CC      IF((IMERI.EQ.-290).AND.(MYNOD.EQ.7).AND.(IR.EQ.551)) THEN
           !CC           WRITE(OUTU,'(A,5I5,6F10.5)')
           !CC     $     'karc:ME,...',
           !CC     $     MYNOD,IMERI,IR,K,KARC,ARCSUM,PIX2
           !CC           IF(K.EQ.7)WRITE(OUTU,'(8F10.5)')(ARCI(IXX),IXX=1,KARC)
           !CC           IF(K.EQ.7)WRITE(OUTU,'(8F10.5)')(ARCF(IXX),IXX=1,KARC)
           !CC      ENDIF
           !
           if (KARC.LE.0) then
              ARCSUM=PIX2
              IB_LOCAL=IB_LOCAL+1
              go to 19
           ENDIF
           !**               General case: sum up arclength and set connectivity code
           IFAIL=0
           CALL SORTAG(ARCI,KARC,ITAG)
           ARCSUM=ARCI(1)
           MI=ITAG(1)
           T=ARCF(MI)
           N=MI
           !
           !     Start the writing where the error is...
           !CC      IF((IMERI.EQ.-290).AND.(MYNOD.EQ.7).AND.(IR.EQ.551)) THEN
           !CC           WRITE(OUTU,'(A,6I5,6F10.5)')
           !CC     $     'SORTAG1:ME,...',
           !CC     $     MYNOD,IMERI,IR,K,KARC,MI,ARCSUM,T
           !CC           IF(K.EQ.7)WRITE(OUTU,'(8F10.5)')(ARCI(IXX),IXX=1,KARC)
           !CC           IF(K.EQ.7)WRITE(OUTU,'(8F10.5)')(ARCF(IXX),IXX=1,KARC)
           !CC      ENDIF
           !
           if (KARC.ne.1) then
              do J=2,KARC
                 M=ITAG(J)
                 if (T.lt.ARCI(J)) then
                    ARCSUM=ARCSUM+ARCI(J)-T
                    EXANG=EXANG+EX(N)
                    JB_LOCAL=JB_LOCAL+1
                    IF(JB_LOCAL.GT.MARC) CALL WRNDIE(-3, &
                         '<ANAREA>','Maximum arc limit exceeded')
                    L=LT(N)
                    IDER(L)=IDER(L)+1
                    SIGN_YDER(L) = SIGN_YDER(L) + 1
                    KENT(JB_LOCAL)=L*1024+K
                    L=LT(M)
                    IDER(L)=IDER(L)+1
                    SIGN_YDER(L) = SIGN_YDER(L) - 1
                    KOUT(JB_LOCAL)=K*1024+L
                 ENDIF
                 TT=ARCF(M)
                 if (TT.ge.T) then
                    T=TT
                    N=M
                 ENDIF
              ENDDO
           ENDIF
           ARCSUM=ARCSUM+PIX2-T
           !
           !     Start the writing where the error is...
           !CC      IF((IMERI.EQ.-290).AND.(MYNOD.EQ.7).AND.(IR.EQ.551))
           !CC     $     WRITE(OUTU,'(A,5I5,6F10.5)')
           !CC     $     'SORTAG1:ME,...',
           !CC     $     MYNOD,IMERI,IR,KARC,MI,ARCSUM,T
           !
           if (.not.LTOP) then
              EXANG=EXANG+EX(N)
              JB_LOCAL=JB_LOCAL+1
              IF(JB_LOCAL.GT.MARC) CALL WRNDIE(-3, &
                   '<ANAREA>','Maximum arc limit exceeded')
              L=LT(N)
              IDER(L)=IDER(L)+1
              SIGN_YDER(L) = SIGN_YDER(L) + 1
              KENT(JB_LOCAL)=L*1024+K
              L=LT(MI)
              IDER(L)=IDER(L)+1
              SIGN_YDER(L) = SIGN_YDER(L) - 1
              KOUT(JB_LOCAL)=K*1024+L
           ENDIF
           !**   Calculate derivatives. Comments refer to the equivalent notation
           !     in Tim Richmond's paper.
           do L=1,IO
              !                    K ; K index is J
              if (IDER(L).ne.0) then
                 !                                         IDER(L)* rho**2
                 RCN=IDER(L)*RRSQ
                 IDER(L)=0
                 UZL=UZ(L)
                 !                                         C(K)
                 GL=GR(L)*RR
                 !                                         G(K)
                 BGL=BG(L)
                 !                                         D(K)*G(K)
                 BSQL=BSQ(L)
                 !                                         D(K)**2
                 RISQL=RISQ(L)
                 !                                         R(K)**2
                 WXLSQ=BSQL-UZL**2
                 !                                         A(K)**2
                 WXL=SQRT(WXLSQ)
                 !                                         A(K)
                 P=BGL-GK*UZL
                 !                                         D(K)*G(K) - G(J)*C(K)
                 V=RISQK*WXLSQ-P**2
                 !                                         E ** 2
                 V = MAX( V, RSMALL)
                 V=SQRT(V)
                 !                                         E
                 T1=RR*(GK*(BGL-BSQL)+UZL*(BGL-RRSQ))/ &
                      (V*RISQL*BSQL)
                 !                                         rho*( G(J)* (D(K)*G(K) - D(K)**2) +
                 !                                         C(K)( D(K)*G(K)-rho**2))
                 !                                         / (E*RX(K)**2 * D(K) **2)
                 DEAL=-WXL*T1
                 !                                         d(omega)/dA(K)
                 DECL=-UZL*T1-RR/V
                 !                                         d(omega)/dC(K)
                 DTKAL=(WXLSQ-P)/(WXL*V)
                 !                                         d( T(lambda+1))/dA(K)
                 DTKCL=(UZL-GK)/V
                 !                                         d( T(lambda+1))/dC(K)
                 S=GK*B(L)-GL*UZL
                 !                                         G(J)*D(K) - G(K)*C(K)
                 T1=2.*GK-UZL
                 !                                         2*(G(J)-C(K)
                 T2=RRSQ-BGL
                 !                                         rho **2 - D(K)*G(K)
                 DTLAL=-(RISQL*WXLSQ*B(L)*T1-S* &
                      (WXLSQ*T2+RISQL*BSQL)) &
                      /(RISQL*WXL*BSQL*V)
                 !                                         d( T(lambda))/dA(K)
                 DTLCL=-(RISQL*B(L)*(UZL*T1-BGL)-UZL*T2*S)/ &
                      (RISQL*BSQL*V)
                 !
                 !                                         d( T(lambda))/dC(K)
                 GACA=RCN*(DEAL-(GK*DTKAL-GL*DTLAL)/RR)/WXL

                 !                                         x component of derivative 
                 !                                         in mu coordinate system
                 GACB = GK - UZL*GL/B(L)
                 !                                         y component of
                 GACB = GACB * SIGN_YDER(L) * RR / WXLSQ
                 !                                         derivative in mu coordinate system
                 SIGN_YDER(L) = 0
                 FACA=UX(L)*GACA - UY(L)*GACB
                 !                                         rotate back around z axis.
                 FACB=UY(L)*GACA + UX(L)*GACB
                 FACC=RCN*(DECL-(GK*DTKCL-GL*DTLCL)/RR)
                 !                                         z component of derivative 
                 !                                         in mu coordinate system
                 DAX=AXX*FACA-AYX*FACB+AZX*FACC
                 !                                         transform back to the
                 DAY=AXY*FACA+AYY*FACB+AZY*FACC
                 !                                         real coordinates
                 DAZ=AZZ*FACC-AXZ*FACA
                 IN=INTAG(L)
                 DX( IR ) = DX( IR ) + ASP_IR * DAX
                 DY( IR ) = DY( IR ) + ASP_IR * DAY
                 DZ( IR ) = DZ( IR ) + ASP_IR * DAZ
                 !
                 DX( IN ) = DX( IN ) - ASP_IR * DAX
                 DY( IN ) = DY( IN ) - ASP_IR * DAY
                 DZ( IN ) = DZ( IN ) - ASP_IR * DAZ
              ENDIF
           ENDDO
19         ARCLEN=ARCLEN+GR(K)*ARCSUM
           !
           !     Start the writing where the error is...
           !CC      IF((IMERI.EQ.-290).AND.(MYNOD.EQ.7).AND.(IR.EQ.551))
           !CC     $     WRITE(OUTU,'(A,4I5,6F10.5)')
           !CC     $     'arclen1:ME,...',
           !CC     $     MYNOD,IMERI,IR,K,ARCSUM,ARCLEN,GR(K),X(IR),Y(IR),Z(IR)
           !
           IN=INTAG(K)
           IF(IN .le. NATOM ) then 
              T1=ARCSUM*RRSQ*(BSQK-RRSQ+RX(IN)**2)/(RRX2*BSQK*BK)
              DX( IN ) = DX( IN ) + ASP_IR * TXK * T1
              DY( IN ) = DY( IN ) + ASP_IR * TYK * T1
              DZ( IN ) = DZ( IN ) + ASP_IR * TZK * T1
           ELSE
              T1=ARCSUM*RRSQ*(BSQK-RRSQ+RR_VIRT**2)/(RRX2*BSQK*BK)
           ENDIF
           DX( IR ) = DX( IR ) - ASP_IR * TXK * T1
           DY( IR ) = DY( IR ) - ASP_IR * TYK * T1
           DZ( IR ) = DZ( IR ) - ASP_IR * TZK * T1
           !
           if (QLONE) go to 56
        ENDIF
        !C            ENDDO
        IF(K.LT.IO) GOTO 17
        !           end of K loop
        if (ARCLEN.eq.ZERO) go to 4
        if (JB_LOCAL.eq.0) go to 56
        !**         Find number of independent boundaries
        J=0
        do K=1,JB_LOCAL
           if (KOUT(K).ne.0) then
              I=K
62            N=KOUT(I)
              KOUT(I)=0
              J=J+1
              do II=1,JB_LOCAL
                 if (N.eq.KENT(II)) then
                    if (II.eq.K) then
                       IB_LOCAL=IB_LOCAL+1
                       if (J.eq.JB_LOCAL) go to 56
                       go to 60
                    ENDIF
                    I=II
                    !C##IF PARALLEL
                    !C                        WRITE(OUTU,'(A,4I6)')'ASPENER>ME,N,KENT,I',
                    !C     $                        MYNOD,N,KENT(I),I
                    !C##ENDIF
                    go to 62
                 ENDIF
              ENDDO
           ENDIF
60         CONTINUE
        ENDDO
        IB_LOCAL=IB_LOCAL+1
        IF(WRNLEV.GE.2) WRITE(OUTU,*) &
             'CONNECTIVITY ERROR ON SPHERE NO. ',IR
56      AREA(IR)=IB_LOCAL*PIX2+EXANG+ARCLEN
        !CC##IF INTEL
        !
        !     Start the writing where the error is...
        !CC      IF((IMERI.EQ.-290).AND.(MYNOD.EQ.7).AND.(IR.EQ.551))
        !CC     $     WRITE(OUTU,'(A,4I5,6F12.6)')
        !CC     $     'bef. dinmod:ME,STEP,AREA,ENER',
        !CC     $     MYNOD,IMERI,IR,IB_LOCAL,PIX2,EXANG,ARCLEN,AREA(IR),RRSQ,PIX4
        !
        !CC            AREA(IR)=DINMOD(AREA(IR),PIX4)
        !CC##ELSE
        AREA(IR)=MOD(AREA(IR),PIX4)
        !CC##ENDIF
16      AREA(IR)=AREA(IR)*RRSQ
        E_DELTA = ASP_IR * AREA(IR)
        if (IMOVE(IR).LE.0) EU = EU + E_DELTA
        IF (QECONT) ECONT(IR) = ECONT(IR) + E_DELTA
        EE=EE+AREA(IR)
        !C            NAME_IR = RES( GETRES( IR, IBASE, NREST))

1001    continue
        IF(PRNLEV.GT.8) THEN
           write (OUTU,'(I10,3(A,F10.5))') IR, &
                ' EASP=',E_DELTA,' AREA=',AREA(IR),' ASPV=',ASP_IR
        ENDIF
     ENDIF
4    CONTINUE
  ENDDO
  !
  !     Start the writing where the error is...
  !CC      IF((IMERI.GT.-3150).AND.(IMERI.LT.4000))
  !CC     $     WRITE(OUTU,'(A,2I5,2F15.8)')
  !CC     $     'End ASPENER:ME,STEP,AREA,ENER',
  !CC     $     MYNOD,IMERI,EE,EU
  !
  !
  !     subtract reference energy from EU
  !     WARNING ***** REFER_SOLVENT_ENER was never initialized
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) EU = EU - REFER_SOLVENT_ENER
#else /**/
  EU = EU - REFER_SOLVENT_ENER
#endif 
  !
  IF(PRNLEV.GT.6) THEN
     EU1=EU
     EE1=EE
#if KEY_PARALLEL==1
     !         Be sure to change the prnlev on all nodes!!!!
     CALL GCOMB(EU1,1)
     CALL GCOMB(EE1,1)
#endif 
     WRITE(OUTU,'(A,F10.5,A,F10.5)') 'Total surface area:  ',EE1, &
          '  Surface energy:',EU1
  ENDIF
  !
  !     call find_asp to reset the asp's of ambiguous atom pairs
  CALL FIND_ASPMB(AREA, NATOM)
  RETURN
END SUBROUTINE ASPEN1MB

!----------------------------------------------------------------------------
SUBROUTINE SURFMEMB_IO(IUNIT)
  !
  !     This routine reads in solvation input data, from unit 83.
  !     Same as original Surface_Io except that ASP values are 10 times
  !     bigger.
  !
  !     vap_to_wat_kd.in  
  !     This is the input file with solvation energy data, using
  !     atomic solvation parameters derived from Wolfenden free
  !     energies of transfer, corrected for standard state by Kyte
  !     and Doolittle.
  !
  !     vap_to_wat_honig.in
  !     This is the input file with solvation energy data, using
  !     atomic solvation parameters derived from Wolfenden free
  !     energies of transfer, adjusted by Sharp, Nicholls, Friedman
  !     and Honig.
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
  use coord
  use deriv
  use ctitla
  use psf
  use string
  use stream
  use parallel
  use surface
  use surfmemb
  !
  !
  implicit none
  integer IUNIT
  !
  CHARACTER(len=8) ATOM_NAME,RES_NAME,SHARE_ATOM
  real(chm_real) ASP_VAL, REF_AREA, VDW_RAD
  INTEGER I, J, IRES
  real(chm_real) EDUM
  real(chm_real),parameter :: ECONT0(1) = (/ ZERO /)
  LOGICAL EOF
  !
  INTEGER LLEN
  INTEGER, PARAMETER :: MAXLIN=150
  CHARACTER(len=150) LINE
  character(len=8) mbdir_idx

  !.............  initial parameters for Implict Membrane

  membrane = .false.
  memb_dir = 0
  L_memb0  = 0.
  mb_center = 0.

  !

  !
  EOF=.FALSE.
  ! turn off
  IF(IUNIT.LE.0) THEN
     IF(PRNLEV.GE.5) WRITE(OUTU,'(A)') &
          'ASP surface area energy term turned off.'
     SOLVMEMB_ENTRD=.FALSE.
     RETURN
  ENDIF
  !
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) THEN
#endif 
     ! Read title
     CALL RDTITL(TITLEB,NTITLB,IUNIT,0)
     CALL WRTITL(TITLEB,NTITLB,OUTU,1)
50   CONTINUE
     CALL RDCMND(LINE,MAXLIN,LLEN,IUNIT,EOF,.FALSE., &
          .FALSE.,'SURFACE>    ')
     IF(EOF) GOTO 200
     CALL TRIME(LINE,LLEN)
     IF(LLEN.EQ.0) GOTO 50
     PROBE_RADIUS=NEXTF(LINE,LLEN)

     !     membrane parameters:
     ! 51   CONTINUE
     !      CALL RDCMND(LINE,MAXLIN,LLEN,IUNIT,EOF,.FALSE.,
     !     1            .FALSE.,'SURFACE>    ')
     !      IF(EOF) GOTO 200
     !      CALL TRIME(LINE,LLEN)
     !      IF(LLEN.EQ.0) GOTO 51

     L_memb0 = NEXTF(LINE,LLEN)
     IF( L_memb0 .gt. ZERO ) THEN 
        membrane  =  .true.
        L_memb0   =  L_memb0 / TWO
        mb_center =  NEXTF(LINE,LLEN)
        mbdir_idx =   NEXTA4(LINE,LLEN)
        If(( mbdir_idx(1:1).eq.'X').OR.(mbdir_idx.eq.'x')) then
           memb_dir = 1
        elseif((mbdir_idx(1:1).eq.'Y').OR.(mbdir_idx.eq.'y')) then
           memb_dir = 2
        elseif((mbdir_idx(1:1).eq.'Z').OR.(mbdir_idx.eq.'z')) then
           memb_dir = 3
        else
           write(outu,*) 'WARNING: NO MEMB_DIR is entered'
           write(outu,*) ' Z direction is assumed'
           memb_dir = 3
        Endif

     ELSEIF( L_memb0 .lt. ZERO ) then
        write(outu,*)  &
             'WARNING: the entered membrane thickness is NEGATIVE !'
        write(outu,*) ' NO MEMBRANE is present'
        membrane = .false.

     ELSE
        write(outu,*) ' NO MEMBRANE is present'
        membrane = .false.       
     ENDIF
     !.......................................................................
     ! set initial flags and hydrogen/lone pair flag array
     DO I = 1, NATOM
        IGNORE(I)=-2
        ASPV(I)=0.0
     ENDDO
     !
100  CONTINUE
     CALL RDCMND(LINE,MAXLIN,LLEN,IUNIT,EOF,.FALSE., &
          .FALSE.,'SURFACE>    ')
     IF(EOF) GOTO 200
     CALL TRIME(LINE,LLEN)
     IF(LLEN.EQ.0) GOTO 100
     !
     RES_NAME =NEXTA8(LINE,LLEN)
     IF(RES_NAME.EQ.'END') GOTO 200
     ATOM_NAME=NEXTA8(LINE,LLEN)
     ASP_VAL  =NEXTF(LINE,LLEN)
     VDW_RAD  =NEXTF(LINE,LLEN)
     REF_AREA =NEXTF(LINE,LLEN)
     SHARE_ATOM=NEXTA8(LINE,LLEN)
     !
     DO IRES = 1, NRES
        IF(RES(IRES).EQ.RES_NAME .OR. RES_NAME.EQ.'ANY') THEN
           DO I = IBASE(IRES)+1, IBASE(IRES+1)
              IF(EQSTWC(ATYPE(I),8,ATOM_NAME,8)) THEN
                 ASPV(I) = ASP_VAL/1000.
                 DX(I) = REF_AREA
                 IF(VDW_RAD.GT.0.0) THEN
                    VDW_SURF(I) = VDW_RAD + PROBE_RADIUS
                    IF(IGNORE(I).EQ.-1 .AND. WRNLEV.GT.2) &
                         WRITE(OUTU,'(A,I5,2(1X,A))') &
                         ' SURFACE_IO: TRYING TO SET AN IGNORED', &
                         ' ATOM:',I,ATYPE(I)(1:idleng), &
                         RES(IRES)(1:idleng)
                    IGNORE(I)=0
                    IF(SHARE_ATOM.NE.' ') THEN
                       DO J = I+1, IBASE(IRES+1)
                          IF(EQSTWC(ATYPE(J),8,SHARE_ATOM,8)) THEN
                             IGNORE(I)=J
                          ENDIF
                       ENDDO
                    ENDIF
                 ELSE
                    IGNORE(I)=-1
                    ASPV(I)=0.0
                    VDW_SURF(I)=0.0
                 ENDIF
              ENDIF
           ENDDO
        ENDIF
     ENDDO
     GOTO 100
     !
200  CONTINUE
     CLOSE(IUNIT)
     !
     ! Assign the vdw radius and asp for each atom, and find the reference
     ! energy for the whole object.
     IF(PRNLEV.GT.8) THEN
        WRITE(OUTU,'(A)') ' Atom TYPE RESN    ASPV   REF_AREA'
     ENDIF
     !
     REFER_SOLVENT_ENER = ZERO
     DO IRES=1,NRES
        DO I = IBASE(IRES) + 1, IBASE(IRES+1)
           IF(IGNORE(I).GE.0) THEN
              REF_AREA=DX(I)
              REFER_SOLVENT_ENER=REFER_SOLVENT_ENER + ASPV(I)*REF_AREA
              IF(PRNLEV.GT.8) THEN
                 WRITE(OUTU,'(I5,2(1X,A),2F10.5)') &
                      I,ATYPE(I)(1:idleng), RES(IRES)(1:idleng), ASPV(I), &
                      REF_AREA
              ENDIF
           ENDIF
           IF(IGNORE(I).GT.I) THEN
              IF(PRNLEV.GT.8) THEN
                 WRITE(OUTU,'(20X,A,I5,2(1X,A))') 'SWAPS WITH:', &
                      IGNORE(I),ATYPE(IGNORE(I))(1:idleng),  &
                      RES(IRES)(1:idleng)
              ENDIF
           ENDIF
           IF(IGNORE(I).EQ.-2) THEN
              IF(WRNLEV.GE.2) THEN
                 WRITE(OUTU,*)' SURFACE_IO: Atom type ', &
                      ATYPE(I)(1:idleng),' of residue type ', &
                      RES(IRES)(1:idleng), &
                      ' has no asp value. Its contribution will be zero.'
              ENDIF
              ASPV(I)=0.0
              VDW_SURF(I)=0.0
           ENDIF
        ENDDO
     ENDDO
     !
     IF(PRNLEV.GE.5) WRITE(OUTU,'(A,F10.5)') &
          'Reference solvent energy is ',REFER_SOLVENT_ENER
     !
#if KEY_PARALLEL==1
  ENDIF
  CALL PSND8(PROBE_RADIUS,1)
  CALL PSND8(REFER_SOLVENT_ENER,1)
  CALL PSND4(IGNORE,NATOM)
  CALL PSND8(ASPV,NATOM)
  CALL PSND8(VDW_SURF,NATOM)
#endif 
  !
  SOLVMEMB_ENTRD=.TRUE.
  DX(1:NATOM)=zero
  DY(1:NATOM)=zero
  DZ(1:NATOM)=zero
  CALL ASPENERMB(EDUM,X,Y,Z,DX,DY,DZ,.FALSE.,ECONT0)
  !
  RETURN
END SUBROUTINE SURFMEMB_IO

subroutine Find_AspMB(AREA, NATOMX)
  !
  !   This routine is called by surface_io_10.for and usere1.for.  It
  !   resets the atomic solvation parameters of shared charge pairs
  !   based on solvent accessible surface area.
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use stream
  use surface
  use surfmemb
  implicit none
  !      definitions of passed variables ...
  integer NATOMX
  real(chm_real) AREA(NATOMX)
  !      local variables ...
  integer IRES, I, J
  real(chm_real)  TEMP

  !
  IF(PRNLEV.GT.6) WRITE(OUTU,'(A)') &
       ' FIND_ASP: Checking for ASP pairs to swap.'
  !
  DO IRES=1,NRES
     DO I = IBASE(IRES) + 1, IBASE(IRES+1)
        IF(IGNORE(I).GT.I) THEN
           ! atoms in this residue may need asp's adjusted.
           J=IGNORE(I)
           IF(AREA(I).GE.AREA(J)) THEN
              IF (ASPV(I).GT.ASPV(J)) THEN
                 TEMP = ASPV(I)
                 ASPV(I) = ASPV(J)
                 ASPV(J) = TEMP
                 IF(PRNLEV.GT.2) WRITE(OUTU,45) IRES, &
                      RES(IRES)(1:idleng), &
                      ATYPE(I)(1:idleng),ATYPE(J)(1:idleng)
45               FORMAT(' FIND_ASP: ASP values exchanged for:', &
                      I6,3(1X,A))
              ENDIF
           ELSE
              IF (ASPV(I).LT.ASPV(J)) THEN
                 TEMP = ASPV(I)
                 ASPV(I) = ASPV(J)
                 ASPV(J) = TEMP
                 IF(PRNLEV.GT.2) WRITE(OUTU,45) IRES,RES(IRES), &
                      ATYPE(I),ATYPE(J)
              ENDIF
           ENDIF
        ENDIF
     ENDDO
  ENDDO
END subroutine Find_AspMB
#else /**/
  SUBROUTINE NULL_ASPMB
    RETURN
  END SUBROUTINE NULL_ASPMB
#endif 


