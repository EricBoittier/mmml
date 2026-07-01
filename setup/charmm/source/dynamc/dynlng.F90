SUBROUTINE DLNGV(GAMMA,IG)
  !
  ! This function routine generates 3*NATOM Gaussian
  ! random deviates of 0.0 mean and standard deviation RF.
  ! The algorithm from Box and Muller.
  !
  ! Bernard R. Brooks   January, 1988
  !
  use chm_kinds
  use clcg_mod,only:random
  use dimens_fcm
  use number
  use consta
  use deriv
  use psf
  use energym
  use fourdm
  use rndnum
  use parallel
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec, &
#if KEY_DOMDEC_GPU==1
       q_gpu, &
#endif
       natoml,atoml
#endif
#if KEY_DOMDEC_GPU==1
  use domdec_random,only:generate_random_buffer_gpu, random_buffer
#endif
  implicit none
  !
  real(chm_real)  GAMMA(*)
  INTEGER IG
  !
  !
  real(chm_real)  A,B,PIS,RDX,RDY,RDZ,RD4
  INTEGER I,K,frstind,lastind
  integer ia
  !
  PIS=PI
  K=0
  if (.not.qoldrng) then     !yw 05-Aug-2008
     IG=1
#if KEY_PARALLEL==1
     IG=MYNODP          
#endif
  endif
  frstind=1
  lastind=natom

#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
  if (q_domdec) then
     frstind = 1
     lastind = natoml
  else
#endif 
#if KEY_PARAFULL==1
     frstind=1+IPARPT(MYNOD)
     lastind=IPARPT(MYNODP)
#endif 
#if KEY_DOMDEC==1
  endif  
#endif
#endif 
  ! parallel random order: this really hurts the parallel performance,
  ! but it is here for langevin dynamics debugging purpose on a
  ! special request from X. Wu :-) But it may (??) also make testing
  ! parallel/parallel methods much easier. MH, May 2012
!  qrandpar=.true.
  if(qrandpar)then
     if(frstind>1)then
        do i=1,frstind-1
           if(imove(i)==0 )then
              if(k==0)then
                 k=1
                 a=random(ig)
                 a=random(ig)
                 a=random(ig)
                 a=random(ig)
              else
                 k=0
                 a=random(ig)
                 a=random(ig)
              endif
           endif
        enddo
     endif
  endif
  ! end of parallel order

#if KEY_DOMDEC_GPU==1
  if (q_domdec .and. q_gpu) then
     call generate_random_buffer_gpu(natoml)
  endif
#endif

  do ia=frstind,lastind
#if KEY_DOMDEC==1
     if (q_domdec) then
        i = atoml(ia)
     else
#endif 
        i = ia
#if KEY_DOMDEC==1
     endif  
#endif
#if KEY_DOMDEC_GPU==1
     if (q_domdec .and. q_gpu) then
        dx(i)=dx(i)+gamma(i)*random_buffer(3*ia-2)
        dy(i)=dy(i)+gamma(i)*random_buffer(3*ia-1)
        dz(i)=dz(i)+gamma(i)*random_buffer(3*ia)
     else
#endif
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
     IF(JPBLOCK(I) == MYNOD) THEN
#endif 
#endif 
        IF(IMOVE(I) == 0) THEN
           IF(K == 0) THEN
              A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
              B=TWO*PIS*RANDOM(IG)
              K=1
              RDX=A*COS(B)
              RDY=A*SIN(B)
              DX(I)=DX(I)+RDX
              DY(I)=DY(I)+RDY
              A=SQRT(MINTWO*LOG(RANDOM(IG)))
              B=TWO*PIS*RANDOM(IG)
              RDZ=GAMMA(I)*A*COS(B)
              DZ(I)=DZ(I)+RDZ
#if KEY_FOURD==1
              IF(DIM4) THEN
                 K=0
                 RD4=GAMMA(I)*A*SIN(B)
                 DFDIM(I)=DFDIM(I)+RD4
              ENDIF
#endif 
           ELSE
              K=0
              RDX=GAMMA(I)*A*SIN(B)
              DX(I)=DX(I)+RDX
              A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
              B=TWO*PIS*RANDOM(IG)
              RDY=A*COS(B)
              RDZ=A*SIN(B)
              DY(I)=DY(I)+RDY
              DZ(I)=DZ(I)+RDZ
           ENDIF
        ENDIF
#if KEY_PARASCAL==1
     ENDIF    
#endif
#if KEY_DOMDEC_GPU==1
     endif
#endif
  ENDDO
  !
  ! again parallel random ordering
  if(qrandpar)then
     if(lastind<natom)then
        do i=lastind+1,natom
           if(imove(i)==0 )then
              if(k==0)then
                 k=1
                 a=random(ig)
                 a=random(ig)
                 a=random(ig)
                 a=random(ig)
              else
                 k=0
                 a=random(ig)
                 a=random(ig)
              endif
           endif
        enddo
     endif
  endif
  !
  RETURN
END SUBROUTINE DLNGV

SUBROUTINE DLNGVdrude(GAMMA,IG)
  !
  ! This function routine generates 3*NATOM Gaussian
  ! random deviates of 0.0 mean and standard deviation RF.
  ! The algorithm from Box and Muller.
  !
  ! Bernard R. Brooks   January, 1988
  !
  ! SB Sep 2014, add drude support; should eventually be merged in dlngv()
  !
  use chm_kinds
  use clcg_mod,only:random
  use dimens_fcm
  use number
  use consta
  use deriv
  use psf
  use energym
  use fourdm
  use rndnum
  use parallel
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec,natoml,atoml  
#endif
  implicit none
  !
  real(chm_real)  GAMMA(*)
  INTEGER IG
  !
  !
  real(chm_real)  A,B,PIS,RDX,RDY,RDZ,RD4
  INTEGER I,K,frstind,lastind
  integer ia
  !
  real(chm_real) :: mt
  real(chm_real) :: gammar,gammad
  real(chm_real) :: rdx1,rdy1,rdz1,rdx2,rdy2,rdz2
  !
  logical q_drude_atom
  !
  PIS=PI
  K=0
  if (.not.qoldrng) then     !yw 05-Aug-2008
     IG=1
#if KEY_PARALLEL==1
     IG=MYNODP          
#endif
  endif
  frstind=1
  lastind=natom

#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
  if (q_domdec) then
     frstind = 1
     lastind = natoml
  else
#endif 
#if KEY_PARAFULL==1
     frstind=1+IPARPT(MYNOD)
     lastind=IPARPT(MYNODP)
#endif 
#if KEY_DOMDEC==1
  endif  
#endif
#endif 
  ! parallel random order: this really hurts the parallel performance,
  ! but it is here for langevin dynamics debugging purpose on a
  ! special request from X. Wu :-) But it may (??) also make testing
  ! parallel/parallel methods much easier. MH, May 2012
!  qrandpar=.true.
  if(qrandpar)then
     if(frstind>1)then
        do i=1,frstind-1
           if(imove(i)==0 )then
              if(k==0)then
                 k=1
                 a=random(ig)
                 a=random(ig)
                 a=random(ig)
                 a=random(ig)
              else
                 k=0
                 a=random(ig)
                 a=random(ig)
              endif
           endif
        enddo
     endif
  endif
  ! end of parallel order

  do ia=frstind,lastind
#if KEY_DOMDEC==1
     if (q_domdec) then
        i = atoml(ia)
     else
#endif 
        i = ia
#if KEY_DOMDEC==1
     endif  
#endif
#if KEY_PARALLEL==1
#if KEY_PARASCAL==1
     IF(JPBLOCK(I) == MYNOD) THEN
#endif 
#endif 
!        IF(IMOVE(I) == 0) THEN
#if KEY_FOURD==1
        IF(DIM4) THEN
           CALL WRNDIE(-2,'<DYNLNG>', &
                '4D and Drudes not supported.')
           K=0
           RD4=GAMMA(I)*A*SIN(B)
           DFDIM(I)=DFDIM(I)+RD4
        ENDIF
#endif 
        IF((IMOVE(I) == 0).and.(.not.isdrude(i))) THEN

           ! APH: Safe drude check
           q_drude_atom = .false.
           if (i < natom) then
              if (isdrude(i+1)) q_drude_atom = .true.
           endif

           drudepair: if (q_drude_atom) then
              !sb we have a nucleus/drude pair (i/i+1)
              !sb assign random forces to the pair!
              gammar=gamma(i)
              gammad=gamma(i+1)
              ! each nucleus/drude pair consumes 6 random numbers, so while we
              ! need the k=0/1 branches, at the end k is the same as before ..
              IF(K == 0) THEN
                 A=SQRT(MINTWO*LOG(RANDOM(IG)))
                 B=TWO*PIS*RANDOM(IG)
                 RDX1=gammar*A*COS(B)
                 RDY1=gammar*A*SIN(B)
                 A=SQRT(MINTWO*LOG(RANDOM(IG)))
                 B=TWO*PIS*RANDOM(IG)
                 RDZ1=gammar*A*COS(B)
                 RDX2=GAMMAd*A*SIN(B)
                 A=SQRT(MINTWO*LOG(RANDOM(IG)))
                 B=TWO*PIS*RANDOM(IG)
                 RDY2=gammad*A*COS(B)
                 RDZ2=gammad*A*SIN(B)
              ELSE
                 RDX1=GAMMAr*A*SIN(B)
                 A=SQRT(MINTWO*LOG(RANDOM(IG)))
                 B=TWO*PIS*RANDOM(IG)
                 RDY1=gammar*A*COS(B)
                 RDZ1=gammar*A*SIN(B)
                 A=SQRT(MINTWO*LOG(RANDOM(IG)))
                 B=TWO*PIS*RANDOM(IG)
                 RDx2=gammad*A*COS(B)
                 RDy2=gammad*A*SIN(B)
                 A=SQRT(MINTWO*LOG(RANDOM(IG)))
                 B=TWO*PIS*RANDOM(IG)
                 rdz2=gammad*A*COS(B)
              ENDIF
              ! have now random forces on c.o.m and in nucleus-drude direction
              ! project back on atomic forces

              mt = amass(i) + amass(i+1)
              dx(i)   = dx(i)   + amass(i)  /mt * rdx1 - rdx2
              dy(i)   = dy(i)   + amass(i)  /mt * rdy1 - rdy2
              dz(i)   = dz(i)   + amass(i)  /mt * rdz1 - rdz2
              dx(i+1) = dx(i+1) + amass(i+1)/mt * rdx1 + rdx2
              dy(i+1) = dy(i+1) + amass(i+1)/mt * rdy1 + rdy2
              dz(i+1) = dz(i+1) + amass(i+1)/mt * rdz1 + rdz2

           else
              ! can use the regular code (particle without Drude) ..
              IF(K == 0) THEN
                 A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
                 B=TWO*PIS*RANDOM(IG)
                 K=1
                 RDX=A*COS(B)
                 RDY=A*SIN(B)
                 DX(I)=DX(I)+RDX
                 DY(I)=DY(I)+RDY
                 A=SQRT(MINTWO*LOG(RANDOM(IG)))
                 B=TWO*PIS*RANDOM(IG)
                 RDZ=GAMMA(I)*A*COS(B)
                 DZ(I)=DZ(I)+RDZ
              ELSE
                 K=0
                 RDX=GAMMA(I)*A*SIN(B)
                 DX(I)=DX(I)+RDX
                 A=GAMMA(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
                 B=TWO*PIS*RANDOM(IG)
                 RDY=A*COS(B)
                 RDZ=A*SIN(B)
                 DY(I)=DY(I)+RDY
                 DZ(I)=DZ(I)+RDZ
              ENDIF
           endif drudepair
        ENDIF
#if KEY_PARASCAL==1
     ENDIF    
#endif
  ENDDO
  !
  ! again parallel random ordering
  if(qrandpar)then
     if(lastind<natom)then
        do i=lastind+1,natom
           if(imove(i)==0 )then
              if(k==0)then
                 k=1
                 a=random(ig)
                 a=random(ig)
                 a=random(ig)
                 a=random(ig)
              else
                 k=0
                 a=random(ig)
                 a=random(ig)
              endif
           endif
        enddo
     endif
  endif
  !
  RETURN
END SUBROUTINE DLNGVdrude


SUBROUTINE LNGFIL(ILANG,IPSTOP,GAMMA,TBATH,DELTA,RBUF, &
     SXREF,SYREF,SZREF,X,Y,Z &
#if KEY_ACE==1
     ,QRADI,BSARR,LLBACE                & 
#endif
     )
  !
  !     This routine fills the GAMMA arrays for Langevin dynamics.
  !
  !    GAMMA(NATOM,1) - RDF (std.dev. of random force)
  !    GAMMA(NATOM,2) - BETA  ( dx scale factor)
  !    GAMMA(NATOM,3) - ALPHA ( x-xold scale factor)
  !    GAMMA(NATOM,4) - Velocity compute scale factor
  !
  !     Charles Brooks III and Axel Brunger, 3-JULY-1983
  !     Modified for a more accurate integration
  !     Bernard R. Brooks  December,1987
  !     ------------------------------------------------
  !
  ! input/output
  use chm_kinds
  use dimens_fcm
  use number
  !
  use euler
  use consta
  use psf
  use cnst_fcm
  use stream
  implicit none
  !
  INTEGER ILANG,IPSTOP
  real(chm_real) GAMMA(*),TBATH,DELTA,RBUF
  real(chm_real) SXREF, SYREF, SZREF
  real(chm_real)  X(*), Y(*), Z(*)
#if KEY_ACE==1
  real(chm_real) QRADI(*),BSARR(*)       
#endif
#if KEY_ACE==1
  LOGICAL LLBACE                 
#endif
  ! local
  INTEGER I,JLANG,NATOM2,NATOM3
  LOGICAL QYES
  real(chm_real)  RFD,GAM,KBT
  real(chm_real)  ROXY, RBUF2, DDX, DDY, DDZ
#if KEY_ACE==1
  real(chm_real)  SOLFAC                 
#endif
  !
  ! begin
  call allocate_cnst(natom)
  rbuf2=rbuf*rbuf
  natom2=natom+natom
  natom3=natom2+natom

  do i=1,natom
     gamma(i)=zero
     if(imove(i) /= 0) then
        gamma(i+natom) =zero
        gamma(i+natom2)=one
     else
        gamma(i+natom)=delta*delta/amass(i)
        gamma(i+natom2)=one
     endif
     gamma(i+natom3)=half/delta
  enddo
  !
#if KEY_TNPACK==1
  IF(ILANG == 0 .AND. .NOT.QEULER) GOTO 200
#else /**/
  IF(ILANG == 0) GOTO 200
#endif 
  JLANG=0
  KBT=KBOLTZ*TBATH
  !
  do i=1,natom
     if(abs(fbeta(i)) > rsmall.and.imove(i) == 0) then
        if(rbuf2 < rsmall) then
           qyes=.true.
        else
           ddx=x(i)-sxref
           ddy=y(i)-syref
           ddz=z(i)-szref
           roxy=ddx*ddx+ddy*ddy+ddz*ddz
           qyes=(roxy > rbuf2)
        endif
        !
        if(qyes) then
#if KEY_ACE==1
           IF (LLBACE) THEN
              !                 scale FBETA with (Ri/bi)^2, where Ri=vdW Radius, and
              !                 bi=Born solvation radius of atom i (see ACE potential)
              SOLFAC=QRADI(IAC(I))/BSARR(I)
              GAM=TIMFAC*SOLFAC*SOLFAC*FBETA(I)*DELTA
           ELSE
#endif 
              GAM=TIMFAC*FBETA(I)*DELTA
#if KEY_ACE==1
           ENDIF                                   
#endif
           RFD=SQRT(2.0*AMASS(I)*GAM*KBT)/DELTA
           JLANG=JLANG+1
           GAMMA(I)=RFD
#if KEY_TNPACK==1
           IF(QEULER) THEN
              GAMMA(I+NATOM)=DELTA*DELTA/((ONE+GAM)*AMASS(I))
              GAMMA(I+NATOM2)=ONE/(ONE+GAM)
              GAMMA(I+NATOM3)=HALF/DELTA
           ELSE
#endif 
              GAMMA(I+NATOM)=DELTA*DELTA/((ONE+GAM*HALF)*AMASS(I))
              GAMMA(I+NATOM2)=(ONE-GAM*HALF)/(ONE+GAM*HALF)
              GAMMA(I+NATOM3)=HALF*SQRT(ONE+GAM*HALF)/DELTA
#if KEY_TNPACK==1
           ENDIF
#endif 
        ENDIF
        !
     ENDIF
  ENDDO
  !
  IF(JLANG > 0 .AND. IPSTOP == 0 .AND. PRNLEV >= 2) THEN
     WRITE(OUTU,9000) TBATH, DELTA, JLANG, RBUF, SXREF, &
          SYREF, SZREF
9000 FORMAT(' LNGFIL: TBATH = ',F12.6,'  DELTA =',F12.6,/, &
          ' LNGFIL: Langevin dynamics setup for ',I7,' atoms',/, &
          ' LNGFIL: RBUF =',F12.6,/, &
          ' LNGFIL: SXREF =',F12.6,' SYREF =',F12.6,' SZREF =',F12.6)
#if KEY_ACE==1
     IF (LLBACE) WRITE(OUTU,9001)                          
#endif
#if KEY_ACE==1
9001 FORMAT(' LNGFIL: Use (Rvdw/Bsolv)^2 to scale FBETA')  
#endif
  ENDIF
  !
  ! if no atoms have any friction, turn off langevin flag?
  !   No!  Don't mess with the flags, just print a warning. -BRB
  !CC   IF(JLANG == 0) ILANG=0
  IF(JLANG == 0 .AND. IPSTOP.EQ.0) THEN
     CALL WRNDIE(-1,'<DYNLNG>', &
          'Langevin integration invoked without friction on any atom.')
  ENDIF
200 CONTINUE

  RETURN
END SUBROUTINE LNGFIL

SUBROUTINE LNGFIL2hb(ILANG,IPSTOP,GAMMA,TBATH,DELTA,RBUF, &
     SXREF,SYREF,SZREF,X,Y,Z&
#if KEY_ACE==1
     ,QRADI,BSARR,LLBACE                & 
#endif
     )
  !
  !     This routine fills the GAMMA arrays for Langevin dynamics.
  !
  !    GAMMA(NATOM,1) - RDF (std.dev. of random force)
  !    GAMMA(NATOM,2) - BETA  ( dx scale factor)
  !    GAMMA(NATOM,3) - ALPHA ( x-xold scale factor)
  !    GAMMA(NATOM,4) - Velocity compute scale factor
  !
  !     Charles Brooks III and Axel Brunger, 3-JULY-1983
  !     Modified for a more accurate integration
  !     Bernard R. Brooks  December,1987
  !     ------------------------------------------------
  !
  !    SB, Sep 2014: takes care of presence of drudes ..
  !    functionality could be integrated in regular lngfil() above    
  !
  ! input/output
  use chm_kinds
  use dimens_fcm
  use number
  !
  use euler
  use consta
  use psf
  use cnst_fcm
  use stream
  implicit none
  !
  INTEGER ILANG,IPSTOP
  real(chm_real) GAMMA(*),TBATH,DELTA,RBUF
  real(chm_real) SXREF, SYREF, SZREF
  real(chm_real)  X(*), Y(*), Z(*)
#if KEY_ACE==1
  real(chm_real) QRADI(*),BSARR(*)       
#endif
#if KEY_ACE==1
  LOGICAL LLBACE                 
#endif
  ! local
  INTEGER I,JLANG,NATOM2,NATOM3
  integer jlangr,jlangd
  LOGICAL QYES
  real(chm_real)  RFD,GAM,KBT,kbtd
  real(chm_real)  ROXY, RBUF2, DDX, DDY, DDZ
#if KEY_ACE==1
  real(chm_real)  SOLFAC                 
#endif
  real(chm_real) masst, massr
  !
  ! begin
  call allocate_cnst(natom)
  rbuf2=rbuf*rbuf
  natom2=natom+natom
  natom3=natom2+natom

  do i=1,natom
     gamma(i)=zero
     if(imove(i) /= 0) then
        gamma(i+natom) =zero
        gamma(i+natom2)=one
     else
        gamma(i+natom)=delta*delta/amass(i)
        gamma(i+natom2)=one
        if (isdrude(i)) then
           ! we need to fix entries for i and i-1
           masst = amass(i) + amass(i-1)
           massr = amass(i) * amass(i-1) / masst
           gamma(i+natom)=delta*delta/massr
           gamma(i-1+natom)=delta*delta/masst
        endif
     endif
     gamma(i+natom3)=half/delta
  enddo
  !
#if KEY_TNPACK==1
  IF(ILANG == 0 .AND. .NOT.QEULER) GOTO 200
#else /**/
  IF(ILANG == 0) GOTO 200
#endif 
  JLANG=0
  jlangr=0
  jlangd=0
  kbtd=kboltz ! implies t_drude = 1
  KBT=KBOLTZ*TBATH
  !
  do i=1,natom
     if(abs(fbeta(i)) > rsmall.and.imove(i) == 0) then
        if(rbuf2 < rsmall) then
           qyes=.true.
        else
           ddx=x(i)-sxref
           ddy=y(i)-syref
           ddz=z(i)-szref
           roxy=ddx*ddx+ddy*ddy+ddz*ddz
           qyes=(roxy > rbuf2)
        endif
        !
        if(qyes) then
#if KEY_ACE==1
           IF (LLBACE) THEN
              !                 scale FBETA with (Ri/bi)^2, where Ri=vdW Radius, and
              !                 bi=Born solvation radius of atom i (see ACE potential)
              SOLFAC=QRADI(IAC(I))/BSARR(I)
              GAM=TIMFAC*SOLFAC*SOLFAC*FBETA(I)*DELTA
           ELSE
#endif 
              GAM=TIMFAC*FBETA(I)*DELTA
#if KEY_ACE==1
           ENDIF                                   
#endif
           RFD=SQRT(2.0*AMASS(I)*GAM*KBT)/DELTA
           JLANG=JLANG+1
           GAMMA(I)=RFD
#if KEY_TNPACK==1
           IF(QEULER) THEN
              GAMMA(I+NATOM)=DELTA*DELTA/((ONE+GAM)*AMASS(I))
              GAMMA(I+NATOM2)=ONE/(ONE+GAM)
              GAMMA(I+NATOM3)=HALF/DELTA
           ELSE
#endif 
              GAMMA(I+NATOM)=DELTA*DELTA/((ONE+GAM*HALF)*AMASS(I))
              GAMMA(I+NATOM2)=(ONE-GAM*HALF)/(ONE+GAM*HALF)
              GAMMA(I+NATOM3)=HALF*SQRT(ONE+GAM*HALF)/DELTA
              if (isdrude(i)) then
                 ! fix up atoms i and i-1
                 jlangd=jlangd+1
                 masst=amass(i)+amass(i-1)
                 massr=amass(i)*amass(i-1)/masst
                 ! the drude rel. d.o.f
                 GAM=TIMFAC*FBETA(I)*DELTA
                 RFD=SQRT(2.0*massr*GAM*KBTd)/DELTA ! sb: note kbtd !!
                 gamma(i)=rfd
                 GAMMA(I+NATOM)=DELTA*DELTA/((ONE+GAM*HALF)*massr)
                 GAMMA(I+NATOM2)=(ONE-GAM*HALF)/(ONE+GAM*HALF)
                 GAMMA(I+NATOM3)=HALF*SQRT(ONE+GAM*HALF)/DELTA
                 ! heavy atom + drude c.o.m. d.o.f.
                 GAM=TIMFAC*FBETA(I-1)*DELTA
                 RFD=SQRT(2.0*masst*GAM*KBT)/DELTA
                 gamma(i-1)=rfd
                 GAMMA(I-1+NATOM)=DELTA*DELTA/((ONE+GAM*HALF)*masst)
                 GAMMA(I-1+NATOM2)=(ONE-GAM*HALF)/(ONE+GAM*HALF)
                 GAMMA(I-1+NATOM3)=HALF*SQRT(ONE+GAM*HALF)/DELTA
              endif
#if KEY_TNPACK==1
           ENDIF
#endif 
        ENDIF
        !
     ENDIF
  ENDDO
  !
  jlangr = jlang - jlangd
  IF(JLANG > 0 .AND. IPSTOP == 0 .AND. PRNLEV >= 2) THEN
     WRITE(OUTU,9000) TBATH, DELTA, JLANG, RBUF, SXREF, &
          SYREF, SZREF
9000 FORMAT(' LNGFIL: TBATH = ',F12.6,'  DELTA =',F12.6,/, &
          ' LNGFIL: Langevin dynamics setup for ',I7,' atoms',/, &
          ' LNGFIL: RBUF =',F12.6,/, &
          ' LNGFIL: SXREF =',F12.6,' SYREF =',F12.6,' SZREF =',F12.6)
     if (jlangd > 0) then
        write(outu,*) ' Drudes present:', jlangd, ' relative coors held at 1K'
     endif
#if KEY_ACE==1
     IF (LLBACE) WRITE(OUTU,9001)                          
#endif
#if KEY_ACE==1
9001 FORMAT(' LNGFIL: Use (Rvdw/Bsolv)^2 to scale FBETA')  
#endif
  ENDIF
  !
  ! if no atoms have any friction, turn off langevin flag?
  !   No!  Don't mess with the flags, just print a warning. -BRB
  !CC   IF(JLANG == 0) ILANG=0
  IF(JLANG == 0 .AND. IPSTOP.EQ.0) THEN
     CALL WRNDIE(-1,'<DYNLNG>', &
          'Langevin integration invoked without friction on any atom.')
  ENDIF
  IF(JLANGD == 0 .AND. IPSTOP.EQ.0) THEN
     CALL WRNDIE(-1,'<DYNLNG>', &
          'Drude Langevin integration invoked without friction on any Drude d.o.f.')
  ENDIF
200 CONTINUE
  RETURN
END SUBROUTINE LNGFIL2hb

#if KEY_BLOCK==1 /*ldm*/
SUBROUTINE LNGFIL2(ILALDM,IPSTOP,GAMMALD &
     ,TBATH,DELTA,BIBLAM,NBLOCK,BMLAMB)
  !
  !     This routine fills the GAMMALD arrays for Langevin dynamics.
  !
  !    GAMMALD(NBLOCK,1) - RDF (std.dev. of random force)
  !    GAMMALD(NBLOCK,2) - BETA  ( dx scale factor)
  !    GAMMALD(NBLOCK,3) - ALPHA ( x-xold scale factor)
  !    GAMMALD(NBLOCK,4) - Velocity compute scale factor
  !
  !     Charles Brooks III and Axel Brunger, 3-JULY-1983
  !     Modified for a more accurate integration
  !     Bernard R. Brooks  December,1987
  !     Shinichi Banba for lambda-dynamics
  !     ------------------------------------------------
  !
  ! input/output
  use chm_kinds
  use dimens_fcm
  use number
  use euler
  use consta
  use psf
  use stream
  implicit none
  !
  LOGICAL ILALDM
  INTEGER IPSTOP
  real(chm_real) TBATH,DELTA
  real(chm_real) GAMMALD(*),BIBLAM(*),BMLAMB(*)
  ! local
  INTEGER I,JLANG,NBLOCK2,NBLOCK3,NBLOCK
  real(chm_real)  RFD,GAM,KBT
  !
  ! begin
  NBLOCK2=NBLOCK+NBLOCK
  NBLOCK3=NBLOCK2+NBLOCK
  !
  DO I=1,NBLOCK
     GAMMALD(I)=ZERO
     GAMMALD(I+NBLOCK)=DELTA/BMLAMB(I)
     GAMMALD(I+NBLOCK2)=ONE
     GAMMALD(I+NBLOCK3)=HALF
  ENDDO
  !
#if KEY_TNPACK==1
  IF(.NOT.ILALDM .AND. .NOT.QEULER) GOTO 200
#else /**/
  IF(.NOT.ILALDM) GOTO 200
#endif 
  JLANG=0
  KBT=KBOLTZ*TBATH
  !
  DO I=1,NBLOCK
     IF(ABS(BIBLAM(I)) > RSMALL)THEN
        GAM=TIMFAC*BIBLAM(I)*DELTA
        RFD=SQRT(2.0*BMLAMB(I)*GAM*KBT)/DELTA
        JLANG=JLANG+1
        GAMMALD(I)=RFD
#if KEY_TNPACK==1
        IF(QEULER) THEN
           GAMMALD(I+NBLOCK)=DELTA*DELTA/((ONE+GAM)*BMLAMB(I))
           GAMMALD(I+NBLOCK2)=ONE/(ONE+GAM)
           GAMMALD(I+NBLOCK3)=HALF/DELTA
        ELSE
#endif /*  LDM*/
           !   GAMMALD(I+NBLOCK) is multiplied by DELTA for next calculation
           GAMMALD(I+NBLOCK)=DELTA/((ONE+GAM*HALF)*BMLAMB(I))
           GAMMALD(I+NBLOCK2)=(ONE-GAM*HALF)/(ONE+GAM*HALF)
           !   GAMMALD(I+NBLOCK3) is multiplied by DELTA for next calculation
           GAMMALD(I+NBLOCK3)=HALF*SQRT(ONE+GAM*HALF)
#if KEY_TNPACK==1
        ENDIF
#endif 
     ENDIF
  ENDDO
  !
  IF(JLANG > 0 .AND. IPSTOP == 0 .AND. PRNLEV >= 2) THEN
     WRITE(OUTU,9000) TBATH, DELTA, JLANG
9000 FORMAT(' LNGFIL: TBATH = ',F12.6,'  DELTA =',F12.6,/, &
          ' LNGFIL: Langevin dynamics setup for ',I7,' blocks',/)
  ENDIF
  !
  ! if no atoms have any friction, turn off langevin flag?
  !   No!  Don't mess with the flags, just print a warning. -BRB
  !CC   IF(JLANG == 0) ILALDM=.FALSE.
  !CC      IF(JLANG == 0 .AND. IPSTOP.EQ.0) THEN
  !CC         CALL WRNDIE(-1,'<DYNLNG>',
  !CC     &'Langevin integration invoked without friction on any lambda.')
  !CC      ENDIF
200 CONTINUE
  !
  RETURN
END SUBROUTINE LNGFIL2

!
SUBROUTINE DLNGV2(GAMMALD,NBLOCK,LSTRT,FRAND)
  !
  ! This function routine generates 3*NATOM Gaussian
  ! random deviates of 0.0 mean and standard deviation RF.
  ! The algorithm from Box and Muller.
  !
  ! Bernard R. Brooks   January, 1988
  ! Shinichi Banba for lambda-dynamics
  !
  use chm_kinds
  use clcg_mod,only:random
  use dimens_fcm
  use number
  use reawri
  use rndnum
  use consta
  use parallel
  !
  implicit none
  real(chm_real)  GAMMALD(*), FRAND(*)
  !
  !
  real(chm_real)  A,B,PIS,RDX,RDY,RDZ,RD4
  INTEGER I,K, NBLOCK,LSTRT,IG
  !
  PIS=PI
  if (.not.qoldrng) then     !yw 05-Aug-2008
     IG=1
#if KEY_PARALLEL==1
     IG=MYNODP          
#endif
  endif
  K=0
  DO I=LSTRT, NBLOCK
     IF(K == 0) THEN
        if (qoldrng) then     !yw 05-Aug-2008
           A=GAMMALD(I)*SQRT(MINTWO*LOG(RANDOM(ISEED)))
           B=TWO*PIS*RANDOM(ISEED)
        else
           A=GAMMALD(I)*SQRT(MINTWO*LOG(RANDOM(IG)))
           B=TWO*PIS*RANDOM(IG)
        endif
        K=1
        FRAND(I)=A*COS(B)
     ELSE
        K=0
        FRAND(I) = A*SIN(B)
     ENDIF
  ENDDO

  RETURN
END SUBROUTINE DLNGV2
#endif /*  LDM*/

SUBROUTINE LNGFIL2THETA(ILALDM,IPSTOP,GAMMALDTHETA &
     ,TBATH,DELTA,THETABIB,THETAM)
  !
  !     This routine fills the GAMMALD arrays for theta of theta-dynamics
  !     in Langevin dynamics.
  !
  !    GAMMALDTHETA(1) - RDF (std.dev. of random force)
  !    GAMMALDTHETA(2) - BETA  ( dx scale factor)
  !    GAMMALDTHETA(3) - ALPHA ( x-xold scale factor)
  !    GAMMALDTHETA(4) - Velocity compute scale factor
  !
  !
  !c   Generated based on SUBROUTINE LNGFIL2()
  !c   Lianqing Zheng and Wei Yang July, 2007
  ! input/output
  use chm_kinds
  use dimens_fcm
  use number
  use euler
  use consta
  use psf
  use stream
  implicit none
  !
  LOGICAL ILALDM
  INTEGER IPSTOP
  real(chm_real) TBATH,DELTA
  real(chm_real) GAMMALDTHETA(*),THETABIB,THETAM
  ! local
  INTEGER I,JLANG
  real(chm_real)  RFD,GAM,KBT

  GAMMALDTHETA(1)=ZERO
  GAMMALDTHETA(2)=DELTA/THETAM
  GAMMALDTHETA(3)=ONE
  GAMMALDTHETA(4)=HALF

  IF(.NOT.ILALDM) GOTO 300
  JLANG=0
  KBT=KBOLTZ*TBATH

  IF(ABS(THETABIB) > RSMALL)THEN
     GAM=TIMFAC*THETABIB*DELTA
     RFD=SQRT(2.0*THETAM*GAM*KBT)/DELTA
     JLANG=JLANG+1
     GAMMALDTHETA(1)=RFD
     GAMMALDTHETA(2)=DELTA/((ONE+GAM*HALF)*THETAM)
     GAMMALDTHETA(3)=(ONE-GAM*HALF)/(ONE+GAM*HALF)
     GAMMALDTHETA(4)=HALF*SQRT(ONE+GAM*HALF)
  ENDIF

  IF(JLANG > 0 .AND. IPSTOP == 0 .AND. PRNLEV >= 2) THEN
     WRITE(OUTU,8000) TBATH, DELTA, JLANG
8000 FORMAT(' LNGFIL: TBATH = ',F12.6,'  DELTA =',F12.6,/, &
          ' LNGFIL: Langevin dynamics setup for ',I7,' blocks',/)
  ENDIF
300 CONTINUE
  !
  RETURN
END SUBROUTINE LNGFIL2THETA


SUBROUTINE DLNGV2THETA(GAMMALDTHETA,FRAND)
  !
  ! This function routine generates 
  ! random deviate of 0.0 mean and standard deviation RF
  ! for theta in theta-dynamics.
  !
  !
  !c   Generated based on SUBROUTINE DLNGV2()
  !c   Lianqing Zheng and Wei Yang July, 2007
  use chm_kinds
  use clcg_mod,only:random
  use dimens_fcm
  use number
  use reawri
  use rndnum
  use consta
  use parallel
  implicit none
  !
  real(chm_real)  GAMMALDTHETA(*), FRAND
  !
  !
  real(chm_real)  A,B,PIS
  INTEGER I,IG
  !
  PIS=PI
  if (qoldrng) then                        !yw 05-Aug-2008
     A=GAMMALDTHETA(1)*SQRT(MINTWO*LOG(RANDOM(ISEED)))
     B=TWO*PIS*RANDOM(ISEED)
  else
     IG=1
#if KEY_PARALLEL==1
     IG=MYNODP          
#endif
     A=GAMMALDTHETA(1)*SQRT(MINTWO*LOG(RANDOM(IG)))
     B=TWO*PIS*RANDOM(IG)
  endif
  FRAND=A*COS(B)
  !
  RETURN
END SUBROUTINE DLNGV2THETA

