!      Set up the K-space tables for the eWald summation for SCC
!      to avoid repetitive work that is independent of the 
!      atom-pair vectors
!       
       subroutine ktabscc(kxvec,kyvec,kzvec,kvec,recbasis,alpha,vol,
     $                    nkvec,kmxscc,ksqmxscc,maxkscc,kxv,kyv,kzv)
#if KEY_PARALLEL==1
       use parallel
#endif 
       implicit real*8(A-H,O-Z) 
       parameter(ONE=1.d0,TWO=2.d0,FOUR=4.d0)
       PARAMETER(PI=3.141592653589793D0,TWOPI=2.0D0*PI)
       integer nkvec,kmxscc,ksqmxscc
       real*8  kxvec(*)
       real*8  kyvec(*)
       real*8  kzvec(*)
       real*8  kvec(*)
       real*8  recbasis(3,3),alpha,vol 
       integer kxv(*),kyv(*),kzv(*)

#if KEY_PARALLEL==1
!      Add common block for saving the distribution of KVECTORS
!      to different nodes
       integer NTRY,I
#endif 

       real*8 B,VFACT,RKSQ
       integer IPT
       integer KX,KY,KZ, KSY,KSZ,KSQ
       real*8  G(3)

!      Set up the K-table for SCC ewald, taking symmetry into account
!      Pretty much in parallel with the SETUPK2 code of CHARMM
!      Except for differences in the constants

      B = ONE/FOUR/alpha/alpha
!     VFACT = TWO*TWOPI/vol
!     ANOTHER FACTOR OF TWO BECAUSE OF SYMMETRY USED in KX,KY,KZ
      VFACT = TWO*TWO*TWOPI/vol
      IPT = 0 
      DO KX = 0, kmxscc
         IF(KX.EQ.0) THEN
            KSY = 0 
         ELSE    
            KSY = -kmxscc
         ENDIF   
         DO KY = KSY, kmxscc
            IF(KX.EQ.0.AND.KY.EQ.0) THEN
               KSZ = 1 
            ELSE    
               KSZ = -kmxscc
            ENDIF   
            DO KZ = KSZ, kmxscc
               KSQ = KX*KX + KY*KY + KZ*KZ 
               IF (KSQ.LE.ksqmxscc.AND. KSQ.NE.0) THEN
                  IPT = IPT + 1 
                  G(1)=KX*recbasis(1,1)+KY*recbasis(2,1)
     $                +KZ*recbasis(3,1)
                  G(2)=KX*recbasis(1,2)+KY*recbasis(2,2)
     $                +KZ*recbasis(3,2)
                  G(3)=KX*recbasis(1,3)+KY*recbasis(2,3)
     $                +KZ*recbasis(3,3)
                  RKSQ = G(1)*G(1)+G(2)*G(2)+G(3)*G(3)
                  KVEC(IPT) = VFACT*EXP((-B*RKSQ))/RKSQ
                  KXVEC(IPT)  = G(1)
                  KYVEC(IPT)  = G(2)
                  KZVEC(IPT)  = G(3)
                  KXV(IPT)  = KX
                  KYV(IPT)  = KY
                  KZV(IPT)  = KZ
               ENDIF   
            ENDDO   
         ENDDO   
      ENDDO   

!     Total number of k-vectors
      nkvec=IPT
      write(*,*) "Total number of vectors: ",nkvec
      if (nkvec.gt.maxkscc) then
        call WRNDIE(-5,'<KTABSCC>','TOO MANY K-VECTORS')
      endif

#if KEY_PARALLEL==1
!     For parallel computations - distribute the kvectors to different
!     NODES
      IKVCSC(0)=0
      DO I=1,NUMNOD
         NTRY=(NKVEC*I)/NUMNOD
         IKVCSC(I)=NTRY
         IF (MYNOD.EQ.0) WRITE(*,*) "NODE ",I," KVEC ",IKVCSC(I)
      ENDDO   
#endif 

      return
      end

!     ------------------------------------------------------------
      subroutine ksum0scc(r,kxvec,kyvec,kzvec,kvec,
     $                    nkvec,reciprocal)
#if KEY_PARALLEL==1
!      Add common block for saving the distribution of KVECTORS
!      to different nodes
       use parallel
#endif 
       implicit real*8(A-H,O-Z) 
       parameter(ZERO=0.d0,ONE=1.d0,TWO=2.d0,FOUR=4.d0)
       PARAMETER(PI=3.141592653589793D0,TWOPI=2.0D0*PI)
       integer nkvec
       real*8  r(3)
       real*8  kxvec(*)
       real*8  kyvec(*)
       real*8  kzvec(*)
       real*8  kvec(*)
       real*8  reciprocal

!           
      integer IPT,ISTART,IFINISH
      real*8  tmp

!     Performs summation in kspace for SCC ewald 
      reciprocal=ZERO
#if KEY_PARALLEL==1
      ISTART=1+IKVCSC(MYNOD)
      IFINISH= IKVCSC(MYNODP)
#else /**/
      ISTART=1
      IFINISH=NKVEC
#endif 
!     DO IPT=1,NKVEC
      DO IPT=ISTART,IFINISH
        tmp = cos(KXVEC(IPT)*r(1)+KYVEC(IPT)*r(2)+KZVEC(IPT)*r(3))
        reciprocal = reciprocal + KVEC(IPT)*tmp
      ENDDO 
#if KEY_PARALLEL==1
      CALL GCOMB(reciprocal,1)
#endif 

      return
      end

      subroutine ksumscc(r,kxvec,kyvec,kzvec,kvec,nkvec,
     $                   qsccewc,qsccews,reciprocal)
#if KEY_PARALLEL==1
!      Add common block for saving the distribution of KVECTORS
!      to different nodes
       use parallel
#endif 
       implicit real*8(A-H,O-Z) 
       parameter(ZERO=0.d0,ONE=1.d0,TWO=2.d0,FOUR=4.d0)
       PARAMETER(PI=3.141592653589793D0,TWOPI=2.0D0*PI)
       integer nkvec
       real*8  r(3)
       real*8  kxvec(*)
       real*8  kyvec(*)
       real*8  kzvec(*)
       real*8  kvec(*)
       real*8  qsccewc(*),qsccews(*)
       real*8  reciprocal

!           
      integer IPT,ISTART,IFINISH
      real*8  tmp
      real*8  tmpc
      real*8  tmps

!     Performs summation in kspace for SCC ewald 
!     NOTE A FACTOR OF TWO USING THIS EXPRESSION (QC: NO!)
      reciprocal=ZERO
#if KEY_PARALLEL==1
      ISTART=1+IKVCSC(MYNOD)
      IFINISH= IKVCSC(MYNODP)
#else /**/
      ISTART=1
      IFINISH=NKVEC
#endif 
!     write(*,*) "ksumscc>",nkvec,kxvec(1),kyvec(1),kzvec(1)
!     write(*,*) "ksumscc2>",qsccewc(1),qsccews(2)
!     DO IPT=1,NKVEC
      DO IPT=ISTART,IFINISH
        tmp  =     KXVEC(IPT)*r(1)+KYVEC(IPT)*r(2)+KZVEC(IPT)*r(3)
        tmpc = cos(tmp)
        tmps = sin(tmp)
        reciprocal = reciprocal + KVEC(IPT)*(tmpc*qsccewc(IPT) +
     $                                       tmps*qsccews(IPT))
      ENDDO 
!     reciprocal = reciprocal/TWO
#if KEY_PARALLEL==1
      CALL GCOMB(reciprocal,1)
#endif 

      return
      end

      subroutine kfsum0scc(r,kxvec,kyvec,kzvec,kvec,
     $                    nkvec,reciprocal)
#if KEY_PARALLEL==1
!      Add common block for saving the distribution of KVECTORS
!      to different nodes
       use parallel
#endif 
       implicit real*8(A-H,O-Z) 
       parameter(ZERO=0.d0,ONE=1.d0,TWO=2.d0,FOUR=4.d0)
       PARAMETER(PI=3.141592653589793D0,TWOPI=2.0D0*PI)
       integer nkvec
       real*8  r(3)
       real*8  kxvec(*)
       real*8  kyvec(*)
       real*8  kzvec(*)
       real*8  kvec(*)
       real*8  reciprocal(3)

!           
      integer IPT,ISTART,IFINISH
      real*8  tmp

!     DO IPT=1,3
!       reciprocal(IPT)=ZERO
!     ENDDO
!     Performs summation in kspace for SCC ewald gradient
#if KEY_PARALLEL==1
      ISTART=1+IKVCSC(MYNOD)
      IFINISH= IKVCSC(MYNODP)
#else /**/
      ISTART=1
      IFINISH=NKVEC
#endif 
!     DO IPT=1,NKVEC
      DO IPT=ISTART,IFINISH
        tmp =-sin(KXVEC(IPT)*r(1)+KYVEC(IPT)*r(2)+KZVEC(IPT)*r(3))
        reciprocal(1) = reciprocal(1) + KVEC(IPT)*tmp*KXVEC(IPT)
        reciprocal(2) = reciprocal(2) + KVEC(IPT)*tmp*KYVEC(IPT)
        reciprocal(3) = reciprocal(3) + KVEC(IPT)*tmp*KZVEC(IPT)
      ENDDO 

#if KEY_PARALLEL==1
      CALL GCOMB(reciprocal,3)
#endif 

      return
      end

      
      subroutine kfsumscc(r,kxvec,kyvec,kzvec,kvec,
     $                    nkvec,qsccewc,qsccews,reciprocal,
     $                    iflag)
#if KEY_PARALLEL==1
!      Add common block for saving the distribution of KVECTORS
!      to different nodes
       use parallel
#endif 
       implicit real*8(A-H,O-Z) 
       parameter(ZERO=0.d0,ONE=1.d0,TWO=2.d0,FOUR=4.d0)
       PARAMETER(PI=3.141592653589793D0,TWOPI=2.0D0*PI)
       integer nkvec
       real*8  r(3)
       real*8  kxvec(*)
       real*8  kyvec(*)
       real*8  kzvec(*)
       real*8  kvec(*)
       real*8  qsccewc(*),qsccews(*)
       real*8  reciprocal(3)
       integer iflag
!      QC: We only split the kvector sum to slave nodes if iflag 
!      .eq. 1

!           
      integer IPT,ISTART,IFINISH
      real*8  tmp,tmpc,tmps,tmp2

!     DO IPT=1,3
!       reciprocal(IPT)=ZERO
!     ENDDO
!     Performs summation in kspace for SCC ewald gradient
#if KEY_PARALLEL==1
      IF (IFLAG.EQ.1) THEN
        ISTART=1+IKVCSC(MYNOD)
        IFINISH= IKVCSC(MYNODP)
      ELSE
        ISTART=1
        IFINISH=NKVEC
      ENDIF
#else /**/
      ISTART=1
      IFINISH=NKVEC
#endif 
!     DO IPT=1,NKVEC
      DO IPT=ISTART,IFINISH
        tmp = KXVEC(IPT)*r(1)+KYVEC(IPT)*r(2)+KZVEC(IPT)*r(3)
        tmpc= cos(tmp)
        tmps= sin(tmp)
        tmp2=tmpc*qsccews(IPT) - tmps*qsccewc(IPT)
        reciprocal(1) = reciprocal(1) + KVEC(IPT)*KXVEC(IPT)*tmp2
        reciprocal(2) = reciprocal(2) + KVEC(IPT)*KYVEC(IPT)*tmp2
        reciprocal(3) = reciprocal(3) + KVEC(IPT)*KZVEC(IPT)*tmp2
      ENDDO 

#if KEY_PARALLEL==1
      IF (IFLAG.EQ.1) CALL GCOMB(reciprocal,3)
#endif 
      return
      end

      

