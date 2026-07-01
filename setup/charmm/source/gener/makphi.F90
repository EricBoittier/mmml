SUBROUTINE MAKPHI
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use psf
  use param
  use stream
  use vangle_mm
  use code
  use memory
  implicit none
  ! local
  INTEGER IPHI,IC
  LOGICAL GO,ERR
  !
  !  Generate arrays for fast vector ephi routines.
  !  Author: Stephen Fleischman
  !
  !  Trigonometric tables introduced to support
  !   the new dihedral energy routines.
  !
  !   by: Arnaud Blondel
  !
  !     determine memory requirement
  NPHIV=0
  DO IPHI = 1,NPHI
     IC=ICP(IPHI)
     IF (IC.NE.0) THEN
        NPHIV = NPHIV+1
        GO =.TRUE.
        DO WHILE(GO)
           IF (CPD(IC).LT.0) THEN
              IC=IC+1
              NPHIV = NPHIV+1
           ELSE
              GO = .FALSE.
           ENDIF
        ENDDO
     ENDIF
  ENDDO
  !     allocate memory for the vector phi arrays
  IF(ALLOCATED(IPV)) THEN
     CALL CHMREALLOC('makphi.src','MAKPHI','IPV',newsz=NPHIV,intg=IPV)
  ELSE
     CALL CHMALLOC('makphi.src','MAKPHI','IPV',size=NPHIV,intg=IPV)
  ENDIF
  IF(ALLOCATED(JPV)) THEN
     CALL CHMREALLOC('makphi.src','MAKPHI','JPV',newsz=NPHIV,intg=JPV)
  ELSE
     CALL CHMALLOC('makphi.src','MAKPHI','JPV',size=NPHIV,intg=JPV)
  ENDIF
  IF(ALLOCATED(KPV)) THEN
     CALL CHMREALLOC('makphi.src','MAKPHI','KPV',newsz=NPHIV,intg=KPV)
  ELSE
     CALL CHMALLOC('makphi.src','MAKPHI','KPV',size=NPHIV,intg=KPV)
  ENDIF
  IF(ALLOCATED(LPV)) THEN
     CALL CHMREALLOC('makphi.src','MAKPHI','LPV',newsz=NPHIV,intg=LPV)
  ELSE
     CALL CHMALLOC('makphi.src','MAKPHI','LPV',size=NPHIV,intg=LPV)
  ENDIF
  IF(ALLOCATED(VIND)) THEN
     CALL CHMREALLOC('makphi.src','MAKPHI','VIND',newsz=NPHIV,intg=VIND)
  ELSE
     CALL CHMALLOC('makphi.src','MAKPHI','VIND',size=NPHIV,intg=VIND)
  ENDIF
  IF (associated(VCPC)) THEN
     call chmrealloc('makphi.src','MAKPHI','VCPC',NPHIV,crlp=VCPC)
  ELSE
     call chmalloc('makphi.src','MAKPHI','VCPC',NPHIV,crlp=VCPC)
  ENDIF
  IF (associated(VCPD)) THEN
     call chmrealloc('makphi.src','MAKPHI','VCPD',NPHIV,intgp=VCPD)
  ELSE
     call chmalloc('makphi.src','MAKPHI','VCPD',NPHIV,intgp=VCPD)
  ENDIF
  IF (associated(VCPB)) THEN
     CALL chmrealloc('makphi.src','MAKPHI','VCPB',NPHIV,crlp=VCPB)
  ELSE
     CALL chmalloc('makphi.src','MAKPHI','VCPB',NPHIV,crlp=VCPB)
  ENDIF
  ! Trigonometric tables. AB.
  IF (associated(VCPCOS)) THEN
     CALL chmrealloc('makphi.src','MAKPHI','VCPCOS',NPHIV,crlp=VCPCOS)
  ELSE
     CALL chmalloc('makphi.src','MAKPHI','VCPCOS',NPHIV,crlp=VCPCOS)
  ENDIF
  IF (associated(VCPSIN)) THEN
     CALL chmrealloc('makphi.src','MAKPHI','VCPSIN',NPHIV,crlp=VCPSIN)
  ELSE
     CALL chmalloc('makphi.src','MAKPHI','VCPSIN',NPHIV,crlp=VCPSIN)
  ENDIF
  ! AB.
  !
  CALL MAKPHI2(NPHIV,IPV,JPV,KPV,LPV,VIND, &
       VCPC,VCPD,VCPB, &
       VCPCOS,VCPSIN,ERR)
  RETURN
END SUBROUTINE MAKPHI
!
SUBROUTINE MAKPHI2(NPHIV,IPV,JPV,KPV,LPV,VIND,VCPC,VCPD,VCPB, &
     VCPCOS,VCPSIN,ERR)
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use psf
  use param
  use stream
  use code
  implicit none
  !
  INTEGER IPV(*),JPV(*),KPV(*),LPV(*),VIND(*),VCPD(*),NPHIV
  real(chm_real) VCPC(*),VCPB(*),VCPCOS(*),VCPSIN(*)
  LOGICAL ERR
  !
  !     indices
  INTEGER IC,IPHI
  !     difference vectors
  LOGICAL GO
  !
  ERR = .FALSE.
  IF (NPHI.EQ.0) RETURN
  !
  NPHIV=0
  DO IPHI = 1,NPHI
     IC=ICP(IPHI)
     IF (IC.NE.0) THEN
        NPHIV=NPHIV+1
        ! Limitation to 6 kept for the parallel routines. AB.
        IF (ABS(CPD(IC)).GT.6 .OR. CPD(IC).EQ.0) THEN
           ! AB.
           CALL WRNDIE(-5,'<MAKPHI>', &
                'Bad periodicity in list for dihedral angles.')
           ERR = .TRUE.
           RETURN
        ENDIF
        GO =.TRUE.
        DO WHILE(GO)
           IPV(NPHIV) = IP(IPHI)
           JPV(NPHIV) = JP(IPHI)
           KPV(NPHIV) = KP(IPHI)
           LPV(NPHIV) = LP(IPHI)
           VIND(NPHIV)=IPHI
           VCPC(NPHIV)=CPC(IC)
           VCPD(NPHIV)=CPD(IC)
           VCPB(NPHIV)=CPB(IC)
           VCPCOS(NPHIV)=COS(VCPB(NPHIV))
           VCPSIN(NPHIV)=SIN(VCPB(NPHIV))
           IF (VCPD(NPHIV).LT.0) THEN
              VCPD(NPHIV)=-VCPD(NPHIV)
              IC=IC+1
              NPHIV=NPHIV+1
           ELSE
              GO = .FALSE.
           ENDIF
        ENDDO
     ENDIF
  ENDDO
  !
  IF(PRNLEV.GT.7) WRITE(OUTU,845) NPHI,NPHIV
845 FORMAT(' MAKPHI:: From',I6,' dihedrals, a total of',I6, &
       ' fast primitive dihedral elements were produced.')
  !
  RETURN
END SUBROUTINE MAKPHI2

SUBROUTINE AUTOGEN(COMLYN,COMLEN)
  !
  ! This routine embodied the AUTOgenerate command which
  ! conditionally deletes all angles and/or dihedrals
  ! and regenerates them for the entire PSF.
  ! This command is intended for the conclusion of a series
  ! of patch commands.
  !                       Bernard R. Brooks, NIH, 18-JUL-1995
  !
  use chm_kinds
  use stream
  use string
  use dimens_fcm
  use psf
  use coord
  use rtf,only:rtfautop
  use memory
  use select,only:selcta
  implicit none

  integer,allocatable,dimension(:,:) :: IATBON
  integer,allocatable,dimension(:) :: NATBON
  integer,allocatable,dimension(:) :: ISLCT
  integer maxwork

  CHARACTER(len=*) COMLYN
  INTEGER COMLEN, I
  LOGICAL LANGLE,LPHI,QDID
  !
  !
  call chmalloc('makphi.src','AUTOGEN','ISLCT',NATOM,intg=ISLCT)
  CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)

  IF(INDXA(COMLYN, COMLEN, 'PATC') .GT. 0) THEN
     RTFAUTOP=.TRUE.
     IF (PRNLEV.GE.2) WRITE(OUTU,*) ' AUTOGEN: turned on for patches.'
  ENDIF
  IF(INDXA(COMLYN, COMLEN, 'NOPA') .GT. 0) THEN
     RTFAUTOP=.FALSE.
     IF (PRNLEV.GE.2) WRITE(OUTU,*) ' AUTOGEN: turned off for patches.'
  ENDIF

  IF(INDXA(COMLYN, COMLEN, 'OFF') .GT. 0) THEN
     DO I=1,NATOM
       ICAAUTO(I)=ior(ICAAUTO(I),96)
     ENDDO
     IF (PRNLEV.GE.2) WRITE(OUTU,*) ' AUTOGEN: turned off for selected atoms.'

  ELSE IF(INDXA(COMLYN, COMLEN, 'ON') .GT. 0) THEN
     DO I=1,NATOM
       ICAAUTO(I)=iand(ICAAUTO(I),31)
     ENDDO
     IF (PRNLEV.GE.2) WRITE(OUTU,*) ' AUTOGEN: turned on for selected atoms.'

  ELSE
    LANGLE=(INDXA(COMLYN, COMLEN, 'ANGL') .GT. 0)
    LPHI  =(INDXA(COMLYN, COMLEN, 'DIHE') .GT. 0)
    IF(LANGLE .OR. LPHI) THEN

     call chmalloc('makphi.src','AUTOGEN','NATBON',NATOM,intg=NATBON)
     call chmalloc('makphi.src','AUTOGEN','IATBON',IATBMX,NATOM,intg=IATBON)
     MAXWORK=MAX(MAXT,MAXP)

     CALL AUTGEN(1,NATBON,IATBON,ISLCT,.true.,0,LANGLE,LPHI,QDID)

     call chmdealloc('makphi.src','AUTOGEN','NATBON',NATOM,intg=NATBON)
     call chmdealloc('makphi.src','AUTOGEN','IATBON',IATBMX,NATOM,intg=IATBON)
     CALL PSFSUM(OUTU)

    ENDIF
  ENDIF

  call chmdealloc('makphi.src','AUTOGEN','ISLCT',NATOM,intg=ISLCT)

  !
  RETURN
END SUBROUTINE AUTOGEN

      SUBROUTINE AUTGEN(NBONDL,NATBON,IATBON,ISLCT,QCLEAN,GENPAT,LANGLE,LPHI,QDID)
!
!     This routine automatically generates the angle and dihedral
!     lists for a set of bonds. This may be done in lieu of specifying
!     angles and dihedrals explicitly in the topology file. This
!     procedure only adds terms, there is no check for duplicate terms
!     from the existing angle or dihedral list. The multiple dihedral
!     potential may be obtained by specifying a dihedral once in the
!     topology file and the second will be generated here.
!
!     For angles, IT(i)<KT(i) For Dihedrals, IP(i)<LP(i)
!
!RCZ - 90/03/22 Ryszard Czerminski
!     subroutine modified to avoid generation of dihedrals
!     between linear bonds
!RCZ
!
!     By Bernard R. Brooks  15-FEB-1984
!
use chm_kinds
use exfunc
use stream
use dimens_fcm
use psf
use param, only: atc
use code
use param
#ifdef KEY_RESIZE
  use resize
#endif  

      implicit none
!
      INTEGER NBONDL
      INTEGER NATBON(*),IATBON(IATBMX,*),ISLCT(*),GENPAT
      LOGICAL QCLEAN,LANGLE,LPHI,QDID
!
      INTEGER I,IBT,JBT,J,IJ,IJA,K,IK,IKA,IPT,JPT
      LOGICAL SKIP
      logical removed,found,qnrtft,keep,qcodes,qdoang,qdodih
#ifdef KEY_RESIZE
      integer na0
#endif
      EXTERNAL  EXCH5
!
!  first let's see if there's anything to actually do here.
       qcodes=.false.
       qdoang=.false.
       qdodih=.false.
       do j=1,natom
          if(islct(j).gt.0) then
             if(iand(ICAAUTO(j),16).eq.0) qcodes=.true. 
             if(iand(ICAAUTO(j),32).eq.0) qdoang=.true. 
             if(iand(ICAAUTO(j),64).eq.0) qdodih=.true. 
          endif
       enddo
       if(.not.qdoang) LANGLE=.false.
       if(.not.qdodih) LPHI=.false.
       if(.not.(LANGLE.or.LPHI)) then
          if(GENPAT.EQ.1) then
            IF (PRNLEV.GE.2) WRITE(OUTU,43) 'segment.'
43          FORMAT(' AUTGEN: There are no autogenerated angles or dihedrals for this ',A)
          elseif(GENPAT.EQ.2) then
            IF (PRNLEV.GE.2) WRITE(OUTU,43) 'patch.'
          else
            CALL WRNDIE(4,'<AUTGEN>','No selected atom has an autogen flag set. Nothing generated.')
          endif
          QDID=.FALSE.
          RETURN
       endif
       QDID=.true.

       if(LANGLE.and.LPHI) then
         IF (PRNLEV.GE.2) WRITE(OUTU,44) 'angles and dihedrals.'
44       FORMAT(' AUTGEN: Autogenerating specified ',A)
       elseif(LANGLE) THEN
         IF (PRNLEV.GE.2) WRITE(OUTU,44) 'angles.'
       else
         IF (PRNLEV.GE.2) WRITE(OUTU,44) 'dihedrals.'
       endif

!qqqqq  
!  write(6,*) ' 1: NTHETA,NPHI ',NTHETA,NPHI

      IF(QCLEAN) THEN 
         ! remove all existing angles and dihedrals for selected atoms

        IF(LANGLE) THEN
          j=0
          do i=1,ntheta
            ! keep any angle involving any unselected atoms
            keep=.false.
            if(ISLCT(IT(i)).eq.0) then
               keep=.true.
            elseif(ISLCT(JT(i)).eq.0) then
               keep=.true.
            elseif(ISLCT(KT(i)).eq.0) then
               keep=.true.
            ! keep any angle involving any atoms that do not autogenerate
            elseif(iand(ICAAUTO(IT(i)),32).gt.0) then
               keep=.true.  ! keep if atom is not autogenerating
            elseif(iand(ICAAUTO(JT(i)),32).gt.0) then
               keep=.true.  
            elseif(iand(ICAAUTO(KT(i)),32).gt.0) then
               keep=.true.  
            endif
            if(keep) then
               j=j+1
               IT(j)=IT(i)
               JT(j)=JT(i)
               KT(j)=KT(i)
            endif
          enddo
          IF(ntheta.gt.j) THEN
             IF (PRNLEV.GE.2) WRITE(OUTU,45) ntheta-j,'angles'
          ENDIF
45 FORMAT(' AUTOGEN:',I6,' ',A,' are removed before regeneration for selected atoms.')
          ntheta=j
        ENDIF

        IF(LPHI) THEN
          j=0
          do i=1,nphi
            ! keep any dihedral involving any unselected atoms
            keep=.false.
            if(ISLCT(IP(i)).eq.0) then
               keep=.true.
            elseif(ISLCT(JP(i)).eq.0) then
               keep=.true.
            elseif(ISLCT(KP(i)).eq.0) then
               keep=.true.
            elseif(ISLCT(LP(i)).eq.0) then
               keep=.true.
            elseif(iand(ICAAUTO(IP(i)),64).gt.0) then
               keep=.true.  ! keep if an atom is not autogenerating
            elseif(iand(ICAAUTO(JP(i)),64).gt.0) then
               keep=.true.
            elseif(iand(ICAAUTO(KP(i)),64).gt.0) then
               keep=.true.
            elseif(iand(ICAAUTO(LP(i)),64).gt.0) then
               keep=.true.
            endif
            if(keep) then
               j=j+1
               IP(j)=IP(i)
               JP(j)=JP(i)
               KP(j)=KP(i)
               LP(j)=LP(i)
            endif
          enddo
          IF(nphi.gt.j) THEN
             IF (PRNLEV.GE.2) WRITE(OUTU,45) nphi-j,'dihedrals'
          ENDIF
          nphi=j
        ENDIF
      ENDIF

!qqqqq     
!  write(6,*) ' 2: NTHETA,NPHI ',NTHETA,NPHI

!
! Construct the cross reference bond list
!
      DO I=1,NATOM
         NATBON(I)=0
      ENDDO
      DO I=NBONDL,NBOND
         IBT=IB(I)
         JBT=JB(I)
         IF(IBT.GT.0 .AND. JBT.GT.0) THEN
            NATBON(IBT)=NATBON(IBT)+1
            IF(NATBON(IBT).GT.IATBMX) THEN
               IF(WRNLEV.GE.2)then
               WRITE(OUTU,335) NATBON(IBT),IBT
 335           FORMAT(' <AUTGEN>: ',I5, &
                      ' is too  many bonds for atom',I5, &
                      ' Check code')
               CALL DIEWRN(-4)
               ENDIF
            ENDIF
            IATBON(NATBON(IBT),IBT)=I
            NATBON(JBT)=NATBON(JBT)+1
            IF(NATBON(JBT).GT.IATBMX) THEN
               IF(WRNLEV.GE.2) WRITE(OUTU,335) JBT
               CALL DIEWRN(-4)
            ENDIF
            IATBON(NATBON(JBT),JBT)=-I
         ENDIF
      ENDDO
!
! Now make the unsorted list of 1-3 interactions by taking bonds
! and extending in a direction one bond.
!
#ifdef KEY_RESIZE
      na0=NTHETA
#endif  
    IF(LANGLE) THEN
        DO I=1,NBOND
          IBT=IB(I)
          JBT=JB(I)
          if(ISLCT(IBT).eq.1) then
           if(ISLCT(JBT).eq.1) then
            if(IAND(ICBAUTO(I),2).eq.0) then ! check bond autogen flag
             if(IAND(ICAAUTO(JBT),34).eq.0) then ! check first atom autogen flag
              if(IAND(ICAAUTO(IBT),33).eq.0) then ! check second atom autogen flag
               DO J=1,NATBON(IBT)
                 IJ=IATBON(J,IBT)
                 if(IAND(ICBAUTO(ABS(IJ)),2).eq.0) then ! check 2nd bond autogen flag
                   IF(IJ.GT.0) THEN
                      IJA=JB(IJ)
                   ELSE
                      IJA=IB(ABS(IJ))
                   ENDIF
                   if(ISLCT(IJA).eq.1) then
                    if(IAND(ICAAUTO(IJA),34).eq.0) then ! check third atom autogen flag
                     IF(IJA.GT.JBT) THEN
                       NTHETA=NTHETA+1
#ifdef KEY_RESIZE
                       if (ntheta .gt. na0) then
                          na0=ntheta+drsz0
                          call resize_psf('genpsf.F90','AUTGEN','NTHETA',na0, .true.)
                       endif
#endif
                       IT(NTHETA)=MIN(JBT,IJA)
                       JT(NTHETA)=IBT
                       KT(NTHETA)=MAX(JBT,IJA)
                     ENDIF
                    endif
                   endif
                 endif
               ENDDO
              endif
             endif

             if(IAND(ICAAUTO(IBT),34).eq.0) then ! check first atom autogen flag
              if(IAND(ICAAUTO(JBT),33).eq.0) then ! check second atom autogen flag
               DO J=1,NATBON(JBT)
                 IJ=IATBON(J,JBT)
                 if(IAND(ICBAUTO(ABS(IJ)),2).eq.0) then ! check 2nd bond autogen flag
                   IF(IJ.GT.0) THEN
                      IJA=JB(IJ)
                   ELSE
                      IJA=IB(ABS(IJ))
                   ENDIF
                   if(ISLCT(IJA).eq.1) then
                    if(IAND(ICAAUTO(IJA),34).eq.0)then ! check third atom autogen flag
                     IF(IJA.GT.IBT) THEN
                       NTHETA=NTHETA+1
#ifdef KEY_RESIZE
                       if (ntheta .gt. na0) then
                          na0=ntheta+drsz0
                          call resize_psf('genpsf.F90','AUTGEN','NTHETA',na0, .true.)
                       endif
#endif
                       IT(NTHETA)=MIN(IBT,IJA)
                       JT(NTHETA)=JBT
                       KT(NTHETA)=MAX(IBT,IJA)
                     ENDIF
                    endif
                   endif
                 endif
               ENDDO
              endif
             endif
            endif
           endif
          endif
        ENDDO
     ENDIF  !  (LANGLE)
!
!qqqqq  
!  write(6,*) ' 3: NTHETA,NPHI ',NTHETA,NPHI


     removed=.false.
!     Now search the extra/deleted list of angles
      DO I=1,NPAUTO
        if(LPAUTO(I).eq.0) then  ! this is an angle type
          if(ISLCT(JPAUTO(I)).eq.1) then  ! check to see if all atoms are selected
            if(ISLCT(IPAUTO(I)).eq.1) then
              if(ISLCT(KPAUTO(I)).eq.1) then
                found=.false.
                DO J=1,NTHETA     ! search new angles to see if angle is found
                  IF(JT(J).EQ.JPAUTO(I)) then ! central atoms match
                    IF(IT(J).EQ.IPAUTO(I)) then ! 1st atoms match
                      IF(KT(J).EQ.KPAUTO(I)) then ! 3rd atoms match!
                        found=.true.
                        IF(IAND(ICPAUTO(I),2).gt.0) then
                          IT(J)=0  ! remove unwanted angle
                          removed=.true.
                        ELSE
                          IF(LANGLE .and. WRNLEV.GE.2)then
                            write(outu,*) 'Requested angle for addition already exists:', &
                                          IPAUTO(I),JPAUTO(I),KPAUTO(I),' Nothing done.'
                          ENDIF
                        ENDIF
                      endif
                    endif
                    IF(KT(J).EQ.IPAUTO(I)) then ! 1st atoms match the other way
                      IF(IT(J).EQ.KPAUTO(I)) then ! 3rd atoms match!
                        found=.true.
                        IF(IAND(ICPAUTO(I),2).gt.0) then
                          IT(J)=0 ! remove unwanted angle
                          removed=.true.
                        ELSE
                          IF(LANGLE .and. WRNLEV.GE.2)then
                            write(outu,*) 'Requested angle for addition already exists:', &
                                          IPAUTO(I),JPAUTO(I),KPAUTO(I),' Nothing done.'
                          ENDIF
                        ENDIF
                      endif
                    endif
                  endif
                ENDDO
                if(.not.found) then
                   IF(IAND(ICPAUTO(I),2).eq.0) then ! add this angle
                       NTHETA=NTHETA+1
#ifdef KEY_RESIZE
                       if (ntheta .gt. na0) then
                          na0=ntheta+drsz0
                          call resize_psf('genpsf.F90','AUTGEN','NTHETA',na0, .true.)
                       endif
#endif
                       IT(NTHETA)=MIN(IPAUTO(I),KPAUTO(I))
                       JT(NTHETA)=JPAUTO(I)
                       KT(NTHETA)=MAX(IPAUTO(I),KPAUTO(I))
                   ELSE   
                                          ! angle to be deleted could not be found.
                      IF(LANGLE .and. WRNLEV.GE.2)then
                        write(outu,*) 'Requested angle for deletion not found:', &
                                      IPAUTO(I),JPAUTO(I),KPAUTO(I),' Nothing done.'
                      ENDIF
                   ENDIF
                endif
              endif
            endif
          endif
        endif
      ENDDO

!qqqqq  
!  write(6,*) ' 4: NTHETA,NPHI ',NTHETA,NPHI


      IF(removed) THEN  ! Remove any unwanted angles that were flagged above
        j=0
        do i=1,ntheta
          ! keep any angle involving unselected atoms
          if(IT(I).ne.0) then
             j=j+1
             IT(j)=IT(i)
             JT(j)=JT(i)
             KT(j)=KT(i) 
          endif
        enddo
        ntheta=j
      ENDIF
#ifdef KEY_RESIZE
  if (ntheta .lt. na0) &
       call resize_psf('genpsf.F90','AUTGEN','NTHETA',ntheta, .true.)
#endif

!qqqqq  
!  write(6,*) ' 5: NTHETA,NPHI ',NTHETA,NPHI

!
! Now make the unsorted list of 1-4 interactions by taking bonds
! and extending in each direction one bond.
!

    IF(LPHI) THEN

#ifdef KEY_RESIZE
      na0=NPHI
#endif    
      DO I=1,NBOND
        IBT=IB(I)
        JBT=JB(I)
        if(ISLCT(IBT).eq.1) then
         if(ISLCT(JBT).eq.1) then
          if(IAND(ICBAUTO(I),4).eq.0) then ! check bond autogen flag for central dihedral exclude
            if(IAND(ICAAUTO(JBT),68).eq.0) then ! check first atom autogen flag
              if(IAND(ICAAUTO(IBT),68).eq.0) then ! check second atom autogen flag
                qnrtft=(iand(ICAAUTO(IBT),16).eq.0 .and. iand(ICAAUTO(JBT),16).eq.0)
                DO J=1,NATBON(IBT)
                  IJ=IATBON(J,IBT)
                  if(IAND(ICBAUTO(ABS(IJ)),8).eq.0) then ! check 2nd bond autogen flag
                    IF(ABS(IJ).NE.I) THEN
                      IF(IJ.GT.0) THEN
                        IJA=JB(IJ)
                      ELSE
                        IJA=IB(ABS(IJ))
                      ENDIF
                      if(ISLCT(IJA).eq.1) then
                       if(IAND(ICAAUTO(IJA),72).eq.0) then ! check outer atom autogen flag
                        DO K=1,NATBON(JBT)
                          IK=IATBON(K,JBT)
                          if(IAND(ICBAUTO(ABS(IK)),8).eq.0) then ! check 3rd bond autogen flag
                            IF(ABS(IK).NE.I) THEN
                              IF(IK.GT.0) THEN
                                IKA=JB(IK)
                              ELSE
                                IKA=IB(ABS(IK))
                              ENDIF
                              if(ISLCT(IKA).eq.1) then
                               if(IAND(ICAAUTO(IKA),72).eq.0) then ! check outer atom autogen flag
                                IF(IKA.EQ.IBT .OR. IJA.EQ.JBT) THEN
                                  CALL WRNDIE(0,'<AUTGEN>', &
                                    'DOUBLE BOND SPECS FOR SOME ATOM PAIRS')
                                ELSE IF(IKA.EQ.IJA) THEN
                                  IF(AMASS(IKA).GT.0.002.AND.AMASS(IBT).GT.0.002.AND.  &
                                     AMASS(JBT).GT.0.002) THEN
                                  !va Skip warning if Lone-Pairs are involved in 3-member ring
                                     IF(WRNLEV.GE.2) write(outu,*) ' AUTGEN: Three member ring found:',IBT,JBT,IKA
                                  ENDIF
                                ELSE
                                  IF(IKA.LT.IJA) THEN
                                    IF(qnrtft) THEN
                                      CALL CHECKDH(IKA,JBT,IBT,IJA,SKIP)
                                    ELSE
                                      SKIP=.false.
                                    ENDIF
                                    IF(.NOT.SKIP) THEN
                                       NPHI=NPHI+1
#ifdef KEY_RESIZE
                                       if (NPHI .gt. na0) then
                                          na0=NPHI+drsz0
                                          call resize_psf('genpsf.F90','AUTGEN','NPHI',na0, .true.)
                                       endif
#endif                 
                                       IP(NPHI)=IKA
                                       JP(NPHI)=JBT
                                       KP(NPHI)=IBT
                                       LP(NPHI)=IJA
                                    ENDIF
                                  ELSE
                                    IF(qnrtft) THEN
                                      CALL CHECKDH(IJA,IBT,JBT,IKA,SKIP)
                                    ELSE
                                      SKIP=.false.
                                    ENDIF
                                    IF(.NOT.SKIP) THEN
                                       NPHI=NPHI+1
#ifdef KEY_RESIZE
                                       if (NPHI .gt. na0) then
                                          na0=NPHI+drsz0
                                          call resize_psf('genpsf.F90','AUTGEN','NPHI',na0, .true.)
                                       endif
#endif                 
                                       IP(NPHI)=IJA
                                       JP(NPHI)=IBT
                                       KP(NPHI)=JBT
                                       LP(NPHI)=IKA
                                    ENDIF
                                  ENDIF
                                ENDIF
                               endif
                              endif
                            ENDIF
                          endif
                        ENDDO
                       endif
                      endif
                    ENDIF
                  ENDIF
                ENDDO
              endif
            endif
          endif
         endif
        endif
      ENDDO
   ENDIF ! (LPHI)

!qqqqq  
!  write(6,*) ' 6: NTHETA,NPHI ',NTHETA,NPHI

     removed=.false.
!     Now search the extra/deleted list of dihedrals
      DO I=1,NPAUTO
        if(ISLCT(JPAUTO(I)).eq.1) then  ! check to see if all atoms are selected
          if(ISLCT(IPAUTO(I)).eq.1) then
            if(ISLCT(KPAUTO(I)).eq.1) then
                                                ! this is an angle type
              if(LPAUTO(I).eq.0) then
                if(IAND(ICPAUTO(I),12).gt.0) then  ! this is an angle type with request to prevent dihedrals

                  found=.false.
                  DO J=1,NPHI     ! search new angles to see if angle is found
                    IF(JP(J).EQ.JPAUTO(I)) then ! 2nd atoms match
                      IF(MIN(IP(J),KP(J)).EQ.MIN(IPAUTO(I),KPAUTO(I))) then ! 1st atoms match
                        IF(MAX(IP(J),KP(J)).EQ.MAX(IPAUTO(I),KPAUTO(I))) then ! 3rd atoms match
                          IP(J)=0  ! remove unwanted dihedral
                          removed=.true.
                        endif
                      endif
                    endif
                    IF(KP(J).EQ.JPAUTO(I)) then ! 2nd atoms match
                      IF(MIN(JP(J),LP(J)).EQ.MIN(IPAUTO(I),KPAUTO(I))) then ! 1st atoms match
                        IF(MAX(JP(J),LP(J)).EQ.MAX(IPAUTO(I),KPAUTO(I))) then ! 3rd atoms match
                          IP(J)=0  ! remove unwanted dihedral
                          removed=.true.
                        endif
                      endif
                    ENDIF
                  ENDDO
                endif
              else
                                                ! this is a dihedral type
                if(ISLCT(LPAUTO(I)).eq.1) then
                  found=.false.
                  DO J=1,NPHI     ! search new dihedral to see if dihedral is found
                    IF(JP(J).EQ.JPAUTO(I)) then ! 2nd atoms match
                      IF(KP(J).EQ.KPAUTO(I)) then ! 3rd atoms match
                        IF(IP(J).EQ.IPAUTO(I)) then ! 1st atoms match
                          IF(LP(J).EQ.LPAUTO(I)) then ! 4th atoms match
                            found=.true.
                            IF(IAND(ICPAUTO(I),12).gt.0) then
                              IP(J)=0 ! remove unwanted angle
                              removed=.true.
                            ELSE
                              IF(LPHI .and. WRNLEV.GE.2)then
                                write(outu,*) 'Requested dihedral for addition already exists:', &
                                              IPAUTO(I),JPAUTO(I),KPAUTO(I),LPAUTO(I),' Nothing done.'
                              ENDIF
                            ENDIF
                          ENDIF
                        ENDIF
                      ENDIF
                    ENDIF
                    IF(KP(J).EQ.JPAUTO(I)) then ! 2nd atoms match
                      IF(JP(J).EQ.KPAUTO(I)) then ! 3rd atoms match
                        IF(LP(J).EQ.IPAUTO(I)) then ! 1st atoms match
                          IF(IP(J).EQ.LPAUTO(I)) then ! 4th atoms match
                            found=.true.
                            IF(IAND(ICPAUTO(I),12).gt.0) then
                              IP(J)=0 ! remove unwanted angle
                              removed=.true.
                            ELSE
                              IF(LPHI .and. WRNLEV.GE.2)then
                                write(outu,*) 'Requested dihedral for addition already exists:', &
                                              IPAUTO(I),JPAUTO(I),KPAUTO(I),LPAUTO(I),' Nothing done.'
                              ENDIF
                            ENDIF
                          ENDIF
                        ENDIF
                      ENDIF
                    ENDIF
                  ENDDO
                  if(.not.found) then
                    IF(IAND(ICPAUTO(I),12).eq.0) then ! add this dihedral
                      NPHI=NPHI+1
#ifdef KEY_RESIZE
                      if (NPHI .gt. na0) then
                         na0=NPHI+drsz0
                         call resize_psf('genpsf.F90','AUTGEN','NPHI',na0, .true.)
                      endif
#endif                 
                      IF(IPAUTO(I).LT.LPAUTO(I)) THEN
                         IP(NPHI)=IPAUTO(I)
                         JP(NPHI)=JPAUTO(I)
                         KP(NPHI)=KPAUTO(I)
                         LP(NPHI)=LPAUTO(I)
                      ELSE
                         LP(NPHI)=IPAUTO(I)
                         KP(NPHI)=JPAUTO(I)
                         JP(NPHI)=KPAUTO(I)
                         IP(NPHI)=LPAUTO(I)
                      ENDIF
                    ELSE   
                                        ! angle to deleted could not be found.
                      IF(LPHI .and. WRNLEV.GE.2)then
                        write(outu,*) 'Requested dihedral to delete not found:', &
                                       IPAUTO(I),JPAUTO(I),KPAUTO(I),LPAUTO(I),' Nothing done.'
                      ENDIF
                    ENDIF
                  endif
                endif
              endif
            endif
          endif
        endif
      ENDDO

!qqqqq  
!  write(6,*) ' 7: NTHETA,NPHI ',NTHETA,NPHI

      IF(removed) THEN  ! Remove any unwanted dihedral that were flagged above
        j=0
        do i=1,nphi
          ! keep any dihedral involving unselected atoms
          if(IP(I).ne.0) then
             j=j+1
             IP(j)=IP(i)
             JP(j)=JP(i)
             KP(j)=KP(i) 
             LP(j)=LP(i) 
          endif
        enddo
        nphi=j
      ENDIF
#ifdef KEY_RESIZE
      if (NPHI .lt. na0) &
           call resize_psf('genpsf.F90','AUTGEN','NPHI',NPHI, .true.)
#endif

!qqqqq  
!  write(6,*) ' 8: NTHETA,NPHI ',NTHETA,NPHI

!
!  Now sort the new dihedral list
!
    IF(LPHI) CALL SORT(NPHI,EXCH5,ORDER5,IP,JP,KP,LP,0,0,0,4)

     ! Now remove any duplicate angles and dihedrals

       IF(LANGLE .and. NTHETA.GT.1) THEN
         CALL SORT(NTHETA,EXCH5,ORDER5,IT,JT,KT,0,0,0,0,3)

         IPT=1
         JPT=1
         DO I=2,NTHETA
           IF(IT(IPT).NE.IT(IPT+1)) THEN 
              JPT=JPT+1
           ELSEIF(JT(IPT).NE.JT(IPT+1)) THEN
              JPT=JPT+1
           ELSEIF(KT(IPT).NE.KT(IPT+1)) THEN
              JPT=JPT+1
           ENDIF
           IPT=IPT+1
           IF(IPT.GT.JPT) THEN
              IT(JPT)=IT(IPT)
              JT(JPT)=JT(IPT)
              KT(JPT)=KT(IPT)
           ENDIF
         ENDDO
!   we just deleted IPT-JPT dihedrals
         IF(IPT.GT.JPT .and. PRNLEV.GE.2) WRITE(OUTU,25) IPT-JPT,'angles'
25   FORMAT(' <AUTGEN>:',I5,' duplicate ',A,' deleted')
         NTHETA=JPT
#ifdef KEY_RESIZE
         call resize_psf('genpsf.F90','AUTGEN','NTHETA',NTHETA, .true.)
#endif
       ENDIF

       IF(LPHI .and. NPHI.GT.1) THEN
         IPT=1
         JPT=1
         DO I=2,NPHI
           IF(IP(IPT).NE.IP(IPT+1)) THEN 
              JPT=JPT+1
           ELSEIF(JP(IPT).NE.JP(IPT+1)) THEN
              JPT=JPT+1
           ELSEIF(KP(IPT).NE.KP(IPT+1)) THEN
              JPT=JPT+1
           ELSEIF(LP(IPT).NE.LP(IPT+1)) THEN
              JPT=JPT+1
           ENDIF
           IPT=IPT+1
           IF(IPT.GT.JPT) THEN
              IP(JPT)=IP(IPT)
              JP(JPT)=JP(IPT)
              KP(JPT)=KP(IPT)
              LP(JPT)=LP(IPT)
           ENDIF
         ENDDO
!   we just deleted IPT-JPT dihedrals
         IF(IPT.GT.JPT .and. PRNLEV.GE.2) WRITE(OUTU,25) IPT-JPT,'dihedrals'
         NPHI=JPT
#ifdef KEY_RESIZE
         call resize_psf('genpsf.F90','AUTGEN','NPHI',NPHI, .true.)
#endif
       ENDIF


      NTHETT=NTHETA
      NPHIT=NPHI

!qqqqq  
!  write(6,*) ' 9: NTHETA,NPHI ',NTHETA,NPHI

!
      RETURN

      END SUBROUTINE AUTGEN

SUBROUTINE CHECKDH(ID,JD,KD,LD,SKIP)
  !
  !RCZ 1990/03/22 checking if there are equilibrium valence angles
  !             equal 180 degrees between ID,JD,KD,LD.
  !BRB 1991/10/16 Modified for efficiency.
  !BRB 2021/07/04 and again
  !
  use consta, only:PI
  use dimens_fcm
  use param
  use psf, only:NATOMT,IMOVE,IAC
  use stream

  INTEGER ID,JD,KD,LD
  LOGICAL SKIP

  INTEGER N,I,J,II(2),JJ(2),KK(2),ICTX(2)
  real(chm_real), parameter :: EPS=0.01
  !
  SKIP=.TRUE.
  !
  ! Test first three atoms
      II(1)=MIN(ID,KD)
      JJ(1)=JD
      KK(1)=MAX(ID,KD)
  ! Test secomd three atoms
      II(2)=MIN(JD,LD)
      JJ(2)=KD
      KK(2)=MAX(JD,LD)
  !
      CALL CODES([0],ICTX,[0],[0], &
               NATOMT,IMOVE,IAC,0,[0],[0], &
               2,II,JJ,KK, &
               0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
               .FALSE.,0,                           & ! DRUDE
#if KEY_CMAP==1
               [0],0,[0],[0],[0],[0],[0],[0],[0],[0],                 & 
#endif
               .TRUE.,.TRUE.)

      DO I=1,2
         J=ICTX(I)
         IF(J.gt.0) THEN
            IF(ABS(CTB(J)-PI).LT.EPS .OR. ABS(CTB(J)).LT.EPS) THEN
               IF(PRNLEV.GE.2) THEN
                  WRITE(OUTU,80) ID,JD,KD,LD
80                FORMAT(' CHECKDH> dihedral with linear angles:',4I5,' will NOT be generated')
               ENDIF
               RETURN
            ENDIF
         ELSE
            IF(PRNLEV.GE.2) WRITE(OUTU,81) ID,JD,KD,LD
81          FORMAT(' CHECKDH> dihedral missing angle parameters:',4I5,' will NOT be generated')
            RETURN
         ENDIF
      ENDDO


      SKIP=.FALSE.
      RETURN
END SUBROUTINE CHECKDH
