!CHARMM Element source/gener/mkpres.src $Revision: 1.2 $
SUBROUTINE MKPRES0
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use exfunc
  use comand
  use stream
  use memory
  implicit none
  integer,allocatable,dimension(:) :: ISLCT1
  integer,allocatable,dimension(:) :: ISLCT2
  integer,allocatable,dimension(:) :: ISLCT3
  integer,allocatable,dimension(:) :: ISLCT4
  integer,allocatable,dimension(:) :: ILST1
  integer,allocatable,dimension(:) :: ILST2
  integer,allocatable,dimension(:) :: ILST3
  integer,allocatable,dimension(:) :: ILST4
  integer,allocatable,dimension(:) :: ILIST
  integer,allocatable,dimension(:) :: IAC2
  real(chm_real),allocatable,dimension(:) :: CG2

  if (prnlev >= 2) then
     write(outu,'(10X,A)') 'MAKE PATCH FOR ALCHEMICAL MUTATIONS'
     write(outu,*)
  endif

  ! Allocate Stack
  call chmalloc('mkpres.src','MKPRES0','ISLCT1',MAXA,intg=ISLCT1)
  call chmalloc('mkpres.src','MKPRES0','ISLCT2',MAXA,intg=ISLCT2)
  call chmalloc('mkpres.src','MKPRES0','ISLCT3',MAXA,intg=ISLCT3)
  call chmalloc('mkpres.src','MKPRES0','ISLCT4',MAXA,intg=ISLCT4)

  call chmalloc('mkpres.src','MKPRES0','ILST1',MAXA,intg=ILST1)
  call chmalloc('mkpres.src','MKPRES0','ILST2',MAXA,intg=ILST2)
  call chmalloc('mkpres.src','MKPRES0','ILST3',MAXA,intg=ILST3)
  call chmalloc('mkpres.src','MKPRES0','ILST4',MAXA,intg=ILST4)
  call chmalloc('mkpres.src','MKPRES0','ILIST',MAXA,intg=ILIST)

  call chmalloc('mkpres.src','MKPRES0','IAC2',MAXA,intg=IAC2)
  call chmalloc('mkpres.src','MKPRES0','CG2',MAXA,crl=CG2)

  CALL MKPRES(ISLCT1,ISLCT2,ISLCT3, &
       ISLCT4,ILST1,ILST2,ILST3, &
       ILST4,ILIST,IAC2,CG2)
  !
  ! Free Stack
  call chmdealloc('mkpres.src','MKPRES0','ISLCT1',MAXA,intg=ISLCT1)
  call chmdealloc('mkpres.src','MKPRES0','ISLCT2',MAXA,intg=ISLCT2)
  call chmdealloc('mkpres.src','MKPRES0','ISLCT3',MAXA,intg=ISLCT3)
  call chmdealloc('mkpres.src','MKPRES0','ISLCT4',MAXA,intg=ISLCT4)

  call chmdealloc('mkpres.src','MKPRES0','ILST1',MAXA,intg=ILST1)
  call chmdealloc('mkpres.src','MKPRES0','ILST2',MAXA,intg=ILST2)
  call chmdealloc('mkpres.src','MKPRES0','ILST3',MAXA,intg=ILST3)
  call chmdealloc('mkpres.src','MKPRES0','ILST4',MAXA,intg=ILST4)
  call chmdealloc('mkpres.src','MKPRES0','ILIST',MAXA,intg=ILIST)

  call chmdealloc('mkpres.src','MKPRES0','IAC2',MAXA,intg=IAC2)
  call chmdealloc('mkpres.src','MKPRES0','CG2',MAXA,crl=CG2)
  RETURN
END SUBROUTINE MKPRES0

SUBROUTINE MKPRES(ISLCT1,ISLCT2,ISLCT3,ISLCT4, &
     ILST1,ILST2,ILST3,ILST4,ILIST,IAC2,CG2)
  !-----------------------------------------------------------------------
  ! This routine writes the PRES for a mutation. Author: Benoit Roux, 1996
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use dimens_fcm
  !     input/output
  !RCZ 10/24/91 to write XPLOR compatible PSF file
  use comand
  use param
  !RCZ
  use coord
  use psf
  use select
  use stream
  use string
  implicit none
  !
  INTEGER ISLCT1(*),ISLCT2(*),ISLCT3(*),ISLCT4(*)
  INTEGER ILST1(*),ILST2(*),ILST3(*),ILST4(*),ILIST(*)
  INTEGER NLST1,NLST2,NLST3,NLST4
  INTEGER IAC2(*)
  real(chm_real) CG2(*)
  !
  !
  !     local
  INTEGER IAT1, ICOUNT, U
  INTEGER I, IRES, ISEG
  !RCZ 91/04/26
  LOGICAL  QSET
  integer ILOOP,J,ISET
  real(chm_real) CGTOT2
  CHARACTER(len=8)  NAME1,NAME2,NAME3,NAME4
  CHARACTER(len=4) WRD
  !
  CALL SELCTA(COMLYN,COMLEN,ISLCT1,X,Y,Z,WMAIN,.FALSE.)
  CALL SELCTA(COMLYN,COMLEN,ISLCT2,X,Y,Z,WMAIN,.FALSE.)
  CALL SELCTA(COMLYN,COMLEN,ISLCT3,X,Y,Z,WMAIN,.FALSE.)
  CALL SELCTA(COMLYN,COMLEN,ISLCT4,X,Y,Z,WMAIN,.FALSE.)
  U = GTRMI(COMLYN,COMLEN,'UNIT',6)
  WRD = NEXTA4(COMLYN,COMLEN)
  write(U,'(A)') '  22     1 '
  write(U,'(A)') 

  DO ISET=1,2
     IF(ISET.EQ.1)THEN
        QSET=.true. 
        WRD(4:4)='0'
     ELSE
        QSET=.false. 
        WRD(4:4)='1'
     ENDIF

     NLST1=0
     NLST2=0
     NLST3=0
     NLST4=0
     DO I=1,NATOM
        ILIST(I)=0
        IF(ISLCT1(I).EQ.1)THEN
           NLST1=NLST1+1
           ILST1(NLST1)=I
        ENDIF
        IF(ISLCT2(I).EQ.1)THEN
           NLST2=NLST2+1
           ILST2(NLST2)=I
        ENDIF
        IF(ISLCT3(I).EQ.1)THEN
           NLST3=NLST3+1
           ILST3(NLST3)=I
        ENDIF
        IF(ISLCT4(I).EQ.1)THEN
           NLST4=NLST4+1
           ILST4(NLST4)=I
        ENDIF
        !     write(outu,'(8I3)') 
        !    ,ISLCT1(I),NLST1,ISLCT2(I),NLST2,ISLCT3(I),NLST3,ISLCT4(I),NLST4
        IAC2(I)=IAC(I)
        CG2(I)=CG(I)
     ENDDO

     if (prnlev >= 2) then
        write(outu,'(A,4I8)') 'the sets are:  ', NLST1,NLST2,NLST3,NLST4
        write(outu,*) 'atoms in the final psf : ',NLST1+NLST2+NLST4
     endif

     IF(NLST1.NE.NLST3)THEN
        CALL WRNDIE(-1,'<MKPRES>', &
             'THE MAIN SEGMENTS DO NOT HAVE THE SAME NUMBER OF ATOMS')
     ENDIF

     if (prnlev >= 2) then
        write(outu,*)
        write(outu,'(6x,a)') 'First set                 Second set'
        write(outu,*)
     endif
     DO I=1,NLST1
        IF ((ATYPE(ILST1(I)) /= ATYPE(ILST3(I))) .OR. &
             (CG(ILST1(I)) /= CG(ILST3(I))) .OR. &
             (IAC(ILST1(I)) /= IAC(ILST3(I))) ) THEN
           if (prnlev >= 2) write(outu,'(6x,2(3A,4X,F5.2,2X,A,2X),a)')  &
                ATYPE(ILST1(I))(1:idleng),' ', &
                ATC(IAC(ILST1(I)))(1:idleng), &
                CG(ILST1(I)),'.ne.', &
                ATYPE(ILST3(I))(1:idleng),' ', &
                ATC(IAC(ILST3(I)))(1:idleng), &
                CG(ILST3(I)), &
                ' THE ATOM TYPES IN THE TWO SEGMENTS DO NOT MATCH PERFECTLY'

           IF(QSET)THEN
              if (prnlev >= 2) write(outu,'(53x,2a)')  &
                   'atom type priority will be given to first set in ',WRD
           ELSE
              if (prnlev >= 2) write(outu,'(53x,2a)')  &
                   'atom type priority will be given to second set in ',WRD
              IAC2(ILST1(I))=IAC(ILST3(I))      
              CG2(ILST1(I))=CG(ILST3(I))      
           ENDIF

        ELSE
           if (prnlev >= 2) write(outu,'(6x,2(3A,4X,F5.2,8X))')  &
                ATYPE(ILST1(I))(1:idleng),' ', &
                ATC(IAC(ILST1(I)))(1:idleng),CG(ILST1(I)), &
                ATYPE(ILST3(I))(1:idleng),' ', &
                ATC(IAC(ILST3(I)))(1:idleng),CG(ILST3(I))

        ENDIF
     ENDDO

     DO I=1,NLST2
        ILIST(ILST2(I))=1
     ENDDO
     DO I=1,NLST4
        ILIST(ILST4(I))=1
     ENDDO

     IF(QSET)THEN
        DO I=1,NLST4
           CG2(ILST4(I))=0.0
        ENDDO
     ELSE
        DO I=1,NLST2
           CG2(ILST2(I))=0.0
        ENDDO
     ENDIF
     if (prnlev >= 2) write(outu,*)

     DO ILOOP=1,2

        do  i=1,nbond
           if(ISLCT4(IB(I))+ISLCT4(JB(I)).GE.1)THEN
              if(ISLCT3(IB(I))+ISLCT3(JB(I)).GE.1)THEN

                 if(ISLCT3(IB(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,IB(I))
                    NAME1= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(IB(I)).eq.1)then
                    NAME1='2'//atype(IB(I))
                    ILIST(IB(I))=1
                 endif

                 if(ISLCT3(JB(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,JB(I))
                    NAME2= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(JB(I)).eq.1)then
                    NAME2='2'//atype(JB(I))
                    ILIST(JB(I))=1
                 endif

                 IF(ILOOP.EQ.2)THEN
                    IF(QSET)THEN
                       write(U,'(10a)') 'BOND  ',NAME1,' ',NAME2
                    ELSE
                       write(U,'(10a)') '!BOND  ',NAME1,' ',NAME2
                    ENDIF
                 ENDIF

              endif
           endif
        enddo

        do  i=1,ntheta
           if(ISLCT4(IT(I))+ISLCT4(JT(I))+ISLCT4(KT(I)).GE.1)THEN
              if(ISLCT3(IT(I))+ISLCT3(JT(I))+ISLCT3(KT(I)).GE.1)THEN

                 if(ISLCT3(IT(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,IT(I))
                    NAME1= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(IT(I)).eq.1)then
                    NAME1='2'//atype(IT(I))
                    ILIST(IT(I))=1
                 endif

                 if(ISLCT3(JT(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,JT(I))
                    NAME2= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(JT(I)).eq.1)then
                    NAME2='2'//atype(JT(I))
                    ILIST(JT(I))=1
                 endif

                 if(ISLCT3(KT(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,KT(I))
                    NAME3= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(KT(I)).eq.1)then
                    NAME3='2'//atype(KT(I))
                    ILIST(KT(I))=1
                 endif

                 IF(ILOOP.EQ.2)THEN 
                    IF(QSET)THEN
                       write(U,'(10a)') 'THETA ',NAME1,' ',NAME2,' ',NAME3
                    ELSE
                       write(U,'(10a)') '!THETA ',NAME1,' ',NAME2,' ',NAME3
                    ENDIF
                 ENDIF

              endif
           endif
        enddo

        do  i=1,nphi
           IF( &
                ISLCT1(IP(I))+ISLCT1(JP(I))+ISLCT1(KP(I))+ISLCT1(LP(I)).GE.2)THEN
              IF( &
                   ISLCT2(IP(I))+ISLCT2(JP(I))+ISLCT2(KP(I))+ISLCT2(LP(I)).GE.1)THEN
                 if(.not.QSET)then
                    NAME1='1'//ATYPE(IP(I))
                    NAME2='1'//ATYPE(JP(I))
                    NAME3='1'//ATYPE(KP(I))
                    NAME4='1'//ATYPE(LP(I))
                    IF(ILOOP.EQ.2)then
                       IF ((ATYPE(IP(I)) == 'N   ') .and. &
                            (ATYPE(JP(I)) == 'CA  ') .and. &
                            (ATYPE(KP(I)) == 'CB  ')) THEN
                          write(U,'(10a)')  &
                               '!DELETE DIHE ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                       ELSEIF ((ATYPE(IP(I)) == 'CA  ') .and. &
                            (ATYPE(JP(I)) == 'CB  ')) THEN
                          write(U,'(10a)')  &
                               '!DELETE DIHE ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                       ELSE
                          write(U,'(10a)')  &
                               'DELETE DIHE ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                       ENDIF
                    ENDIF
                 endif
              endif
           endif
        enddo

        do  i=1,nphi
           IF( &
                ISLCT4(IP(I))+ISLCT4(JP(I))+ISLCT4(KP(I))+ISLCT4(LP(I)).GE.1)THEN
              IF( &
                   ISLCT3(IP(I))+ISLCT3(JP(I))+ISLCT3(KP(I))+ISLCT3(LP(I)).GE.1)THEN

                 if(ISLCT3(IP(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,IP(I))
                    NAME1= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(IP(I)).eq.1)then
                    NAME1='2'//atype(IP(I))
                    ILIST(IP(I))=1
                 endif

                 if(ISLCT3(JP(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,JP(I))
                    NAME2= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(JP(I)).eq.1)then
                    NAME2='2'//atype(JP(I))
                    ILIST(JP(I))=1
                 endif

                 if(ISLCT3(KP(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,KP(I))
                    NAME3= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(KP(I)).eq.1)then
                    NAME3='2'//atype(KP(I))
                    ILIST(KP(I))=1
                 endif

                 if(ISLCT3(LP(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,LP(I))
                    NAME4= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(LP(I)).eq.1)then
                    NAME4='2'//atype(LP(I))
                    ILIST(LP(I))=1
                 endif

                 IF(ILOOP.EQ.2)THEN
                    if(qset)then

                       IF( &
                            ISLCT3(IP(I))+ISLCT3(JP(I))+ISLCT3(KP(I))+ISLCT3(LP(I)).GE.2)THEN
                          IF ((ATYPE(IP(I)) == 'N   ') .and. &
                               (ATYPE(JP(I)) == 'CA  ') .and. &
                               (ATYPE(KP(I)) == 'CB  ')) THEN
                             write(U,'(10a)') 'DIHE  ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                          ELSEIF ((ATYPE(IP(I)) == 'CA  ') .and. &
                               (ATYPE(JP(I)) == 'CB  ')) THEN
                             write(U,'(10a)') 'DIHE  ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                          ELSE
                             write(U,'(10a)') '!DIHE  ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                          ENDIF
                       else
                          write(U,'(10a)') 'DIHE  ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                       endif

                    else

                       IF( &
                            ISLCT3(IP(I))+ISLCT3(JP(I))+ISLCT3(KP(I))+ISLCT3(LP(I)).GE.2)THEN
                          IF ((ATYPE(IP(I)) == 'N   ') .and. &
                               (ATYPE(JP(I)) == 'CA  ') .and. &
                               (ATYPE(KP(I)) == 'CB  ')) THEN
                             write(U,'(10a)') '!DIHE  ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                          ELSEIF ((ATYPE(IP(I)) == 'CA  ') .and. &
                               (ATYPE(JP(I)) == 'CB  ')) THEN
                             write(U,'(10a)') '!DIHE  ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                          ELSE
                             write(U,'(10a)') 'DIHE  ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                          ENDIF
                       else
                          write(U,'(10a)') '!DIHE  ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                       endif

                    endif
                 ENDIF

              endif
           endif

        enddo

        do  i=1,nimphi
           IF( &
                ISLCT4(IM(I))+ISLCT4(JM(I))+ISLCT4(KM(I))+ISLCT4(LM(I)).GE.1)THEN
              IF( &
                   ISLCT3(IM(I))+ISLCT3(JM(I))+ISLCT3(KM(I))+ISLCT3(LM(I)).GE.1)THEN

                 if(ISLCT3(IM(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,IM(I))
                    NAME1= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(IM(I)).eq.1)then
                    NAME1='2'//atype(IM(I))
                    ILIST(IM(I))=1
                 endif

                 if(ISLCT3(JM(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,JM(I))
                    NAME2= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(JM(I)).eq.1)then
                    NAME2='2'//atype(JM(I))
                    ILIST(JM(I))=1
                 endif

                 if(ISLCT3(KM(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,KM(I))
                    NAME3= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(KM(I)).eq.1)then
                    NAME3='2'//atype(KM(I))
                    ILIST(KM(I))=1
                 endif

                 if(ISLCT3(LM(I)).eq.1)then
                    CALL MKPRES2(ILST1,ILST3,NLST1,IAT1,LM(I))
                    NAME4= '1'//atype(IAT1)
                    ILIST(IAT1)=1
                 elseif(ISLCT4(LM(I)).eq.1)then
                    NAME4='2'//atype(LM(I))
                    ILIST(LM(I))=1
                 endif

                 IF(ILOOP.EQ.2) then
                    IF(QSET) THEN
                       write(U,'(10a)') 'IMPR  ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                    ELSE
                       write(U,'(10a)') '!IMPR  ',NAME1,' ',NAME2,' ',NAME3,' ',NAME4
                    ENDIF
                 endif

              endif
           endif
        enddo

        if (prnlev >= 2) write(outu,*)
        CGTOT2=0.0
        do i=1,natom
           IF(ILIST(I)+ISLCT2(I).GE.1)THEN

              IF(ISLCT1(I)+ISLCT2(I).GT.0)then
                 CGTOT2=CGTOT2+CG2(I)
              endif

              IF(ISLCT4(I).EQ.1)then
                 CGTOT2=CGTOT2+CG2(I)
              endif

           ENDIF
        ENDDO

        !yw      IF(ILOOP.EQ.1.and.QSET) write(U,'(2a,f5.2)') 'PRES  ',WRD,CGTOT2
        !yw      IF(ILOOP.EQ.1.and..not.QSET)THEN
        !yw      write(U,'(2a,f5.2)') 'PRES  ',WRD,CGTOT2
        !yw      ENDIF
        IF(ILOOP.EQ.1) write(U,'(2a,1x,f6.2)') 'PRES  ',WRD,CGTOT2
        write(U,*)

        if(ILOOP.EQ.1)THEN
           do i=1,natom
              IF(ILIST(I)+ISLCT2(I).GE.1)THEN

                 IF(ISLCT1(I).GT.0)then
                    write(U,'(4A,3x,f5.2)') 'ATOM   1',ATYPE(I)(1:idleng),' ', &
                         ATC(IAC2(I))(1:idleng),CG2(I)
                    CGTOT2=CGTOT2+CG2(I)
                 endif

                 IF(ISLCT2(I).GT.0)then
                    write(U,'(4A,3x,f5.2)') 'ATOM   1',ATYPE(I)(1:idleng),' ', &
                         ATC(IAC2(I))(1:idleng),CG2(I)
                    CGTOT2=CGTOT2+CG2(I)
                 endif

                 IF(ISLCT4(I).EQ.1)then
                    write(U,'(4A,3x,f5.2,2x,30(A,1X))')  &
                         'ATOM   2',ATYPE(I)(1:idleng),' ', &
                         ATC(IAC2(I))(1:idleng),CG2(I), &
                         ('1'//ATYPE(ILST2(J))(1:idleng),J=1,NLST2)
                    CGTOT2=CGTOT2+CG2(I)
                 endif

              ENDIF
           ENDDO
        endif

     ENDDO
  ENDDO
  write(U,'(A)') 'END'
  !
  RETURN
END SUBROUTINE MKPRES

SUBROUTINE MKPRES2(ILST1,ILST3,NLST1,IAT1,IAT3)
  INTEGER ILST1(*),ILST3(*),NLST1,IAT1,IAT3,I
  DO I=1,NLST1
     IF(ILST3(I).EQ.IAT3)THEN
        IAT1=ILST1(I)
        RETURN
     ENDIF
  ENDDO
END SUBROUTINE MKPRES2

