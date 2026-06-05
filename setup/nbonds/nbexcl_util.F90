module nbexcl_util

  implicit none
  private

  public makgrp

contains

  SUBROUTINE MAKGRP(ISGRP,IFGRP,NGRP,IGLO14,ING14,NNG14,MAXNNG, &
#ifdef KEY_RESIZE
       IBLO14,INB14,IGPBS,IGR,CMPLTD,IPT,ISTOP,mxaim0)
#else
       IBLO14,INB14,IGPBS,IGR,CMPLTD,IPT,ISTOP)
#endif    
    !-----------------------------------------------------------------------
    !     THIS ROUTINE MAKES A GROUP-GROUP EXCLUSION LIST
    !
    !     ISGRP - FIRST GROUP OF LIST TO MAKE (EITHER 1 OR NGRP+1)
    !     IFGRP - LAST GROUP OF LIST TO MAKE (EITHER NGRP OR NGRPT)
    !     NGRP  - NUMBER OF GROUPS IN PSF (NOT INCLUDING IMAGES)
    !     IGLO14(IFGRP) - INBLO ARRAY FOR GROUPS (POINTERS INTO ING14)
    !     ING14(NNG14)   - GROUP INDICIES OF GROUPS WITH EXCLUSIONS
    !     NNG14 - NUMBER OF GROUP EXCLUSIONS
    !     MAXNNG - MAXIMUM NUMBER OF GROUP EXCLUSIONS ALLOWED
    !     IBLO14(NATOMT) - ATOM EXCLUSION POINTER LIST
    !     INB14(NNB14) - ATOM EXCLUSION LIST
    !     IGPBS(IFGRP) - GROUP BASE POINTER LIST
    !     IGR(NATOM) - TEMP LIST FOR BACKPOINTER FROM ATOMS TO GROUPS
    !
    !     BERNARD R. BROOKS  8/31/84
    !
    use chm_kinds
    use dimens_fcm
    use stream
    use timerm
    implicit none
    !
    !-----------------------------------------------------------------------
    !  Miscellaneous:
    !
    !  MAXING - The maximum number of atoms in any electrostatic group.
    !
    integer,parameter :: MAXING=1000
    !-----------------------------------------------------------------------
    INTEGER ISGRP,IFGRP,NGRP,NNG14,MAXNNG
    INTEGER IGLO14(*),ING14(*),IBLO14(*),INB14(*)
    INTEGER IGPBS(*),IGR(*)
    LOGICAL CMPLTD
    INTEGER IPT(MAXING),ISTOP(MAXING)
    !
    INTEGER IGRP,JGRP,IS,IQ,NA,I,J,K,ILAST,MINV,MINI
#ifdef KEY_RESIZE
    integer,intent(in):: mxaim0
#endif
    !
    CMPLTD=.FALSE.
    !
    DO IGRP=1,NGRP
       IS=IGPBS(IGRP)+1
       IQ=IGPBS(IGRP+1)
       DO I=IS,IQ
          IGR(I)=IGRP
       ENDDO
    ENDDO
    !
    IF(ISGRP > 1) THEN
       DO IGRP=1,ISGRP
          IGLO14(IGRP)=0
       ENDDO
    ENDIF
    !
    !
    NNG14=0
    DO IGRP=ISGRP,IFGRP
       IS=IGPBS(IGRP)+1
       IQ=IGPBS(IGRP+1)
       NA=IQ-IS+1
       IF(NA > MAXING) THEN
          CALL WRNDIE(-3,'<MAKGRP>', &
               'maximum number of atoms per group exceeded')
       ENDIF
       !
       J=0
       DO I=IS,IQ
          J=J+1
          IF(I == 1) THEN
             IPT(J)=1
          ELSE
             IPT(J)=IBLO14(I-1)+1
          ENDIF
          ISTOP(J)=IBLO14(I)
       ENDDO
       !
       ILAST=0
       MINV=0
#ifdef KEY_RESIZE       
       DO WHILE(MINV /= MXAIM0+1)
          MINV=MXAIM0+1
#else          
       DO WHILE(MINV /= MAXAIM+1)
          MINV=MAXAIM+1
#endif
          DO I=1,NA
             IF(IPT(I) <= ISTOP(I)) THEN
                K=INB14(IPT(I))
                IF(K < 0) K=-K
                IF(K < MINV) THEN
                   MINV=K
                   MINI=I
                ENDIF
             ENDIF
          ENDDO
#ifdef KEY_RESIZE       
          IF(MINV <= MXAIM0) THEN
#else
          IF(MINV <= MAXAIM) THEN
#endif                
             JGRP=IGR(MINV)
             IF(JGRP > ILAST .AND. JGRP /= IGRP) THEN
                NNG14=NNG14+1
                IF(NNG14 > MAXNNG) RETURN
                ING14(NNG14)=JGRP
                ILAST=JGRP
             ENDIF
             !
             IPT(MINI)=IPT(MINI)+1
          ENDIF
       ENDDO
       !
       IGLO14(IGRP)=NNG14
    ENDDO
    CMPLTD=.TRUE.
    IF(PRNLEV >= 3) THEN
       IF(ISGRP == 1) THEN
          WRITE(OUTU,432) NNG14
       ELSE
          IF(PRNLEV >= 5) WRITE(OUTU,433) NNG14
       ENDIF
432    FORMAT(' <MAKGRP> found',I7,' group exclusions.')
433    FORMAT(' <MAKGRP> found',I7,' image group exclusions.')
    ENDIF
    !
    RETURN
  END SUBROUTINE MAKGRP

end module nbexcl_util
