module resdist

  use chm_kinds
  use dimens_fcm
  use resdist_ltm
  implicit none

contains

  SUBROUTINE REDSET
    !
    ! syntax:
    !
    ! RESDistance [ RESEt ] [ SCALE real ] [ KVAL real  RVAL real [EVAL integer] -
    !
    !         [ POSITIVE ] [ IVAL integer ]  repeat( real  first-atom  second-atom ) ]
    !         [ NEGATIVE ]
    !
    !  (default EVAL is 2, EVAL must be positive)
    !
    !     E = 1/EVAL *  Kval * Dref**EVAL
    !
    !  Where:
    !
    !     Dref =  K1*R1**Ival + K2*R2**Ival + ... + Kn*Rn**Ival - Rval
    !
    !  Where K1,K2,...Kn are the real values in the repeat section and
    !  R1,R2,...Rn are the distances between specified pair of atoms.
    !      E =  1/EVAL  * Kval ( K1*R1 + K2*R2 + ... + Kn*Rn - Rval ) ** EVAL
    !
    !
    !          by Bernard R. Brooks - NIH - March, 1995
    !
  use number
  use psf
  use comand
  use select
  use stream
  use string
  use chutil
#if KEY_OPENMM==1
  use omm_ctrl, only : omm_system_changed
#endif
#if KEY_DOMDEC==1
  use domdec_common,only:domdec_system_changed
#endif
  use memory

  implicit none

    ! local
    integer,allocatable,dimension(:) :: ISLCT
    INTEGER I,J,N,II,NI
    character(len=8) SIDI,RIDI,RENI,ACI
    character(len=8) SIDJ,RIDJ,RENJ,ACJ
    real(chm_real) TEMPR
    !
    ! begin
    !
#if KEY_NOMISC==1
    CALL WRNDIE(-1,'<REDSET>', &
         'Distance restraint code is not compiled.')
    RETURN
#else /**/
    ! If this routine is called, signle to openmm the system has 
    ! chagnged.
#if KEY_OPENMM==1
    call omm_system_changed()
#endif
#if KEY_DOMDEC==1
    call domdec_system_changed()
#endif
    !
    ! Process initialization and reset option
    IF(INDXA(COMLYN,COMLEN,'RESE') > 0) THEN
       REDNUM=0
       IF(PRNLEV >= 2) WRITE(OUTU,610)
610    FORMAT('  RESDIST: Restrained distances restraints reset.')
    ENDIF
    ! initialize counters and scalars
    IF(REDNUM == 0) THEN
       REDNM2=0
       REDSCA=ONE
       REDIPT(1)=0
    ENDIF
    !
    ! Parse overall scale factor
    TEMPR=REDSCA
    REDSCA=GTRMF(COMLYN,COMLEN,'SCAL',REDSCA)
    IF(TEMPR /= REDSCA) THEN
       IF(PRNLEV >= 2) WRITE(OUTU,620) REDSCA
620    FORMAT('  RESDIST: Scale factor set to',F12.4)
    ENDIF
    !
    CALL TRIMA(COMLYN,COMLEN)
    IF(COMLEN == 0) THEN
       IF(PRNLEV >= 2) WRITE(OUTU,230) REDNUM
       RETURN
    ENDIF
    !
    ! Parse a new restraint if anything is left on the command line
    !
    IF(REDNUM >= REDMAX) THEN
       CALL WRNDIE(0,'<REDSET>', &
            'Max number of restraints exceeded. Ignored.')
       IF(PRNLEV >= 2) WRITE(OUTU,230) REDNUM
       RETURN
    ENDIF
    !
    REDNUM=REDNUM+1
    !
    REDKVAL(REDNUM)=GTRMF(COMLYN,COMLEN,'KVAL',ZERO)
    REDRVAL(REDNUM)=GTRMF(COMLYN,COMLEN,'RVAL',ZERO)
    REDEVAL(REDNUM)=GTRMI(COMLYN,COMLEN,'EVAL',2)
    REDIVAL(REDNUM)=GTRMI(COMLYN,COMLEN,'IVAL',1)
    REDMVAL(REDNUM)=0
    IF(INDXA(COMLYN,COMLEN,'NEGA') > 0) REDMVAL(REDNUM)=-1
    IF(INDXA(COMLYN,COMLEN,'POSI') > 0) THEN
       IF(REDMVAL(REDNUM) == 0) THEN
          REDMVAL(REDNUM)=1
       ELSE
          CALL WRNDIE(0,'<REDSET>', &
               'Cannot specify both POSItive and NEGAtive')
          REDMVAL(REDNUM)=0
       ENDIF
    ENDIF
    !
    IF(REDKVAL(REDNUM) == ZERO) THEN
       CALL WRNDIE(0,'<REDSET>', &
            'No force constant (or zero) specified. Ignored.')
       REDNUM=REDNUM-1
       IF(PRNLEV >= 2) WRITE(OUTU,230) REDNUM
       RETURN
    ENDIF
    IF(REDEVAL(REDNUM) <= 0) THEN
       CALL WRNDIE(0,'<REDSET>', &
            'Illegal exponent specified. Ignored.')
       REDNUM=REDNUM-1
       IF(PRNLEV >= 2) WRITE(OUTU,230) REDNUM
       RETURN
    ENDIF
    !
    N=REDNM2
    !
    call chmalloc('resdist.src','REDSET','ISLCT',NATOM,intg=ISLCT)
    !---------------------------------------------------------------
    ! Parsing loop for atom pairs
    !
100 CONTINUE
    !
    !c      write(6,888) N,comlyn(1:40)
    !c 888  format(' RESD parsing:::',i5,' comlyn:"',A)
    !
    IF(N >= REDMX2) THEN
       CALL WRNDIE(0,'<REDSET>', &
            'Max number of restraint pairs. Ignored.')
       REDNUM=REDNUM-1
       IF(PRNLEV >= 2) WRITE(OUTU,230) REDNUM
       RETURN
    ENDIF
    !
    N=N+1
    REDKLIS(N)=NEXTF(COMLYN,COMLEN)
    CALL NXTATM(REDILIS(1,N),NI,2,COMLYN,COMLEN,ISLCT, &
         SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
    !
    IF(NI /= 2) THEN
       CALL WRNDIE(0,'<REDSET>', &
            'Wrong number of atoms specified. Check syntax.')
       REDNUM=REDNUM-1
       IF(PRNLEV >= 2) WRITE(OUTU,230) REDNUM
       RETURN
    ENDIF
    !
    IF(REDKLIS(N) == ZERO) THEN
       CALL WRNDIE(0,'<REDSET>', &
            'No distance (or zero) scale factor specified. Check syntax.')
       REDNUM=REDNUM-1
       IF(PRNLEV >= 2) WRITE(OUTU,230) REDNUM
       RETURN
    ENDIF
    !
    CALL TRIME(COMLYN,COMLEN)
    IF(COMLEN > 0) GOTO 100
    !---------------------------------------------------------------
    !
    call chmdealloc('resdist.src','REDSET','ISLCT',NATOM,intg=ISLCT)
    !
    IF(PRNLEV >= 2) THEN
       WRITE(OUTU,510) REDNUM,REDKVAL(REDNUM),REDRVAL(REDNUM), &
            N-REDNM2,REDEVAL(REDNUM), &
            REDIVAL(REDNUM),REDMVAL(REDNUM)
510    FORMAT(' REDSET: Adding restraint ',I5,'  KVAL=',F10.3, &
            ' RVAL=',F10.3,/,'     Number of atom pairs=',I4, &
            ' Global Exponent=',I4,' Internal Exponent=',I4, &
            ' One sided mode=',I2)
       !
       DO II=REDNM2+1,N
          I=REDILIS(1,II)
          J=REDILIS(2,II)
          CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
          CALL ATOMID(J,SIDJ,RIDJ,RENJ,ACJ)
          WRITE(OUTU,512) &
               I,SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng), &
               J,SIDJ(1:idleng),RIDJ(1:idleng),ACJ(1:idleng), &
               REDKLIS(II)
512       FORMAT('      atom pair: ',2(I5,1X,A,1X,A,1X,A), &
               '   distance factor=',F12.4)
       ENDDO
       !
       WRITE(OUTU,230) REDNUM
230    FORMAT(' RESDIST:  Current number of restraints=',I4)
    ENDIF
    !
    REDNM2=N
    REDIPT(REDNUM+1)=N
    !
    RETURN
  END SUBROUTINE REDSET

  SUBROUTINE REDWRI(IUNIT)
    !
    ! Print a list of current restraints
    ! By BRB - February 1995
    !
  use number
  use coord
  use stream
  use chutil,only:atomid

    INTEGER IUNIT
    !
    ! local
    character(len=8) SIDI,RIDI,RENI,ACI
    character(len=8) SIDJ,RIDJ,RENJ,ACJ
    INTEGER N,II,I,J,IPT,IVAL,IMODE
    real(chm_real) SIJ,DEL,DF,EIJ,XIJ,YIJ,ZIJ
    real(chm_real) ATOT
    ! begin
    !
    IF(PRNLEV < 2) RETURN
    WRITE(IUNIT,27) REDNUM,REDSCA
27  FORMAT('REDWRI:  Number of distance restraints=',I4, &
         '  Overall scale factor=',F12.4/)
    !
    DO N=1,REDNUM
       !
       IVAL=REDIVAL(N)
       IMODE=REDMVAL(N)
       ATOT=ZERO
       DO IPT=REDIPT(N)+1,REDIPT(N+1)
          I=REDILIS(1,IPT)
          J=REDILIS(2,IPT)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          SIJ=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
          ATOT=ATOT+SQRT(SIJ)**IVAL*REDKLIS(IPT)
       ENDDO
       !
       DEL= ATOT-REDRVAL(N)
       IF(DEL > ZERO .AND. IMODE < 0) THEN
          DF = ZERO
          EIJ= ZERO
       ELSE IF(DEL < ZERO .AND. IMODE > 0) THEN
          DF = ZERO
          EIJ= ZERO
       ELSE
          DF = REDSCA*REDKVAL(N)*DEL**(REDEVAL(N)-1)
          EIJ= REDSCA*REDKVAL(N)*DEL**REDEVAL(N)/REDEVAL(N)
       ENDIF
       !
       WRITE(IUNIT,510) REDNUM,REDKVAL(N),REDRVAL(N), &
            REDIPT(N+1)-REDIPT(N),REDEVAL(N), &
            REDIVAL(REDNUM),REDMVAL(REDNUM)
510    FORMAT(' REDWRI: Evaluating restraint ',I5,'  KVAL=',F10.3, &
            ' RVAL=',F10.3,/,'    Number of atom pairs=',I4, &
            ' Global Exponent=',I4,' Internal Exponent=',I4, &
            ' One sided mode=',I2)
       !
       DO II=REDIPT(N)+1,REDIPT(N+1)
          I=REDILIS(1,II)
          J=REDILIS(2,II)
          CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
          CALL ATOMID(J,SIDJ,RIDJ,RENJ,ACJ)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          SIJ=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
          ATOT=SQRT(SIJ)
          WRITE(IUNIT,512) &
               I,SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng), &
               J,SIDJ(1:idleng),RIDJ(1:idleng),ACJ(1:idleng), &
               REDKLIS(II),ATOT
512       FORMAT('      atom pair: ',2(I5,1X,A,1X,A,1X,A), &
               '   distance factor=',F12.4, &
               '  Current distance=',F14.6)
       ENDDO
       !
       WRITE(IUNIT,514) DEL,EIJ,DF
514    FORMAT('      Distance deviation=',F14.6,'  Energy=',F14.5, &
            ' Force=',F14.5/)
       !
    ENDDO

    WRITE(IUNIT,230) REDNUM
230 FORMAT(' REDWRI:  Current number of restraints=',I4/)
    !
    RETURN
  END SUBROUTINE REDWRI

  SUBROUTINE REDCNS(EN,DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
       REDNUM,REDSCA,REDIPT,REDILIS,REDKLIS, &
       REDKVAL,REDRVAL,REDEVAL,REDIVAL,REDMVAL, &
       DD1,IUPT,QSECD)
    !
    ! Routine computes force field for restraints
    !
    !    By B. Brooks, NIH, February 1995
    !    Tim Miller, NIH, December 2006 added partial 2nd deriv. code
    !
    use number
    use dimens_fcm
    use reawri
    use parallel
    !
    ! I/O
    real(chm_real) EN
    real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
    LOGICAL QECONT
    real(chm_real) ECONT(*)
    INTEGER REDNUM
    real(chm_real) REDSCA
    INTEGER REDIPT(*), REDILIS(2,*)
    real(chm_real) REDKLIS(*), REDRVAL(*), REDKVAL(*)
    INTEGER REDEVAL(*),REDIVAL(*), REDMVAL(*)
    real(chm_real) DD1(*)
    INTEGER IUPT(*)
    LOGICAL QSECD
    !
    ! local
    real(chm_real) XIJ,YIJ,ZIJ,SIJ,RIJ,EIJ,DF,DEL,RLIM,DDF,DFA,DDFA
    real(chm_real) RX,RY,RZ,A,RIJU,AEXP,ATOT,DFDEL,DDFDEL
    INTEGER I,J,N,II,JJ,MM,IADD,NIJ,IVAL,IMODE
    INTEGER NWARN
    ! begin
    NWARN=0
    !
    DO N=MYNODP,REDNUM,NUMNOD
       !
       ! loop over all pairs of atoms belonging to restraint N and
       ! compute the energies
       !

       ATOT=ZERO
       IVAL=REDIVAL(N)
       IMODE=REDMVAL(N)
       NIJ=0
       DO MM=REDIPT(N)+1,REDIPT(N+1)
          I=REDILIS(1,MM)
          J=REDILIS(2,MM)
          XIJ=X(I)-X(J)
          YIJ=Y(I)-Y(J)
          ZIJ=Z(I)-Z(J)
          SIJ=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
          ATOT=ATOT+SQRT(SIJ)**IVAL*REDKLIS(MM)
          NIJ=NIJ+2
       ENDDO
       !
       DEL= ATOT-REDRVAL(N)
       IF(DEL > ZERO .AND. IMODE < 0) THEN
          DFDEL = ZERO
          DDFDEL = ZERO
          EIJ= ZERO
       ELSE IF(DEL < ZERO .AND. IMODE > 0) THEN
          DFDEL = ZERO
          DDFDEL = ZERO
          EIJ= ZERO
       ELSE
          DFDEL = REDSCA*REDKVAL(N)*DEL**(REDEVAL(N)-1)
          EIJ= REDSCA*REDKVAL(N)*DEL**REDEVAL(N)/REDEVAL(N)

          IF(REDEVAL(N) >= 2) THEN
             DDFDEL = REDSCA*REDKVAL(N)*(REDEVAL(N)-1)* &
                  DEL**(REDEVAL(N)-2)
          ELSE
             DDFDEL = ZERO
             DDF = ZERO
          ENDIF
       ENDIF
       !
       ! Compute forces
       !
       DO MM=REDIPT(N)+1,REDIPT(N+1)
          I=REDILIS(1,MM)
          J=REDILIS(2,MM)
          RX=X(I)-X(J)
          RY=Y(I)-Y(J)
          RZ=Z(I)-Z(J)
          SIJ=RX*RX+RY*RY+RZ*RZ
          RIJ=SQRT(SIJ)
          IF(IVAL == 1) THEN
             DFA=REDKLIS(MM)/RIJ
          ELSE IF(IVAL == 2) THEN
             DFA=TWO*REDKLIS(MM)
          ELSE
             DFA=IVAL*REDKLIS(MM)*RIJ**(IVAL-2)
          ENDIF
          DF=DFDEL*DFA
          !
          XIJ=RX*DF
          YIJ=RY*DF
          ZIJ=RZ*DF
          DX(I)=DX(I)+XIJ
          DX(J)=DX(J)-XIJ
          DY(I)=DY(I)+YIJ
          DY(J)=DY(J)-YIJ
          DZ(I)=DZ(I)+ZIJ
          DZ(J)=DZ(J)-ZIJ

          IF(QSECD) THEN
             IF(NIJ == 2) THEN
                IF(J < I) THEN
                   JJ=3*I-2
                   II=3*J-2
                ELSE
                   JJ=3*J-2
                   II=3*I-2
                ENDIF

                !         Calculate DDF
                IF(IVAL >= 2) THEN
                   DDFA=REDKLIS(MM)*IVAL*(IVAL-1)*RIJ**(IVAL-2)
                ELSE
                   DDFA=ZERO
                ENDIF
                DDF=DFDEL*(DDFA-DFA)/SIJ+DDFDEL*(DFA**2)

                A=RX*RX*DDF+DF
                IADD=IUPT(II)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ
                DD1(IADD)=DD1(IADD)+A

                A=RY*RY*DDF+DF
                IADD=IUPT(II+1)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+II+1
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+1)+JJ+1
                DD1(IADD)=DD1(IADD)+A

                A=RZ*RZ*DDF+DF
                IADD=IUPT(II+2)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+2)+JJ+2
                DD1(IADD)=DD1(IADD)+A

                A=RX*RY*DDF
                IADD=IUPT(II)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II+1
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ+1
                DD1(IADD)=DD1(IADD)+A

                A=RX*RZ*DDF
                IADD=IUPT(II)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+JJ
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ)+JJ+2
                DD1(IADD)=DD1(IADD)+A

                A=RY*RZ*DDF
                IADD=IUPT(II+1)+JJ+2
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+2)+JJ+1
                DD1(IADD)=DD1(IADD)-A
                IADD=IUPT(II+1)+II+2
                DD1(IADD)=DD1(IADD)+A
                IADD=IUPT(JJ+1)+JJ+2
                DD1(IADD)=DD1(IADD)+A
             ELSE
                NWARN=NWARN+1
             ENDIF
          ENDIF
       ENDDO
       !
       EN=EN+EIJ
       !
       IF(QECONT) THEN
          EIJ=EIJ/NIJ
          DO MM=REDIPT(N)+1,REDIPT(N+1)
             I=REDILIS(1,MM)
             J=REDILIS(2,MM)
             ECONT(I)=ECONT(I)+EIJ
             ECONT(J)=ECONT(J)+EIJ
          ENDDO
       ENDIF
    ENDDO
    !
    IF(NWARN > 0) CALL WRNDIE(-2,'<REDCNS>', &
         'There is no Hessian code for multiple distance RESD terms.')
    !
    RETURN
#endif 
  END SUBROUTINE REDCNS
end module resdist

