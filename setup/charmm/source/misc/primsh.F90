module primsh
  use chm_kinds
  implicit none

  integer,allocatable,dimension(:) :: LPOINT, LSTSHL, LPROT, LPWH, NONPOLAR
  integer,allocatable,dimension(:) :: TEMPLIST, NUMNEI, WATLIST
  integer,allocatable,dimension(:,:) :: NEILIS

  logical :: QBHEL, QSHEL
  integer :: NTBHEL, NTSHEL, UPDF
  integer :: CHFR, SPACE, NPROT, NPWH
  real(chm_real) :: DRSH, FOCO1, FOCO2, CUT, MOLVOL, RWEL, XALER
  real(chm_real) :: SCO, CHCO, PFINAL, WATV

contains

#if KEY_PRIMSH==0
  SUBROUTINE BHELP(ISLCT)
    INTEGER ISLCT(*)
    RETURN
  END SUBROUTINE BHELP

  SUBROUTINE SHELP(ISLCT)
    INTEGER ISLCT(*)
    RETURN
  END SUBROUTINE SHELP
#else /**/
  subroutine primsh_iniall()
#if KEY_PRIMSH==1
    qbhel = .false.     
#endif
#if KEY_PRIMSH==1
    qshel = .false.     
#endif
    return
  end subroutine primsh_iniall

  !     ---------------------------------------------------------------
  SUBROUTINE BHELP(ISLCT)
    !
    use dimens_fcm
    use number
    use comand
    use psf
    use select
    use stream
    use string
    use coord
    use memory

    implicit none
    INTEGER ISLCT(*)
    !
    !-----------------------------------------------------------------
    ! Local variables
    CHARACTER(len=4) WRD
    INTEGER IMODE,I
    IF(INDXA(COMLYN,COMLEN,'RESE').GT.0)THEN
       IF(QBHEL)THEN
          if (prnlev >= 2) WRITE(OUTU,100) 'BHEL RESET TO ZERO'
100       FORMAT(1x,A)
          call chmdealloc('primsh.src','BHELP','LPOINT',NATOM,intg=LPOINT)
          call chmdealloc('primsh.src','BHELP','LPROT',NATOM,intg=LPROT)
          call chmdealloc('primsh.src','BHELP','LPWH',NATOM,intg=LPWH)
          call chmdealloc('primsh.src','BHELP','NONPOLAR',NATOM,intg=NONPOLAR)
          QBHEL  = .FALSE.
       ELSE
          CALL WRNDIE(0,'<BHEL>','BHEL NOT SETUP')
       ENDIF
       ! Get parameter values
    ELSE
       ! Initialize if necessary
       IF(.NOT.QBHEL)THEN
          call chmalloc('primsh.src','BHELP','LPOINT',NATOM,intg=LPOINT)
          call chmalloc('primsh.src','BHELP','LPROT',NATOM,intg=LPROT)
          call chmalloc('primsh.src','BHELP','LPWH',NATOM,intg=LPWH)
          call chmalloc('primsh.src','BHELP','NONPOLAR',NATOM,intg=NONPOLAR)
          QBHEL  = .TRUE.
       ENDIF
       IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
       CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0)THEN
          CALL WRNDIE(-1,'<BHEL>','ATOM SELECTION PARSING ERROR')
       ENDIF
       CALL STBHEL(NATOM,nonpolar,nres,res,ibase,ISLCT,LPOINT,NTBHEL)
       call nprotatm(natom,atype,lprot,lpwh,nprot,npwh)
    ENDIF
    if (prnlev >= 2) write(OUTU,*)'BHEL',QBHEL
    RETURN
  END SUBROUTINE BHELP
  !     ---------------------------------------------------------------
  SUBROUTINE SHELP(ISLCT)
    !
    use dimens_fcm
    use number
    use comand
    use psf
    use select
    use stream
    use string
    use coord
    use memory

    implicit none
    INTEGER ISLCT(*)
    !
    !-----------------------------------------------------------------
    ! Local variables
    CHARACTER(len=4) WRD
    INTEGER IMODE,I

    DRSH = 7.0
    DRSH = GTRMF(COMLYN,COMLEN,'DRSH',DRSH)
    XALER=0.00001
    XALER= GTRMF(COMLYN,COMLEN,'RELA',XALER)
    !
    PFINAL = 1.0
    PFINAL = GTRMF(COMLYN,COMLEN,'PFINAL',PFINAL)
    FOCO1 = 15.0
    FOCO1 = GTRMF(COMLYN,COMLEN,'FOCO1',FOCO1)
    FOCO2 = 15.0
    FOCO2 = GTRMF(COMLYN,COMLEN,'FOCO2',FOCO2)
    CUT = 2.0
    CUT = GTRMF(COMLYN,COMLEN,'CUT',CUT)
    RWEL = 0.25
    RWEL = GTRMF(COMLYN,COMLEN,'RWEL',RWEL)
    SCO = 0.0
    SCO = GTRMF(COMLYN,COMLEN,'SCO',SCO)
    CHFR = 1000
    CHFR = GTRMI(COMLYN,COMLEN,'CHFR',CHFR)
    CHCO = 0.00001
    CHCO = GTRMF(COMLYN,COMLEN,'CHCO',CHCO)
    MOLVOL = 29.9
    MOLVOL = GTRMF(COMLYN,COMLEN,'MOLVOL',MOLVOL)
    SPACE = 1000000
    SPACE = GTRMI(COMLYN,COMLEN,'SPACE',SPACE)
    UPDF = 10
    UPDF = GTRMI(COMLYN,COMLEN,'UPDF',UPDF)
    !
    !      FERF = 0.9
    !      FERF = GTRMF(COMLYN,COMLEN,'FREF',FERF)
    !      RSLV = GTRMF(COMLYN,COMLEN,'RSLV',ZERO)
    !      FOCO = 3.0
    !      FOCO = GTRMF(COMLYN,COMLEN,'FOCO',FOCO)

    IF(INDXA(COMLYN,COMLEN,'RESE').GT.0)THEN

       IF(QSHEL)THEN
          if (prnlev >= 2) WRITE(OUTU,100) 'SHEL RESET TO ZERO'
100       FORMAT(1x,A)
          call chmdealloc('primsh.src','SHELP','LSTSHL',NATOM,intg=LSTSHL)
          call chmdealloc('primsh.src','SHELP','NEILIS',ntshel,natom, &
               intg=neilis)
          call chmdealloc('primsh.src','SHELP','TEMPLIST',NATOM,intg=templist)
          call chmdealloc('primsh.src','SHELP','NUMNEI',NATOM,intg=numnei)
          call chmdealloc('primsh.src','SHELP','WATLIST',NTSHEL,intg=watlist)
          QSHEL  = .FALSE.
       ELSE
          CALL WRNDIE(0,'<SHEL>','SHEL NOT SETUP')
       ENDIF
       ! Get parameter values
    ELSE
       !
       ! Initialize if necessary
       IF (.NOT. QSHEL) THEN
          call chmalloc('primsh.src','SHELP','LSTSHL',NATOM,intg=LSTSHL)
          call chmalloc('primsh.src','SHELP','TEMPLIST',NATOM,intg=templist)
          call chmalloc('primsh.src','SHELP','NUMNEI',NATOM,intg=numnei)
       ENDIF
       IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
       CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0)THEN
          CALL WRNDIE(-1,'<SHEL>','ATOM SELECTION PARSING ERROR')
       ENDIF
       CALL STSHEL(NATOM,ISLCT,LSTSHL,NTSHEL,WATV,MOLVOL)
       IF (.NOT. QSHEL) THEN
          call chmalloc('primsh.src','SHELP','NEILIS',ntshel,natom, &
               intg=neilis)
          call chmalloc('primsh.src','SHELP','WATLIST',NTSHEL,intg=watlist)
          QSHEL = .TRUE.
       ENDIF

    ENDIF
    if (prnlev >= 2) write(OUTU,*)'SHEL',QSHEL
    RETURN
  END SUBROUTINE SHELP

  SUBROUTINE STSHEL(NATOM,ISLCT,LSTSHL,NTSHEL,watv,molvol)

    use stream
    implicit none
    INTEGER NATOM, ISLCT(*), LSTSHL(*), NTSHEL
    real(chm_real) watv,molvol

    ! Local variables
    INTEGER I

    NTSHEL=0
    DO  I=1,NATOM
       IF(ISLCT(I).EQ.1)THEN
          NTSHEL=NTSHEL+1
          LSTSHL(NTSHEL)=I
       ENDIF
    enddo
    watv = molvol*ntshel

    RETURN
  END SUBROUTINE STSHEL
  !     -------------------------------------------
  SUBROUTINE STBHEL(NATOM,nonpolar,nres,res,ibase,ISLCT,LPOINT,NTBHEL)
    use stream
    implicit none
    INTEGER NATOM,ISLCT(*),LPOINT(*),NTBHEL,nonpolar(*),nres,ibase(*)
    character(len=8) res(*)
    ! Local variables
    INTEGER I,j,k

    NTBHEL=0
    DO  I=1,NATOM
       nonpolar(i) = 0
       IF(ISLCT(I).EQ.1)THEN
          NTBHEL=NTBHEL+1
          LPOINT(NTBHEL)=I
       ENDIF
    enddo

    DO J=1,NRES
       IF((RES(J).EQ.'ALA').OR.(RES(J).EQ.'GLY').OR.(RES(J).EQ.'ILE') &
            .OR.(RES(J).EQ.'LEU').OR.(RES(J).EQ.'MET') &
            .OR.(RES(J).EQ.'PHE').OR.(RES(J).EQ.'PRO') &
            .OR.(RES(J).EQ.'TRP').OR.(RES(J).EQ.'VAL'))THEN
          DO K=IBASE(J)+1,IBASE(J+1)
             IF(ISLCT(K).EQ.1)THEN
                NONPOLAR(K)=1
             ENDIF
          ENDDO
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE STBHEL

  !     -------------------------------------------
  SUBROUTINE NPROTATM(NATOM,ATYPE,LPROT,LPWH,NPROT,NPWH)
    use stream

    implicit none

    INTEGER NATOM, NPROT, NPWH, LPROT(*), LPWH(*)
    CHARACTER(len=8) ATYPE(*)
    ! Local variables
    INTEGER I


    NPROT=0
    NPWH=0
    DO I=1,NATOM
       IF (ATYPE(I) /= 'OH2' .AND. ATYPE(I) /= 'H1' &
            .AND. ATYPE(I) /= 'H2') THEN
          NPROT=NPROT+1
          LPROT(NPROT)=I
          IF (ATYPE(I)(1:1) /= 'H') THEN
             NPWH=NPWH+1
             LPWH(NPWH)=I
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE NPROTATM

  !     -------------------------------------------
  SUBROUTINE CHDRSH(X,Y,Z,LPROT,LPWH,NPROT, &
       NPWH,VDWR,IAC,ITC,DRSH,SPACE,WATV,XALER,TOTV)
    use stream
    use memory
    use corman2,only: volume
    use param_store, only: get_param, set_param

    implicit none

    INTEGER NPROT,NPWH,IAC(*),ITC(*)
    INTEGER SPACE
    INTEGER LPROT(*),LPWH(*)
    REAL(chm_real) X(*),Y(*),Z(*),VDWR(*),DRSH,WATV,XALER,TOTV

    ! Local variables

    INTEGER I,J,K
    REAL(chm_real) PROTV
    INTEGER,allocatable,dimension(:) :: ISLCT1,ISLCT2
    REAL(chm_real),allocatable,dimension(:) :: WPROT,XI,YI,ZI
    REAL(chm_real),allocatable,dimension(:) :: WTOT,XJ,YJ,ZJ

    !maybe:
    call chmalloc('primsh.src','CHDRSH','LPROT',NPROT,intg=ISLCT1)
    call chmalloc('primsh.src','CHDRSH','LPROT',NPWH,intg=ISLCT2)
    call chmalloc('primsh.src','CHDRSH','XI',NPROT,crl=XI)
    call chmalloc('primsh.src','CHDRSH','YI',NPROT,crl=YI)
    call chmalloc('primsh.src','CHDRSH','ZI',NPROT,crl=ZI)
    call chmalloc('primsh.src','CHDRSH','XJ',NPWH,crl=XJ)
    call chmalloc('primsh.src','CHDRSH','YJ',NPWH,crl=YJ)
    call chmalloc('primsh.src','CHDRSH','ZJ',NPWH,crl=ZJ)
    call chmalloc('primsh.src','CHDRSH','WPROT',NPROT,crl=WPROT)
    call chmalloc('primsh.src','CHDRSH','WTOT',NPWH,crl=WTOT)

    DO I=1,NPROT
       WPROT(I)=VDWR(ITC(IAC(LPROT(I))))
       ISLCT1(I)=1
       XI(I)=X(LPROT(I))
       YI(I)=Y(LPROT(I))
       ZI(I)=Z(LPROT(I))
    ENDDO
    DO J=1,NPWH
       WTOT(J)=VDWR(ITC(IAC(LPWH(J))))+DRSH
       ISLCT2(J)=1
       XJ(J)=X(LPWH(J))
       YJ(J)=Y(LPWH(J))
       ZJ(J)=Z(LPWH(J))
    ENDDO
    CALL VOLUME(NPROT,XI,YI,ZI,WPROT,ISLCT1,WPROT,WPROT,SPACE,.FALSE.)
    call get_param('VOLU', PROTV)
    CALL VOLUME(NPWH,XJ,YJ,ZJ,WTOT,ISLCT2,WTOT,WTOT,SPACE,.FALSE.)
    call get_param('VOLU', TOTV)
    DRSH=DRSH+XALER*(WATV-(TOTV-PROTV))
    DO K=1,NPWH
       WTOT(K)=VDWR(ITC(IAC(LPWH(K))))+DRSH
    ENDDO
    CALL VOLUME(NPWH,XJ,YJ,ZJ,WTOT,ISLCT2,WTOT,WTOT,SPACE,.FALSE.)
    call get_param('VOLU', TOTV)
    call set_param('DRSH',DRSH)
    !maybe:
    call chmdealloc('primsh.src','CHDRSH','ISLCT1',NPROT,intg=ISLCT1)
    call chmdealloc('primsh.src','CHDRSH','ISLCT2',NPWH,intg=ISLCT2)
    call chmdealloc('primsh.src','CHDRSH','XI',NPROT,crl=XI)
    call chmdealloc('primsh.src','CHDRSH','YI',NPROT,crl=YI)
    call chmdealloc('primsh.src','CHDRSH','ZI',NPROT,crl=ZI)
    call chmdealloc('primsh.src','CHDRSH','XJ',NPWH,crl=XJ)
    call chmdealloc('primsh.src','CHDRSH','YJ',NPWH,crl=YJ)
    call chmdealloc('primsh.src','CHDRSH','ZJ',NPWH,crl=ZJ)
    call chmdealloc('primsh.src','CHDRSH','WTOT',NPWH,crl=WTOT)
    call chmdealloc('primsh.src','CHDRSH','WPROT',NPROT,crl=WPROT)

    RETURN
  END SUBROUTINE CHDRSH


  !     -------------------------------------
  subroutine PSHEL(VDWR,IAC,ITC,X,Y,Z, &
       ESHEL,DX,DY,DZ,nwat,NPATM,TOTV)
    !     -------------------------------------

    use stream
    use number
    use energym
    use contrl
    use param_store, only: set_param

    implicit none
    integer IAC(*),ITC(*)

    integer NWAT,NPATM

    real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),VDWR(*)
    REAL(chm_real) ESHEL,TOTV

    integer lsi,i,j,k,p,q,lpj,imi
    logical update
    real(chm_real) dimra,dxi,dyi,dzi,di,ra,dimi,dxmi
    real(chm_real) dymi,dzmi,dr,g,gx,gy,gz,FOCO

    IF(DYNAMQ)THEN
       MDSTEP=0
       MINXYZ=.FALSE.
    ENDIF
    IF((DYNAMQ).OR.((MOD(MDSTEP,CHFR).EQ.0) &
         .AND.(MDSTEP.GT.0).AND.(.NOT.MINXYZ)))THEN
       CALL CHDRSH(X,Y,Z,LPROT,LPWH,NPROT, &
            NPWH,VDWR,IAC,ITC,DRSH,SPACE,WATV,XALER,TOTV)
       if (prnlev >= 2) WRITE(OUTU,*) 'DRSH',DRSH
       if (prnlev >= 2) WRITE(OUTU,*) 'SCO',SCO
    ELSE
       SCO=SCO+CHCO*(EPROP(PRESSE)-PFINAL)
       call set_param('SCO',SCO)
    ENDIF
    IF((DYNAMQ).OR.((MOD(MDSTEP,UPDF).EQ.0)))THEN
       UPDATE = .TRUE.
    ELSE
       UPDATE=.FALSE.
    ENDIF
    IF (UPDATE) THEN
       NWAT=0
       do i=1,NTSHEL
          dimra=10000.
          LSI=LSTSHL(i)
          NPATM=0
          do j=1,NTBHEL
             LPJ=LPOINT(j)
             dxi=x(LSI)-x(LPJ)
             dyi=y(LSI)-y(LPJ)
             dzi=z(LSI)-z(LPJ)
             di=sqrt(dxi**2+dyi**2+dzi**2)
             if (di.lt.cut*DRSH) then
                NPATM=NPATM+1
                TEMPLIST(NPATM)=LPJ
                ra=VDWR(ITC(IAC(LPJ)))
                if(di-ra.lt.dimra)then
                   dimra=di-ra
                   dimi=di
                   imi=LPJ
                   dxmi=dxi
                   dymi=dyi
                   dzmi=dzi
                endif
             endif
          enddo
          IF(NONPOLAR(imi).EQ.1)THEN
             FOCO=FOCO1
          ELSE
             FOCO=FOCO2
          ENDIF
          dr=dimra-DRSH
          if(dr.gt.(- RWEL*DRSH))then
             NWAT=NWAT+1
             do  k=1, NPATM
                NEILIS(NWAT,k)=TEMPLIST(k)
             enddo
             NUMNEI(NWAT)=NPATM
             WATLIST(NWAT)=LSI
             if (dr.gt.0.) then
                eshel=eshel+0.5*FOCO*dr**2
                g=FOCO*dr
             else
                eshel=eshel+0.5*SCO*FOCO*dr**2
                g=SCO*FOCO*dr
             endif
             g=g/dimi
             gx=g*dxmi
             gy=g*dymi
             gz=g*dzmi
             DX(lsi)=DX(lsi)+gx
             DY(lsi)=DY(lsi)+gy
             DZ(lsi)=DZ(lsi)+gz
             DX(imi)=DX(imi)-gx
             dy(imi)=dy(imi)-gy
             dz(imi)=dz(imi)-gz
          endif
       enddo
    else
       do p=1,NWAT
          dimra=10000.
          LSI=WATLIST(p)
          do q=1,NUMNEI(p)
             LPJ=NEILIS(p,q)
             dxi=x(LSI)-x(LPJ)
             dyi=y(LSI)-y(LPJ)
             dzi=z(LSI)-z(LPJ)
             di=sqrt(dxi**2+dyi**2+dzi**2)
             ra=VDWR(ITC(IAC(LPJ)))
             if (di-ra.lt.dimra) then
                dimra=di-ra
                dimi=di
                imi=LPJ
                dxmi=dxi
                dymi=dyi
                dzmi=dzi
             endif
          enddo
          IF(NONPOLAR(imi).EQ.1)THEN
             FOCO=FOCO1
          ELSE
             FOCO=FOCO2
          ENDIF
          dr=dimra-DRSH
          if(dr.gt.(- RWEL*DRSH))then
             if (dr.gt.0.) then
                eshel=eshel+0.5*FOCO*dr**2
                g=FOCO*dr
             else
                eshel=eshel+0.5*SCO*FOCO*dr**2
                g=SCO*FOCO*dr
             endif
             g=g/dimi
             gx=g*dxmi
             gy=g*dymi
             gz=g*dzmi
             DX(lsi)=DX(lsi)+gx
             DY(lsi)=DY(lsi)+gy
             DZ(lsi)=DZ(lsi)+gz
             DX(imi)=DX(imi)-gx
             dy(imi)=dy(imi)-gy
             dz(imi)=dz(imi)-gz
          endif
       enddo
    endif
    DYNAMQ=.FALSE.
    if(DRSH.lt.0.)write(outu,*)'PSHEL WARNING: DRSH .lt. 0  ',drsh
    return
  END subroutine PSHEL

#endif 

end module primsh

