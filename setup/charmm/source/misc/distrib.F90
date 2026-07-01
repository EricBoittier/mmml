module distrib_module

contains
  SUBROUTINE DISTRIxxx
    !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use exfunc
  use comand
  use psf
  use stream
  use string
  use coord
#if KEY_FLUCQ==1
  use flucq
#endif 
    implicit none
    !
    !-----------------------------------------------------------------------
    ! Temporary arrays to read trajectory files
    !
    !-----------------------------------------------------------------------
    !
    INTEGER      DVECT,RR,VR,GR,NR

    DVECT  = GTRMI(COMLYN,COMLEN,'DVEC',5000)
    !
    IF(INDXA(COMLYN, COMLEN, 'DIPO') > 0) THEN
       CALL DISTRI3(NATOM,NATOMT,X,Y,Z,WMAIN, &
            CG)
    ELSE
       CALL DISTRI2(NATOM,NATOMT,X,Y,Z,WMAIN, &
            DVECT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID &
#if KEY_FLUCQ==1 || KEY_CHEQ==1
            ,CG    & 
#endif
            )
    ENDIF
    !


    RETURN
  END SUBROUTINE DISTRIXXX

  SUBROUTINE  DISTRI2(NATOM,NATOMT,X,Y,Z,WMAIN, &
       DVECT,IBASE,NICTOT,NSEGT,RESID,RES,SEGID &
#if KEY_FLUCQ==1 || KEY_CHEQ==1
       ,CG    & 
#endif
       )
    !

    !-----------------------------------------------------------------------
    !
    !  DISTRIBUTION FUNCTION ANALYSIS MAIN SUBROUTINE
    !

#if KEY_CHEQ==1
  use cheq,only:qcg                    
#endif
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor               
#endif
  use chm_kinds
  use chm_types
  use bases_fcm
  use dimens_fcm
  use number
  use consta
  use exfunc
  use stream
  use string
  use comand
  use cvio, only: trjspc, readcv
  use image
  use image_routines_module, only: transo
  use deriv
  use econtmod
  use ctitla
#if KEY_FLUCQ==1
  use flucq     
#endif
  use memory
  use select, only: selcta, selrpn

    implicit none
#if KEY_FLUCQ==1 || KEY_CHEQ==1
    real(chm_real)  CG(MAXAIM)   
#endif
    !-----------------------------------------------------------------------
    ! General
    INTEGER       NATOM, NATOMT
    real(chm_real)        X(*),Y(*),Z(*),WMAIN(*)
    INTEGER       DVECT
    INTEGER       IBASE(*),NICTOT(*),NSEGT
    CHARACTER(len=*) RESID(*),RES(*),SEGID(*)

    !-----------------------------------------------------------------------
    !  Miscelaneous Local variables
    INTEGER,allocatable,dimension(:) :: ISLCT,IPART1,IPART2,IFREE
    INTEGER       IMODE
    INTEGER       IOMODE,ICNTRL(20)
    LOGICAL       DONE,done0
    INTEGER       NCOORD,I,I1,I2
    real(chm_real)        RMAX
    LOGICAL       QWEIGHT, QIMAGE, QTWODIM, QRSYM
    !
    integer npart1,npart2, NATOMX
    integer iat1,iat2
    real(chm_real) dr,rho
    integer iungr,ir, nmax
    real(chm_real) rij
    real(chm_real) x1min,x1max,y1min,y1max,z1min,z1max
    real(chm_real) x2min,x2max,y2min,y2max,z2min,z2max
    real(chm_real) d2dfact,dnpart1
    real(chm_real) RDEF,RMARK

    ! General stuff needed for READCV
    INTEGER NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP
    INTEGER NFREAT,IUNIT,IFILE,ISTEP,ISTATS,NDEGF,NSAVV
    real(chm_real)  DELTA,DELTA2, dummy(NATOM)
    CHARACTER(len=4) :: HDR1='COOR',HDR2='CORD'
    real(chm_real),allocatable,dimension(:) :: RR,VR,GR,NR
    real(chm_real4),allocatable,dimension(:) :: R4TEMP

    DONE=.FALSE.

    call chmalloc('distrib.src','DISTRI','ISLCT', natomt,intg=ISLCT)
    call chmalloc('distrib.src','DISTRI','IPART1',natomt,intg=IPART1)
    call chmalloc('distrib.src','DISTRI','IPART2',natomt,intg=IPART2)
    call chmalloc('distrib.src','DISTRI','IFREE', natomt,intg=IFREE)

    ! Allocate
    call chmalloc('distrib.src','DISTRI','R4TEMP',NATOM,cr4=R4TEMP)
    call chmalloc('distrib.src','DISTRI','RR',DVECT,crl=RR)
    call chmalloc('distrib.src','DISTRI','VR',DVECT,crl=VR)
    call chmalloc('distrib.src','DISTRI','GR',DVECT,crl=GR)
    call chmalloc('distrib.src','DISTRI','NR',DVECT,crl=NR)

    WRITE(OUTU,*)
    WRITE(OUTU, '(6X,2A)') 'RADIAL DISTRIBUTION FUNCTION ANALYSIS SUBROUTINE'

    IUNGR  = GTRMI(COMLYN,COMLEN,'IUNG',OUTU)
    RDEF=6.10D0
    RMAX   = GTRMF(COMLYN,COMLEN,'RMAX',RDEF)
    RDEF=0.0334D0
    RHO    = GTRMF(COMLYN,COMLEN,'RHO',RDEF)
    RDEF=0.025D0
    DR     = GTRMF(COMLYN,COMLEN,'DR',RDEF)
    RMARK=9.9D9
    x1min  = GTRMF(COMLYN,COMLEN,'X1MI',-RMARK)
    x1max  = GTRMF(COMLYN,COMLEN,'X1MA',RMARK)
    y1min  = GTRMF(COMLYN,COMLEN,'Y1MI',-RMARK)
    y1max  = GTRMF(COMLYN,COMLEN,'Y1MA',RMARK)
    z1min  = GTRMF(COMLYN,COMLEN,'Z1MI',-RMARK)
    z1max  = GTRMF(COMLYN,COMLEN,'Z1MA',RMARK)
    x2min  = GTRMF(COMLYN,COMLEN,'X2MI',-RMARK)
    x2max  = GTRMF(COMLYN,COMLEN,'X2MA',RMARK)
    y2min  = GTRMF(COMLYN,COMLEN,'Y2MI',-RMARK)
    y2max  = GTRMF(COMLYN,COMLEN,'Y2MA',RMARK)
    z2min  = GTRMF(COMLYN,COMLEN,'Z2MI',-RMARK)
    z2max  = GTRMF(COMLYN,COMLEN,'Z2MA',RMARK)
    QWEIGHT = (INDXA(COMLYN, COMLEN, 'WEIG')  >  0)
    QIMAGE  = (INDXA(COMLYN, COMLEN, 'IMAG')  >  0)
    QTWODIM  = (INDXA(COMLYN, COMLEN, 'TWOD')  >  0)
    QRSYM  = (INDXA(COMLYN, COMLEN, 'RSYM')  >  0)
    if(QRSYM)  WRITE(outu,*) 'Symmetric Restrictions to selections.'
    d2dfact = 1.d0
    if(QTWODIM) then
       d2dfact = 0.d0
       WRITE(outu,*) 'Two dimensional RDF requested in XY plane.'
    endif

    WRITE(outu,*) 'output unit ',IUNGR
    WRITE(outu,*) 'rmax        ',RMAX
    WRITE(outu,*) 'density rho ',RHO
    WRITE(outu,*) 'delta r     ',DR
    IF(QWEIGHT) WRITE(OUTU,'(A)') 'THE WMAIN ARRAY WILL BE USED'


    IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
    CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOMT,1,IMODE, &
         .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEGT, &
         .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
    IF(IMODE /= 0)THEN
       CALL WRNDIE(0,'<SEL1AT>','ATOM SELECTION PARSING ERROR')
    ENDIF
    npart1=0
    IF(QIMAGE)THEN
       NATOMX=NATOMT
    ELSE
       NATOMX=NATOM
    ENDIF
    loop10: do i=1,natomx
       if(islct(i) == 1)then
          npart1=npart1+1
          ipart1(npart1)=i
       endif
    enddo loop10
    IF(npart1 /= 0)then
       write(outu,*) 'npart1 ',npart1
    ELSEIF(npart1 == 0)then
       WRITE(OUTU,*) 'The g(r) is generated with respect to the origin'
    ENDIF


    IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
    CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOMT,1,IMODE, &
         .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEGT, &
         .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
    IF(IMODE /= 0)THEN
       CALL WRNDIE(0,'<SEL1AT>','ATOM SELECTION PARSING ERROR')
    ENDIF
    !     CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    npart2=0
    do i=1,natomx
       if(islct(i) == 1)then
          npart2=npart2+1
          ipart2(npart2)=i
       endif
    enddo
    write(outu,*) 'npart2 ',npart2
    !
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP)

    ! Initialize average properties
    dummy = ZERO
    NCOORD=0
    IUNIT=FIRSTU
    NFREAT=NATOM
    ISTATS=1
    DONE=.FALSE.
    nmax=int(rmax/dr)
    if(nmax > dvect)then
       CALL WRNDIE(-4,'DISTRI>  ','NMAX LARGER THAN DVECT')
    endif
    ! Volume elements for 2 and 3d cases:
    if(QTWODIM) then
       rr(1)=dr
       vr(1)=pi*rr(1)**2
       nr(1)=0.0d0
       do i=2,nmax
          rr(i)=i*dr
          vr(i)=pi*(rr(i)**2-rr(i-1)**2)
          nr(i)=0.0d0
       enddo
    else
       rr(1)=dr
       vr(1)=4*pi/3*rr(1)**3
       nr(1)=0.0d0
       do i=2,nmax
          rr(i)=i*dr
          vr(i)=4*pi/3*(rr(i)**3-rr(i-1)**3)
          nr(i)=0.0d0
       enddo
    endif

    dnpart1=zero   ! To work out mean number of solutes considered.
    DONE0=(IUNIT == 0).OR.(NUNIT == 0)

    done_loop: do while(.not. done)
       done0_block: IF(DONE0)THEN
          write(OUTU,*) 'The main coordinate set is used'
          done=done0
       else done0_block

          !.......................................................................
          ! Main loop for reading trajectory file

          CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
               CG,QCG,                                 & 
#endif
               R4TEMP,NATOM,IFREE,NFREAT, &
               FIRSTU,NUNIT,IUNIT,IFILE, &
               ISTEP,ISTATS,NDEGF,DELTA2, &
               NBEGIN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
               TITLEB,NTITLB,.FALSE., dummy,.FALSE.)
          DONE=(ISTATS < 0)
       ENDIF done0_block
       DELTA=TIMFAC*DELTA2*NSKIP
       NCOORD=NCOORD+1
       IF(PRNLEV > 8) THEN
          WRITE(OUTU,'(1X,A,F8.3,1X,A,I8,1X,A,I8)') &
               'TIME = ',ISTEP*DELTA2*TIMFAC, &
               'NCOORD = ',NCOORD, &
               'STEP = ',ISTEP
       ENDIF

       ! Construct coordinates for all image atoms.
       ! Do it here so routines will work with images.
       !
       IF(NTRANS > 0) THEN
          CALL TRANSO(X,Y,Z,DX,DY,DZ,.TRUE.,QECONT,ECONT,NATOM,NTRANS, &
               IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR,NOROT,NATIM &
#if KEY_FLUCQ==1
               ,QFLUC,CG,FQCFOR  & 
#endif
               )
       ENDIF
       !
       IF(PRNLEV > 8) THEN
          do  i=1,NATOMT
             write(outu,'(1x,i4,3f10.5)') i,x(i),y(i),z(i)
          enddo
       ENDIF

       IF(npart1 /= 0)then
          loop31: do i1=1,npart1
             iat1=ipart1(i1)
             if(QRSYM) then
                if(abs(x(iat1)) < x1min .or. abs(x(iat1)) > x1max) cycle loop31
                if(abs(y(iat1)) < y1min .or. abs(y(iat1)) > y1max) cycle loop31
                if(abs(z(iat1)) < z1min .or. abs(z(iat1)) > z1max) cycle loop31
             else
                if(x(iat1) < x1min .or. x(iat1) > x1max) cycle loop31
                if(y(iat1) < y1min .or. y(iat1) > y1max) cycle loop31
                if(z(iat1) < z1min .or. z(iat1) > z1max) cycle loop31
             endif
             dnpart1=dnpart1+1.d0
             loop30: do i2=1,npart2
                iat2=ipart2(i2)
                if(QRSYM) then
                   if(abs(x(iat2)) < x2min .or. abs(x(iat2)) > x2max) cycle loop30
                   if(abs(y(iat2)) < y2min .or. abs(y(iat2)) > y2max) cycle loop30
                   if(abs(z(iat2)) < z2min .or. abs(z(iat2)) > z2max) cycle loop30
                else
                   if(x(iat2) < x2min .or. x(iat2) > x2max) cycle loop30
                   if(y(iat2) < y2min .or. y(iat2) > y2max) cycle loop30
                   if(z(iat2) < z2min .or. z(iat2) > z2max) cycle loop30
                endif
                if(abs(x(iat1)-x(iat2)) > rmax) cycle loop30
                if(abs(y(iat1)-y(iat2)) > rmax) cycle loop30
                if(abs(z(iat1)-z(iat2)) > rmax) cycle loop30
                rij = sqrt((x(iat1)-x(iat2))**2+ &
                     (y(iat1)-y(iat2))**2+ &
                     d2dfact*(z(iat1)-z(iat2))**2)

                IF(PRNLEV > 8)THEN
                   write(outu,*) iat1,iat2, 'rij ',rij
                   i=iat1
                   write(outu,'(1x,i4,3f10.5)') i,x(i),y(i),z(i)
                   i=iat2
                   write(outu,'(1x,i4,3f10.5)') i,x(i),y(i),z(i)
                ENDIF

                if(rij > rmax) cycle loop30
                ir=int(rij/dr)+1
                IF(QWEIGHT)THEN
                   nr(ir)=nr(ir)+wmain(iat2)
                ELSE
                   nr(ir)=nr(ir)+1.0D0
                ENDIF
             end do loop30
          end do loop31

       ELSEIF(npart1 == 0)THEN
          loop32: do i2=1,npart2
             iat2=ipart2(i2)
             if((abs(x(iat2))) > rmax) cycle loop32
             if((abs(y(iat2))) > rmax) cycle loop32
             if((abs(z(iat2))) > rmax) cycle loop32
             if(QRSYM) then
                if(abs(x(iat2)) < x2min .or. abs(x(iat2)) > x2max) cycle loop32
                if(abs(y(iat2)) < y2min .or. abs(y(iat2)) > y2max) cycle loop32
                if(abs(z(iat2)) < z2min .or. abs(z(iat2)) > z2max) cycle loop32
             else
                if(x(iat2) < x2min .or. x(iat2) > x2max) cycle loop32
                if(y(iat2) < y2min .or. y(iat2) > y2max) cycle loop32
                if(z(iat2) < z2min .or. z(iat2) > z2max) cycle loop32
             endif
             rij = sqrt((x(iat2))**2+ &
                  (y(iat2))**2+ &
                  d2dfact*(z(iat2))**2)

             if(rij > rmax) cycle loop32
             ir=int(rij/dr)+1
             IF(QWEIGHT)THEN
                nr(ir)=nr(ir)+wmain(iat2)
             ELSE
                nr(ir)=nr(ir)+1.0D0
             ENDIF
          end do loop32
       ENDIF
    enddo done_loop

    WRITE(OUTU,*) 'A TOTAL OF ',NCOORD,' FRAMES WILL BE USED'

    IF(npart1 /= 0)THEN
       dnpart1 = dnpart1 / real(ncoord)
       if(dnpart1 >=  1.d-6) then
          WRITE(OUTU,*) 'The N(r) is normalized by n_solutes=',dnpart1
          nr(1)=nr(1)/ncoord/dnpart1
          do ir=2,nmax
             nr(ir)=nr(ir)/ncoord/dnpart1
             gr(ir)=nr(ir)/vr(ir)/rho
             nr(ir)=nr(ir)+nr(ir-1)
          end do
       else
          WRITE(OUTU,*) 'The mean n_solutes < 1.d-6. Setting RDF to zero.'
          nr(2:nmax)=zero
          gr(2:nmax)=zero
          nr(2:nmax)=zero
       endif
    ELSEIF(npart1 == 0)THEN
       nr(1)=nr(1)/ncoord
       do ir=2,nmax
          nr(ir)=nr(ir)/ncoord
          gr(ir)=nr(ir)/vr(ir)/rho
          nr(ir)=nr(ir)+nr(ir-1)
       end do
    ENDIF

    do ir=2,nmax
       write(IUNGR,'(1x,4f15.8)') rr(ir)-dr/2,gr(ir),nr(ir)
    enddo


    call chmdealloc('distrib.src','DISTRI','ISLCT', natomt,intg=ISLCT)
    call chmdealloc('distrib.src','DISTRI','IPART1',natomt,intg=IPART1)
    call chmdealloc('distrib.src','DISTRI','IPART2',natomt,intg=IPART2)
    call chmdealloc('distrib.src','DISTRI','IFREE', natomt,intg=IFREE)
    call chmdealloc('distrib.src','DISTRI','R4TEMP',NATOM,cr4=R4TEMP)
    call chmdealloc('distrib.src','DISTRI','RR',DVECT,crl=RR)
    call chmdealloc('distrib.src','DISTRI','VR',DVECT,crl=VR)
    call chmdealloc('distrib.src','DISTRI','GR',DVECT,crl=GR)
    call chmdealloc('distrib.src','DISTRI','NR',DVECT,crl=NR)

    RETURN
  END SUBROUTINE DISTRI2

  SUBROUTINE  DISTRI3(NATOM,NATOMT,X,Y,Z,WMAIN,CG)

    !-----------------------------------------------------------------------
    !
    !  DIPOLE DISTRIBUTION ANALYSIS MAIN SUBROUTINE
    !

#if KEY_CHEQ==1
  use cheq,only:qcg                   
#endif

  use chm_kinds
  use bases_fcm
  use dimens_fcm
  use number
  use consta
  use exfunc
  use stream
  use string
  use comand
  use cvio, only: trjspc, readcv  
  use image
  use deriv
  use econtmod
  use ctitla
  use memory

    implicit none
    !-----------------------------------------------------------------------
    ! General
    INTEGER       NATOM, NATOMT
    real(chm_real)        X(*),Y(*),Z(*),WMAIN(*)
    real(chm_real)        CG(*)

    !-----------------------------------------------------------------------
    !  Miscelaneous Local variables
    LOGICAL       DONE,done0
    INTEGER       NCOORD,I,I1,I2,IUNGR
    real(chm_real)        RMIN,RMAX
    !
    integer       ncost,nsum
    real(chm_real)        ro,uxo,uyo,uzo,rhh,uxhh,uyhh,uzhh,cost
    real(chm_real)        nang(30)
    !

    ! General stuff needed for READCV
    integer,allocatable,dimension(:) :: ifree
    real(chm_real4),allocatable,dimension(:) :: r4temp
    INTEGER NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP
    INTEGER NFREAT,IUNIT,IFILE,ISTEP,ISTATS,NDEGF,NSAVV
    real(chm_real)  DELTA,DELTA2, dummy(NATOM)
    CHARACTER(len=4) :: HDR1='COOR', HDR2='CORD'

    dummy = ZERO
    call chmalloc('distrib.src','DISTRI','IFREE', natomt,intg=IFREE)
    call chmalloc('distrib.src','DISTRI','R4TEMP',NATOM,cr4=R4TEMP)

    WRITE(OUTU,*)
    WRITE(OUTU,100) 'DIPOLE DISTRIBUTION ANALYSIS SUBROUTINE'
100 FORMAT(6X,2A)

    IUNGR  = GTRMI(COMLYN,COMLEN,'IUNG',OUTU)
    RMIN   = GTRMF(COMLYN,COMLEN,'RMIN',ZERO)
    RMAX   = GTRMF(COMLYN,COMLEN,'RMAX',ZERO)

    WRITE(outu,*) 'output unit ',IUNGR
    WRITE(outu,*) 'rmax        ',RMAX
    WRITE(outu,*) 'rmin        ',RMIN

    !
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP)

    ! Initialize average properties
    NCOORD=0
    IUNIT=FIRSTU
    NFREAT=NATOM
    ISTATS=1
    DONE=.FALSE.
    do i=1,20
       nang(i)=0.0d0
    enddo

    DONE0=(IUNIT == 0).OR.(NUNIT == 0)
    done_loop: do while(.not. done)
       done0_block: IF(DONE0)THEN
          write(OUTU,*) 'The main coordinate set is used'
          done=done0
       else

          !.......................................................................
          ! Main loop for reading trajectory file
          CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
               CG,QCG,                               & 
#endif
               R4TEMP,NATOM,IFREE,NFREAT, &
               FIRSTU,NUNIT,IUNIT,IFILE, &
               ISTEP,ISTATS,NDEGF,DELTA2, &
               NBEGIN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
               TITLEB,NTITLB,.FALSE., dummy,.FALSE.)
          DONE=(ISTATS < 0)
       ENDIF done0_block

       DELTA=TIMFAC*DELTA2*NSKIP
       NCOORD=NCOORD+1
       IF(PRNLEV > 8) THEN
          WRITE(OUTU,'(1X,A,F8.3,1X,A,I8,1X,A,I8)') &
               'TIME = ',ISTEP*DELTA2*TIMFAC, &
               'NCOORD = ',NCOORD, &
               'STEP = ',ISTEP
       ENDIF

       IF(PRNLEV > 8) THEN
          do i=1,NATOMT
             write(6,'(1x,i4,3f10.5)') i,x(i),y(i),z(i)
          enddo
       ENDIF

       loop90: do i=1,NATOMT,3
          ro = sqrt(x(i)**2+y(i)**2+z(i)**2)
          if(ro > rmax.or.ro <= rmin) cycle loop90
          uxo=x(i)/ro
          uyo=y(i)/ro
          uzo=z(i)/ro
          uxhh=(x(i+1)+x(i+2))/2.0-x(i)
          uyhh=(y(i+1)+y(i+2))/2.0-y(i)
          uzhh=(z(i+1)+z(i+2))/2.0-z(i)
          rhh= sqrt(uxhh*uxhh+uyhh*uyhh+uzhh*uzhh)
          uxhh=uxhh/rhh
          uyhh=uyhh/rhh
          uzhh=uzhh/rhh
          cost=uxo*uxhh+uyo*uyhh+uzo*uzhh+1.0
          ncost=int(cost/0.1)+1
          nang(ncost)=nang(ncost)+1.0
       enddo loop90
    enddo done_loop

    WRITE(OUTU,*) 'A TOTAL OF ',NCOORD,' FRAMES WILL BE USED'

    nsum=0.0
    do i=1,20
       nsum=nsum+nang(i)
    enddo
    do i=1,20
       write(iungr,'(2f10.3)') (i-10.0)/10.0-0.05,nang(i)/nsum
    enddo

    call chmdealloc('distrib.src','DISTRI','R4TEMP',NATOM,cr4=R4TEMP)
    call chmdealloc('distrib.src','DISTRI','IFREE', natomt,intg=IFREE)
    RETURN
  END SUBROUTINE DISTRI3

end module distrib_module

module distmatrix

contains

  SUBROUTINE DSTMT0
  use chm_kinds
  use dimens_fcm
  use exfunc
  use comand
  use stream
  use string
  use number
  use psf
  use coord
  use memory
  use select, only: selcta

    implicit none
    integer,allocatable,dimension(:) :: ISLCT,IPART1,IPART2
    !
    real(chm_real4),allocatable,dimension(:) :: R4TEMP
    real(chm_real),allocatable,dimension(:) :: DSTMT
    integer,allocatable,dimension(:) :: IFREE
    !
    !-----------------------------------------------------------------------
    !
    INTEGER      IUNMT
    !     real(chm_real)       RMAX
    integer i,npart1,npart2
    logical QAVER


    call chmalloc('distrib.src','DSTMT0','ISLCT', natom,intg=ISLCT)
    call chmalloc('distrib.src','DSTMT0','IPART1',natom,intg=IPART1)
    call chmalloc('distrib.src','DSTMT0','IPART2',natom,intg=IPART2)

    IUNMT  = GTRMI(COMLYN,COMLEN,'IUNM',OUTU)
    !     RMAX   = GTRMF(COMLYN,COMLEN,'RMAX',HUNDRD)
    QAVER = .TRUE. .or. (INDXA(COMLYN, COMLEN, 'AVER')  >  0)
    IF(INDXA(COMLYN, COMLEN, 'SQUA')  >  0) QAVER = .FALSE.

    if(qaver)then
       write(outu,'(6X,2A)') 'Matrix of average distance will be calculated'
    else
       write(outu,'(6X,2A)') &
            'Matrix of average distance squared will be calculated'
    endif

    ! select the first group of atoms
    CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    npart1=0
    do i=1,natom
       if(islct(i) == 1)then
          npart1=npart1+1
          ipart1(npart1)=i
       endif
    enddo
    write(outu,*) 'npart1 ',npart1

    ! select the second group of atoms
    CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    npart2=0
    do i=1,natom
       if(islct(i) == 1)then
          npart2=npart2+1
          ipart2(npart2)=i
       endif
    enddo
    write(outu,*) 'npart2 ',npart2
    write(outu,*) 'matrix of',npart1*npart2

    ! Allocate 
    call chmalloc('distrib.src','DSTMT1','IFREE',MAXA,intg=IFREE)
    call chmalloc('distrib.src','DSTMT1','R4TEMP',NATOM,cr4=R4TEMP)
    call chmalloc('distrib.src','DSTMT1','DSTMT',NPART1*NPART2+1,crl=DSTMT)

    CALL DSTMT2(NATOM,X,Y,Z,WMAIN,QAVER, &
         IPART1,NPART1,IPART2,NPART2,IUNMT, &
         IFREE,R4TEMP,DSTMT,CG)

    ! Free 
    call chmdealloc('distrib.src','DSTMT1','IFREE',MAXA,intg=IFREE)
    call chmdealloc('distrib.src','DSTMT1','R4TEMP',NATOM,cr4=R4TEMP)
    call chmdealloc('distrib.src','DSTMT1','DSTMT',NPART1*NPART2+1,crl=DSTMT)

    call chmdealloc('distrib.src','DSTMT0','ISLCT', natom,intg=ISLCT)
    call chmdealloc('distrib.src','DSTMT0','IPART1',natom,intg=IPART1)
    call chmdealloc('distrib.src','DSTMT0','IPART2',natom,intg=IPART2)

    RETURN
  END SUBROUTINE DSTMT0

  SUBROUTINE DSTMT2(NATOM,X,Y,Z,WMAIN,QAVER, &
       IPART1,NPART1,IPART2,NPART2,IUNMT, &
       IFREE,R4TEMP,DSTMT,CG)

    !-----------------------------------------------------------------------
    !
    !  DISTANCE MATRIX FOR AVERAGE AND RMS FLUCTUATIONS OF DISTANCES
    !

#if KEY_CHEQ==1
  use cheq,only:  qcg                       
#endif

  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use exfunc
  use stream
  use comand
  use cvio, only: trjspc, readcv
  use ctitla
    implicit none
    !-----------------------------------------------------------------------
    ! General
    INTEGER       NATOM
    real(chm_real)        X(*),Y(*),Z(*),WMAIN(*)
    LOGICAL       QAVER
    INTEGER       IPART1(*),NPART1,IPART2(*),NPART2
    integer       iunmt
    INTEGER       IFREE(*)
    REAL(CHM_REAL4)        R4TEMP(*)
    INTEGER       DVECT
    real(chm_real)        DSTMT(*)
    real(chm_real)        CG(*)

    !-----------------------------------------------------------------------
    !  Miscelaneous Local variables
    INTEGER       IOMODE,ICNTRL(20)
    LOGICAL       DONE,done0
    INTEGER       NCOORD,I,I1,I2
    !
    integer iat1,iat2,idstmt
    real(chm_real) rij

    ! General stuff needed for READCV
    INTEGER NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP
    INTEGER NFREAT,IUNIT,IFILE,ISTEP,ISTATS,NDEGF,NSAVV
    real(chm_real)  DELTA,DELTA2, dummy(NATOM)
    CHARACTER(len=4) :: HDR1='COOR',HDR2='CORD'

    dummy = ZERO
    DONE=.FALSE.
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP)

    ! Initialize average properties
    NCOORD=0
    IUNIT=FIRSTU
    NFREAT=NATOM
    ISTATS=1
    DONE=.FALSE.

    dstmt(1:npart1*npart2)=zero

    !.......................................................................
    ! Main loop for reading trajectory file
    do while(.not. done)
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CG,QCG,                               & 
#endif
            R4TEMP,NATOM,IFREE,NFREAT, &
            FIRSTU,NUNIT,IUNIT,IFILE, &
            ISTEP,ISTATS,NDEGF,DELTA2, &
            NBEGIN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
            TITLEB,NTITLB,.FALSE., dummy,.FALSE.)
       DELTA=TIMFAC*DELTA2*NSKIP
       NCOORD=NCOORD+1
       DONE=(ISTATS < 0)
       IF(PRNLEV > 8)THEN
          WRITE(OUTU,'(1X,A,F8.3,1X,A,I8,1X,A,I8)') &
               'TIME = ',ISTEP*DELTA2*TIMFAC, &
               'NCOORD = ',NCOORD, &
               'STEP = ',ISTEP
       ENDIF

       IF(PRNLEV > 8)THEN
          do i=1,natom
             write(outu,'(1x,i4,3f10.5)') i,x(i),y(i),z(i)
          enddo
       ENDIF

       do i1=1,npart1
          iat1=ipart1(i1)
          do i2=1,npart2
             iat2=ipart2(i2)

             idstmt=i2+(i1-1)*npart2

             ! squared distance
             rij = (x(iat1)-x(iat2))**2+ &
                  (y(iat1)-y(iat2))**2+ &
                  (z(iat1)-z(iat2))**2

             IF(QAVER)THEN
                rij=sqrt(rij)
             ENDIF

             dstmt(idstmt)=dstmt(idstmt)+rij

             IF(PRNLEV > 8)THEN
                write(outu,*) iat1,iat2, 'rij ',rij, 'idstmt',idstmt
                write(outu,*) 'dstmt ',dstmt(idstmt)
                write(outu,'(1x,i4,3f10.5)') iat1,x(iat1),y(iat1),z(iat1)
                write(outu,'(1x,i4,3f10.5)') iat2,x(iat2),y(iat2),z(iat2)
             ENDIF
          enddo
       enddo
    enddo

    WRITE(OUTU,*) 'A TOTAL OF ',NCOORD,' FRAMES WILL BE USED'
    do i1=1,npart1
       do i2=1,npart2
          idstmt=i2+(i1-1)*npart2
          dstmt(idstmt)=dstmt(idstmt)/ncoord
          iat1=ipart1(i1)
          iat2=ipart2(i2)
          write(iunmt,*) idstmt,dstmt(idstmt),iat1,iat2
       enddo
    enddo

    RETURN
  END SUBROUTINE DSTMT2

  SUBROUTINE PROFIL
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use comand
  use psf
  use stream
  use string
  use coord
#if KEY_CHEQ==1
  use cheq,only: qcg               
#endif
  use cvio, only: trjspc, readcv
  use number
  use consta
  use comand
  use ctitla
  use memory
  use select, only: selcta

    implicit none
  
    !-----------------------------------------------------------------------
    !  Miscelaneous Local variables
    INTEGER       IOMODE,ICNTRL(20)
    LOGICAL       DONE,done0
    INTEGER       NCOORD,I,I1,I2
    !
    real(chm_real)        ZMAX, ZMIN
    integer npart1,npart2
    integer iat1,iat2
    real(chm_real) dz,area,wtot,zref
    integer iungz,iz, nmax, izs
    logical qtrans
    logical QWEIGHT, QSYMM

    ! General stuff needed for READCV
    INTEGER NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP
    INTEGER NFREAT,IUNIT,IFILE,ISTEP,ISTATS,NDEGF,NSAVV
    real(chm_real)  DELTA,DELTA2, dummy(NATOM)
    CHARACTER(len=4) :: HDR1='COOR',HDR2='CORD'

    !-----------------------------------------------------------------------
    ! Temporary arrays to read trajectory files
    real(chm_real4),allocatable,dimension(:) :: R4TEMP
    real(chm_real),allocatable,dimension(:) :: ZZ, GZ
    integer,allocatable,dimension(:) :: ISLCT,IPART1,IPART2,IFREE
    !
    !-----------------------------------------------------------------------
    !
    INTEGER      DVECT

    DVECT  = GTRMI(COMLYN,COMLEN,'DVEC',5000)
    ! Allocate 
    call chmalloc('distrib.src','PROFIL','ISLCT', natom,intg=ISLCT)
    call chmalloc('distrib.src','PROFIL','IPART1',natom,intg=IPART1)
    call chmalloc('distrib.src','PROFIL','IPART2',natom,intg=IPART2)
    call chmalloc('distrib.src','PROFIL','IFREE', natom,intg=IFREE)

    call chmalloc('distrib.src','PROFIL','R4TEMP',NATOM,cr4=R4TEMP)
    call chmalloc('distrib.src','PROFIL','ZZ',DVECT,crl=ZZ)
    call chmalloc('distrib.src','PROFIL','GZ',DVECT,crl=GZ)

    dummy = ZERO
    DONE=.FALSE.

    WRITE(OUTU,*)
    WRITE(OUTU,100) 'DENSITY PROFILE ANALYSIS SUBROUTINE'
100 FORMAT(6X,2A)

    IUNGZ  = GTRMI(COMLYN,COMLEN,'IUNG',OUTU)
    ZMIN   = GTRMF(COMLYN,COMLEN,'ZMIN',-THIRTY)
    ZMAX   = GTRMF(COMLYN,COMLEN,'ZMAX',THIRTY)
    DZ     = GTRMF(COMLYN,COMLEN,'DZ',ONE)
    AREA   = GTRMF(COMLYN,COMLEN,'AREA',ONE)
    QTRANS = (INDXA(COMLYN, COMLEN, 'TRAN')  >  0)
    QWEIGHT = (INDXA(COMLYN, COMLEN, 'WEIG')  >  0)
    QSYMM   = (INDXA(COMLYN, COMLEN, 'SYMM')  >  0)

    WRITE(OUTU,'(6X,A,F8.3)') 'ZMAX ',ZMAX
    WRITE(OUTU,'(6X,A,F8.3)') 'ZMIN ',ZMIN
    WRITE(OUTU,'(6X,A,F8.3)') 'DZ   ',DZ
    WRITE(OUTU,'(6X,A,F8.3)') 'AREA ',AREA
    IF(QWEIGHT) WRITE(OUTU,'(A)') 'THE WMAIN ARRAY WILL BE USED'

    ! Selection:  the atoms for computing the profile along Z
    CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    npart1=0
    do i=1,natom
       if(islct(i) == 1)then
          npart1=npart1+1
          ipart1(npart1)=i
       endif
    enddo
    write(outu,*) 'Density Profile computed for ',npart1,' atoms'

    zref=zero
    IF(QTRANS)THEN
       write(outu,100) 'Global translation of the coordinates along Z'
       write(outu,100) 'weighted according to the WMAIN array'
       ! Second selection:  the atoms for resetting the Z-coordinate
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       npart2=0
       Wtot=zero
       do i=1,natom
          if(islct(i) == 1)then
             npart2=npart2+1
             ipart2(npart2)=i
             Wtot=Wtot+wmain(i)
             zref=zref+z(i)*wmain(i)
          endif
       enddo
       write(outu,*) 'Wtot   ',Wtot
       zref=zref/Wtot
       write(outu,*) 'Zref   ',Zref
    ENDIF

    CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,NBEGIN,NSKIP,NSTOP)

    ! Initialize average properties
    NCOORD=0
    IUNIT=FIRSTU
    NFREAT=NATOM
    ISTATS=1
    DONE=.FALSE.
    nmax=int((zmax-zmin)/dz)
    if(nmax > dvect)then
       CALL WRNDIE(-4,'PROFIL>  ','NMAX LARGER THAN DVECT')
    endif
    do iz=1,nmax
       ZZ(iz)=zmin+iz*dZ
       GZ(iz)=ZERO
    enddo
    !
    DONE0=(IUNIT == 0).OR.(NUNIT == 0)

    done_loop: do while(.not. done)
       done0_block: IF(DONE0)THEN
          WRITE(OUTU,*) 'No unit was specified ',nunit
          WRITE(OUTU,*) 'The main coordinate set is used'
       else done0_block

          !.......................................................................
          ! Main loop for reading trajectory file
          CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
               CG,QCG,                                & 
#endif
               R4TEMP,NATOM,IFREE,NFREAT, &
               FIRSTU,NUNIT,IUNIT,IFILE, &
               ISTEP,ISTATS,NDEGF,DELTA2, &
               NBEGIN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
               TITLEB,NTITLB,.FALSE., dummy,.FALSE.)
          DONE=(ISTATS < 0)
       ENDIF done0_block
       DELTA=TIMFAC*DELTA2*NSKIP
       NCOORD=NCOORD+1

       if(qtrans)then
          zref=zero
          do i2=1,npart2
             iat2=ipart2(i2)
             zref=zref+z(iat2)*wmain(iat2)
          enddo
          zref=zref/wtot
       ENDIF

       IF(PRNLEV > 5)THEN
          WRITE(OUTU,'(1X,A,F8.3,1X,A,I8,1X,A,I8,1X,A,F8.3)') &
               'TIME = ',ISTEP*DELTA2*TIMFAC, &
               'NCOORD = ',NCOORD, &
               'STEP = ',ISTEP, &
               'ZREF = ',ZREF
       ENDIF

       IF(PRNLEV > 8)THEN
          do i=1,natom
             write(outu,'(1x,i4,4f10.5)') i,x(i),y(i),z(i),wmain(i)
          enddo
       ENDIF

       loop30: do i1=1,npart1
          iat1=ipart1(i1)
          if((z(iat1)-zref) > zmax) cycle loop30
          iz=int(((z(iat1)-zref)-zmin)/dz)+1
          if(QSYMM) izs=int(((-z(iat1)-zref)-zmin)/dz)+1
          if(QWEIGHT)then
             gz(iz)=gz(iz)+wmain(iat1)
             if(QSYMM) gz(izs)=gz(izs)+wmain(iat1)
          else
             gz(iz)=gz(iz)+ONE
             if(QSYMM) gz(izs)=gz(izs)+ONE
          endif
       enddo loop30
    enddo done_loop

    WRITE(OUTU,*) 'A TOTAL OF ',NCOORD,' FRAMES WILL BE USED'
    do iz=1,nmax
       gz(iz)=gz(iz)/(ncoord*area*dz)
       if(QSYMM) gz(iz)=gz(iz)/2
    enddo
    do iz=1,nmax
       WRITE(IUNGZ,'(1x,3f12.7)') zz(iz)-dz/2,gz(iz)
    enddo

    call chmdealloc('distrib.src','PROFIL','ISLCT', natom,intg=ISLCT)
    call chmdealloc('distrib.src','PROFIL','IPART1',natom,intg=IPART1)
    call chmdealloc('distrib.src','PROFIL','IPART2',natom,intg=IPART2)
    call chmdealloc('distrib.src','PROFIL','IFREE', natom,intg=IFREE)
            
    call chmdealloc('distrib.src','PROFIL','R4TEMP',NATOM,cr4=R4TEMP)
    call chmdealloc('distrib.src','PROFIL','ZZ',DVECT,crl=ZZ)
    call chmdealloc('distrib.src','PROFIL','GZ',DVECT,crl=GZ)

    RETURN
  END SUBROUTINE PROFIL
end module distmatrix

