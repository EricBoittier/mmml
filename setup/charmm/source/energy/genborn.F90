module genborn

  use cff_fcm
  use chm_kinds
  use gb_common, only:p,r_gb,vol_gb,alph_gb,sigx_gb,sigy_gb, &
     sigz_gb,t_gb,qanalys,eps_gb,cutalph,gb_atm,igentype

  implicit none

  !  Note the polarization energy Gpol is given by:
  !                                                     q q
  !                              N   N                   i j
  !    G   =  -C  (1-1/eps){1/2 sum sum ------------------------------------ }
  !     pol     el              i=1 j=1 [r^2 + alpha *alpha exp(-D  )]^(0.5)
  !                                       ij        i      j      ij
  !
  !
  !  GB_BLOCK    =   Heap pointer to NBLOCK length array containing the GB energy
  !                  which are related to the atoms belong to block n
  !
  !  GB_LAMB     =   Heap pointer to NATOM*NBLOCK lenght array containing the
  !                  atom-based GB energy when IGenType = 1.
  !                  Caustion : GB_LAMB is used only when IgenType = 1.
  !
  !  GBLDM       =   Heap pointer to NBLOCK length array containing the partial
  !                  terms which are used to get the first derivative of
  !                  lambda in lambda-dynamics.
  !                  Caustion : GBLDM is used only when IGenType = 1 and
  !                             QLDM is true.
  !
  !  QGenBorn    =   Logical flag for GB calculations.
  !
  !  D_Aij       =   Heap pointer to NAngl length array containing angle
  !                  distances needed in GB force calculation.

  Integer :: IAlph_GB, IR_GB, IVol_GB, d_Aij
  !--------------variables moved to pert/block.src---------------------
  ! real(chm_real),allocatable,dimension(:),save :: gb_block,gb_lamb, &
  !      gbldm
  !----------------------------------------------------------------------
  Logical QGenBorn

  integer :: alloc_err

contains

  subroutine genborn_init()
    QGenBorn = .false.
    return
  end subroutine genborn_init

  Subroutine Genborn_Set(comlyn, comlen)
    !-----------------------------------------------------------------------
    !  This routine reads in and sets up the data structures for GB
    !  calculations.

  use stream
  use string
  use dimens_fcm
  use psf
  use memory
  use number
  use coord
#if KEY_BLOCK==1
  use block_fcm
  use lambdam
#endif /*  BLOCK*/
  use inbnd

    Character(len=*) :: Comlyn
    Integer Comlen
    Logical QWeight
    !RS...B980923.rs Move the INTEGER statement above the DATA statement
    Integer i
    Character(len=2) :: C(6)
    Data C /'P1','P2','P3','P4','P5','P6'/

    If (IndxA(Comlyn, Comlen,'CLEA') > 0) then
       If ( .not. QGenBorn ) Then
          Call WrnDie(-1,'<genborn.src>GENBORN_SET', &
               'Called Clear w/o GBorn being active')
          Return
       Else
          QGenBorn = .false.
       Endif
       ! Clear up heap space and return

       If ( Prnlev  >=  2 ) then
          Write(Outu,'(A)')' Clearing Generalized Born Arrays'
       Endif
#if KEY_BLOCK==1
       IF (QBLOCK) THEN
          IF (IGenType  == 1) THEN
             IGenType = 0
             call chmdealloc('genborn.src','Genborn_Set','gb_block',nblock,crl=gb_block)
!ldm
             IF (QLDM) THEN
                call chmdealloc('genborn.src','Genborn_Set','gb_lamb',natom*nblock,crl=gb_lamb)
                call chmdealloc('genborn.src','Genborn_Set','gbldm',nblock,crl=gbldm)
             ENDIF
! LDM

          ELSE IF (IGenType  ==  2) THEN
             IGenType = 0
             call chmdealloc('genborn.src','Genborn_Set','r_gb',natom,crl=r_gb)
             call chmdealloc('genborn.src','Genborn_Set','vol_gb',natom,crl=vol_gb)
             call chmdealloc('genborn.src','Genborn_Set','alph_gb',natom*nblock,crl=alph_gb)

             call chmdealloc('genborn.src','Genborn_Set','sigx_gb',natom*nblock,crl=sigx_gb)
             call chmdealloc('genborn.src','Genborn_Set','sigy_gb',natom*nblock,crl=sigy_gb)
             call chmdealloc('genborn.src','Genborn_Set','sigz_gb',natom*nblock,crl=sigz_gb)
             call chmdealloc('genborn.src','Genborn_Set','t_gb',natom*nblock,crl=t_gb)
             call chmdealloc('genborn.src','Genborn_Set','gb_block',nblock,crl=gb_block)

             If ( QAnalys ) Then
                call chmdealloc('genborn.src','Genborn_Set','gb_atm',natom,crl=gb_atm)
             Else
                call chmdealloc('genborn.src','Genborn_Set','gb_atm',1,crl=gb_atm)
             Endif

             QAnalys = .false.

             Return
          ENDIF
       ENDIF
#endif /*  BLOCK*/

       call chmdealloc('genborn.src','Genborn_Set','r_gb',natom,crl=r_gb)
       call chmdealloc('genborn.src','Genborn_Set','vol_gb',natom,crl=vol_gb)
       call chmdealloc('genborn.src','Genborn_Set','alph_gb',natom,crl=alph_gb)

       call chmdealloc('genborn.src','Genborn_Set','sigx_gb',natom,crl=sigx_gb)
       call chmdealloc('genborn.src','Genborn_Set','sigy_gb',natom,crl=sigy_gb)
       call chmdealloc('genborn.src','Genborn_Set','sigz_gb',natom,crl=sigz_gb)
       call chmdealloc('genborn.src','Genborn_Set','t_gb',natom,crl=t_gb)

       If ( QAnalys ) Then
          call chmdealloc('genborn.src','Genborn_Set','gb_atm',natom,crl=gb_atm)
       Else
          call chmdealloc('genborn.src','Genborn_Set','gb_atm',1,crl=gb_atm)
       Endif

       QAnalys = .false.
       Return

    Endif

    QGenBorn = .true.

    IGenType = GTRMI(COMLYN,COMLEN,'GBTY',0)

    Eps_GB = GtrmF(Comlyn, Comlen, 'EPSILON', Eighty)
    Cutalph = GtrmF(Comlyn, Comlen, 'CUTA', Mega)
    QWeight = (IndxA(Comlyn, Comlen,'WEIG') > 0)
    QAnalys = (IndxA(Comlyn, Comlen,'ANAL') > 0)

    Do i = 1, 5
       P(i) = GtrmF(Comlyn, Comlen, C(i), Zero)
    Enddo
    P(6) = GtrmF(Comlyn, Comlen, 'LAMBDA', One)

    If ( Prnlev  >=  2 ) then
       Write(Outu,'(A)') &
            ' Generalized Born energy and force terms  will be used'
       Write(Outu,'(A)') &
            ' The parameters are (P(6) = Lambda)'
       Write(Outu,'(3(A,A,f12.5,2x))') &
            (C(i),' =',P(i),i=1,6)
       Write(Outu,'(A, f12.5)') &
            ' The solvent dielectric constant is eps = ', Eps_GB
       Write(Outu,'(A, f15.5)') &
            ' The maximum allowed GB radius, alpha = ', CutAlph
       If (Qweight) then
          Write(Outu,'(A)') &
               ' Radii for taken from WMAIN'
       Endif
       If (QAnalys) then
          Write(Outu,'(A)') &
               'Atomic contributions available in GBAtm through scalar commands'
       Endif
    Endif

#if KEY_BLOCK==1
    IF (QBLOCK) THEN
       IF(IGenType  ==  1) THEN
          call chmalloc('genborn.src','Genborn_Set','gb_block',nblock,crl=gb_block)

!ldm
          IF (QLDM) THEN
             call chmalloc('genborn.src','Genborn_Set','gb_lamb',natom*nblock,crl=gb_lamb)
             call chmalloc('genborn.src','Genborn_Set','gbldm',nblock,crl=gbldm)
          ENDIF
! LDM
       ELSE IF(IGenType  ==  2) THEN
          call chmalloc('genborn.src','Genborn_Set','r_gb',natom,crl=r_gb)
          call chmalloc('genborn.src','Genborn_Set','vol_gb',natom,crl=vol_gb)
          call chmalloc('genborn.src','Genborn_Set','alph_gb',natom*nblock,crl=alph_gb)

          call chmalloc('genborn.src','Genborn_Set','sigx_gb',natom*nblock,crl=sigx_gb)
          call chmalloc('genborn.src','Genborn_Set','sigy_gb',natom*nblock,crl=sigy_gb)
          call chmalloc('genborn.src','Genborn_Set','sigz_gb',natom*nblock,crl=sigz_gb)
          call chmalloc('genborn.src','Genborn_Set','t_gb',natom*nblock,crl=t_gb)
          call chmalloc('genborn.src','Genborn_Set','gb_block',nblock,crl=gb_block)

          If ( QAnalys ) Then
             call chmalloc('genborn.src','Genborn_Set','gb_atm',natom,crl=gb_atm)
          Else
             call chmalloc('genborn.src','Genborn_Set','gb_atm',1,crl=gb_atm)
          Endif

          Call Fill_GB(R_GB, Vol_GB, WMain, &
               QWeight,IGenType)

          RETURN
       ENDIF
    ENDIF
#endif /*  BLOCK*/

    ! Allocate heap space for needed arrays

    call chmalloc('genborn.src','Genborn_Set','r_gb',natom,crl=r_gb)
    call chmalloc('genborn.src','Genborn_Set','vol_gb',natom,crl=vol_gb)
    call chmalloc('genborn.src','Genborn_Set','alph_gb',natom,crl=alph_gb)

    call chmalloc('genborn.src','Genborn_Set','sigx_gb',natom,crl=sigx_gb)
    call chmalloc('genborn.src','Genborn_Set','sigy_gb',natom,crl=sigy_gb)
    call chmalloc('genborn.src','Genborn_Set','sigz_gb',natom,crl=sigz_gb)
    call chmalloc('genborn.src','Genborn_Set','t_gb',natom,crl=t_gb)

    If ( QAnalys ) Then
       call chmalloc('genborn.src','Genborn_Set','gb_atm',natom,crl=gb_atm)
    Else
       call chmalloc('genborn.src','Genborn_Set','gb_atm',1,crl=gb_atm)
    Endif

    ! Fill static arrays

    Call Fill_GB(R_GB, Vol_GB, WMain, &
         QWeight,IGenType)

    Return
  End Subroutine Genborn_Set

  Subroutine Fill_GB(R, V, W, QWei, GBType)
  use number
  use consta
  use dimens_fcm
  use ffieldm
  use mmffm
  use cff_fcm
  use param
  use psf
  use stream
  use code

    real(chm_real) R(*), V(*), W(*)
    Integer I, itmp, mtmp
    Logical Qwei
    Integer GBType

    Do I=1,Natom
       If (Qwei) then
          R(I) = W(I)
#if KEY_MMFF==1
       ElseIf (FFIELD  ==  mmff) Then
          R(I) = RSTAR(MTYPE(I)*(MTYPE(I)+1)/2) / Two
          W(I) = R(I)
#endif 
#if KEY_CFF==1
       ElseIf (FFIELD  ==  cff) Then
          itmp = ITC(IAC(I))
          mtmp = MNO(itmp, itmp)
          ! XXX why not CNBD1/CNBD2?
          R(I) = (NINE * CNBA(mtmp) / (SIX * CNBB(mtmp)))**(Third) / Two
          W(I) = R(I)
#endif 
       Else
          R(I) = VDWR(ITC(IAC(I)))
       Endif

       V(I) = FOUR*PI*R(I)**3/THREE
    Enddo

#if KEY_MMFF==1 || KEY_CFF==1
    If ( Prnlev  >=  2 ) then
       If ( FFIELD  ==  mmff ) Write(Outu,'(A)') &
            ' FFIELD=MMFF active, Atom vdW radii put in wmain'
       If ( FFIELD  ==  cff ) Write(Outu,'(A)') &
            ' FFIELD=CFF active, Atom vdW radii put in wmain'
    Endif
#endif 
    Return
  End Subroutine Fill_GB


  Real(chm_real) Function Alphij(r2, Ri, Rj, P, P5)

  use consta
  use number

    real(chm_real) r2, Ri, Rj, P, P5, r4, C, Rcomp

    C = ONE
    If ( P5  >  ZERO ) then
       Rcomp = r2 / ( (Ri+Rj)*(Ri+Rj) )
       If ( Rcomp  <=  (ONE/P5) ) then
          C = - Cos(Rcomp*P5*PI)
          C = C + ONE
          C = HALF * C
          C = C * C
       Endif
    Endif

    r4 = r2 * r2
    Alphij = P * C / r4

    Return
  End Function Alphij

  real(chm_real) Function Sigij(r, Ri, Rj, P, P5)

  use consta
  use number

    real(chm_real) r, r2, r4, Ri, Rj, P, P5, C, Rcomp, dC

    r2 = r * r
    C = ONE
    dC = Zero
    If ( P5  >  ZERO ) then
       Rcomp = r2 / ( (Ri+Rj)*(Ri+Rj) )
       If ( Rcomp  <=  (ONE/P5) ) then
          C = - Cos(Rcomp*P5*PI)
          C = C + ONE
          dC = C*P5*Pi*Rcomp*Sin(Rcomp*P5*Pi)/r
          C = HALF * C
          C = C * C
       Endif
    Endif

    r4 = r2 * r2
    Sigij = P * dC / ( r4 * r ) &
         - Four * P * C / ( r4 * r2 )

    Return
  End Function Sigij

  Subroutine GBSolv(GBEnr, X, Y, Z, DX, DY, DZ, &
       JNbl, INbl, Inbl14, INblo14, &
       QInte, Islct, Jslct &
#if KEY_GBFIXAT==1
       ,GbFixed                                & 
#endif
       )
  use number
  use consta
  use dimens_fcm
  use stream
  use psf
  use inbnd
  use parallel
  use gbswit, only: qgbswit, louter, c2ofnb, gbswit_setup, es_switch
    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*)
    Real(chm_real) GBEnr, X(*), Y(*), Z(*), DX(*), DY(*), DZ(*)
    Integer ISlct(*), JSlct(*)
#if KEY_GBFIXAT==1
    Integer GbFixed                            
#endif
    Logical QInte


    ! Local variables
    Integer i, ic, j, k
    real(chm_real) rij, xij, yij, zij, Sqrtrij
    real(chm_real) Dij, Sij, SqrtSij, Sij3half, expDij
    real(chm_real) Aij, Aji, SigijDif &
#if KEY_GBFIXAT==0
         ,Bij, Bji  
#else
         ;          
#endif
    real(chm_real) Factor, Dxi, Dyi, Dzi, GBtmp
    Logical LOK

    Integer ITemp, NPr, jpr

    real(chm_real) FSw, DFSw

#if KEY_GBINLINE==1
    real(chm_real) dAij, Ratio 
#endif

    GBEnr = Zero
    Factor = CCELEC * ( One - One / Eps_gb )

    call gbswit_setup()

    ! Loop over bonded atoms first:

    Do ic = MynodP, Nbond, Numnod
       LOK = .true.
       i = ib(ic)
       j = jb(ic)

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then
          ! Check if both in pair are fixed!
          If ( .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then
             xij = x(i) - x(j)
             yij = y(i) - y(j)
             zij = z(i) - z(j)

             rij = xij * xij + yij * yij + zij * zij
             Sqrtrij = Sqrt(rij)

             SigijDif = Sigij(Sqrtrij,r_gb(i),r_gb(j),P(2),Zero)
             ! Subtract contributions from non-bonded terms to correct for over counting
             SigijDif=SigijDif-Sigij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))

             If ( imove(i)  ==  0 ) then
                Aij = &
                     + Factor * ( Cg(j) * Cg(j) + T_GB(j) ) &
                     * SiGijDif * VOL_GB(i) / CCELEC


                DX(i) = DX(i) + xij * Aij
                DY(i) = DY(i) + yij * Aij
                DZ(i) = DZ(i) + zij * Aij
             Endif

             If ( imove(j)  ==  0 ) then
                Aji = &
                     + Factor * ( Cg(i) * Cg(i) + T_GB(i) ) &
                     * SiGijDif * VOL_GB(j) / CCELEC

                DX(j) = DX(j) - xij * Aji
                DY(j) = DY(j) - yij * Aji
                DZ(j) = DZ(j) - zij * Aji
             Endif

          Endif
       Endif

    Enddo

    ! Next atoms connected through angles

    Do ic = MynodP, NTheta, NumNod

       LOK = .true.
       i = it(ic)
       j = kt(ic)

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then

          ! Check if both in pair are fixed!
          If ( .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then

             xij = x(i) - x(j)
             yij = y(i) - y(j)
             zij = z(i) - z(j)

             rij = xij * xij + yij * yij + zij * zij
             Sqrtrij = Sqrt(rij)

             SiGijDif = Sigij(Sqrtrij,r_gb(i),r_gb(j),P(3),Zero)
             ! Subtract contributions from non-bonded terms to correct for over counting
             SigijDif=SigijDif-Sigij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))

             If ( imove(i)  ==  0 ) then
                Aij = &
                     + Factor * ( Cg(j) * Cg(j) + T_GB(j) ) &
                     * SiGijDif * VOL_GB(i) / CCELEC


                DX(i) = DX(i) + xij * Aij
                DY(i) = DY(i) + yij * Aij
                DZ(i) = DZ(i) + zij * Aij
             Endif

             If ( imove(j)  ==  0 ) then
                Aji = &
                     + Factor * ( Cg(i) * Cg(i) + T_GB(i) ) &
                     * SiGijDif * VOL_GB(j) / CCELEC

                DX(j) = DX(j) - xij * Aji
                DY(j) = DY(j) - yij * Aji
                DZ(j) = DZ(j) - zij * Aji
             Endif

          Endif
       Endif
    Enddo

    ! Now we need to get the atoms in the exclusion list
    Itemp = 0
    ! Define the atom bounds for this processor.
    Do i = MyNodP, Natom, NumNod

       If (i  /=  1 ) ITemp = INblo14(i - 1)
       Dxi = Zero
       Dyi = Zero
       Dzi = Zero
       GBtmp = Zero

       NPr = INblo14(i) - ITemp
       Do jpr = 1, NPr
          LOK = .true.
          k = Inbl14(Itemp+jpr)
          j = Abs(k)
          If ( k  >  0 .and. &
               .not. ( imove(i)  ==  1 &
               .and. imove(j)  ==  1 ) ) then
             If ( QInte ) LOK = &
                  ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
             If (LOK) Then

                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)

                rij = xij * xij + yij * yij + zij * zij
                Sqrtrij = Sqrt(rij)

                SiGijDif = SiGij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))

                Aij = Factor * ( Cg(j) * Cg(j) + T_GB(j) ) &
                     * SiGijDif * VOL_GB(i) / CCELEC

                Aji = Factor * ( Cg(i) * Cg(i) + T_GB(i) ) &
                     * SiGijDif * VOL_GB(j) / CCELEC

#if KEY_GBFIXAT==0
                Bij = Zero                       
#endif
#if KEY_GBFIXAT==0
                Bji = Zero                       
#endif

                If ( CG(j) * CG(i)  /=  Zero ) then

                   Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                   expDij = exp(-Dij)

                   Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                   SqrtSij = Sqrt(Sij)
                   Sij3half = Sij * SqrtSij

                   Sij = CG(i) * CG(j) * Factor / Sij3half

                   Aij = Aij + Sij - Sij * PT25 * expDij
#if KEY_GBFIXAT==0
                   Bij = Sij * Alph_Gb(i) * expDij * &
                        ( Alph_Gb(i) * Alph_Gb(j) &
                        + PT25 * rij ) / CCELEC
                   Bji = Sij * Alph_Gb(j) * expDij * &
                        ( Alph_Gb(j) * Alph_Gb(i) &
                        + PT25 * rij ) / CCELEC
#endif 
                   Aji = Sij - Sij * PT25 * expDij &
                        + Factor * ( Cg(i) * Cg(i) + T_GB(i) ) &
                        * SiGijDif * VOL_GB(j) / CCELEC

                   GBtmp = GBtmp - Sij * SqrtSij * SqrtSij

                   If (Qanalys) Then
                      GB_Atm(i) = GB_Atm(i) - &
                           Half * Sij * SqrtSij * SqrtSij
                      GB_Atm(j) = GB_Atm(j) - &
                           Half * Sij * SqrtSij * SqrtSij
                   Endif

                Endif

                If ( imove(i)  ==  0 ) then
                   Dxi = Dxi + xij * Aij &
#if KEY_GBFIXAT==0
                        + Sigx_Gb(i) * Bij  
#endif
#if KEY_GBFIXAT==1
                        ;                   
#endif
                   Dyi = Dyi + yij * Aij &
#if KEY_GBFIXAT==0
                        + Sigy_Gb(i) * Bij  
#endif
#if KEY_GBFIXAT==1
                        ;                   
#endif
                   Dzi = Dzi + zij * Aij &
#if KEY_GBFIXAT==0
                        + Sigz_Gb(i) * Bij  
#endif
#if KEY_GBFIXAT==1
                        ;                   
#endif
                Endif

                If ( imove(j)  ==  0 ) then
                   DX(j) = DX(j) - xij * Aji &
#if KEY_GBFIXAT==0
                        + Sigx_Gb(j) * Bji  
#endif
#if KEY_GBFIXAT==1
                        ;                   
#endif
                   DY(j) = DY(j) - yij * Aji &
#if KEY_GBFIXAT==0
                        + Sigy_Gb(j) * Bji  
#endif
#if KEY_GBFIXAT==1
                        ;                   
#endif
                   DZ(j) = DZ(j) - zij * Aji &
#if KEY_GBFIXAT==0
                        + Sigz_Gb(j) * Bji  
#endif
#if KEY_GBFIXAT==1
                        ;                   
#endif
                Endif

             Endif
          Endif
       Enddo

       If (NPr > 0) then
          GBEnr = GBEnr + GBtmp

          DX(i) = DX(i) + Dxi
          DY(i) = DY(i) + Dyi
          DZ(i) = DZ(i) + Dzi
       Endif

    Enddo

    ! Finally do pairs of nonbonded terms
    ! For lists use different construct here

    Itemp = 0
#if KEY_IMCUBES==1
    if(lbycbim)then
       if( (INBL(Natom)-INBL(Natom+natom))  > 0)then
          write(outu,*) INBL(Natom),INBL(Natom-1)
          Call WrnDie(-3,'<genborn.f>GBSOLV', &
               'GBsolv (genborn.src), last atom has pairs')
       endif
    endif
#endif
    Do i = 1, Natom-1
#if KEY_IMCUBES==1
       if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif

       Dxi = Zero
       Dyi = Zero
       Dzi = Zero
       GBtmp = Zero

       NPr = INbl(i) - ITemp
       Do jpr = 1, NPr
          k = JNbl(Itemp+jpr)
          j = Abs(k)

          xij = x(i) - x(j)
          yij = y(i) - y(j)
          zij = z(i) - z(j)

          rij = xij * xij + yij * yij + zij * zij

          If (rij  <  C2OfNb) Then
             call es_switch(rij, FSw, DfSw) ! electrostatic switch func

             Sqrtrij = Sqrt(rij)
#if KEY_GBINLINE==0
             SiGijDif = SiGij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))
#else /**/
             dAij = Zero
             Aij = One
             Ratio = rij /  &
                  (( r_gb(i) + r_gb(j) ) * ( r_gb(i) + r_gb(j) ))
             If ( Ratio  <=  One/P(5) ) then
                Aij = Aij - Cos( Ratio * P(5) * Pi )
                dAij = dAij + Aij &
                     * P(5) * Pi * Ratio * Sin( Ratio * P(5) * Pi ) &
                     / Sqrtrij
                Aij = Half * Aij
                Aij = Aij * Aij
             Endif
             Aij = P(4) * Aij / ( rij * rij )
             SiGijDif = P(4) * dAij / ( rij * rij * Sqrtrij ) &
                  - Four * Aij /  rij
#endif 

             Aij = Factor * ( Cg(j) * Cg(j) + T_GB(j) ) &
                  * SiGijDif * VOL_GB(i) / CCELEC

             Aji = Factor * ( Cg(i) * Cg(i) + T_GB(i) ) &
                  * SiGijDif * VOL_GB(j) / CCELEC

#if KEY_GBFIXAT==0
             Bij = Zero             
#endif
#if KEY_GBFIXAT==0
             Bji = Zero             
#endif
             Sij = Zero

             If ( CG(j) * CG(i)  /=  Zero ) then

                Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                expDij = exp(-Dij)

                Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                SqrtSij = Sqrt(Sij)
                Sij3half = Sij * SqrtSij

                Sij = CG(i) * CG(j) * Factor / Sij3half * FSw

                Aij = Aij + Sij - Sij * PT25 * expDij

#if KEY_GBFIXAT==0
                Bij = Sij * Alph_Gb(i) * expDij *  &
                     ( Alph_Gb(i) * Alph_Gb(j) &
                     + PT25 * rij ) / CCELEC
                Bji = Sij * Alph_Gb(j) * expDij *  &
                     ( Alph_Gb(j) * Alph_Gb(i) &
                     + PT25 * rij ) / CCELEC

#endif 
                Aji = Sij - Sij * PT25 * expDij &
                     + Factor * ( Cg(i) * Cg(i) + T_GB(i) ) &
                     * SiGijDif * VOL_GB(j) / CCELEC

                Sij = Sij * SqrtSij * SqrtSij

             Endif

             if ( qgbSwit .and. louter ) then
                Aij = Aij  - Sij * DFSw
                Aji = Aji  - Sij * DFSw
             endif

             GBtmp = GBtmp - Sij

             If (Qanalys) Then
                GB_Atm(i) = GB_Atm(i) - Half * Sij
                GB_Atm(j) = GB_Atm(j) - Half * Sij
             Endif

             If ( imove(i)  ==  0 ) then
                Dxi = Dxi + xij * Aij &
#if KEY_GBFIXAT==0
                     + Sigx_Gb(i) * Bij  
#endif
#if KEY_GBFIXAT==1
                     ;                   
#endif
                Dyi = Dyi + yij * Aij &
#if KEY_GBFIXAT==0
                     + Sigy_Gb(i) * Bij  
#endif
#if KEY_GBFIXAT==1
                     ;                   
#endif
                Dzi = Dzi + zij * Aij &
#if KEY_GBFIXAT==0
                     + Sigz_Gb(i) * Bij  
#endif
#if KEY_GBFIXAT==1
                     ;                   
#endif
             Endif

             If ( imove(j)  ==  0 ) then
                DX(j) = DX(j) - xij * Aji &
#if KEY_GBFIXAT==0
                     + Sigx_Gb(j) * Bji  
#endif
#if KEY_GBFIXAT==1
                     ;                   
#endif
                DY(j) = DY(j) - yij * Aji &
#if KEY_GBFIXAT==0
                     + Sigy_Gb(j) * Bji  
#endif
#if KEY_GBFIXAT==1
                     ;                   
#endif
                DZ(j) = DZ(j) - zij * Aji &
#if KEY_GBFIXAT==0
                     + Sigz_Gb(j) * Bji  
#endif
#if KEY_GBFIXAT==1
                     ;                   
#endif
             Endif

          Endif
       Enddo

       If (NPr > 0) then
          GBEnr = GBEnr + GBtmp

          DX(i) = DX(i) + Dxi
          DY(i) = DY(i) + Dyi
          DZ(i) = DZ(i) + Dzi
       Endif

       ITemp = INbl(i)
    Enddo
#if KEY_GBFIXAT==1 /*gb_fixed_atoms*/
    !-----------------------------------------------------------------------
    ! Add the GB terms of fixed atom - fixed atom
    !
    ! ***  Caution  ***
    ! All GB terms of fixed atom - fixed atom are assumed to be nonbonded.
    ! Therefore, CtOnNb shoulb be longer than 1-2, 1-3 interactions
    If ( GbFixed  >  1 ) Then
       call gbswit_setup()
       Do i = 1, Natom-1
          IF (CG(I)  /=  ZERO) THEN
             GBtmp = Zero
             Do j = i+1, Natom
                IF (CG(J)  /=  ZERO) THEN

                   If ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) THEN

                      xij = x(i) - x(j)
                      yij = y(i) - y(j)
                      zij = z(i) - z(j)
                      rij = xij * xij + yij * yij + zij * zij
                      If (rij  <  c2ofnb) Then
                         call es_switch(rij, FSw) ! electrostatic switch func
                         Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                         expDij = exp(-Dij)
                         Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                         SqrtSij = Sqrt(Sij)

                         Sij = CG(i) * CG(j) * Factor / SqrtSij * FSw

                         GBtmp = GBtmp - Sij
                         If (Qanalys) Then
                            GB_Atm(i) = GB_Atm(i) - Half * Sij
                            GB_Atm(j) = GB_Atm(j) - Half * Sij
                         Endif
                      EndIf
                   EndIF
                ENDIF
             Enddo
             GBEnr = GBEnr + GBtmp
          ENDIF
       Enddo

    ENDIF
#endif /* (gb_fixed_atoms)*/

    ! Finally add in self-terms

    ! Define the atom bounds for this processor.
    Do i = MynodP, Natom, NumNod

       LOK = .true.
       If ( Qinte ) LOK = Islct(i)  ==  1
       If ( LOK ) Then
          If ( Cg(i)  /=  Zero &
#if KEY_GBFIXAT==0
               .and. Imove(i)  /=  1 &  
#endif
               ) then

             GBEnr = GBEnr - CG(i) * CG(i) * Factor * Half/Alph_Gb(i)

             If (Qanalys) Then
                GB_Atm(i) = GB_Atm(i) &
                     - CG(i) * CG(i) * Factor * Half / Alph_Gb(i)
             Endif

#if KEY_GBFIXAT==1
             IF (imove(i)  /=  1 ) THEN
                DX(i) = DX(i) &
                     + Factor*(CG(i)*CG(i)+T_GB(i))*Sigx_Gb(i)/CCELEC
                DY(i) = DY(i) &
                     + Factor*(CG(i)*CG(i)+T_GB(i))*Sigy_Gb(i)/CCELEC
                DZ(i) = DZ(i) &
                     + Factor*(CG(i)*CG(i)+T_GB(i))*Sigz_Gb(i)/CCELEC

             Endif
#else /**/
             DX(i) = DX(i) &
                  + Factor * CG(i) * CG(i) * Sigx_Gb(i) / CCELEC
             DY(i) = DY(i) &
                  + Factor * CG(i) * CG(i) * Sigy_Gb(i) / CCELEC
             DZ(i) = DZ(i) &
                  + Factor * CG(i) * CG(i) * Sigz_Gb(i) / CCELEC
#endif 
          Endif    ! cg(i) /= 0
       Endif        ! LOK

    Enddo

    ! Note need to GCOMB GB_Atm(i) if QAnalys true.
#if KEY_PARALLEL==1 /*paraend*/
    ! Combine GB_Atm
    If(Qanalys) Call GComb(GB_atm, Natom)
#endif /* (paraend)*/
    Return
  End Subroutine GBSolv

  Subroutine FillAlpGB( X, Y, Z, &
       JNbl, INbl, Inbl14, INblo14, &
       QInte, Islct, JSlct &
#if KEY_GBFIXAT==1
       ,GbFixed                                & 
#endif
       )
  use number
  use consta
  use dimens_fcm
  use stream
  use psf
  use inbnd
  use parallel
#if KEY_PARALLEL==1
  use memory        
#endif

  use gbswit, only: gbswit_setup, es_switch

    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*), &
         Islct(*), Jslct(*)

#if KEY_PARALLEL==1
    Integer AtFrst, AtLast
#endif 
    Real(chm_real) X(*), Y(*), Z(*)
    Logical Qinte
#if KEY_PARALLEL==1
    Real(chm_real),allocatable,dimension(:) :: Temp 
#endif

    ! Local variables
    Integer i, ic, j, k
    integer GbFixed
    real(chm_real) rij, xij, yij, zij, Aij, Tij
    real(chm_real) Dij, Sij, SqrtSij, Sij3half, expDij
    real(chm_real) C2OfNb
    Logical LOK
    Integer ITemp, NPr, jpr

    real(chm_real) FSw

#if KEY_GBINLINE==1
    real(chm_real) dAij, Ratio 
#endif

    ! Function declarations
    ! real(chm_real) Alphij, Sigij

    call gbswit_setup()

    GbFixed = 0
    Do i = 1, Natom
       Alph_Gb(i) = Zero
       Sigx_Gb(i)  = Zero
       Sigy_Gb(i)  = Zero
       Sigz_Gb(i)  = Zero
       T_GB(i)     = Zero
       If ( QAnalys ) GB_Atm(i) = Zero
       If ( imove(i)  ==  1 ) GbFixed = GbFixed + 1
    Enddo

    If ( GbFixed  >  0 ) Then
#if KEY_GBFIXAT==0
       Call WrnDie(-3,'<FillAlpGB>', &
            'Generalized Born not implemented w/ fixed atoms')
#else /**/
       Call WrnDie(0,'<FillAlpGB>', &
            'Generalized Born runs slowly w/ fixed atoms')
#endif 
    Endif

    ! Loop over bonded atoms first:

    Do ic = MynodP, Nbond, Numnod

       LOK = .true.
       i = ib(ic)
       j = jb(ic)

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then

          xij = x(i) - x(j)
          yij = y(i) - y(j)
          zij = z(i) - z(j)

          rij = xij * xij + yij * yij + zij * zij

          Aij = Alphij(rij, r_gb(i), r_gb(j), P(2), Zero)
          ! Subtract contributions from non-bonded terms to correct for over counting

          Aij = Aij - Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))

          Alph_Gb(i) = Alph_Gb(i) + Aij * VOL_GB(j)
          Alph_Gb(j) = Alph_Gb(j) + Aij * VOL_GB(i)

          rij = sqrt ( rij )

          ! Now do sigma contributions
          Aij = Sigij(rij, r_gb(i), r_gb(j), P(2), Zero)
          ! Subtract contributions from non-bonded terms to correct for over counting
          Aij = Aij - Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))

          If ( imove(i)  ==  0 ) then
             Sigx_Gb(i) = Sigx_Gb(i) + Aij * VOL_GB(j) * xij
             Sigy_Gb(i) = Sigy_Gb(i) + Aij * VOL_GB(j) * yij
             Sigz_Gb(i) = Sigz_Gb(i) + Aij * VOL_GB(j) * zij
          Endif

          If ( imove(j)  ==  0 ) then
             Sigx_Gb(j) = Sigx_Gb(j) - Aij * VOL_GB(i) * xij
             Sigy_Gb(j) = Sigy_Gb(j) - Aij * VOL_GB(i) * yij
             Sigz_Gb(j) = Sigz_Gb(j) - Aij * VOL_GB(i) * zij
          Endif

       Endif

    Enddo

    ! Next atoms connected through angles

    Do ic = MynodP, NTheta, NumNod

       LOK = .true.
       i = it(ic)
       j = kt(ic)

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then
          xij = x(i) - x(j)
          yij = y(i) - y(j)
          zij = z(i) - z(j)

          rij = xij * xij + yij * yij + zij * zij

          Aij = Alphij(rij, r_gb(i), r_gb(j), P(3), Zero)

          ! Subtract contributions from non-bonded terms to correct for over counting
          Aij = Aij - Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))

          Alph_Gb(i) = Alph_Gb(i) + Aij * VOL_GB(j)
          Alph_Gb(j) = Alph_Gb(j) + Aij * VOL_GB(i)

          rij = sqrt ( rij )

          ! Now do sigma contributions
          Aij = Sigij(rij, r_gb(i), r_gb(j), P(3), Zero)

          ! Subtract contributions from non-bonded terms to correct for over counting
          Aij = Aij - Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))

          If ( imove(i)  ==  0 ) then
             Sigx_Gb(i) = Sigx_Gb(i) + Aij * VOL_GB(j) * xij
             Sigy_Gb(i) = Sigy_Gb(i) + Aij * VOL_GB(j) * yij
             Sigz_Gb(i) = Sigz_Gb(i) + Aij * VOL_GB(j) * zij
          Endif

          If ( imove(j)  ==  0 ) then
             Sigx_Gb(j) = Sigx_Gb(j) - Aij * VOL_GB(i) * xij
             Sigy_Gb(j) = Sigy_Gb(j) - Aij * VOL_GB(i) * yij
             Sigz_Gb(j) = Sigz_Gb(j) - Aij * VOL_GB(i) * zij
          Endif

       Endif
    Enddo

    ! Do atoms on exclusion list
    Itemp = 0
    ! Define the atom bounds for this processor.
    Do i = MyNodP, Natom, NumNod

       If (i  /=  1 ) ITemp = INblo14(i - 1)
       NPr = INblo14(i) - ITemp
       Do jpr = 1, NPr

          LOK = .true.
          k = Inbl14(Itemp+jpr)
          j = Abs(k)
          ! Don't do fixed atoms from exclusion list here, get them below.
          If ( k  >  0 &
               .and. .not. (imove(i)  ==  1 &
               .and. imove(j)  ==  1) ) then
             If ( QInte ) LOK = &
                  ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
             If (LOK) Then
                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)

                rij = xij * xij + yij * yij + zij * zij

                Aij =  Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))

                Alph_Gb(i) = Alph_Gb(i) + Aij * VOL_GB(j)
                Alph_Gb(j) = Alph_Gb(j) + Aij * VOL_GB(i)

                rij = sqrt ( rij )

                ! Now do sigma contributions
                Aij =  Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))

                If ( imove(i)  ==  0 ) then
                   Sigx_Gb(i) = Sigx_Gb(i) + Aij * VOL_GB(j) * xij
                   Sigy_Gb(i) = Sigy_Gb(i) + Aij * VOL_GB(j) * yij
                   Sigz_Gb(i) = Sigz_Gb(i) + Aij * VOL_GB(j) * zij
                Endif

                If ( imove(j)  ==  0 ) then
                   Sigx_Gb(j) = Sigx_Gb(j) - Aij * VOL_GB(i) * xij
                   Sigy_Gb(j) = Sigy_Gb(j) - Aij * VOL_GB(i) * yij
                   Sigz_Gb(j) = Sigz_Gb(j) - Aij * VOL_GB(i) * zij
                Endif

             Endif
          Endif
       Enddo

    Enddo

    c2ofnb = ctofnb * ctofnb
    ! For lists us different construct here
    Itemp = 0
    Do i = 1, Natom-1
#if KEY_IMCUBES==1
       if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif

       NPr = INbl(i) - ITemp
       Do jpr = 1, NPr
          k = JNbl(Itemp+jpr)
          j = Abs(k)
          xij = x(i) - x(j)
          yij = y(i) - y(j)
          zij = z(i) - z(j)

          rij = xij * xij + yij * yij + zij * zij

          If (rij  <  c2ofnb) Then

#if KEY_GBINLINE==0
             Aij =  Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
             Aij = One
             Ratio = rij /  &
                  (( r_gb(i) + r_gb(j) ) * ( r_gb(i) + r_gb(j) ))
             If ( Ratio  <=  One/P(5) ) then
                Aij = Aij - Cos( Ratio * P(5) * Pi )
                Aij = Half * Aij
                Aij = Aij * Aij
             Endif
             Aij = P(4) * Aij / ( rij * rij )
#endif 

             Alph_Gb(i) = Alph_Gb(i) + Aij * VOL_GB(j)
             Alph_Gb(j) = Alph_Gb(j) + Aij * VOL_GB(i)

             rij = sqrt ( rij )

             ! Now do sigma contributions
#if KEY_GBINLINE==0
             Aij =  Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
             dAij = Zero
             If ( Ratio  <=  One/P(5) ) then
                dAij = dAij + (  One - Cos( Ratio * P(5) * Pi ) ) &
                     * P(5) * Pi * Ratio * Sin( Ratio * P(5) * Pi ) / rij
             Endif
             Aij = P(4) * dAij / ( rij * rij * rij * rij * rij ) &
                  - Four * Aij / ( rij * rij )
#endif 

             If ( imove(i)  ==  0 ) then
                Sigx_Gb(i) = Sigx_Gb(i) + Aij * VOL_GB(j) * xij
                Sigy_Gb(i) = Sigy_Gb(i) + Aij * VOL_GB(j) * yij
                Sigz_Gb(i) = Sigz_Gb(i) + Aij * VOL_GB(j) * zij
             Endif

             If ( imove(j)  ==  0 ) then
                Sigx_Gb(j) = Sigx_Gb(j) - Aij * VOL_GB(i) * xij
                Sigy_Gb(j) = Sigy_Gb(j) - Aij * VOL_GB(i) * yij
                Sigz_Gb(j) = Sigz_Gb(j) - Aij * VOL_GB(i) * zij
             Endif

          Endif
       Enddo

       ITemp = INbl(i)
    Enddo

    ! If we use lists and there are fixed atoms, add contributions from
    ! fixed atom lists to Alpha, Sigx_Gb(Y,Z).
    If ( GbFixed  >  1 ) Then
       C2OfNB = CtOfNb * CtOfNb
       Do i = 1, Natom-1
          Do j = i+1, Natom

             If ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) then

                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)

                rij = xij * xij + yij * yij + zij * zij

                If (rij  <  c2ofnb) Then
#if KEY_GBINLINE==0
                   Aij =  Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                   Aij = One
                   Ratio = rij /  &
                        (( r_gb(i) + r_gb(j) ) * ( r_gb(i) + r_gb(j) ))
                   If ( Ratio  <=  One/P(5) ) then
                      Aij = Aij - Cos( Ratio * P(5) * Pi )
                      Aij = Half * Aij
                      Aij = Aij * Aij
                   Endif
                   Aij = P(4) * Aij / ( rij * rij )
#endif 

                   Alph_Gb(i) = Alph_Gb(i) + Aij * VOL_GB(j)
                   Alph_Gb(j) = Alph_Gb(j) + Aij * VOL_GB(i)

                Endif
             Endif
          Enddo
       Enddo

    Endif

#if KEY_PARALLEL==1 /*paralpha*/
    ! Get Alpha's updated on all processors
    Call GComb(Alph_gb, Natom)
#endif /* (paralpha)*/
    Do i = 1, Natom

       Alph_Gb(i) = Alph_Gb(i) + CCELEC*P(1)/(Two*r_gb(i)*r_gb(i)) &
            - CCELEC/(Two*r_gb(i)*P(6))

       Alph_Gb(i) = - CCELEC / ( Alph_Gb(i) * Two )

       If ( Alph_Gb(i)  <=  Zero )  Alph_Gb(i) = CutAlph
       If ( Alph_Gb(i)  >  CutAlph )  Alph_Gb(i) = CutAlph

    Enddo

    ! ***Calculation of T***
    ! First get the atoms on exclusion lists
    Itemp = 0
    ! Define the atom bounds for this processor.
    Do i = MynodP, Natom, NumNod
       If (i  /=  1 ) ITemp = INblo14(i - 1)
       NPr = INblo14(i) - Itemp
       If ( Cg(i)  /=  Zero .and. Npr  >  0 ) then

          Tij = Zero
          Do jpr = 1, NPr
             LOK = .true.
             k = Inbl14(Itemp+jpr)
             j = Abs(k)
             If ( ( Cg(j)  /=  Zero .and. k  >  0 ) .and. &
                  .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then

                If ( QInte ) LOK = &
                     ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
                If (LOK) Then
                   xij = x(i) - x(j)
                   yij = y(i) - y(j)
                   zij = z(i) - z(j)

                   rij = xij * xij + yij * yij + zij * zij
                   Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                   expDij = exp(-Dij)

                   Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                   SqrtSij = Sqrt(Sij)
                   Sij3half = Sij * SqrtSij

                   Sij = CG(i) * CG(j) / Sij3half

#if KEY_GBFIXAT==0
                   If (imove(i) == 0)  & 
#endif
                   Tij = Tij  &
                        + Sij * Alph_Gb(i) * expDij*  &
                        ( Alph_Gb(i) * Alph_Gb(j)  &
                        + PT25 * rij )

#if KEY_GBFIXAT==0
                   If (imove(j) == 0)  & 
#endif
                   T_GB(j) = T_GB(j)  &
                        + Sij * Alph_Gb(j) * expDij*  &
                        ( Alph_Gb(j) * Alph_Gb(i)  &
                        + PT25 * rij )

                Endif
             Endif
          Enddo

          T_GB(i) = T_GB(i) + Tij
       Endif
    Enddo
    ! Now non-bonded contributions
    call gbswit_setup()
    Itemp = 0
    Do i = 1, Natom-1
#if KEY_IMCUBES==1
       if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif
       NPr = INbl(i) - ITemp
       If ( Cg(i)  /=  Zero .and. Npr  >  0 ) then

          Tij = Zero
          Do jpr = 1, NPr
             k = JNbl(Itemp+jpr)
             j = Abs(k)

             If ( Cg(j)  /=  Zero ) then
                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)

                rij = xij * xij + yij * yij + zij * zij
                If (rij  <  c2ofnb) Then
                   call es_switch(rij, FSw) ! Electrostatic Switch function
                   Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                   expDij = exp(-Dij)

                   Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                   SqrtSij = Sqrt(Sij)
                   Sij3half = Sij * SqrtSij

                   Sij = CG(i) * CG(j) / Sij3half * FSw

#if KEY_GBFIXAT==0
                   If (imove(i) == 0)  & 
#endif
                   Tij = Tij  &
                        + Sij * Alph_Gb(i) * expDij*  &
                        ( Alph_Gb(i) * Alph_Gb(j)  &
                        + PT25 * rij )

#if KEY_GBFIXAT==0
                   If (imove(j) == 0)  & 
#endif
                   T_GB(j) = T_GB(j)  &
                        + Sij * Alph_Gb(j) * expDij*  &
                        ( Alph_Gb(j) * Alph_Gb(i)  &
                        + PT25 * rij )

                Endif
             Endif
          Enddo
          T_GB(i) = T_GB(i) + Tij
       Endif
       ITemp = INbl(i)
    Enddo
#if KEY_GBFIXAT==1 /*fixed_atom_contr*/

    !------------------------------------------------------------------
    ! contribution from fixed atom - fixed atom to T_GB(i)
    ! *** CAUTION  ***
    ! In this loop, 1-2, 1-3 interactions are also imposed to the swich function,
    ! Therefore, CtOnNb shoulb be longer than the longest 1-2, or 1-3 interactions.
    !
    If ( GbFixed  >  1 ) Then
       call gbswit_setup()
       Do i = 1, Natom-1
          IF (CG(I)  /=  ZERO) THEN
             Tij = Zero
             Do j = i+1, Natom
                IF(CG(J)  /=  ZERO) THEN

                   If ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) then

                      xij = x(i) - x(j)
                      yij = y(i) - y(j)
                      zij = z(i) - z(j)
                      rij = xij * xij + yij * yij + zij * zij

                      If (rij  <  c2ofnb) Then
                         Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                         expDij = exp(-Dij)

                         Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                         SqrtSij = Sqrt(Sij)
                         Sij3half = Sij * SqrtSij

                         call es_switch(rij, FSw) ! Electrostatic Switch func
                         Sij = CG(i) * CG(j) / Sij3half * FSw

                         Tij = Tij &
                              + Sij * Alph_Gb(i) * expDij*  &
                              ( Alph_Gb(i) * Alph_Gb(j) &
                              + PT25 * rij )

                         T_GB(j) = T_GB(j) &
                              + Sij * Alph_Gb(j) * expDij*  &
                              ( Alph_Gb(j) * Alph_Gb(i) &
                              + PT25 * rij )

                      Endif
                   Endif
                ENDIF
             Enddo
             T_GB(i) = T_GB(i) + Tij
          ENDIF
       ENDDO
    Endif
#endif /*  (fixed_atom_contr)*/

#if KEY_PARALLEL==1 /*paraend*/
    call chmalloc('genborn.src','FillAlpGB','temp',4*natom,crl=temp)
    Do i = 1, Natom
       Temp(i) = T_GB(i)
       Temp(Natom + i) = Sigx_Gb(i)
       Temp(2*Natom + i) = Sigy_Gb(i)
       Temp(3*Natom + i) = Sigz_Gb(i)
    Enddo
    ! Combine T and Sigmas
    Call GComb(Temp, 4*Natom)
    Do i = 1, Natom
       T_GB(i) = Temp(i)
       Sigx_Gb(i) = Temp(Natom + i)
       Sigy_Gb(i) = Temp(2*Natom + i)
       Sigz_Gb(i) = Temp(3*Natom + i)
    Enddo
    call chmdealloc('genborn.src','FillAlpGB','temp',4*natom,crl=temp)
#endif /* (paraend)*/
    Return
  End Subroutine FillAlpGB

#if KEY_BLOCK==1 /*block_main*/
  !==================================================================
  !==================================================================
  !==================================================================
  !
  Subroutine GBSolv1(GBEnr, X, Y, Z, DX, DY, DZ, &
       JNbl, INbl, Inbl14, INblo14, &
       QInte, Islct, Jslct, GbFixed)
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
    ! Generalized Born with Block (e.g. Lambda-dynamics, FEP, LEP)
    ! Type 1 ( Born radius depend on the coefficient of block)
    ! S.B. 10/30/99
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  use number
  use consta
  use dimens_fcm
  use stream
  use psf
  use block_fcm
  use inbnd
  use parallel
  use lambdam
  use gbswit, only: qgbswit, louter, c2ofnb, gbswit_setup, es_switch

    implicit none

    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*)
    real(chm_real) GBEnr, X(*), Y(*), Z(*), DX(*), DY(*), DZ(*)
    Integer ISlct(*), JSlct(*), GbFixed
    Logical QInte
    real(chm_real) Self

    ! Local variables
    Integer i, ic, j, k
    real(chm_real) rij, xij, yij, zij, Sqrtrij
    real(chm_real) Dij, Sij, SqrtSij, Sij3half, expDij
    real(chm_real) Aij, Aji, SigijDif
    real(chm_real) Factor, Dxi, Dyi, Dzi, GBtmp
    Logical LOK
    INTEGER IBL, JBL, KBL, KK
    real(chm_real) COEF, CGI2, CGJ2, CGIJ
    real(chm_real) fsw, dfsw ! gbswit es_switch return vals
    Integer ITemp, NPr, jpr

#if KEY_GBINLINE==1
    real(chm_real) dAij, Ratio 
#endif

    ! Initialized energy terms
    GBEnr = Zero
    DO i = 1, NBLOCK
       GB_BLOCK(i) = Zero
    ENDDO
#if KEY_BLOCK==1 /*ldm*/
    IF (QLDM) THEN
       DO i = 1, NBLOCK
          GBLDM(i) = Zero
       ENDDO
    ENDIF
#endif /*  LDM*/

    Factor = CCELEC * ( One - One / Eps_gb )

    call gbswit_setup()
 
    ! Loop over bonded atoms first:
    !-------------------------------------------------------------------------

    Do ic = MynodP, Nbond, Numnod
       LOK = .true.
       i = ib(ic)
       j = jb(ic)
       !CCCCCCCCCCCCC
       ! getting the number of block and coefficient
       IBL = IBLCKP(I)
       JBL = IBLCKP(J)
       IF (JBL < IBL) THEN
          COEF = BLCOEB(JBL+IBL*(IBL-1)/2)
       ELSE
          COEF = BLCOEB(IBL+JBL*(JBL-1)/2)
       ENDIF
       !CCCCCCCCCCCCC

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then
          ! Check if both in pair are fixed!
          If ( .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then
             xij = x(i) - x(j)
             yij = y(i) - y(j)
             zij = z(i) - z(j)

             rij = xij * xij + yij * yij + zij * zij
             Sqrtrij = Sqrt(rij)

             SigijDif = Sigij(Sqrtrij,r_gb(i),r_gb(j),P(2),Zero)
             ! Subtract contributions from non-bonded terms to correct for over counting
             SigijDif=SigijDif-Sigij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))

             CGI2 = CG(I)*CG(I)
             CGJ2 = CG(J)*CG(J)

             If ( imove(i)  ==  0 ) then
                Aij=Factor*(CGJ2+T_GB(j)) &
                     * SiGijDif * VOL_GB(i) / CCELEC

                DX(i) = DX(i) + COEF * xij * Aij
                DY(i) = DY(i) + COEF * yij * Aij
                DZ(i) = DZ(i) + COEF * zij * Aij
             Endif

             If ( imove(j)  ==  0 ) then
                Aji=Factor*(CGI2+T_GB(i)) &
                     * SiGijDif * VOL_GB(j) / CCELEC

                DX(j) = DX(j) - COEF * xij * Aji
                DY(j) = DY(j) - COEF * yij * Aji
                DZ(j) = DZ(j) - COEF * zij * Aji
             Endif

          Endif
       Endif

    Enddo

    ! Next atoms connected through angles
    !----------------------------------------------------------------------------------
    Do ic = MynodP, NTheta, NumNod

       LOK = .true.
       i = it(ic)
       j = kt(ic)
       KK = jt(ic)
       !CCCCCCCCCCCCC
       ! getting the number of block and coefficient
       IBL = IBLCKP(I)
       JBL = IBLCKP(J)
       KBL = IBLCKP(KK)
       IF (IBL  ==  JBL) JBL = KBL
       IF (JBL < IBL) THEN
          COEF = BLCOEB(JBL+IBL*(IBL-1)/2)
       ELSE
          COEF = BLCOEB(IBL+JBL*(JBL-1)/2)
       ENDIF
       !CCCCCCCCCCCCC

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then

          ! Check if both in pair are fixed!
          If ( .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then

             xij = x(i) - x(j)
             yij = y(i) - y(j)
             zij = z(i) - z(j)

             rij = xij * xij + yij * yij + zij * zij
             Sqrtrij = Sqrt(rij)

             SiGijDif = Sigij(Sqrtrij,r_gb(i),r_gb(j),P(3),Zero)
             ! Subtract contributions from non-bonded terms to correct for over counting
             SigijDif=SigijDif-Sigij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))

             CGI2 = CG(I)*CG(I)
             CGJ2 = CG(J)*CG(J)


             If ( imove(i)  ==  0 ) then
                Aij=Factor*(CGJ2 + T_GB(j)) &
                     * SiGijDif * VOL_GB(i) / CCELEC


                DX(i) = DX(i) + COEF * xij * Aij
                DY(i) = DY(i) + COEF * yij * Aij
                DZ(i) = DZ(i) + COEF * zij * Aij
             Endif

             If ( imove(j)  ==  0 ) then
                Aji=Factor*(CGI2+T_GB(i)) &
                     * SiGijDif * VOL_GB(j) / CCELEC

                DX(j) = DX(j) - COEF * xij * Aji
                DY(j) = DY(j) - COEF * yij * Aji
                DZ(j) = DZ(j) - COEF * zij * Aji
             Endif

          Endif
       Endif
    Enddo

    ! Now we need to get the atoms in the exclusion list
    !----------------------------------------------------------------------------------------
    Itemp = 0
    ! Define the atom bounds for this processor.
    Do i = MyNodP, Natom, NumNod

       If (i  /=  1 ) ITemp = INblo14(i - 1)
       Dxi = Zero
       Dyi = Zero
       Dzi = Zero
       GBtmp = Zero

       NPr = INblo14(i) - ITemp
       Do jpr = 1, NPr
          LOK = .true.
          k = Inbl14(Itemp+jpr)
          j = Abs(k)
          If ( k  >  0 .and. &
               .not. ( imove(i)  ==  1 &
               .and. imove(j)  ==  1 ) ) then
             If ( QInte ) LOK = &
                  ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
             If (LOK) Then
                !CCCCCCCCCCCCC
                ! getting the number of block and coefficient
                IBL = IBLCKP(I)
                JBL = IBLCKP(J)
                IF (JBL < IBL) THEN
                   KK = JBL
                   JBL = IBL
                   IBL = KK
                ENDIF
                COEF = BLCOEB(IBL+JBL*(JBL-1)/2)
                !CCCCCCCCCCCCC
                IF (.Not. (IBL /= 1 .AND. IBL .NE. JBL) ) THEN
                   ! IF both i & j do not belong to block 1 and  block(i) is not equal to
                   ! block(j), skipp next lines. (COEF is always zero)
                   xij = x(i) - x(j)
                   yij = y(i) - y(j)
                   zij = z(i) - z(j)

                   rij = xij * xij + yij * yij + zij * zij
                   Sqrtrij = Sqrt(rij)

                   SiGijDif = SiGij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))

                   CGI2 = CG(I)*CG(I)
                   CGJ2 = CG(J)*CG(J)
                   CGIJ = CG(I)*CG(J)

                   Aij = ZERO
                   Aji = ZERO

                   If (CGIJ /= Zero) then
                      Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                      expDij = exp(-Dij)

                      Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                      SqrtSij = Sqrt(Sij)
                      Sij3half = Sij * SqrtSij

                      Sij = CGIJ * Factor / Sij3half

                      Aij = Sij - Sij * PT25 * expDij

                      Aji = Aij

                      Sij = Sij * SqrtSij * SqrtSij

                      GBtmp = GBtmp - COEF * Sij

                      IF (JBL  ==  1) THEN

                         If (Qanalys) Then
                            GB_Atm(i) = GB_Atm(i) - Half * Sij
                            GB_Atm(j) = GB_Atm(j) - Half * Sij
                         Endif
                      ELSE
                         GB_BLOCK(JBL) = GB_BLOCK(JBL) - Sij
                         If (Qanalys) Then
                            GB_Atm(i) = GB_Atm(i) - COEF * Half * Sij
                            GB_Atm(j) = GB_Atm(j) - COEF * Half * Sij
                         Endif
                      ENDIF
                   Endif

                   Aij = Aij + Factor * ( CGJ2 + T_GB(j) ) &
                        * SiGijDif * VOL_GB(i) / CCELEC

                   Aji = Aji + Factor * ( CGI2 + T_GB(i) ) &
                        * SiGijDif * VOL_GB(j) / CCELEC

                   If ( imove(i)  ==  0 ) then
                      Dxi = Dxi + COEF * xij * Aij
                      Dyi = Dyi + COEF * yij * Aij
                      Dzi = Dzi + COEF * zij * Aij
                   Endif

                   If ( imove(j)  ==  0 ) then
                      DX(j) = DX(j) - COEF * xij * Aji
                      DY(j) = DY(j) - COEF * yij * Aji
                      DZ(j) = DZ(j) - COEF * zij * Aji
                   Endif
                Endif
             Endif
          Endif
       Enddo

       If (NPr > 0) then
          GBEnr = GBEnr + GBtmp

          DX(i) = DX(i) + Dxi
          DY(i) = DY(i) + Dyi
          DZ(i) = DZ(i) + Dzi
       Endif

    Enddo

    ! Finally do pairs of nonbonded terms
    ! For lists use different construct here
    Itemp = 0
#if KEY_IMCUBES==1
    if(lbycbim)then
       if( (INBL(Natom)-INBL(Natom+natom))  > 0)then
          write(outu,*) INBL(Natom),INBL(Natom-1)
          Call WrnDie(-3,'<genborn.src>GBSOLV1', &
               'GBsolv (genborn.src), last atom has pairs')
       endif
    endif
#endif 
    Do i = 1, Natom-1
#if KEY_IMCUBES==1
       if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif

       Dxi = Zero
       Dyi = Zero
       Dzi = Zero
       GBtmp = Zero

       NPr = INbl(i) - ITemp
       Do jpr = 1, NPr
          k = JNbl(Itemp+jpr)
          j = Abs(k)
          !CCCCCCCCCCCCC
          ! getting the number of block and coefficient
          IBL = IBLCKP(I)
          JBL = IBLCKP(J)
          IF (JBL < IBL) THEN
             KK = JBL
             JBL = IBL
             IBL = KK
          ENDIF
          COEF = BLCOEB(IBL+JBL*(JBL-1)/2)
          !CCCCCCCCCCCCC
          IF (.NOT. (IBL /= 1 .AND. JBL.NE.IBL) ) THEN
             ! IF both i & j do not belong to block 1 and  block(i) is not equal to
             ! block(j), skipp next lines. (COEF is always zero)

             xij = x(i) - x(j)
             yij = y(i) - y(j)
             zij = z(i) - z(j)

             rij = xij * xij + yij * yij + zij * zij
             If (rij  <  C2OfNb) Then
                call es_switch(rij, FSw, DFSw) ! Electrostatic Switch func
                Sqrtrij = Sqrt(rij)
#if KEY_GBINLINE==0
                SiGijDif = SiGij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))
#else /* GBINLINE */
                dAij = Zero
                Aij = One
                Ratio = rij /  &
                     (( r_gb(i) + r_gb(j) ) * ( r_gb(i) + r_gb(j) ))
                If ( Ratio  <=  One/P(5) ) then
                   Aij = Aij - Cos( Ratio * P(5) * Pi )
                   dAij = dAij + Aij &
                        * P(5) * Pi * Ratio * Sin( Ratio * P(5) * Pi ) &
                        / Sqrtrij
                   Aij = Half * Aij
                   Aij = Aij * Aij
                Endif
                Aij = P(4) * Aij / ( rij * rij )
                SiGijDif = P(4) * dAij / ( rij * rij * Sqrtrij ) &
                     - Four * Aij /  rij
#endif /* GBINLINE */

                CGI2 = CG(I)*CG(I)
                CGJ2 = CG(J)*CG(J)
                CGIJ = CG(I)*CG(J)

                Aij = ZERO
                Aji = ZERO
                Sij = Zero

                If ( CG(j) * CG(i)  /=  Zero ) then
                   Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                   expDij = exp(-Dij)
                   Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                   SqrtSij = Sqrt(Sij)
                   Sij3half = Sij*SqrtSij
                   Sij = CGIJ * Factor / Sij3half * FSw
                   Aij = Sij - Sij * PT25 * expDij

                   Aji = Aij

                   Sij = Sij * SqrtSij * SqrtSij
                   GBtmp = GBtmp - COEF * Sij
                   If (Qanalys) Then
                      GB_Atm(i) = GB_Atm(i) - Half * COEF * Sij
                      GB_Atm(j) = GB_Atm(j) - Half * COEF * Sij
                   Endif

                Endif
                Aij = Aij + Factor * ( CGJ2 + T_GB(j) ) &
                     * SiGijDif * VOL_GB(i) / CCELEC

                Aji = Aji + Factor * ( CGI2 + T_GB(i) ) &
                     * SiGijDif * VOL_GB(j) / CCELEC

                if ( qgbswit .and. louter ) then
                   Aij = Aij  - Sij * DFSw
                   Aji = Aji  - Sij * DFSw
                endif

                if (JBL .ne.  1) &
                     GB_BLOCK(JBL) = GB_BLOCK(JBL) - Sij

                If ( imove(i)  ==  0 ) then
                   Dxi = Dxi + COEF * xij * Aij
                   Dyi = Dyi + COEF * yij * Aij
                   Dzi = Dzi + COEF * zij * Aij
                Endif

                If ( imove(j)  ==  0 ) then
                   DX(j) = DX(j) - COEF * xij * Aji
                   DY(j) = DY(j) - COEF * yij * Aji
                   DZ(j) = DZ(j) - COEF * zij * Aji
                Endif
             ENDIF
          Endif
       Enddo

       If (NPr > 0) then
          GBEnr = GBEnr + GBtmp

          DX(i) = DX(i) + Dxi
          DY(i) = DY(i) + Dyi
          DZ(i) = DZ(i) + Dzi
       Endif

       ITemp = INbl(i)
    Enddo
    !-----------------------------------------------------------------------
    ! Add the GB terms of fixed atom - fixed atom
    !
    ! ***  Caution  ***
    ! All GB terms of fixed atom - fixed atom are assumed to be nonbonded.
    ! Therefore, CtOnNb shoulb be longer than 1-2, 1-3 interactions
    If ( GbFixed  >  1 ) Then
       call gbswit_setup()
       Do i = 1, Natom-1
          GBtmp = Zero
          Do j = i+1, Natom
             If ( ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) .and. &
                  ( CG(j) * CG(i)  /=  Zero ) ) THEN
                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)
                rij = xij * xij + yij * yij + zij * zij
                If (rij  <  c2ofnb ) Then
                   !CCCCCCCCCCCCC
                   ! getting the number of block and coefficient
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)
                   IF (JBL < IBL) THEN
                      KK = JBL
                      JBL = IBL
                      IBL = KK
                   ENDIF
                   COEF = BLCOEB(IBL+JBL*(JBL-1)/2)
                   !CCCCCCCCCCCCC
                   IF (.NOT. (IBL /= 1 .AND. JBL.NE.IBL) ) THEN
                      call es_switch(rij, fsw) ! Electrostatic Switch func
                      Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                      expDij = exp(-Dij)
                      Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                      SqrtSij = Sqrt(Sij)
                      CGIJ = CG(I) * CG(J)

                      Sij = CGIJ * Factor / SqrtSij * FSw

                      GBtmp = GBtmp - COEF * Sij
                      If (Qanalys) Then
                         GB_Atm(i) = GB_Atm(i) - COEF * Half * Sij
                         GB_Atm(j) = GB_Atm(j) - COEF * Half * Sij
                      Endif
#if KEY_BLOCK==1 /*ldm*/
                      IF (JBL >= LSTRT) THEN
                         GB_BLOCK(JBL) = GB_BLOCK(JBL) - COEF * Sij
                      ENDIF
#endif 
                   EndIf
                EndIF
             ENDIF
          Enddo
          GBEnr = GBEnr + GBtmp
       Enddo
    ENDIF

    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
    ! Finally add in self-terms (including the first derivative of lambda)
    !----------------------------------------------------------------------------
    ! initialized
    ! Define the atom bounds for this processor.
    Do i = MynodP, Natom, NumNod
       !CCCCCCCCCCCCC
       ! getting the number of block and coefficient
       IBL = IBLCKP(I)
       KK=IBL+IBL*(IBL-1)/2
       COEF = BLCOEV(KK)

       LOK = .true.
       If ( Qinte ) LOK = Islct(i)  ==  1
       If ( LOK ) Then
          If ( Cg(i)  /=  Zero ) Then
             IF (IBL  ==  1) THEN
                CGI2 = CG(I) * CG(I)

                Self = - COEF * CGI2 * Factor * Half / Alph_Gb(i)
                GBEnr = GBEnr + Self

                If (Qanalys) Then
                   GB_Atm(i) = GB_Atm(i) + Self
                Endif
#if KEY_BLOCK==1 /*ldm*/
                IF (QLDM) THEN
                   DO J = LSTRT, NBLOCK
                      GBLDM(j) = GBLDM(j) &
                           + (CgI2 + T_GB(i) ) * GB_LAMB(Natom*(j-1)+i)
                   ENDDO
                ENDIF
#endif /*  LDM*/

             ELSE
                Self = - CG(i)*CG(i) * Factor * Half / Alph_Gb(i)
                GBEnr = GBEnr + COEF * Self

                GB_BLOCK(IBL) = GB_BLOCK(IBL) + Self

                If (Qanalys) Then
                   GB_Atm(i) = GB_Atm(i) + COEF * Self
                Endif

             ENDIF

             IF (IMOVE(I)  /=  1) THEN
                DX(i) = DX(i) &
                     + Factor*COEF * ( CG(i)*CG(i) + T_GB(i) ) &
                     * Sigx_Gb(i)/CCELEC
                DY(i) = DY(i) &
                     + Factor*COEF * ( CG(i)*CG(i) + T_GB(i) ) &
                     * Sigy_Gb(i)/CCELEC
                DZ(i) = DZ(i) &
                     + Factor*COEF * ( CG(i)*CG(i) + T_GB(i) ) &
                     * Sigz_Gb(i)/CCELEC
             ENDIF
          Endif
       Endif

    Enddo

    ! Finally, put gether the GB energy
    IF (QPRNTV)  THEN
       VBGENB(1) = VBGENB(1) + GBEnr
    ENDIF
    DO I = LSTRT, NBLOCK
#if KEY_BLOCK==1 /*ldm*/
       IF (QLDM) THEN
          BIFLAM(I) = BIFLAM(I) + GB_BLOCK(I)
          GBLDM(I) = FACTOR / CCELEC * GBLDM(I)
          BIFLAM(I) = BIFLAM(I) + GBLDM(I)
       ELSE IF (QLMC) THEN
          BIFLAM(I) = BIFLAM(I) + GB_BLOCK(I)
       ENDIF
#endif /*  (LDM)*/
       IF (QPRNTV)  THEN
          VBGENB(I) = VBGENB(I) + GB_BLOCK(I)
       ENDIF
    ENDDO

    ! Note need to GCOMB GB_Atm(i) if QAnalys true.
#if KEY_PARALLEL==1 /*paraend*/
    ! Combine GB_Atm
    If(Qanalys) Call GComb(GB_atm, Natom)
#endif /* (paraend)*/
    Return
  End Subroutine GBSolv1

  Subroutine FillAlpGB1( X, Y, Z, &
       JNbl, INbl, Inbl14, INblo14, &
       QInte, Islct, JSlct, GbFixed)
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
    ! Generalized Born with Block (e.g. Lambda-dynamics, FEP, LEP)
    ! Type 1 ( Born radius depend on the coefficient of block)
    ! S.B. 10/30/99
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  use number
  use consta
  use dimens_fcm
  use psf
  use block_fcm
  use parallel
  use inbnd, only: ctofnb, lbycbim
  use gbswit, only: gbswit_setup, es_switch, c2ofnb
#if KEY_BLOCK==1
  use lambdam   /*ldm*/
#endif
    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*), &
         Islct(*), Jslct(*)
#if KEY_PARALLEL==1
    Integer AtFrst, AtLast
#endif 
    real(chm_real) X(*), Y(*), Z(*)
    INTEGER GbFixed
    Logical Qinte

    ! Local variables
    Integer i, ic, j, k
    real(chm_real) rij, xij, yij, zij, Aij, Tij
    real(chm_real) Dij, Sij, SqrtSij, Sij3half, expDij
    real(chm_real) CGIJ
    real(chm_real) COEF1, COEF2, fsw, dfsw
    INTEGER IBL, JBL, KBL, KK
    LOgical LOK
    Integer ITemp, NPr, jpr
#if KEY_GBINLINE==1
    real(chm_real) dAij, Ratio 
#endif

    GbFixed = 0
    Do i = 1, Natom
       Alph_Gb(i) = Zero
       Sigx_Gb(i)  = Zero
       Sigy_Gb(i)  = Zero
       Sigz_Gb(i)  = Zero
       T_gb(i)     = Zero
       If ( QAnalys ) GB_Atm(i) = Zero
       If ( imove(i)  ==  1 ) GbFixed = GbFixed + 1
#if KEY_BLOCK==1 /*ldm*/
       IF (QLDM) THEN
          DO j = 0, NBLOCK-1
             GB_LAMB(j*Natom+i) = Zero
          ENDDO
       ENDIF
#endif /*  LDM*/
    Enddo

    ! If ( GbFixed  >  0 ) Then
    !    Call WrnDie(0,'<genborn.src>FillAlpGB', &
    !         'Generalized Born only ran slowly w/ fixed atoms')
    ! Endif

    call gbswit_setup()
    ! Loop over bonded atoms first:
    !-----------------------------------------------------------------------
    Do ic = MynodP, Nbond, Numnod

       LOK = .true.
       i = ib(ic)
       j = jb(ic)

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then
          !CCCCCCCCCCCCC
          ! getting the number of block
          IBL = IBLCKP(I)
          JBL = IBLCKP(J)
          IF (JBL < IBL) THEN
             KK=JBL+IBL*(IBL-1)/2
          ELSE
             KK=IBL+JBL*(JBL-1)/2
          ENDIF
          ! Check the number of the block that atom i and j belong to
          ! And decide the coefficient
          k = 0
          IF (IBL  ==  1) THEN
             IF (JBL  ==  1 ) THEN
                COEF1 = One
                COEF2 = One
             ELSE
                COEF1 = BLCOEB(KK)
                COEF2 = One
                k = i
             ENDIF
          ELSE
             IF (JBL  ==  1) THEN
                COEF1 = One
                COEF2 = BLCOEB(KK)
                k = j
             ELSE
                COEF1 = One
                COEF2 = One
             ENDIF
          ENDIF
          !CCCCCCCCCCCCC
          IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1).AND.(IBL.NE.JBL)) ) THEN
             ! IF both i & j do not belong to block 1 and  block(i) is not equal to
             ! block(j), skipp next lines. (COEF is always zero)
             xij = x(i) - x(j)
             yij = y(i) - y(j)
             zij = z(i) - z(j)

             rij = xij * xij + yij * yij + zij * zij

             Aij = Alphij(rij, r_gb(i), r_gb(j), P(2), Zero)
             ! Subtract contributions from non-bonded terms to correct for over counting
             Aij = Aij - Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))

             Alph_Gb(i) = Alph_Gb(i) + COEF1 * Aij * VOL_GB(j)
             Alph_Gb(j) = Alph_Gb(j) + COEF2 * Aij * VOL_GB(i)
#if KEY_BLOCK==1 /*ldm*/
             IF (QLDM .AND. k  /=  0) THEN
                IF (k == i) GB_LAMB(Natom*(JBL-1)+i) = &
                     GB_LAMB(Natom*(JBL-1)+i) + Aij*VOL_GB(j)
                IF (k == j) GB_LAMB(Natom*(IBL-1)+j) = &
                     GB_LAMB(Natom*(IBL-1)+j) + Aij*VOL_GB(i)
             ENDIF
#endif /*  LDM*/

             rij = sqrt ( rij )

             ! Now do sigma contributions
             Aij = Sigij(rij, r_gb(i), r_gb(j), P(2), Zero)
             ! Subtract contributions from non-bonded terms to correct for over counting
             Aij = Aij - Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))

             If ( imove(i)  ==  0 ) then
                Sigx_Gb(i) = Sigx_Gb(i) + COEF1 * Aij * VOL_GB(j) * xij
                Sigy_Gb(i) = Sigy_Gb(i) + COEF1 * Aij * VOL_GB(j) * yij
                Sigz_Gb(i) = Sigz_Gb(i) + COEF1 * Aij * VOL_GB(j) * zij
             Endif

             If ( imove(j)  ==  0 ) then
                Sigx_Gb(j) = Sigx_Gb(j) - COEF2 * Aij * VOL_GB(i) * xij
                Sigy_Gb(j) = Sigy_Gb(j) - COEF2 * Aij * VOL_GB(i) * yij
                Sigz_Gb(j) = Sigz_Gb(j) - COEF2 * Aij * VOL_GB(i) * zij
             Endif
          ENDIF
       ENDIF
    Enddo

    ! Next atoms connected through angles
    !--------------------------------------------------------------------
    Do ic = MynodP, NTheta, NumNod

       LOK = .true.
       i = it(ic)
       j = kt(ic)
       KK = jt(ic)

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then
          !CCCCCCCCCCCCC
          ! getting the number of block
          IBL = IBLCKP(I)
          JBL = IBLCKP(J)
          KBL = IBLCKP(KK)
          IF (IBL  ==  JBL) JBL = KBL
          IF (JBL < IBL) THEN
             KK=JBL+IBL*(IBL-1)/2
          ELSE
             KK=IBL+JBL*(JBL-1)/2
          ENDIF
          ! Check the number of the block that atom i and j belong to
          ! And decide the coefficient
          k = 0
          IF (IBL  ==  1) THEN
             IF (JBL  ==  1 ) THEN
                COEF1 = One
                COEF2 = One
             ELSE
                COEF1 = BLCOEB(KK)
                COEF2 = One
                k = i
             ENDIF
          ELSE
             IF (JBL  ==  1) THEN
                COEF1 = One
                COEF2 = BLCOEB(KK)
                k = j
             ELSE
                COEF1 = One
                COEF2 = One
             ENDIF
          ENDIF
          !CCCCCCCCCCCCC
          IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1).AND.(IBL.NE.JBL)) ) THEN
             ! IF both i & j do not belong to block 1 and  block(i) is not equal to
             ! block(j), skipp next lines. (COEF is always zero)
             xij = x(i) - x(j)
             yij = y(i) - y(j)
             zij = z(i) - z(j)

             rij = xij * xij + yij * yij + zij * zij

             Aij = Alphij(rij, r_gb(i), r_gb(j), P(3), Zero)
             ! Subtract contributions from non-bonded terms to correct for over counting
             Aij = Aij - Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))

             Alph_Gb(i) = Alph_Gb(i) + COEF1 * Aij * VOL_GB(j)
             Alph_Gb(j) = Alph_Gb(j) + COEF2 * Aij * VOL_GB(i)

#if KEY_BLOCK==1 /*ldm*/
             IF (QLDM .AND. k  /=  0) THEN
                IF (k == i) GB_LAMB(Natom*(JBL-1)+i) = &
                     GB_LAMB(Natom*(JBL-1)+i) + Aij *VOL_GB(j)
                IF (k == j) GB_LAMB(Natom*(IBL-1)+j) = &
                     GB_LAMB(Natom*(IBL-1)+j) + Aij *VOL_GB(i)
             ENDIF
#endif /*  LDM*/

             rij = sqrt ( rij )
             ! Now do sigma contributions
             Aij = Sigij(rij, r_gb(i), r_gb(j), P(3), Zero)

             ! Subtract contributions from non-bonded terms to correct for over counting
             Aij = Aij - Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))

             If ( imove(i)  ==  0 ) then
                Sigx_Gb(i) = Sigx_Gb(i) + COEF1 * Aij * VOL_GB(j) * xij
                Sigy_Gb(i) = Sigy_Gb(i) + COEF1 * Aij * VOL_GB(j) * yij
                Sigz_Gb(i) = Sigz_Gb(i) + COEF1 * Aij * VOL_GB(j) * zij
             Endif

             If ( imove(j)  ==  0 ) then
                Sigx_Gb(j) = Sigx_Gb(j) - COEF2 * Aij * VOL_GB(i) * xij
                Sigy_Gb(j) = Sigy_Gb(j) - COEF2 * Aij * VOL_GB(i) * yij
                Sigz_Gb(j) = Sigz_Gb(j) - COEF2 * Aij * VOL_GB(i) * zij
             Endif

          ENDIF
       Endif
    Enddo

    ! Do atoms on exclusion list
    !--------------------------------------------------------------------
    Itemp = 0
    ! Define the atom bounds for this processor.
    Do i = MyNodP, Natom, NumNod

       If (i  /=  1 ) ITemp = INblo14(i - 1)
       NPr = INblo14(i) - ITemp
       Do jpr = 1, NPr

          LOK = .true.
          k = Inbl14(Itemp+jpr)
          j = Abs(k)
          ! Don't do fixed atoms from exclusion list here, get them below.
          If ( k  >  0 &
               .and. .not. (imove(i)  ==  1 &
               .and. imove(j)  ==  1) ) then
             If ( QInte ) LOK = &
                  ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
             If (LOK) Then
                !CCCCCCCCCCCCC
                ! getting the number of block
                IBL = IBLCKP(I)
                JBL = IBLCKP(J)
                IF (JBL < IBL) THEN
                   KK=JBL+IBL*(IBL-1)/2
                ELSE
                   KK=IBL+JBL*(JBL-1)/2
                ENDIF
                ! Check the number of the block that atom i and j belong to
                ! And decide the coefficient
                k = 0
                IF (IBL  ==  1) THEN
                   IF (JBL  ==  1 ) THEN
                      COEF1 = One
                      COEF2 = One
                   ELSE
                      COEF1 = BLCOEB(KK)
                      COEF2 = One
                      k = i
                   ENDIF
                ELSE
                   IF (JBL  ==  1) THEN
                      COEF1 = One
                      COEF2 = BLCOEB(KK)
                      k = j
                   ELSE
                      COEF1 = One
                      COEF2 = One
                   ENDIF
                ENDIF
                !CCCCCCCCCCCCC
                IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1).AND.(IBL.NE.JBL)) ) &
                     THEN
                   ! IF both i & j do not belong to block 1 and  block(i) is not equal to
                   ! block(j), skipp next lines. (COEF is always zero)
                   xij = x(i) - x(j)
                   yij = y(i) - y(j)
                   zij = z(i) - z(j)

                   rij = xij * xij + yij * yij + zij * zij

#if KEY_GBINLINE==0
                   Aij =  Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                   Aij = One
                   Ratio = rij /  &
                        (( r_gb(i) + r_gb(j) ) * ( r_gb(i) + r_gb(j) ))
                   If ( Ratio  <=  One/P(5) ) then
                      Aij = Aij - Cos( Ratio * P(5) * Pi )
                      Aij = Half * Aij
                      Aij = Aij * Aij
                   Endif
                   Aij = P(4) * Aij / ( rij * rij )
#endif 
                   Alph_Gb(i) = Alph_Gb(i) + COEF1 * Aij * VOL_GB(j)
                   Alph_Gb(j) = Alph_Gb(j) + COEF2 * Aij * VOL_GB(i)
#if KEY_BLOCK==1 /*ldm*/
                   IF (QLDM .AND. k  /=  0) THEN
                      IF (k == i) GB_LAMB(Natom*(JBL-1)+i) = &
                           GB_LAMB(Natom*(JBL-1)+i) + Aij*VOL_GB(j)
                      IF (k == j) GB_LAMB(Natom*(IBL-1)+j) = &
                           GB_LAMB(Natom*(IBL-1)+j) + Aij*VOL_GB(i)
                   ENDIF
#endif /*  LDM*/

                   rij = sqrt ( rij )

                   ! Now do sigma contributions
#if KEY_GBINLINE==0
                   Aij =  Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                   dAij = Zero
                   If ( Ratio  <=  One/P(5) ) then
                      dAij = dAij + (  One - Cos( Ratio * P(5) * Pi ) ) &
                           * P(5) * Pi * Ratio * Sin( Ratio * P(5) * Pi )/rij
                   Endif
                   Aij = P(4) * dAij / ( rij * rij * rij * rij * rij ) &
                        - Four * Aij / ( rij * rij )
#endif 
                   If ( imove(i)  ==  0 ) then
                      Sigx_Gb(i) = Sigx_Gb(i) + COEF1 * Aij*VOL_GB(j)*xij
                      Sigy_Gb(i) = Sigy_Gb(i) + COEF1 * Aij*VOL_GB(j)*yij
                      Sigz_Gb(i) = Sigz_Gb(i) + COEF1 * Aij*VOL_GB(j)*zij
                   Endif

                   If ( imove(j)  ==  0 ) then
                      Sigx_Gb(j) = Sigx_Gb(j) - COEF2 * Aij*VOL_GB(i)*xij
                      Sigy_Gb(j) = Sigy_Gb(j) - COEF2 * Aij*VOL_GB(i)*yij
                      Sigz_Gb(j) = Sigz_Gb(j) - COEF2 * Aij*VOL_GB(i)*zij
                   Endif

                ENDIF
             Endif
          Endif
       Enddo

    Enddo

    ! For lists us different construct here
    Itemp = 0
    Do i = 1, Natom-1
#if KEY_IMCUBES==1
       if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif

       NPr = INbl(i) - ITemp
       Do jpr = 1, NPr
          k = JNbl(Itemp+jpr)
          j = Abs(k)

          xij = x(i) - x(j)
          yij = y(i) - y(j)
          zij = z(i) - z(j)

          rij = xij * xij + yij * yij + zij * zij

          If (rij  <  c2ofnb) Then
             !CCCCCCCCCCCCC
             ! getting the number of block
             IBL = IBLCKP(I)
             JBL = IBLCKP(J)
             IF (JBL < IBL) THEN
                KK=JBL+IBL*(IBL-1)/2
             ELSE
                KK=IBL+JBL*(JBL-1)/2
             ENDIF
             ! Check the number of the block that atom i and j belong to
             ! And decide the coefficient
             k = 0
             IF (IBL  ==  1) THEN
                IF (JBL  ==  1 ) THEN
                   COEF1 = One
                   COEF2 = One
                ELSE
                   COEF1 = BLCOEB(KK)
                   COEF2 = One
                   k = i
                ENDIF
             ELSE
                IF (JBL  ==  1) THEN
                   COEF1 = One
                   COEF2 = BLCOEB(KK)
                   k = j
                ELSE
                   COEF1 = One
                   COEF2 = One
                ENDIF
             ENDIF
             !CCCCCCCCCCCCC
             IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1).AND.(IBL.NE.JBL)) ) &
                  THEN
                ! IF both i & j do not belong to block 1 and  block(i) is not equal to
                ! block(j), skipp next lines. (COEF is always zero)

#if KEY_GBINLINE==0
                Aij =  Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                Aij = One
                Ratio = rij /  &
                     (( r_gb(i) + r_gb(j) ) * ( r_gb(i) + r_gb(j) ))
                If ( Ratio  <=  One/P(5) ) then
                   Aij = Aij - Cos( Ratio * P(5) * Pi )
                   Aij = Half * Aij
                   Aij = Aij * Aij
                Endif
                Aij = P(4) * Aij / ( rij * rij )
#endif 

                Alph_Gb(i) = Alph_Gb(i) + COEF1 * Aij * VOL_GB(j)
                Alph_Gb(j) = Alph_Gb(j) + COEF2 * Aij * VOL_GB(i)

#if KEY_BLOCK==1 /*ldm*/
                IF (QLDM .AND. k  /=  0) THEN
                   IF (k == i) GB_LAMB(Natom*(JBL-1)+i) = &
                        GB_LAMB(Natom*(JBL-1)+i) + Aij*VOL_GB(j)
                   IF (k == j) GB_LAMB(Natom*(IBL-1)+j) = &
                        GB_LAMB(Natom*(IBL-1)+j) + Aij*VOL_GB(i)
                ENDIF
#endif /*  LDM*/

                rij = sqrt ( rij )

                ! Now do sigma contributions
#if KEY_GBINLINE==0
                Aij =  Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                dAij = Zero
                If ( Ratio  <=  One/P(5) ) then
                   dAij = dAij + (  One - Cos( Ratio * P(5) * Pi ) ) &
                        * P(5) * Pi * Ratio * Sin( Ratio * P(5) * Pi )/rij
                Endif
                Aij = P(4) * dAij / ( rij * rij * rij * rij * rij ) &
                     - Four * Aij / ( rij * rij )
#endif 
                If ( imove(i)  ==  0 ) then
                   Sigx_Gb(i) = Sigx_Gb(i) + COEF1 * Aij*VOL_GB(j)*xij
                   Sigy_Gb(i) = Sigy_Gb(i) + COEF1 * Aij*VOL_GB(j)*yij
                   Sigz_Gb(i) = Sigz_Gb(i) + COEF1 * Aij*VOL_GB(j)*zij
                Endif

                If ( imove(j)  ==  0 ) then
                   Sigx_Gb(j) = Sigx_Gb(j) - COEF2 * Aij*VOL_GB(i)*xij
                   Sigy_Gb(j) = Sigy_Gb(j) - COEF2 * Aij*VOL_GB(i)*yij
                   Sigz_Gb(j) = Sigz_Gb(j) - COEF2 * Aij*VOL_GB(i)*zij
                Endif
             ENDIF
          Endif
       Enddo

       ITemp = INbl(i)
    Enddo

    !----------------------------------------------------
    ! If we use lists and there are fixed atoms, add contributions from
    ! fixed atom lists to Alpha
    If ( GbFixed  >  1 ) Then
       C2OfNB = CtOfNb * CtOfNb
       Do i = 1, Natom-1
          Do j = i+1, Natom

             If ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) then

                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)

                rij = xij * xij + yij * yij + zij * zij
                If (rij  <  c2ofnb) Then
                   !CCCCCCCCCCCCC
                   ! getting the number of block
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)
                   IF (JBL < IBL) THEN
                      KK=JBL+IBL*(IBL-1)/2
                   ELSE
                      KK=IBL+JBL*(JBL-1)/2
                   ENDIF
                   ! Check the number of the block that atom i and j belong to
                   ! And decide the coefficient
                   k = 0
                   IF (IBL  ==  1) THEN
                      IF (JBL  ==  1 ) THEN
                         COEF1 = One
                         COEF2 = One
                      ELSE
                         COEF1 = BLCOEB(KK)
                         COEF2 = One
                         k = i
                      ENDIF
                   ELSE
                      IF (JBL  ==  1) THEN
                         COEF1 = One
                         COEF2 = BLCOEB(KK)
                         k = j
                      ELSE
                         COEF1 = One
                         COEF2 = One
                      ENDIF
                   ENDIF
                   !CCCCCCCCCCCCC
                   IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1) &
                        .AND.(IBL /= JBL)) ) THEN
#if KEY_GBINLINE==0
                      Aij =  Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                      Aij = One
                      Ratio = rij /  &
                           (( r_gb(i) + r_gb(j) )*( r_gb(i) + r_gb(j) ))
                      If ( Ratio  <=  One/P(5) ) then
                         Aij = Aij - Cos( Ratio * P(5) * Pi )
                         Aij = Half * Aij
                         Aij = Aij * Aij
                      Endif
                      Aij = P(4) * Aij / ( rij * rij )
#endif 
                      Alph_Gb(i) = Alph_Gb(i) + COEF1 * Aij * VOL_GB(j)
                      Alph_Gb(j) = Alph_Gb(j) + COEF2 * Aij * VOL_GB(i)
#if KEY_BLOCK==1 /*ldm*/
                      IF (QLDM .AND. k  /=  0) THEN
                         IF (k == i) GB_LAMB(Natom*(JBL-1)+i) = &
                              GB_LAMB(Natom*(JBL-1)+i) + Aij*VOL_GB(j)
                         IF (k == j) GB_LAMB(Natom*(IBL-1)+j) = &
                              GB_LAMB(Natom*(IBL-1)+j) + Aij*VOL_GB(i)
                      ENDIF
#endif /*  LDM*/
                   ENDIF
                Endif
             Endif
          Enddo
       Enddo

    Endif

#if KEY_PARALLEL==1 /*paralpha*/
    ! Get Alpha's updated on all processors
    Call GComb(Alph_gb, Natom)
#endif /* (paralpha)*/
    ! finally getting the alpha
    !----------------------------------------------------
    Do i = 1, Natom

       Alph_Gb(i) = Alph_Gb(i) + CCELEC*P(1)/(Two*r_gb(i)*r_gb(i)) &
            - CCELEC/(Two*r_gb(i)*P(6))

       Alph_Gb(i) = - CCELEC / ( Alph_Gb(i) * Two )

       If ( Alph_Gb(i)  <=  Zero )  Alph_Gb(i) = CutAlph
       If ( Alph_Gb(i)  >  CutAlph )  Alph_Gb(i) = CutAlph

    Enddo


    ! ***Calculation of T***
    ! First get the atoms on exclusion lists
    !--------------------------------------------------------
    Itemp = 0
    ! Define the atom bounds for this processor.
    Do i = MynodP, Natom, NumNod
       If (i  /=  1 ) ITemp = INblo14(i - 1)
       NPr = INblo14(i) - Itemp
       If ((CG(i) /= Zero).and.Npr > 0) then

          Tij = Zero
          Do jpr = 1, NPr
             LOK = .true.
             k = Inbl14(Itemp+jpr)
             j = Abs(k)
             If (( (CG(j) /= Zero).and. k  >  0 ) .and. &
                  .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then

                If ( QInte ) LOK = &
                     ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
                If (LOK) Then
                   !CCCCCCCCCCCCC
                   ! getting the number of block
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)
                   IF (JBL < IBL) THEN
                      KK=JBL+IBL*(IBL-1)/2
                   ELSE
                      KK=IBL+JBL*(JBL-1)/2
                   ENDIF
                   ! Check the number of the block that atom i and j belong to
                   ! And decide the coefficient
                   IF (IBL  ==  1) THEN
                      IF (JBL  ==  1 ) THEN
                         COEF1 = One
                         COEF2 = One
                      ELSE
                         COEF1 = BLCOEB(KK)
                         COEF2 = One
                      ENDIF
                   ELSE
                      IF (JBL  ==  1) THEN
                         COEF1 = One
                         COEF2 = BLCOEB(KK)
                      ELSE
                         COEF1 = One
                         COEF2 = One
                      ENDIF
                   ENDIF
                   !CCCCCCCCCCCCC
                   CGIJ = CG(I) * CG(J)

                   IF(.NOT.((IBL /= 1.AND.JBL.NE.1).AND.(IBL.NE.JBL))) &
                        THEN
                      xij = x(i) - x(j)
                      yij = y(i) - y(j)
                      zij = z(i) - z(j)

                      rij = xij * xij + yij * yij + zij * zij
                      Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                      expDij = exp(-Dij)

                      Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                      SqrtSij = Sqrt(Sij)
                      Sij3half = Sij * SqrtSij

                      Sij = CGIJ / Sij3half

                      Tij = Tij + COEF1 * &
                           Sij * Alph_Gb(i) * expDij*  &
                           ( Alph_Gb(i) * Alph_Gb(j) &
                           + PT25 * rij )

                      T_gb(j) = T_gb(j)  + COEF2 * &
                           Sij * Alph_Gb(j) * expDij*  &
                           ( Alph_Gb(j) * Alph_Gb(i) &
                           + PT25 * rij )
                   ENDIF
                Endif
             Endif
          Enddo

          T_gb(i) = T_gb(i) + Tij
       Endif
    Enddo

    ! Now non-bonded contributions
    Itemp = 0
    Do i = 1, Natom-1
#if KEY_IMCUBES==1
       if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif
       NPr = INbl(i) - ITemp
       If ((CG(i)  /=  Zero ) .and. Npr  >  0 ) then

          Tij = Zero
          Do jpr = 1, NPr
             k = JNbl(Itemp+jpr)
             j = Abs(k)
             If ( Cg(j)  /=  Zero ) then
                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)

                rij = xij * xij + yij * yij + zij * zij
                If (rij  <  c2ofnb) Then
                   !CCCCCCCCCCCCC
                   ! getting the number of block
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)
                   IF (JBL < IBL) THEN
                      KK=JBL+IBL*(IBL-1)/2
                   ELSE
                      KK=IBL+JBL*(JBL-1)/2
                   ENDIF
                   ! Check the number of the block that atom i and j belong to
                   ! And decide the coefficient
                   IF (IBL  ==  1) THEN
                      IF (JBL  ==  1 ) THEN
                         COEF1 = One
                         COEF2 = One
                      ELSE
                         COEF1 = BLCOEB(KK)
                         COEF2 = One
                         k = i
                      ENDIF
                   ELSE
                      IF (JBL  ==  1) THEN
                         COEF1 = One
                         COEF2 = BLCOEB(KK)
                         k = j
                      ELSE
                         COEF1 = One
                         COEF2 = One
                      ENDIF
                   ENDIF
                   !CCCCCCCCCCCCC
                   CGIJ = CG(I) * CG(J)

                   IF(.NOT.((IBL /= 1.AND.JBL.NE.1).AND.(IBL.NE.JBL))) &
                        THEN
                      Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                      expDij = exp(-Dij)

                      Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                      SqrtSij = Sqrt(Sij)
                      Sij3half = Sij * SqrtSij

                      ! Electrostatic Switch function
                      call es_switch(rij, FSw)
                      Sij = CGIJ / Sij3half * FSw

                      Tij = Tij + COEF1 &
                           * Sij * Alph_Gb(i) * expDij*  &
                           ( Alph_Gb(i) * Alph_Gb(j) &
                           + PT25 * rij )

                      T_gb(j) = T_gb(j) +COEF2 &
                           * Sij * Alph_Gb(j) * expDij*  &
                           ( Alph_Gb(j) * Alph_Gb(i) &
                           + PT25 * rij )

                   ENDIF
                Endif
             Endif
          Enddo
          T_gb(i) = T_gb(i) + Tij
       Endif
       ITemp = INbl(i)
    Enddo

    !------------------------------------------------------------------
    ! contribution from fixed atom - fixed atom to T_gb(i)
    ! *** CAUTION  ***
    ! In this loop, 1-2, 1-3 interactions are also added with the swich function,
    ! Therefore, CtOnNb should be longer than the longest distance of 1-2,
    ! or 1-3 interactions.
    !
    If ( GbFixed  >  0 ) Then
       call gbswit_setup()
       Do i = 1, Natom-1
          Tij = Zero
          Do j = i+1, Natom

             If ( ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) .and. &
                  ( CGIJ  /=  ZERO) ) THEN

                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)

                rij = xij * xij + yij * yij + zij * zij

                If (rij  <  c2ofnb) Then
                   !CCCCCCCCCCCCC
                   ! getting the number of block
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)
                   ! Check the number of the block that atom i and j belong to
                   ! And decide the coefficient
                   IF (IBL  ==  1) THEN
                      IF (JBL  ==  1 ) THEN
                         COEF1 = One
                         COEF2 = One
                      ELSE
                         COEF1 = BLCOEB(IBL+JBL*(JBL-1)/2)
                         COEF2 = One
                         k = i
                      ENDIF
                   ELSE
                      IF (JBL  ==  1) THEN
                         COEF1 = One
                         COEF2 = BLCOEB(JBL+IBL*(IBL-1)/2)
                         k = j
                      ELSE
                         COEF1 = One
                         COEF2 = One
                      ENDIF
                   ENDIF
                   !CCCCCCCCCCCCC
                   CGIJ = CG(I) * CG(J)

                   IF(.NOT.((IBL /= 1.AND.JBL.NE.1).AND.(IBL.NE.JBL)))THEN
                      Dij = rij / ( Four * Alph_Gb(i) * Alph_Gb(j) )
                      expDij = exp(-Dij)

                      Sij = rij + Alph_Gb(i) * Alph_Gb(j) * expDij
                      SqrtSij = Sqrt(Sij)
                      Sij3half = Sij * SqrtSij

                      ! Electrostatic Switch function
                      call es_switch(rij, FSw)
                      Sij = CGIJ / Sij3half * FSw

                      Tij = Tij + COEF1 &
                           * Sij * Alph_Gb(i) * expDij*  &
                           ( Alph_Gb(i) * Alph_Gb(j) + PT25 * rij )

                      T_gb(j) = T_gb(j) +COEF2 &
                           * Sij * Alph_Gb(j) * expDij*  &
                           ( Alph_Gb(j) * Alph_Gb(i) + PT25 * rij )

                   ENDIF
                Endif
             Endif
          Enddo
          T_gb(i) = T_gb(i) + Tij
       ENDDO
    Endif

#if KEY_PARALLEL==1 /*paraend*/
    ! Combine T
    Call GComb(T_gb, Natom)
    ! Combine Sigma
    Call GComb(Sigx_Gb, Natom)
    Call GComb(Sigy_Gb, Natom)
    Call GComb(Sigz_Gb, Natom)
#endif /* (paraend)*/
    Return
  End Subroutine FillAlpGB1

  Subroutine GBSolv2(GBEnr, X, Y, Z, DX, DY, DZ, &
       JNbl, INbl, Inbl14, INblo14, &
       QInte, Islct, Jslct, GbFixed)
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
    ! Generalized Born with Block (e.g. Lambda-dynamics, FEP, LEP)
    ! Type 2 ( Born radius depend on the coefficient of block)
    ! S.B. 10/30/99
    !   Faster Version (All procedure is in line)
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  use number
  use consta
  use dimens_fcm
  use stream
  use psf
  use block_fcm
  use inbnd
  use parallel
  use lambdam
  use gbswit, only: qgbswit, louter, c2ofnb, gbswit_setup, es_switch

    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*)
    Real(chm_real) GBEnr, X(*), Y(*), Z(*), DX(*), DY(*), DZ(*)
    Integer ISlct(*), JSlct(*), GbFixed
    Logical QInte

    ! Local variables
    Integer i, ic, j, k
    real(chm_real) rij, xij, yij, zij, Sqrtrij
    real(chm_real) Dij, Sij, SqrtSij, Sij3half, expDij
    real(chm_real) Aij, Aji, SigijDif
    real(chm_real) Factor, Dxi, Dyi, Dzi, GBtmp
    Logical LOK
    INTEGER IBL, JBL, KBL, KK, k1, k2
    real(chm_real) COEF, fsw, dfsw
    Integer ITemp, NPr, jpr

#if KEY_GBINLINE==1
    real(chm_real) dAij, Ratio 
#endif

    ! Initialized GB potential energy
    DO i = 1, NBLOCK
       GB_BLOCK(i) = Zero
    ENDDO

    Factor = CCELEC * ( One - One / Eps_gb )

    call gbswit_setup()

    ! Loop over bonded atoms first:
    !----------------------------------------------------------
    Do ic = MynodP, Nbond, Numnod
       LOK = .true.
       i = ib(ic)
       j = jb(ic)
       !CCCCCCCCCCCCC
       ! getting the number of block and coefficient
       IBL = IBLCKP(I)
       JBL = IBLCKP(J)
       IF (JBL < IBL) THEN
          KK=JBL+IBL*(IBL-1)/2
       ELSE
          KK=IBL+JBL*(JBL-1)/2
       ENDIF
       COEF = BLCOEB(KK)
       !CCCCCCCCCCCCC

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then
          ! Check if both in pair are fixed!
          If ( .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then
             IF (.NOT. ((IBL /= 1.AND.JBL.NE.1).AND.(IBL.NE.JBL)) ) THEN
                ! IF both i & j do not belong to block 1 and  block(i) is not equal to
                ! block(j), skipp next lines. (COEF is always zero)
                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)
                rij = xij * xij + yij * yij + zij * zij
                Sqrtrij = Sqrt(rij)
                SigijDif = Sigij(Sqrtrij,r_gb(i),r_gb(j),P(2),Zero)
                ! Subtract contributions from non-bonded terms to correct for over counting
                SigijDif=SigijDif- &
                     Sigij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))

                IF (IBL * JBL  ==  1 )THEN
                   ! Both i & j belong to block 1
                   Aij = Zero
                   Aji = Zero

                   DO k = LSTRT, NBLOCK
                      COEF = BLCOEB(1+k*(k-1)/2)

                      If ( imove(i)  ==  0 ) then
                         Aij = Aij + COEF * T_GB(Natom*(k-1)+j)
                      Endif

                      If ( imove(j)  ==  0 ) then
                         Aji = Aji + COEF * T_GB(Natom*(k-1)+i)
                      Endif
                   ENDDO

                   Aij = Factor * (Cg(j) * Cg(j) + Aij) &
                        * SiGijDif * VOL_GB(i) / CCELEC
                   Aji = Factor * (Cg(i) * Cg(i) + Aji) &
                        * SiGijDif * VOL_GB(j) / CCELEC

                   If ( imove(i)  ==  0 ) then
                      DX(i) = DX(i) + xij * Aij
                      DY(i) = DY(i) + yij * Aij
                      DZ(i) = DZ(i) + zij * Aij
                   Endif
                   If ( imove(j)  ==  0 ) then
                      DX(j) = DX(j) - xij * Aji
                      DY(j) = DY(j) - yij * Aji
                      DZ(j) = DZ(j) - zij * Aji
                   Endif

                ELSE
                   k1 = i
                   k2 = j
                   IF (IBL  ==  1) k1 = Natom * (JBL - 1) + i
                   IF (JBL  ==  1) k2 = Natom * (IBL - 1) + j

                   If ( imove(i)  ==  0 ) then
                      Aij = &
                           + Factor * ( Cg(j) * Cg(j) + T_GB(k2) ) &
                           * SiGijDif * VOL_GB(i) / CCELEC

                      DX(i) = DX(i) + COEF * xij * Aij
                      DY(i) = DY(i) + COEF * yij * Aij
                      DZ(i) = DZ(i) + COEF * zij * Aij
                   Endif

                   If ( imove(j)  ==  0 ) then
                      Aji = &
                           + Factor * ( Cg(i) * Cg(i) + T_GB(k1) ) &
                           * SiGijDif * VOL_GB(j) / CCELEC

                      DX(j) = DX(j) - COEF * xij * Aji
                      DY(j) = DY(j) - COEF * yij * Aji
                      DZ(j) = DZ(j) - COEF * zij * Aji
                   Endif
                ENDIF
             ENDIF
          Endif
       Endif
    Enddo

    ! Next atoms connected through angles
    !----------------------------------------------------------
    Do ic = MynodP, NTheta, NumNod

       LOK = .true.
       i = it(ic)
       j = kt(ic)
       KK = jt(ic)
       !CCCCCCCCCCCCC
       ! getting the number of block and coefficient
       ! second atom to make a angle is ignored ( jt(ic) )
       IBL = IBLCKP(I)
       JBL = IBLCKP(J)
       KBL = IBLCKP(KK)
       IF (IBL  ==  JBL) JBL = KBL
       IF (JBL < IBL) THEN
          KK=JBL+IBL*(IBL-1)/2
       ELSE
          KK=IBL+JBL*(JBL-1)/2
       ENDIF
       COEF = BLCOEA(KK)
       !CCCCCCCCCCCCC

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then

          ! Check if both in pair are fixed!
          If ( .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then
             !CCCC
             IF (.NOT. ((IBL /= 1.AND.JBL.NE.1).AND.(IBL.NE.JBL)) ) THEN
                ! IF both i & j do not belong to block 1 and  block(i) is not equal to
                ! block(j), skipp next lines. (COEF is always zero)
                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)
                rij = xij * xij + yij * yij + zij * zij
                Sqrtrij = Sqrt(rij)
                SigijDif = Sigij(Sqrtrij,r_gb(i),r_gb(j),P(3),Zero)
                ! Subtract contributions from non-bonded terms to correct for over counting
                SigijDif=SigijDif- &
                     Sigij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))

                IF (IBL * JBL  ==  1 ) THEN
                   ! Both i & j belong to block 1
                   Aij = Zero
                   Aji = Zero

                   DO k = LSTRT, NBLOCK
                      COEF = BLCOEB(1+k*(k-1)/2)
                      If ( imove(i)  ==  0 ) then
                         Aij = Aij + COEF * T_GB(Natom*(k-1)+j)
                      Endif
                      If ( imove(j)  ==  0 ) then
                         Aji = Aji + COEF * T_GB(Natom*(k-1)+i)
                      Endif
                   ENDDO

                   Aij = Factor * ( Cg(j) * Cg(j) + Aij) &
                        * SiGijDif * VOL_GB(i) / CCELEC
                   Aji = Factor * ( Cg(i) * Cg(i) + Aji) &
                        * SiGijDif * VOL_GB(j) / CCELEC

                   If ( imove(i)  ==  0 ) then
                      DX(i) = DX(i) +  xij * Aij
                      DY(i) = DY(i) +  yij * Aij
                      DZ(i) = DZ(i) +  zij * Aij
                   Endif
                   If ( imove(j)  ==  0 ) then
                      DX(j) = DX(j) - xij * Aji
                      DY(j) = DY(j) - yij * Aji
                      DZ(j) = DZ(j) - zij * Aji
                   Endif
                ELSE
                   k1 = i
                   k2 = j
                   IF (IBL  ==  1) k1 = Natom * (JBL - 1) + i
                   IF (JBL  ==  1) k2 = Natom * (IBL - 1) + j

                   If ( imove(i)  ==  0 ) then
                      Aij = &
                           + Factor * ( Cg(j) * Cg(j) + T_GB(k2) ) &
                           * SiGijDif * VOL_GB(i) / CCELEC


                      DX(i) = DX(i) + COEF * xij * Aij
                      DY(i) = DY(i) + COEF * yij * Aij
                      DZ(i) = DZ(i) + COEF * zij * Aij
                   Endif

                   If ( imove(j)  ==  0 ) then
                      Aji = &
                           + Factor * ( Cg(i) * Cg(i) + T_GB(k1) ) &
                           * SiGijDif * VOL_GB(j) / CCELEC

                      DX(j) = DX(j) - COEF * xij * Aji
                      DY(j) = DY(j) - COEF * yij * Aji
                      DZ(j) = DZ(j) - COEF * zij * Aji
                   Endif
                ENDIF
             ENDIF
          Endif
       Endif
    Enddo

    ! Now we need to get the atoms in the exclusion list
    !-----------------------------------------------------
    Itemp = 0
    ! Define the atom bounds for this processor.
    Do i = MyNodP, Natom, NumNod

       If (i  /=  1 ) ITemp = INblo14(i - 1)
       Dxi = Zero
       Dyi = Zero
       Dzi = Zero

       NPr = INblo14(i) - ITemp
       Do jpr = 1, NPr
          LOK = .true.
          k = Inbl14(Itemp+jpr)
          j = Abs(k)
          If ( k  >  0 .and. &
               .not. ( imove(i)  ==  1 &
               .and. imove(j)  ==  1 ) ) then
             If ( QInte ) LOK = &
                  ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
             If (LOK) Then
                !CCCCCCCCCCCCC
                ! getting the number of block and coefficient
                IBL = IBLCKP(I)
                JBL = IBLCKP(J)
                IF (JBL < IBL) THEN
                   KK=JBL+IBL*(IBL-1)/2
                ELSE
                   KK=IBL+JBL*(JBL-1)/2
                ENDIF
                COEF = BLCOEA(KK)
                !CCCCCCCCCCCCC
                IF(.NOT.((IBL /= 1.AND.JBL.NE.1).AND.(IBL.NE.JBL)))THEN
                   ! IF both i & j do not belong to block 1 and  block(i) is not equal to
                   ! block(j), skipp next lines. (COEF is always zero)
                   xij = x(i) - x(j)
                   yij = y(i) - y(j)
                   zij = z(i) - z(j)
                   rij = xij * xij + yij * yij + zij * zij
                   Sqrtrij = Sqrt(rij)
                   SiGijDif = SiGij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))

                   IF (IBL * JBL  ==  1 ) THEN
                      ! Both i & j belong to block 1
                      Aij = Zero
                      Aji = Zero

                      DO k = LSTRT, NBLOCK
                         COEF = BLCOEB(1+k*(k-1)/2)
                         Aij = Aij + Factor * COEF * SigijDif * VOL_GB(i) &
                              * (Cg(j)*Cg(j) + T_GB(Natom*(k-1)+j))/CCELEC

                         Aji = Aji + Factor * COEF * SigijDif * VOL_GB(j) &
                              * (Cg(i)*Cg(i) + T_GB(Natom*(k-1)+i))/CCELEC

                         If ( CG(j) * CG(i)  /=  Zero ) then
                            Dij=rij/( Four*Alph_Gb(Natom*(k-1)+i) &
                                 *Alph_Gb(Natom*(k-1)+j))
                            expDij = exp(-Dij)
                            Sij=rij+Alph_Gb(Natom*(k-1)+i) &
                                 *Alph_Gb(Natom*(k-1)+j)*expDij
                            SqrtSij = Sqrt(Sij)
                            Sij3half = Sij * SqrtSij
                            Sij = CG(i) * CG(j) * Factor / Sij3half

                            Aij = Aij + COEF * Sij * (One - PT25 * expDij)
                            Aji = Aji + COEF * Sij * (One - PT25 * expDij)

                            Sij = Sij * SqrtSij * SqrtSij
                            IF (QPRNTV) VBGENB(1) = VBGENB(1) - Sij
                            GB_BLOCK(k) = GB_BLOCK(k) - Sij

                            If (Qanalys) Then
                               GB_Atm(i) = GB_Atm(i) - Half * COEF * Sij
                               GB_Atm(j) = GB_Atm(j) - Half * COEF * Sij
                            Endif
                         Endif
                      ENDDO

                      If ( imove(i)  ==  0 ) then
                         Dxi = Dxi + xij * Aij
                         Dyi = Dyi + yij * Aij
                         Dzi = Dzi + zij * Aij
                      Endif

                      If ( imove(j)  ==  0 ) then
                         DX(j) = DX(j) - xij * Aji
                         DY(j) = DY(j) - yij * Aji
                         DZ(j) = DZ(j) - zij * Aji
                      Endif
                   ELSE

                      k1 = i
                      k2 = j
                      IF (IBL  ==  1) k1 = Natom * (JBL - 1) + i
                      IF (JBL  ==  1) k2 = Natom * (IBL - 1) + j
                      IF (JBL  <  IBL) JBL = IBL

                      Aij = Factor * ( Cg(j) * Cg(j) + T_GB(k2) ) &
                           * SiGijDif * VOL_GB(i) / CCELEC

                      Aji = Factor * ( Cg(i) * Cg(i) + T_GB(k1) ) &
                           * SiGijDif * VOL_GB(j) / CCELEC

                      If ( CG(j) * CG(i)  /=  Zero ) then
                         Dij = rij / ( Four * Alph_Gb(k1) * Alph_Gb(k2) )
                         expDij = exp(-Dij)
                         Sij = rij + Alph_Gb(k1) * Alph_Gb(k2) * expDij
                         SqrtSij = Sqrt(Sij)
                         Sij3half = Sij * SqrtSij
                         Sij = CG(i) * CG(j) * Factor / Sij3half

                         Aij = Aij + Sij - Sij * PT25 * expDij
                         Aji = Aji + Sij - Sij * PT25 * expDij

                         Sij = Sij * SqrtSij * SqrtSij
                         IF (QPRNTV) VBGENB(Max(IBL,JBL)) = VBGENB(Max(IBL,JBL)) - Sij
                         GB_BLOCK(JBL) = GB_BLOCK(JBL) - Sij

                         If (Qanalys) Then
                            GB_Atm(i) = GB_Atm(i) - Half * COEF * Sij
                            GB_Atm(j) = GB_Atm(j) - Half * COEF * Sij
                         Endif
                      Endif

                      If ( imove(i)  ==  0 ) then
                         Dxi = Dxi + COEF * xij * Aij
                         Dyi = Dyi + COEF * yij * Aij
                         Dzi = Dzi + COEF * zij * Aij
                      Endif

                      If ( imove(j)  ==  0 ) then
                         DX(j) = DX(j) - COEF * xij * Aji
                         DY(j) = DY(j) - COEF * yij * Aji
                         DZ(j) = DZ(j) - COEF * zij * Aji
                      Endif

                   ENDIF
                ENDIF
             Endif
          Endif
       Enddo

       If (NPr > 0) then
          DX(i) = DX(i) + Dxi
          DY(i) = DY(i) + Dyi
          DZ(i) = DZ(i) + Dzi
       Endif

    Enddo

    ! Finally do pairs of nonbonded terms
    ! For lists use different construct here
    Itemp = 0
#if KEY_IMCUBES==1
    if(lbycbim)then
       if( (INBL(Natom)-INBL(Natom+natom))  > 0)then
          write(outu,*) INBL(Natom),INBL(Natom-1)
          Call WrnDie(-3,'<genborn.src>GBSOLV2', &
               'GBsolv (genborn.src), last atom has pairs')
       endif
    endif
#endif 
    Do i = 1, Natom-1
#if KEY_IMCUBES==1
       if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif

       Dxi = Zero
       Dyi = Zero
       Dzi = Zero

       NPr = INbl(i) - ITemp
       Do jpr = 1, NPr
          k = JNbl(Itemp+jpr)
          j = Abs(k)
          xij = x(i) - x(j)
          yij = y(i) - y(j)
          zij = z(i) - z(j)

          rij = xij * xij + yij * yij + zij * zij
          If (rij  <  C2OfNb) Then
             !CCCCCCCCCCCCC
             ! getting the number of block and coefficient
             IBL = IBLCKP(I)
             JBL = IBLCKP(J)
             IF (JBL < IBL) THEN
                KK=JBL+IBL*(IBL-1)/2
             ELSE
                KK=IBL+JBL*(JBL-1)/2
             ENDIF
             COEF = BLCOEA(KK)
             !CCCCCCCCCCCCC
             IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1) &
                  .AND.(IBL /= JBL)) ) THEN
                ! IF both i & j do not belong to block 1 and  block(i)
                !    is not equal to  block(j),
                !    skipp next lines. (COEF is always zero)
                Sqrtrij = Sqrt(rij)
#if KEY_GBINLINE==0
                SiGijDif = SiGij(Sqrtrij,r_gb(i),r_gb(j),P(4),P(5))
#else /**/
                dAij = Zero
                Aij = One
                Ratio = rij /  &
                     (( r_gb(i) + r_gb(j) ) * ( r_gb(i) + r_gb(j) ))
                If ( Ratio  <=  One/P(5) ) then
                   Aij = Aij - Cos( Ratio * P(5) * Pi )
                   dAij = dAij + Aij &
                        * P(5) * Pi * Ratio * Sin( Ratio * P(5) * Pi ) &
                        / Sqrtrij
                   Aij = Half * Aij
                   Aij = Aij * Aij
                Endif
                Aij = P(4) * Aij / ( rij * rij )
                SiGijDif = P(4) * dAij / ( rij * rij * Sqrtrij ) &
                     - Four * Aij /  rij
#endif 

                call es_switch(rij, FSw, DfSw) ! Electrostatic Switch func
                IF (IBL * JBL  ==  1) THEN
                   ! Both i & j belong to block 1
                   Aij = Zero
                   Aji = Zero
                   DO k = LSTRT, NBLOCK
                      COEF = BLCOEB(1+k*(k-1)/2)
                      Aij = Aij + Factor * COEF * SiGijDif * VOL_GB(i) &
                           * ( Cg(j) * Cg(j)+T_GB(Natom*(k-1)+j))/CCELEC

                      Aji = Aji + Factor * COEF * SiGijDif * VOL_GB(j) &
                           * ( Cg(i) * Cg(i)+T_GB(Natom*(k-1)+i))/CCELEC

                      If ( CG(j) * CG(i)  /=  Zero ) then
                         Dij=rij/(Four*Alph_Gb(Natom*(k-1)+i) &
                              *Alph_Gb(Natom*(k-1)+j))
                         expDij = exp(-Dij)
                         Sij=rij+Alph_Gb(Natom*(k-1)+i) &
                              *Alph_Gb(Natom*(k-1)+j)*expDij
                         SqrtSij = Sqrt(Sij)
                         Sij3half = Sij * SqrtSij
                         Sij = CG(i) * CG(j) * Factor / Sij3half * FSw

                         Aij = Aij + COEF * Sij * (One - PT25 * expDij)
                         Aji = Aji + COEF * Sij * (One - PT25 * expDij)

                         Sij = Sij * SqrtSij * SqrtSij

                         if ( qgbswit .and. louter ) then
                            Aij = Aij  - COEF * Sij * DFSw
                            Aji = Aji  - COEF * Sij * DFSw
                         endif

                         IF (QPRNTV) VBGENB(1) = VBGENB(1) - Sij
                         GB_BLOCK(k) = GB_BLOCK(k) - Sij

                         If (Qanalys) Then
                            GB_Atm(i) = GB_Atm(i) - Half * COEF * Sij
                            GB_Atm(j) = GB_Atm(j) - Half * COEF * Sij
                         Endif
                      Endif
                   ENDDO

                   If ( imove(i)  ==  0 ) then
                      Dxi = Dxi + xij * Aij
                      Dyi = Dyi + yij * Aij
                      Dzi = Dzi + zij * Aij
                   Endif

                   If ( imove(j)  ==  0 ) then
                      DX(j) = DX(j) - xij * Aji
                      DY(j) = DY(j) - yij * Aji
                      DZ(j) = DZ(j) - zij * Aji
                   Endif

                ELSE
                   k1 = i
                   k2 = j
                   IF (IBL  ==  1) k1 = Natom * (JBL - 1) + i
                   IF (JBL  ==  1) k2 = Natom * (IBL - 1) + j
                   IF (JBL  <  IBL) JBL = IBL

                   Aij = Factor * ( Cg(j) * Cg(j) + T_GB(k2) ) &
                        * SiGijDif * VOL_GB(i) / CCELEC

                   Aji = Factor * ( Cg(i) * Cg(i) + T_GB(k1) ) &
                        * SiGijDif * VOL_GB(j) / CCELEC

                   Sij = Zero
                   If ( CG(j) * CG(i)  /=  Zero ) then
                      Dij = rij / ( Four * Alph_Gb(k1) * Alph_Gb(k2) )
                      expDij = exp(-Dij)
                      Sij = rij + Alph_Gb(k1) * Alph_Gb(k2) * expDij
                      SqrtSij = Sqrt(Sij)
                      Sij3half = Sij * SqrtSij

                      Sij = CG(i) * CG(j) * Factor / Sij3half * FSw

                      Aij = Aij + Sij - Sij * PT25 * expDij
                      Aji = Aji + Sij - Sij * PT25 * expDij
                      Sij = Sij * SqrtSij * SqrtSij

                      if ( qgbswit .and. louter ) then
                         Aij = Aij  - Sij * DFSw
                         Aji = Aji  - Sij * DFSw
                      endif


                      IF (QPRNTV) VBGENB(Max(IBL,JBL)) = VBGENB(Max(IBL,JBL)) - Sij
                      GB_BLOCK(JBL) = GB_BLOCK(JBL) - Sij

                      If (Qanalys) Then
                         GB_Atm(i) = GB_Atm(i) - Half * COEF * Sij
                         GB_Atm(j) = GB_Atm(j) - Half * COEF * Sij
                      Endif
                   Endif

                   If ( imove(i)  ==  0 ) then
                      Dxi = Dxi + COEF * xij * Aij
                      Dyi = Dyi + COEF * yij * Aij
                      Dzi = Dzi + COEF * zij * Aij
                   Endif

                   If ( imove(j)  ==  0 ) then
                      DX(j) = DX(j) - COEF * xij * Aji
                      DY(j) = DY(j) - COEF * yij * Aji
                      DZ(j) = DZ(j) - COEF * zij * Aji
                   Endif
                ENDIF
             ENDIF
          ENDIF
       Enddo

       If (NPr > 0) then
          DX(i) = DX(i) + Dxi
          DY(i) = DY(i) + Dyi
          DZ(i) = DZ(i) + Dzi
       Endif

       ITemp = INbl(i)
    Enddo

    !-----------------------------------------------------------------------
    ! Add the GB terms of fixed atom - fixed atom
    !
    ! ***  Causion  ***
    ! All GB terms of fixed atom - fixed atom are assumed to be nonbonded.
    ! Therefore, CtOnNb shoulb be longer than 1-2, 1-3 interactions
    If ( GbFixed  >  1 ) Then
       call gbswit_setup()
       Do i = 1, Natom-1
          Do j = i+1, Natom

             If ( ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) .and. &
                  ( CG(j) * CG(i)  /=  Zero ) ) THEN
                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)
                rij = xij * xij + yij * yij + zij * zij
                If (rij  <  c2ofnb ) Then
                   !CCCCCCCCCCCCC
                   ! getting the number of block and coefficient
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)
                   IF (JBL < IBL) THEN
                      KK=JBL+IBL*(IBL-1)/2
                   ELSE
                      KK=IBL+JBL*(JBL-1)/2
                   ENDIF
                   COEF = BLCOEA(KK)
                   !CCCCCCCCCCCCC
                   IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1) &
                        .AND.(IBL /= JBL)) ) THEN
                      call es_switch(rij, FSw) ! Electrostatic Switch func
                      IF (IBL * JBL  ==  1) THEN
                         ! Both i & j belong to block 1
                         Aij = Zero
                         Aji = Zero
                         DO k = LSTRT, NBLOCK
                            COEF = BLCOEB(1+k*(k-1)/2)
                            If ( CG(j) * CG(i)  /=  Zero ) then
                               Dij=rij/(Four*Alph_Gb(Natom*(k-1)+i) &
                                    *Alph_Gb(Natom*(k-1)+j))
                               expDij = exp(-Dij)
                               Sij=rij+Alph_Gb(Natom*(k-1)+i) &
                                    *Alph_Gb(Natom*(k-1)+j)*expDij
                               SqrtSij = Sqrt(Sij)
                               Sij = CG(i) * CG(j) * Factor / SqrtSij * FSw
                               IF (QPRNTV) VBGENB(1) = VBGENB(1) - Sij * COEF
                               GB_BLOCK(k) = GB_BLOCK(k) - Sij
                               If (Qanalys) Then
                                  GB_Atm(i) = GB_Atm(i) - Half * COEF * Sij
                                  GB_Atm(j) = GB_Atm(j) - Half * COEF * Sij
                               Endif
                            Endif
                         ENDDO
                      ELSE
                         k1 = i
                         k2 = j
                         IF (IBL  ==  1) k1 = Natom * (JBL - 1) + i
                         IF (JBL  ==  1) k2 = Natom * (IBL - 1) + j
                         IF (JBL  <  IBL) JBL = IBL

                         Dij = rij / ( Four * Alph_Gb(k1) * Alph_Gb(k2) )
                         expDij = exp(-Dij)
                         Sij = rij + Alph_Gb(k1) * Alph_Gb(k2) * expDij
                         SqrtSij = Sqrt(Sij)
                         Sij = CG(i) * CG(j) * Factor / SqrtSij * FSw
                         IF (QPRNTV) VBGENB(JBL) = VBGENB(JBL) - Sij
                         GB_BLOCK(JBL) = GB_BLOCK(JBL) - Sij

                         If (Qanalys) Then
                            GB_Atm(i) = GB_Atm(i) - Half * COEF * Sij
                            GB_Atm(j) = GB_Atm(j) - Half * COEF * Sij
                         Endif
                      ENDIF
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
       ENDDO
    ENDIF

    ! Finally add in self-terms
    !------------------------------------------------------------
    ! Define the atom bounds for this processor.
    Do i = MynodP, Natom, NumNod
       !CCCCCCCCCCCCC
       ! getting the number of block and coefficient
       IBL = IBLCKP(I)
       KK=IBL+IBL*(IBL-1)/2
       COEF = BLCOEA(KK)
       !CCCCCCCCCCCCC
       LOK = .true.
       If ( Qinte ) LOK = Islct(i)  ==  1
       If ( LOK ) Then
          If ( Cg(i)  /=  Zero ) Then
             GBtmp = Factor * Half * Cg(i)  * Cg(i)
             IF (IBL  ==  1) THEN
                DO j = LSTRT, NBLOCK
                   COEF = BLCOEA(1+j*(j-1)/2)
                   IF (IMOVE(I)  /=  1) THEN
                      DX(i) = DX(i) + FACTOR * COEF / CCELEC &
                           * (Cg(i)*Cg(i) + T_GB(Natom*(j-1)+i) ) &
                           * Sigx_Gb(Natom*(j-1)+i)
                      DY(i) = DY(i) + FACTOR * COEF / CCELEC &
                           * (Cg(i)*Cg(i) + T_GB(Natom*(j-1)+i) ) &
                           * Sigy_Gb(Natom*(j-1)+i)
                      DZ(i) = DZ(i) + FACTOR * COEF / CCELEC &
                           * (Cg(i)*Cg(i) + T_GB(Natom*(j-1)+i) ) &
                           * Sigz_Gb(Natom*(j-1)+i)
                   ENDIF
                   IF (QPRNTV)  &
                        VBGENB(1) = VBGENB(1) - GBtmp / Alph_Gb(Natom*(j-1)+i) * COEF
                   GB_BLOCK(j) = GB_BLOCK(j) - GBtmp / Alph_Gb(Natom*(j-1)+i)
                ENDDO
             ELSE
                IF (QPRNTV) VBGENB(IBL) = VBGENB(IBL) - GBtmp / Alph_Gb(i)
                GB_BLOCK(IBL) = GB_BLOCK(IBL) - GBtmp / Alph_Gb(i)
                IF (IMOVE(I)  /=  1 ) THEN
                   DX(i) = DX(i) + Factor * COEF / CCELEC &
                        * (Cg(i) * Cg(i) +T_GB(i) ) * Sigx_Gb(i)
                   DY(i) = DY(i) + Factor * COEF / CCELEC &
                        * (Cg(i) * Cg(i) +T_GB(i) ) * Sigy_Gb(i)
                   DZ(i) = DZ(i) + Factor * COEF / CCELEC &
                        * (Cg(i) * Cg(i) +T_GB(i) ) * Sigz_Gb(i)
                ENDIF
             ENDIF

             If (Qanalys) Then
                IF (IBL  ==  1) THEN
                   DO j = LSTRT, NBLOCK
                      GB_Atm(i) = GB_Atm(i) &
                           - COEF * CG(i)*CG(i)*Factor*Half/ &
                           Alph_Gb(Natom*(j-1)+i)
                   ENDDO
                ELSE
                   GB_Atm(i) = GB_Atm(i) &
                        - COEF * CG(i) * CG(i) * Factor * Half/Alph_Gb(i)
                ENDIF
             Endif
          Endif
       Endif

    Enddo
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    GBEnr = Zero
    DO I = LSTRT, NBLOCK
       COEF = BLCOEA(I+I*(I-1)/2)
       GBEnr = GBEnr + COEF * GB_BLOCK(I)
#if KEY_BLOCK==1 /*ldm*/
       IF (QLDM.or.QLMC) BIFLAM(I) = BIFLAM(I) + GB_BLOCK(I)
#endif /*  LDM*/
    ENDDO
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
    ! Note need to GCOMB GB_Atm(i) if QAnalys true.
#if KEY_PARALLEL==1 /*paraend*/
    ! Combine GB_Atm
    If(Qanalys) Call GComb(GB_atm, Natom)
#endif /* (paraend)*/
    Return
  End Subroutine GBSolv2

  Subroutine FillAlpGB2( X, Y, Z, &
       JNbl, INbl, Inbl14, INblo14, &
       QInte, Islct, JSlct, GbFixed)
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    !
    ! Generalized Born with Block (e.g. Lambda-dynamics, FEP, LEP)
    ! Type 2 ( Born radius depend on the coefficient of block)
    ! S.B. 10/30/99
    !
    !CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  use number
  use consta
  use dimens_fcm
  use psf
  use block_fcm
  use inbnd
  use parallel
  use lambdam
  use gbswit, only: gbswit_setup, es_switch
    implicit none

    Integer JNbl(*), INbl(*), Inbl14(*), INblo14(*), &
         Islct(*), Jslct(*)
    Real(chm_real) X(*), Y(*), Z(*)
    Logical Qinte
    ! Local variables
#if KEY_PARALLEL==1
    Integer AtFrst, AtLast 
#endif
    Integer i, ic, j, k, GbFixed
    real(chm_real) rij, xij, yij, zij, Aij, Tij
    real(chm_real) Dij, Sij, SqrtSij, Sij3half, expDij
    real(chm_real) C2OfNb
    LOgical LOK
    INTEGER IBL, JBL, KBL, k1, k2
    real(chm_real) COEF, FSw
    Integer ITemp, NPr, jpr
#if KEY_GBINLINE==1
    real(chm_real) dAij, Ratio 
#endif

    GbFixed = 0
    Do i = 1, Natom
       If ( QAnalys ) GB_Atm(i) = Zero
       If ( imove(i)  ==  1 ) GbFixed = GbFixed + 1
    Enddo

    DO i = 1, NATOM * NBLOCK
       Alph_Gb(i) = Zero
       Sigx_Gb(i)  = Zero
       Sigy_Gb(i)  = Zero
       Sigz_Gb(i)  = Zero
       T_GB(i)     = Zero
    ENDDO

    call gbswit_setup()

    ! Loop over bonded atoms first
    Do ic = MynodP, Nbond, Numnod

       LOK = .true.
       i = ib(ic)
       j = jb(ic)

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then
          !CCCCCCCCCCCCC
          ! getting the number of block
          IBL = IBLCKP(I)
          JBL = IBLCKP(J)
          IF (IBL  ==  1) THEN
             IF (JBL  ==  1 ) THEN
                k1 = i
                k2 = j
             ELSE
                k1 = Natom * (JBL - 1) + i
                k2 = j
             ENDIF
          ELSE
             IF (JBL  ==  1) THEN
                k1 = i
                k2 = Natom * (IBL - 1) + j
             ELSE
                k1 = i
                k2 = j
             ENDIF
          ENDIF
          !CCCCCCCCCCCCC
          IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1).AND.(IBL.NE.JBL)) ) THEN
             ! IF both i & j do not belong to block 1 and  block(i) is not equal to
             ! block(j), skipp next lines. (COEF is always zero)
             xij = x(i) - x(j)
             yij = y(i) - y(j)
             zij = z(i) - z(j)

             rij = xij * xij + yij * yij + zij * zij

             Aij = Alphij(rij, r_gb(i), r_gb(j), P(2), Zero)
             ! Subtract contributions from non-bonded terms to correct for over counting
             Aij = Aij - Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))

             Alph_Gb(k1) = Alph_Gb(k1) + Aij * VOL_GB(j)
             Alph_Gb(k2) = Alph_Gb(k2) + Aij * VOL_GB(i)

             rij = sqrt ( rij )

             ! Now do sigma contributions
             Aij = Sigij(rij, r_gb(i), r_gb(j), P(2), Zero)
             ! Subtract contributions from non-bonded terms to correct for over counting
             Aij = Aij - Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))

             If ( imove(i)  ==  0 ) then
                Sigx_Gb(k1) = Sigx_Gb(k1) + Aij * VOL_GB(j) * xij
                Sigy_Gb(k1) = Sigy_Gb(k1) + Aij * VOL_GB(j) * yij
                Sigz_Gb(k1) = Sigz_Gb(k1) + Aij * VOL_GB(j) * zij
             Endif

             If ( imove(j)  ==  0 ) then
                Sigx_Gb(k2) = Sigx_Gb(k2) - Aij * VOL_GB(i) * xij
                Sigy_Gb(k2) = Sigy_Gb(k2) - Aij * VOL_GB(i) * yij
                Sigz_Gb(k2) = Sigz_Gb(k2) - Aij * VOL_GB(i) * zij
             Endif

          ENDIF
       ENDIF
    Enddo

    ! Next atoms connected through angles
    !--------------------------------------------------------------------
    Do ic = MynodP, NTheta, NumNod

       LOK = .true.
       i = it(ic)
       j = kt(ic)
       K = jt(ic)

       If ( QInte ) LOK = ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
       If (LOK) Then
          !CCCCCCCCCCCCC
          ! getting the number of block
          IBL = IBLCKP(I)
          JBL = IBLCKP(J)
          KBL = IBLCKP(K)
          IF (IBL  ==  JBL) JBL = KBL
          IF (IBL  ==  1) THEN
             IF (JBL  ==  1 ) THEN
                k1 = i
                k2 = j
             ELSE
                k1 = Natom * (JBL - 1) + i
                k2 = j
             ENDIF
          ELSE
             IF (JBL  ==  1) THEN
                k1 = i
                k2 = Natom * (IBL - 1) + j
             ELSE
                k1 = i
                k2 = j
             ENDIF
          ENDIF
          !CCCCCCCCCCCCC
          IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1).AND.(IBL.NE.JBL)) ) THEN
             ! IF both i & j do not belong to block 1 and  block(i) is not equal to
             ! block(j), skipp next lines. (COEF is always zero)
             xij = x(i) - x(j)
             yij = y(i) - y(j)
             zij = z(i) - z(j)

             rij = xij * xij + yij * yij + zij * zij

             Aij = Alphij(rij, r_gb(i), r_gb(j), P(3), Zero)
             ! Subtract contributions from non-bonded terms to correct for over counting
             Aij = Aij - Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))

             Alph_Gb(k1) = Alph_Gb(k1) + Aij * VOL_GB(j)
             Alph_Gb(k2) = Alph_Gb(k2) + Aij * VOL_GB(i)

             rij = sqrt ( rij )
             ! Now do sigma contributions
             Aij = Sigij(rij, r_gb(i), r_gb(j), P(3), Zero)

             ! Subtract contributions from non-bonded terms to correct for over counting
             Aij = Aij - Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))

             If ( imove(i)  ==  0 ) then
                Sigx_Gb(k1) = Sigx_Gb(k1) + Aij * VOL_GB(j) * xij
                Sigy_Gb(k1) = Sigy_Gb(k1) + Aij * VOL_GB(j) * yij
                Sigz_Gb(k1) = Sigz_Gb(k1) + Aij * VOL_GB(j) * zij
             Endif

             If ( imove(j)  ==  0 ) then
                Sigx_Gb(k2) = Sigx_Gb(k2) - Aij * VOL_GB(i) * xij
                Sigy_Gb(k2) = Sigy_Gb(k2) - Aij * VOL_GB(i) * yij
                Sigz_Gb(k2) = Sigz_Gb(k2) - Aij * VOL_GB(i) * zij
             Endif
          ENDIF
       Endif
    Enddo

    ! Do atoms on exclusion list
    !--------------------------------------------------------------------
    Itemp = 0
    ! Define the atom bounds for this processor.
    Do i = MyNodP, Natom, NumNod

       If (i  /=  1 ) ITemp = INblo14(i - 1)
       NPr = INblo14(i) - ITemp
       Do jpr = 1, NPr

          LOK = .true.
          k = Inbl14(Itemp+jpr)
          j = Abs(k)
          ! Don't do fixed atoms from exclusion list here, get them below.
          If ( k  >  0 &
               .and. .not. (imove(i)  ==  1 &
               .and. imove(j)  ==  1) ) then
             If ( QInte ) LOK = &
                  ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
             If (LOK) Then
                !CCCCCCCCCCCCC
                ! getting the number of block
                IBL = IBLCKP(I)
                JBL = IBLCKP(J)
                IF (IBL  ==  1) THEN
                   IF (JBL  ==  1 ) THEN
                      k1 = i
                      k2 = j
                   ELSE
                      k1 = Natom * (JBL - 1) + i
                      k2 = j
                   ENDIF
                ELSE
                   IF (JBL  ==  1) THEN
                      k1 = i
                      k2 = Natom * (IBL - 1) + j
                   ELSE
                      k1 = i
                      k2 = j
                   ENDIF
                ENDIF
                !CCCCCCCCCCCCC
                IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1) &
                     .AND.(IBL /= JBL)) ) THEN
                   ! IF both i & j do not belong to block 1 and  block(i)
                   !    is not equal to  block(j),
                   !    skipp next lines. (COEF is always zero)
                   xij = x(i) - x(j)
                   yij = y(i) - y(j)
                   zij = z(i) - z(j)

                   rij = xij * xij + yij * yij + zij * zij

#if KEY_GBINLINE==0
                   Aij =  Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                   Aij = One
                   Ratio = rij /  &
                        ((r_gb(i) + r_gb(j) ) * ( r_gb(i) + r_gb(j) ))
                   If ( Ratio  <=  One/P(5) ) then
                      Aij = Aij - Cos( Ratio * P(5) * Pi )
                      Aij = Half * Aij
                      Aij = Aij * Aij
                   Endif
                   Aij = P(4) * Aij / ( rij * rij )
#endif 
                   Alph_Gb(k1) = Alph_Gb(k1) + Aij * VOL_GB(j)
                   Alph_Gb(k2) = Alph_Gb(k2) + Aij * VOL_GB(i)

                   rij = sqrt ( rij )

                   ! Now do sigma contributions
#if KEY_GBINLINE==0
                   Aij =  Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                   dAij = Zero
                   If ( Ratio  <=  One/P(5) ) then
                      dAij = dAij + (  One - Cos( Ratio * P(5) * Pi ) ) &
                           * P(5) * Pi * Ratio * Sin( Ratio * P(5) * Pi )/rij
                   Endif
                   Aij = P(4) * dAij / ( rij * rij * rij * rij * rij ) &
                        - Four * Aij / ( rij * rij )
#endif 
                   If ( imove(i)  ==  0 ) then
                      Sigx_Gb(k1) = Sigx_Gb(k1) + Aij * VOL_GB(j) * xij
                      Sigy_Gb(k1) = Sigy_Gb(k1) + Aij * VOL_GB(j) * yij
                      Sigz_Gb(k1) = Sigz_Gb(k1) + Aij * VOL_GB(j) * zij
                   Endif

                   If ( imove(j)  ==  0 ) then
                      Sigx_Gb(k2) = Sigx_Gb(k2) - Aij * VOL_GB(i) * xij
                      Sigy_Gb(k2) = Sigy_Gb(k2) - Aij * VOL_GB(i) * yij
                      Sigz_Gb(k2) = Sigz_Gb(k2) - Aij * VOL_GB(i) * zij
                   Endif
                ENDIF
                ! i belongs to block 1
             Endif
          Endif
       Enddo

    Enddo

    C2OfNB = CtOfNb * CtOfNb
    ! For lists us different construct here
    Itemp = 0
    Do i = 1, Natom-1
#if KEY_IMCUBES==1
       if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif

       NPr = INbl(i) - ITemp
       Do jpr = 1, NPr
          k = JNbl(Itemp+jpr)
          j = Abs(k)

          xij = x(i) - x(j)
          yij = y(i) - y(j)
          zij = z(i) - z(j)

          rij = xij * xij + yij * yij + zij * zij

          If (rij  <  c2ofnb) Then
             !CCCCCCCCCCCCC
             ! getting the number of block
             IBL = IBLCKP(I)
             JBL = IBLCKP(J)
             IF (IBL  ==  1) THEN
                IF (JBL  ==  1 ) THEN
                   k1 = i
                   k2 = j
                ELSE
                   k1 = Natom * (JBL - 1) + i
                   k2 = j
                ENDIF
             ELSE
                IF (JBL  ==  1) THEN
                   k1 = i
                   k2 = Natom * (IBL - 1) + j
                ELSE
                   k1 = i
                   k2 = j
                ENDIF
             ENDIF
             !CCCCCCCCCCCCC
             IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1) &
                  .AND.(IBL /= JBL)) ) THEN
                ! IF both i & j do not belong to block 1 and  block(i)
                ! is not equal to  block(j),
                ! skipp next lines. (COEF is always zero)
                ! Both i & j belong to block 1
                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)

                rij = xij * xij + yij * yij + zij * zij

#if KEY_GBINLINE==0
                Aij =  Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                Aij = One
                Ratio = rij /  &
                     (( r_gb(i) + r_gb(j) ) * ( r_gb(i) + r_gb(j) ))
                If ( Ratio  <=  One/P(5) ) then
                   Aij = Aij - Cos( Ratio * P(5) * Pi )
                   Aij = Half * Aij
                   Aij = Aij * Aij
                Endif
                Aij = P(4) * Aij / ( rij * rij )
#endif 

                Alph_Gb(k1) = Alph_Gb(k1) + Aij * VOL_GB(j)
                Alph_Gb(k2) = Alph_Gb(k2) + Aij * VOL_GB(i)

                rij = sqrt ( rij )

                ! Now do sigma contributions
#if KEY_GBINLINE==0
                Aij =  Sigij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                dAij = Zero
                If ( Ratio  <=  One/P(5) ) then
                   dAij = dAij + (  One - Cos( Ratio * P(5) * Pi ) ) &
                        * P(5) * Pi * Ratio * Sin( Ratio * P(5) * Pi ) / rij
                Endif
                Aij = P(4) * dAij / ( rij * rij * rij * rij * rij ) &
                     - Four * Aij / ( rij * rij )
#endif 
                If ( imove(i)  ==  0 ) then
                   Sigx_Gb(k1) = Sigx_Gb(k1) + Aij * VOL_GB(j) * xij
                   Sigy_Gb(k1) = Sigy_Gb(k1) + Aij * VOL_GB(j) * yij
                   Sigz_Gb(k1) = Sigz_Gb(k1) + Aij * VOL_GB(j) * zij
                Endif

                If ( imove(j)  ==  0 ) then
                   Sigx_Gb(k2) = Sigx_Gb(k2) - Aij * VOL_GB(i) * xij
                   Sigy_Gb(k2) = Sigy_Gb(k2) - Aij * VOL_GB(i) * yij
                   Sigz_Gb(k2) = Sigz_Gb(k2) - Aij * VOL_GB(i) * zij
                Endif
             ENDIF
          Endif
       Enddo

       ITemp = INbl(i)
    Enddo

    !---------------------------------------------------------------------
    ! If we use lists and there are fixed atoms, add contributions from
    ! fixed atom lists to Alpha
    If ( GbFixed  >  1 ) Then
       C2OfNB = CtOfNb * CtOfNb
       Do i = 1, Natom-1
          Do j = i+1, Natom

             If ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) then

                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)

                rij = xij * xij + yij * yij + zij * zij
                If (rij  <  c2ofnb) Then
                   !CCCCCCCCCCCCC
                   ! getting the number of block
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)
                   IF (IBL  ==  1) THEN
                      IF (JBL  ==  1 ) THEN
                         k1 = i
                         k2 = j
                      ELSE
                         k1 = Natom * (JBL - 1) + i
                         k2 = j
                      ENDIF
                   ELSE
                      IF (JBL  ==  1) THEN
                         k1 = i
                         k2 = Natom * (IBL - 1) + j
                      ELSE
                         k1 = i
                         k2 = j
                      ENDIF
                   ENDIF
                   !CCCCCCCCCCCCC
                   IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1) &
                        .AND.(IBL /= JBL)) ) THEN
                      ! IF both i & j do not belong to block 1 and  block(i)
                      !    is not equal to  block(j),
                      !    skipp next lines. (COEF is always zero)
#if KEY_GBINLINE==0
                      Aij =  Alphij(rij, r_gb(i), r_gb(j), P(4), P(5))
#else /**/
                      Aij = One
                      Ratio = rij /  &
                           (( r_gb(i) + r_gb(j) ) * ( r_gb(i) + r_gb(j) ))
                      If ( Ratio  <=  One/P(5) ) then
                         Aij = Aij - Cos( Ratio * P(5) * Pi )
                         Aij = Half * Aij
                         Aij = Aij * Aij
                      Endif
                      Aij = P(4) * Aij / ( rij * rij )
#endif 
                      Alph_Gb(k1) = Alph_Gb(k1) + Aij * VOL_GB(j)
                      Alph_Gb(k2) = Alph_Gb(k2) + Aij * VOL_GB(i)

                   ENDIF
                Endif
             Endif
          Enddo
       Enddo

    Endif

#if KEY_PARALLEL==1 /*paralpha*/
    ! Get Alpha's updated on all processors
    Call GComb(Alph_gb, Natom * NBlock)
#endif /* (paralpha)*/
    ! finally getting the alpha
    !----------------------------------------------------
    Do i = 1, Natom
       IBL = IBLCKP(I)

       Alph_Gb(i) = Alph_Gb(i) + CCELEC*P(1)/(Two*r_gb(i)*r_gb(i)) &
            - CCELEC/(Two*r_gb(i)*P(6))

       IF (IBL  ==  1) THEN
          DO k = LSTRT, NBLOCK
             j = Natom*(k-1)+i
             Alph_Gb(j)  = Alph_Gb(j) + Alph_Gb(i)
             Alph_Gb(j) = - CCELEC / (Alph_Gb(j) * Two)
             If ( Alph_Gb(j)  <=  Zero )  Alph_Gb(j) = CutAlph
             If ( Alph_Gb(j)  >  CutAlph )  Alph_Gb(j) = CutAlph

             IF (IMOVE(I)  /=  1) THEN
                Sigx_Gb(j) = Sigx_Gb(j) + Sigx_Gb(i)
                Sigy_Gb(j) = Sigy_Gb(j) + Sigy_Gb(i)
                Sigz_Gb(j) = Sigz_Gb(j) + Sigz_Gb(i)
             ENDIF
          ENDDO
       ENDIF

       Alph_Gb(i) = - CCELEC / ( Alph_Gb(i) * Two )

       If ( Alph_Gb(i)  <=  Zero )  Alph_Gb(i) = CutAlph
       If ( Alph_Gb(i)  >  CutAlph )  Alph_Gb(i) = CutAlph

    Enddo


    ! ***Calculation of T***
    ! First get the atoms on exclusion lists
    !--------------------------------------------------------
    Itemp = 0
    ! Define the atom bounds for this processor.
    Do i = MynodP, Natom, NumNod
       If (i  /=  1 ) ITemp = INblo14(i - 1)
       NPr = INblo14(i) - Itemp
       If ( Cg(i)  /=  Zero .and. Npr  >  0 ) then

          Do jpr = 1, NPr
             LOK = .true.
             k = Inbl14(Itemp+jpr)
             j = Abs(k)
             If ( ( Cg(j)  /=  Zero .and. k  >  0 ) .and. &
                  .not. ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) ) then

                If ( QInte ) LOK = &
                     ( Islct(i)  ==  1 ) .and. ( Jslct(j) .eq. 1 )
                If (LOK) Then
                   !CCCCCCCCCCCCC
                   ! getting the number of block
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)
                   !CCCCCCCCCCCCC
                   IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1) &
                        .AND.(IBL /= JBL)) ) THEN
                      ! IF both i & j do not belong to block 1 and  block(i)
                      ! is not equal to  block(j),
                      ! skipp next lines. (COEF is always zero)
                      xij = x(i) - x(j)
                      yij = y(i) - y(j)
                      zij = z(i) - z(j)
                      rij = xij * xij + yij * yij + zij * zij

                      IF (IBL * JBL  ==  1) THEN
                         ! Both i & j belong to block 1

                         DO k = LSTRT, NBLOCK
                            k1 = NATOM * (k-1) + i
                            k2 = NATOM * (k-1) + j
                            Dij = rij / ( Four * Alph_Gb(k1)*Alph_Gb(k2))
                            expDij = exp(-Dij)
                            Sij = rij + Alph_Gb(k1) * Alph_Gb(k2) * expDij
                            SqrtSij = Sqrt(Sij)
                            Sij3half = Sij * SqrtSij

                            Sij = CG(i) * CG(j) / Sij3half

                            T_GB(k1) = T_GB(k1) &
                                 + Sij * Alph_Gb(k1) * expDij &
                                 * (Alph_Gb(k1) * Alph_Gb(k2) + PT25 * rij)

                            T_GB(k2) = T_GB(k2) &
                                 + Sij * Alph_Gb(k2) * expDij &
                                 * (Alph_Gb(k2) * Alph_Gb(k1) + PT25 * rij)
                         ENDDO

                      ELSE

                         k1 = i
                         k2 = j
                         IF (IBL  ==  1 ) k1 = Natom * (JBL - 1) + i
                         IF (JBL  ==  1 ) k2 = Natom * (IBL - 1) + j
                         ! IF (JBL  ==  1 ) k2 = Natom * (JBL - 1) + i

                         ! k = Natom * (JBL - 1) + i
                         Dij = rij / ( Four * Alph_Gb(k1) * Alph_Gb(k2) )
                         expDij = exp(-Dij)

                         Sij = rij + Alph_Gb(k1) * Alph_Gb(k2) * expDij
                         SqrtSij = Sqrt(Sij)
                         Sij3half = Sij * SqrtSij

                         Sij = CG(i) * CG(j) / Sij3half

                         T_GB(k1) = T_GB(k1) &
                              + Sij * Alph_Gb(k1) * expDij &
                              * ( Alph_Gb(k1) * Alph_Gb(k2) + PT25 * rij )

                         T_GB(k2) = T_GB(k2) &
                              + Sij * Alph_Gb(k2) * expDij &
                              * ( Alph_Gb(k2) * Alph_Gb(k1) + PT25 * rij )
                      ENDIF
                   ENDIF
                Endif
             Endif
          Enddo
       Endif
    Enddo

    ! Now non-bonded contributions
    Itemp = 0
    Do i = 1, Natom-1
#if KEY_IMCUBES==1
       if(lbycbim) ITEMP=INBL(I+NATOM)  
#endif
       NPr = INbl(i) - ITemp
       If ( Cg(i)  /=  Zero .and. Npr  >  0 ) then

          Tij = Zero
          Do jpr = 1, NPr
             k = JNbl(Itemp+jpr)
             j = Abs(k)
             If ( Cg(j)  /=  Zero ) then
                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)

                rij = xij * xij + yij * yij + zij * zij
                If (rij  <  c2ofnb) Then
                   ! getting the number of block and coefficient
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)

                   IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1) &
                        .AND.(IBL /= JBL)) ) THEN
                      ! IF both i & j do not belong to block 1 and  block(i)
                      ! is not equal to  block(j),
                      ! skipp next lines. (COEF is always zero)

                      call es_switch(rij, FSw) ! Electrostatic Switch func
                      IF (IBL * JBL  ==  1) THEN
                         ! Both i & j belong to block 1
                         DO k = LSTRT, NBLOCK
                            k1 = Natom * (k-1) + i
                            k2 = Natom * (k-1) + j
                            Dij = rij / ( Four * Alph_Gb(k1) * Alph_Gb(k2) )
                            expDij = exp(-Dij)
                            Sij = rij + Alph_Gb(k1) * Alph_Gb(k2) * expDij
                            SqrtSij = Sqrt(Sij)
                            Sij3half = Sij * SqrtSij
                            Sij = CG(i) * CG(j) / Sij3half * FSw

                            T_GB(k1) = T_GB(k1) &
                                 + Sij * Alph_Gb(k1) * expDij &
                                 * ( Alph_Gb(k1) * Alph_Gb(k2) + PT25 * rij )

                            T_GB(k2) = T_GB(k2) &
                                 + Sij * Alph_Gb(k2) * expDij &
                                 * ( Alph_Gb(k2) * Alph_Gb(k1) + PT25 * rij )
                         ENDDO

                      ELSE
                         k1 = i
                         k2 = j
                         IF (IBL  ==  1)  k1 = Natom * (JBL - 1) + i
                         IF (JBL  ==  1)  k2 = Natom * (IBL - 1) + j

                         Dij = rij / ( Four * Alph_Gb(k1) * Alph_Gb(k2) )
                         expDij = exp(-Dij)

                         Sij = rij + Alph_Gb(k1) * Alph_Gb(k2) * expDij
                         SqrtSij = Sqrt(Sij)
                         Sij3half = Sij * SqrtSij

                         Sij = CG(i) * CG(j) / Sij3half * FSw

                         T_GB(k1) = T_GB(k1) &
                              + Sij * Alph_Gb(k1) * expDij &
                              * ( Alph_Gb(k1) * Alph_Gb(k2) + PT25 * rij )

                         T_GB(k2) = T_GB(k2) &
                              + Sij * Alph_Gb(k2) * expDij &
                              * ( Alph_Gb(k2) * Alph_Gb(k1) + PT25 * rij )
                      ENDIF
                   ENDIF

                Endif
             Endif
          Enddo
       Endif
       ITemp = INbl(i)
    Enddo
    !-----------------------------------------------------------------------
    ! Add the GB terms of fixed atom - fixed atom
    !
    ! ***  Caution  ***
    ! All GB terms of fixed atom - fixed atom are assumed to be nonbonded.
    ! Therefore, CtOnNb shoulb be longer than 1-2, 1-3 interactions
    If ( GbFixed  >  1 ) Then
       call gbswit_setup()
       Do i = 1, Natom-1
          Do j = i+1, Natom

             If ( ( imove(i)  ==  1 .and. imove(j) .eq. 1 ) .and. &
                  ( CG(j) * CG(i)  /=  Zero ) ) THEN
                xij = x(i) - x(j)
                yij = y(i) - y(j)
                zij = z(i) - z(j)
                rij = xij * xij + yij * yij + zij * zij
                If (rij  <  c2ofnb ) Then
                   !CCCCCCCCCCCCC
                   ! getting the number of block and coefficient
                   IBL = IBLCKP(I)
                   JBL = IBLCKP(J)
                   !CCCCCCCCCCCCC
                   IF (.NOT. ((IBL /= 1 .AND. JBL.NE.1) &
                        .AND.(IBL /= JBL)) ) THEN
                      call es_switch(rij, FSw) ! Electrostatic Switch func
                      IF (IBL * JBL  ==  1) THEN
                         ! Both i & j belong to block 1
                         DO k = LSTRT-1, NBLOCK - 1
                            Dij = rij / ( Four * Alph_Gb(Natom*k+i) &
                                 * Alph_Gb(Natom*k+j) )
                            expDij = exp(-Dij)
                            Sij = rij + Alph_Gb(Natom*k+i) &
                                 * Alph_Gb(Natom*k+j) * expDij
                            SqrtSij = Sqrt(Sij)
                            Sij3half = Sij * SqrtSij
                            Sij = CG(i) * CG(j) / Sij3half * FSw

                            T_GB(Natom*k+i) = T_GB(Natom*k+i) &
                                 + Sij * Alph_Gb(Natom*k+i) * expDij &
                                 * ( Alph_Gb(Natom*k+i) * Alph_Gb(Natom*k+j) &
                                 + PT25 * rij )

                            T_GB(Natom*k+j) = T_GB(Natom*k+j) &
                                 + Sij * Alph_Gb(Natom*k+j) * expDij &
                                 * ( Alph_Gb(Natom*k+j)*Alph_Gb(Natom*k+i) &
                                 + PT25 * rij )
                         ENDDO

                      ELSE
                         k1 = i
                         k2 = j
                         IF (IBL  ==  1)  k1 = Natom * (JBL - 1) + i
                         IF (JBL  ==  1)  k2 = Natom * (IBL - 1) + j

                         Dij = rij / ( Four * Alph_Gb(k1) * Alph_Gb(k2) )
                         expDij = exp(-Dij)

                         Sij = rij + Alph_Gb(k1) * Alph_Gb(k2) * expDij
                         SqrtSij = Sqrt(Sij)
                         Sij3half = Sij * SqrtSij
                         Sij = CG(i) * CG(j) / Sij3half * FSw

                         T_GB(k1) = T_GB(k1) &
                              + Sij * Alph_Gb(k1) * expDij &
                              * ( Alph_Gb(k1) * Alph_Gb(k2) + PT25 * rij )

                         T_GB(k2) = T_GB(k2) &
                              + Sij * Alph_Gb(k2) * expDij &
                              * ( Alph_Gb(k2) * Alph_Gb(k1) + PT25 * rij )
                      ENDIF
                   ENDIF
                Endif
             Endif
          Enddo
       Enddo
    Endif

#if KEY_PARALLEL==1 /*paraend*/
    ! Combine T
    Call GComb(T_gb, Natom * NBlock)
    ! Combine Sigma
    Call GComb(Sigx_Gb, Natom * NBlock)
    Call GComb(Sigy_Gb, Natom * NBlock)
    Call GComb(Sigz_Gb, Natom * NBlock)
#endif /* (paraend)*/

    Return
  END Subroutine FillAlpGB2
#endif /* (block_main)*/
end module genborn

