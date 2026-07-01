!> This module provides an interface between the OpenMM implemented energy function
!> and the CHARMM BLOCK facility. Currently implemented:
!>  BLOCK scaled bonds, urey bradley, angles, torsions, 
!> improper torsions, vdws and electrostatics
!> Written by C.L. Brooks III, January, 2013

module omm_block
#if KEY_OPENMM==1
  use chm_kinds
  use number
  use stream, only: OUTU, PRNLEV
  use block_ltm, only : QBLOCK, NBLOCK, IBLCKP, &
       BLCOEB, BLCOEA, BLCOED, BLCOEP, BLCOEE, BLCOEV, QSOFT_OMM
  use block_fcm, only : QNOBO, QNOAN, QNOPH, QNOIM, QBVSPLT
  implicit none
  integer, parameter :: MAX_EXCEPTIONS = int(1e+7)
  integer :: blockOneAtom = 0
contains

  real*8 function getlambda()
  !> This function gets the lambda value between the environment and the product atoms assuming
  !> the overall potential is scaled as V_total = V_env + (1-lambda)*V_reac + lambda*V_prod
  !>   kk = ibl + jbl * ( jbl - 1 ) / 2, ibl = 1, jbl = 3, kk=4; lambda=bleocv(kk)
  !> Note that if nblock == 2 we use the ibl,jbl = 1,2, kk=2 and lambda = 1 - bleocv(kk)  
    real*8 :: lambda
    lambda = blcoev(4)
    if (nblock == 2) lambda = 1.0 - blcoev(2)
    getlambda = lambda
  end function getlambda
    
    
  !> scale bond force constants if block is in effect
  real*8 function blockscale_bond(i, j, kbond)
    integer*4 :: i, j, ibl, jbl, kk
    real*8 :: kbond, coef
    coef = ONE
#if KEY_BLOCK==1
    if(qblock .and. .not. qnobo) then
       ibl = IBLCKP(i)
       jbl = IBLCKP(j)
       if (jbl < ibl) then
          kk = jbl
          jbl = ibl
          ibl = kk
       endif
       kk = ibl + jbl * ( jbl - 1 ) / 2
       coef = BLCOEB(kk)
    endif
#endif 
    blockscale_bond = kbond * coef
  end function blockscale_bond
  
  !> scale urey-bradley force constants if block is in effect
  real*8 function blockscale_ub(i, k, kbond)
    integer*4 :: i, k, ibl, jbl, kk
    real*8 :: kbond, coef
    coef = ONE
#if KEY_BLOCK==1
    if(qblock .and. .not. qnoan) then
       ibl = IBLCKP(i)
       jbl = IBLCKP(k)
       if (jbl < ibl) then
          kk = jbl
          jbl = ibl
          ibl = kk
       endif
       kk = ibl + jbl * ( jbl - 1 ) / 2
       coef = BLCOEA(kk)
    endif
#endif 
    blockscale_ub = kbond * coef
  end function blockscale_ub
    
  ! scale angle force constants if block is in effect
  real*8 function blockscale_angle(i, j, k, kangle)
    integer*4 :: i, j, k, ibl, jbl, kk
    real*8 :: kangle, coef
    coef = ONE
#if KEY_BLOCK==1
    if(qblock .and. .not. qnoan) then
       ibl = IBLCKP(i)
       jbl = IBLCKP(j)
       kk = IBLCKP(k)
       if(ibl == jbl) jbl = kk
       if (jbl < ibl) then
          kk = jbl
          jbl = ibl
          ibl = kk
       endif
       kk = ibl + jbl * ( jbl - 1 ) / 2
       coef = BLCOEA(kk)
    endif
#endif 
    blockscale_angle = kangle * coef
  end function blockscale_angle
  
  ! scale torsion/improper force constants if block is in effect
  real*8 function blockscale_dihe(i, j, k, l, cpd, kdihe)
    integer*4 :: i, j, k, l, ibl, jbl, kkk, lll, cpd
    real*8 :: kdihe, coef
    coef = ONE
#if KEY_BLOCK==1
    if( qblock .and. .not. &
         ( (qnoim .and. (cpd == 0)) .or. (qnoph .and. (cpd .ne. 0)) ) ) then
       ibl = IBLCKP(i)
       jbl = IBLCKP(j)
       kkk = IBLCKP(k)
       lll = IBLCKP(l)
       if(ibl == jbl) jbl = kkk
       if(ibl == jbl) jbl = lll
       
       if (jbl < ibl) then
          kkk = jbl
          jbl = ibl
          ibl = kkk
       endif
       kkk = ibl + jbl * ( jbl - 1 ) / 2
! LNI bugfix 2016-04-28
       coef = BLCOED(kkk)
    endif
#endif 
    blockscale_dihe = kdihe * coef
  end function blockscale_dihe

   !> Check if block is turned on and qnoph is not
  subroutine checkblock_cmap
#if KEY_BLOCK==1
    if(qblock .and. .not. qnoph) then
       if(prnlev >= 2) write(OUTU, '(A)' ) &
            'CHARMM> OpenMM - Warning BLOCK in use, CMAP not supported w/ BLOCK in OpenMM'
    endif
    if(qblock .and. qnoph) then
       if(prnlev >= 2) write(OUTU, '(A)' ) &
            'CHARMM> OpenMM - Warning BLOCK/QNOPH in use, CMAP not scaled in OpenMM'
    endif
#endif 
    return
  end subroutine checkblock_cmap

  !> scale electrostatic and vdw interaction parameters if block 
  !> is in effect.
  !> Note that at present we assume there are only three blocks
  !> with the environment represent block 1, reactant and product
  !> occupying blocks 2 and 3. We also assume blocks 2 and 3 are
  !> excluded from one another. 
  subroutine blockscale_nonbond(j, charge, epsilon)
    integer*4 :: j
    real*8 :: charge, epsilon, scaleElec, scalevdW
#if KEY_BLOCK==1
    if(qblock) then
       call blockscale_nbpair(blockOneAtom, j, scaleElec, scalevdW)
       ! scaled by single factor since energy is qi*qj
       charge = charge * scaleElec              
       ! scaled by square since energy is ~ sqrt( eps_i*eps_j )
       epsilon = epsilon * scalevdW * scalevdW
    endif
#endif 
    return
  end subroutine blockscale_nonbond

  !> scale electrostatic and vdw interaction parameters if block 
  !> is in effect.
  !> Note that at present we assume there are only three blocks
  !> with the environment represent block 1, reactant and product
  !> occupying blocks 2 and 3. We also assume blocks 2 and 3 are
  !> excluded from one another. 
  subroutine blockscale_nbpair(i, j, scaleElec, scalevdW)
    use fast, only : LOWTP
    integer*4 :: i, j, ibl, jbl, kk
    real*8 :: scaleElec, scalevdW
    scaleElec = ONE
    scalevdW = ONE
#if KEY_BLOCK==1
    if(qblock) then
       ibl = blocknumber(i)
       jbl = blocknumber(j)
       kk = lowtp(max(ibl,jbl)) + ibl + jbl
       scaleElec = BLCOEE(kk)
       scalevdW = BLCOEV(kk)
       !write(*,'(i4,1x,i4,1x,i4,1x,i4,2x,f5.2,1x,f5.2)') &
       !      i, j, ibl, jbl, scaleElec, scalevdW
    endif
#endif 
    return
  end subroutine blockscale_nbpair
  integer function FindblockOneAtom(natom)
    integer :: iatom, natom
    FindblockOneAtom = 0
#if KEY_BLOCK==1
    if(qblock) then
       do iatom=1, natom
          if( IBLCKP(iatom) == 1 ) then
             FindblockOneAtom = iatom
             return
          endif
       enddo
    endif
#endif
  end function FindblockOneAtom
  integer function blocknumber(iatom)
    integer :: iatom
    blocknumber = 1
#if KEY_BLOCK==1
    if(qblock) blocknumber = IBLCKP(iatom) 
#endif
  end function blocknumber
#endif /*  */
end module omm_block

