SUBROUTINE NULL_FQQ
  return
END SUBROUTINE NULL_FQQ

! These routines handle the addition of contributions to FQ charge forces
! from QM/MM interaction. To disable these contributions, it is necessary
! to disable the QM energy calculation with the "SKIP QMEL" command.
!
! Implementation is for GAMESS, CADPAC, and QUANTUM
! Implementation for MNDO97 and SQUANTM is not done, and it should be included here.
!
! In the interests of memory conservation, the FlucQ QM/MM interaction is
! calculated by (essentially) regenerating the necessary 1-electron interactions
! between the QM electrons and the MM atoms, after the SCF, and then factoring
! in the density matrix. This is a similar procedure to that followed by the
! main QM/MM interface code for calculating gradients. The relevant code is
! "borrowed" from the 1-electron codes of the QM packages.
!
! From within each QM package, two FlucQ routines must be called:-
!
!    FQQCOR   Calculates the FlucQ QM/MM core term - i.e. the contribution to
!             charge forces from the QM nuclei (ab initio) or cores
!             (semi-empirical)
!    FQQMMM   Calculates the main FlucQ QM/MM 1-electron interaction - i.e. the
!             contribution to charge forces from the QM electron density


! QUANTUM interface; in this case the interaction is divided between the
! "core" (nucleus+inner electrons) term and the "valence" term. The first
! is accounted for by FQQCOR, the second by FQQMMM. QUANTUM uses eV for
! energy, so these must be converted to CHARMM units by means of the
! conversion factor 23.061d0

#if KEY_FLUCQ==1 /*flucq*/
#if KEY_QUANTUM==1 /*quantum_main*/
SUBROUTINE FQQCOR(CHMAT,VALUE)
  !
  !     Adds the contribution from the QM core repulsion energy VALUE to
  !     the electrostatic potential on atom CHMAT
  !
  !     Author: Ben Webb, 1999
  !
  use flucqm,only: fqcfor
  use chm_kinds
  use dimens_fcm
  use flucq
  implicit none
  INTEGER CHMAT
  real(chm_real) VALUE
  IF (QFLUC) FQCFOR(CHMAT) = FQCFOR(CHMAT) + VALUE * 23.061d0
  RETURN
END SUBROUTINE FQQCOR

SUBROUTINE FQQMMM (DENS, COORDS, INBX, JNBX, EXFACT, &
     XG, YG, ZG, QGRPMM)
  !
  !     Adds the contribution from the QM 1-electron valence terms to the
  !     FlucQ charge forces, and converts from MOPAC units (eV) to CHARMM
  !
  !     Modified from the QUANTUM interface subroutine EAMPCE
  !     DENS = passed density matrix from SCF
  !
  !     Author: Ben Webb, 2000
  !
  use flucqm,only: fqcfor
  use chm_kinds
  use dimens_fcm
  use memory
  use quantm
  use number
  use psf
  use flucq
  implicit none
  !
  ! . Passed variables.
  Integer  inbx(*), jnbx(*)
  Logical  qgrpmm(*)
  real(chm_real)   DENS(*), coords(3,*), exfact, xg(*), yg(*), zg(*)
  !DCC+
  ! . Local variables.
  Integer   I, ndim2, nlen, space
  Logical   am1
  ! . Pointers.
  integer,allocatable,dimension(:) :: jnbl
  real(chm_real),allocatable,dimension(:) :: cgl, &
       dxpppp, dypppp, dzpppp, dxpzpz, dypzpz, dzpzpz, &
       e1b, fact0, fact1m, &
       fact1p, fact20, fact2m, fact2p, fact4p, pppp, &
       pzpz, rqm, rqm2, rqm2b, rqmb, rqmi, scale, &
       sfct1m, sfct1p, sfct20, sfct2m, sfct2p, sfct4p, spz, &
       ss, work1, work2, work3, work4, xn, yn, zn, &
       xn2, yn2, zn2, xqm, yqm, zqm
  !
  IF (.NOT. QFLUC) RETURN

  am1   = Index(keywrd,'AM1') .gt. 0
  ! . Allocate space for the arrays.
  ndim2 = natom - natqm
  If (ndim2 .gt. 0) then
     nlen   = ndim2
     space  = ndim2
     call chmalloc('fluqqmmm.src','FQQMMM','jnbl',space,intg=jnbl)
     call chmalloc('fluqqmmm.src','FQQMMM','cgl',nlen,crl=cgl)
     call chmalloc('fluqqmmm.src','FQQMMM','e1b',nlen,crl=e1b)
     call chmalloc('fluqqmmm.src','FQQMMM','fact0',nlen,crl=fact0)
     call chmalloc('fluqqmmm.src','FQQMMM','fact1m',nlen,crl=fact1m)
     call chmalloc('fluqqmmm.src','FQQMMM','fact1p',nlen,crl=fact1p)
     call chmalloc('fluqqmmm.src','FQQMMM','fact20',nlen,crl=fact20)
     call chmalloc('fluqqmmm.src','FQQMMM','fact2m',nlen,crl=fact2m)
     call chmalloc('fluqqmmm.src','FQQMMM','fact2p',nlen,crl=fact2p)
     call chmalloc('fluqqmmm.src','FQQMMM','fact4p',nlen,crl=fact4p)
     call chmalloc('fluqqmmm.src','FQQMMM','pppp',nlen,crl=pppp)
     call chmalloc('fluqqmmm.src','FQQMMM','pzpz',nlen,crl=pzpz)
     call chmalloc('fluqqmmm.src','FQQMMM','rqm2',nlen,crl=rqm2)
     call chmalloc('fluqqmmm.src','FQQMMM','rqm2b',nlen,crl=rqm2b)
     call chmalloc('fluqqmmm.src','FQQMMM','rqm',nlen,crl=rqm)
     call chmalloc('fluqqmmm.src','FQQMMM','rqmb',nlen,crl=rqmb)
     call chmalloc('fluqqmmm.src','FQQMMM','rqmi',nlen,crl=rqmi)
     call chmalloc('fluqqmmm.src','FQQMMM','scale',nlen,crl=scale)
     call chmalloc('fluqqmmm.src','FQQMMM','sfct1m',nlen,crl=sfct1m)
     call chmalloc('fluqqmmm.src','FQQMMM','sfct1p',nlen,crl=sfct1p)
     call chmalloc('fluqqmmm.src','FQQMMM','sfct20',nlen,crl=sfct20)
     call chmalloc('fluqqmmm.src','FQQMMM','sfct2m',nlen,crl=sfct2m)
     call chmalloc('fluqqmmm.src','FQQMMM','sfct2p',nlen,crl=sfct2p)
     call chmalloc('fluqqmmm.src','FQQMMM','sfct4p',nlen,crl=sfct4p)
     call chmalloc('fluqqmmm.src','FQQMMM','spz',nlen,crl=spz)
     call chmalloc('fluqqmmm.src','FQQMMM','ss',nlen,crl=ss)
     call chmalloc('fluqqmmm.src','FQQMMM','work1',nlen,crl=work1)
     call chmalloc('fluqqmmm.src','FQQMMM','work2',nlen,crl=work2)
     call chmalloc('fluqqmmm.src','FQQMMM','work3',nlen,crl=work3)
     call chmalloc('fluqqmmm.src','FQQMMM','work4',nlen,crl=work4)
     call chmalloc('fluqqmmm.src','FQQMMM','xn',nlen,crl=xn)
     call chmalloc('fluqqmmm.src','FQQMMM','yn',nlen,crl=yn)
     call chmalloc('fluqqmmm.src','FQQMMM','zn',nlen,crl=zn)
     call chmalloc('fluqqmmm.src','FQQMMM','xn2',nlen,crl=xn2)
     call chmalloc('fluqqmmm.src','FQQMMM','yn2',nlen,crl=yn2)
     call chmalloc('fluqqmmm.src','FQQMMM','zn2',nlen,crl=zn2)
     call chmalloc('fluqqmmm.src','FQQMMM','xqm',nlen,crl=xqm)
     call chmalloc('fluqqmmm.src','FQQMMM','yqm',nlen,crl=yqm)
     call chmalloc('fluqqmmm.src','FQQMMM','zqm',nlen,crl=zqm)
     call chmalloc('fluqqmmm.src','FQQMMM','dxpppp',10*nlen,crl=dxpppp)
     call chmalloc('fluqqmmm.src','FQQMMM','dypppp',nlen,crl=dypppp)
     call chmalloc('fluqqmmm.src','FQQMMM','dzpppp',nlen,crl=dzpppp)
     call chmalloc('fluqqmmm.src','FQQMMM','dxpzpz',nlen,crl=dxpzpz)
     call chmalloc('fluqqmmm.src','FQQMMM','dypzpz',nlen,crl=dypzpz)
     call chmalloc('fluqqmmm.src','FQQMMM','dzpzpz',nlen,crl=dzpzpz)
     ! . Calculate the integrals
     Call FQQMM2(DENS, coords, inbx, jnbx, exfact, am1, &
          xg, yg, zg, qgrpmm, ndim2, &
          jnbl,cgl,e1b, &
          fact0, fact1m, fact1p, &
          fact20, fact2m, fact2p, fact4p, &
          pppp, pzpz, rqm2, rqm2b,rqm, &
          rqmb, rqmi, scale, sfct1m, &
          sfct1p, sfct20, sfct2m, sfct2p, &
          sfct4p, spz, ss, work1, &
          work2, work3, work4, &
          xn, yn, zn, xn2, yn2, &
          zn2, xqm, yqm, zqm, &
          dxpppp, dypppp, dzpppp, &
          dxpzpz, dypzpz, dzpzpz, FQCFOR)
     !DCC+
     ! . Free storage space.
     call chmdealloc('fluqqmmm.src','FQQMMM','jnbl',space,intg=jnbl)
     call chmdealloc('fluqqmmm.src','FQQMMM','cgl',nlen,crl=cgl)
     call chmdealloc('fluqqmmm.src','FQQMMM','e1b',nlen,crl=e1b)
     call chmdealloc('fluqqmmm.src','FQQMMM','fact0',nlen,crl=fact0)
     call chmdealloc('fluqqmmm.src','FQQMMM','fact1m',nlen,crl=fact1m)
     call chmdealloc('fluqqmmm.src','FQQMMM','fact1p',nlen,crl=fact1p)
     call chmdealloc('fluqqmmm.src','FQQMMM','fact20',nlen,crl=fact20)
     call chmdealloc('fluqqmmm.src','FQQMMM','fact2m',nlen,crl=fact2m)
     call chmdealloc('fluqqmmm.src','FQQMMM','fact2p',nlen,crl=fact2p)
     call chmdealloc('fluqqmmm.src','FQQMMM','fact4p',nlen,crl=fact4p)
     call chmdealloc('fluqqmmm.src','FQQMMM','pppp',nlen,crl=pppp)
     call chmdealloc('fluqqmmm.src','FQQMMM','pzpz',nlen,crl=pzpz)
     call chmdealloc('fluqqmmm.src','FQQMMM','rqm2',nlen,crl=rqm2)
     call chmdealloc('fluqqmmm.src','FQQMMM','rqm2b',nlen,crl=rqm2b)
     call chmdealloc('fluqqmmm.src','FQQMMM','rqm',nlen,crl=rqm)
     call chmdealloc('fluqqmmm.src','FQQMMM','rqmb',nlen,crl=rqmb)
     call chmdealloc('fluqqmmm.src','FQQMMM','rqmi',nlen,crl=rqmi)
     call chmdealloc('fluqqmmm.src','FQQMMM','scale',nlen,crl=scale)
     call chmdealloc('fluqqmmm.src','FQQMMM','sfct1m',nlen,crl=sfct1m)
     call chmdealloc('fluqqmmm.src','FQQMMM','sfct1p',nlen,crl=sfct1p)
     call chmdealloc('fluqqmmm.src','FQQMMM','sfct20',nlen,crl=sfct20)
     call chmdealloc('fluqqmmm.src','FQQMMM','sfct2m',nlen,crl=sfct2m)
     call chmdealloc('fluqqmmm.src','FQQMMM','sfct2p',nlen,crl=sfct2p)
     call chmdealloc('fluqqmmm.src','FQQMMM','sfct4p',nlen,crl=sfct4p)
     call chmdealloc('fluqqmmm.src','FQQMMM','spz',nlen,crl=spz)
     call chmdealloc('fluqqmmm.src','FQQMMM','ss',nlen,crl=ss)
     call chmdealloc('fluqqmmm.src','FQQMMM','work1',nlen,crl=work1)
     call chmdealloc('fluqqmmm.src','FQQMMM','work2',nlen,crl=work2)
     call chmdealloc('fluqqmmm.src','FQQMMM','work3',nlen,crl=work3)
     call chmdealloc('fluqqmmm.src','FQQMMM','work4',nlen,crl=work4)
     call chmdealloc('fluqqmmm.src','FQQMMM','xn',nlen,crl=xn)
     call chmdealloc('fluqqmmm.src','FQQMMM','yn',nlen,crl=yn)
     call chmdealloc('fluqqmmm.src','FQQMMM','zn',nlen,crl=zn)
     call chmdealloc('fluqqmmm.src','FQQMMM','xn2',nlen,crl=xn2)
     call chmdealloc('fluqqmmm.src','FQQMMM','yn2',nlen,crl=yn2)
     call chmdealloc('fluqqmmm.src','FQQMMM','zn2',nlen,crl=zn2)
     call chmdealloc('fluqqmmm.src','FQQMMM','xqm',nlen,crl=xqm)
     call chmdealloc('fluqqmmm.src','FQQMMM','yqm',nlen,crl=yqm)
     call chmdealloc('fluqqmmm.src','FQQMMM','zqm',nlen,crl=zqm)
     call chmdealloc('fluqqmmm.src','FQQMMM','dxpppp',10*nlen,crl=dxpppp)
     call chmdealloc('fluqqmmm.src','FQQMMM','dypppp',nlen,crl=dypppp)
     call chmdealloc('fluqqmmm.src','FQQMMM','dzpppp',nlen,crl=dzpppp)
     call chmdealloc('fluqqmmm.src','FQQMMM','dxpzpz',nlen,crl=dxpzpz)
     call chmdealloc('fluqqmmm.src','FQQMMM','dypzpz',nlen,crl=dypzpz)
     call chmdealloc('fluqqmmm.src','FQQMMM','dzpzpz',nlen,crl=dzpzpz)

  Endif
  !
  Return
END SUBROUTINE FQQMMM

SUBROUTINE FQQMM2(DENS, COORDS, INBX, JNBX, EXFACT, AM1, &
     XG, YG, ZG, QGRPMM_local, NDIM2, JNBL, CGL, E1B, &
     FACT0, FACT1M, FACT1P, FACT20, FACT2M, &
     FACT2P, FACT4P, PPPP, PZPZ, RQM2, RQM2B, RQM, &
     RQMB, RQMI, SCALE, SFCT1M, SFCT1P, SFCT20, &
     SFCT2M, SFCT2P, SFCT4P, SPZ, SS, WORK1, WORK2, &
     WORK3, WORK4, XN, YN, ZN, XN2, YN2, &
     ZN2, XQM, YQM, ZQM, &
     DXPPPP, DYPPPP, DZPPPP, &
     DXPZPZ, DYPZPZ, DZPZPZ, FQCFOR)
  !
  !     Does the work of FQQMMM. Taken almost verbatim from the QUANTUM
  !     interface routine EAMPE2; nuclear terms and gradients have been
  !     removed, and the density matrix is factored in.
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use coord
  use deriv
  use inbnd
  use psf
  use quantm
  use am1parm
  use nbndqm_mod
  use sizes
  use qmlinkm

  use stream
  implicit none
  !

  ! . Passed variables.
  Integer   inbx(*), jnbx(*), jnbl(*), ndim2
  Logical   am1, qgrpmm_local(*)
  real(chm_real)    DENS(*), coords(3,*), exfact
  !DCC+
  !
  real(chm_real)    cgl(*), e1b(ndim2,*), fact0(*), fact1m(*), &
       fact1p(*), fact20(*), fact2m(*), fact2p(*), &
       fact4p(*), pppp(*), pzpz(*), rqm2(*), rqm2b(*), &
       rqm(*), rqmb(*), rqmi(*), scale(*), sfct1m(*), &
       sfct1p(*), sfct20(*), sfct2m(*), sfct2p(*), &
       sfct4p(*), spz(*), ss(*), work1(*), work2(*), &
       work3(*), work4(*), xn(*), yn(*), zn(*), &
       xn2(*), yn2(*), zn2(*), xqm(*), yqm(*), zqm(*), &
       dxpppp(*), dypppp(*), &
       dzpppp(*), dxpzpz(*), dypzpz(*), dzpzpz(*), &
       xg(*), yg(*), zg(*)
  real(chm_real)    FQCFOR(*)
  ! . Local variables.
  Integer  i, ii, ipair, j, k, jj, n, n1, n2, n3, matm, mgrp, mnum, &
       mstrt, mstop, nbg, npr, nqm, numqm, qatom, &
       qfirst, qgrp, qnum, qstop, qstrt, QQSTOP, CHMAT
  Logical  qswitr
  real(chm_real)   alpha, c2ofnb, c2onnb, cgqm, d1, d2, d4,  &
       deltx, delty, &
       deltz, dfm, dfn, dfq, en, f1, f2, f3, funct, &
       rho0, rho1, rho2, rijl, riju, rqm2l, rqm2u, rul3, rul12, &
       scent, sfact, temp1, temp2, temp3, temp4, tpppp, tpzpz, &
       trqm2b, xq, yq, zq
  !
  real(chm_real)   four, half, one, quartr, thirty, three, twelve,  &
       two, zero
  Parameter (zero = 0.D0, quartr = 0.25D0, half = 0.5D0, &
       one = 1.D0, two = 2.D0, three = 3.D0, four = 4.D0, &
       twelve = 12.D0, thirty = 30.D0)
  !
  real(chm_real)   atobh2, atobhr, confac, ctoev
  Parameter (atobhr = 1.D0 / BOHRR, atobh2 = atobhr * atobhr, &
       confac = 23.061D0, ctoev = 27.21D0)
  !DCC-
  integer  itemp1, mstop1(maxgrp), nmgrp
  !DCC+
  ! . Define some constants.
  c2ofnb = ctofnb * ctofnb
  c2onnb = ctonnb * ctonnb
  rul3   = one / (c2ofnb - c2onnb)**3
  rul12  = twelve * rul3
  ! . Loop over the group non-bond lists - the QM groups are first.
  nbg    = 0
  numqm  = 0
  qfirst = 0
  loop520: Do qgrp = 1,ngrp
     npr    = inbx(qgrp) - qfirst
     qfirst = inbx(qgrp)
     qstrt  = igpbs(qgrp) + 1
     qstop  = igpbs(qgrp + 1)
     qnum   = qstop - qstrt + 1
     ! . Loop over the MM groups.
     n = 0
     If (npr .le. 0) then
        If (.not. qgrpmm_local(qgrp)) numqm = numqm + qnum
        cycle loop520
     Endif
     !DCC-
     nmgrp = 0
     itemp1 = 0
     !DCC+
     loop20:Do ipair = 1,npr
        nbg = nbg + 1
        mgrp = jnbx(nbg)
        If (mgrp .lt. 0) then
           sfact =  exfact
           mgrp  = -mgrp
        Else
           sfact =  one
        Endif
        mstrt = igpbs(mgrp) + 1
        mstop = igpbs(mgrp + 1)
        mnum  = mstop - mstrt + 1
        funct = 23.061d0
        dfm   = zero
        dfq   = zero
        !
        deltx = xg(qgrp) - xg(mgrp)
        delty = yg(qgrp) - yg(mgrp)
        deltz = zg(qgrp) - zg(mgrp)
        scent = deltx*deltx + delty*delty + deltz*deltz
        If (scent .lt. c2ofnb) then
           !DCC-
           nmgrp = nmgrp + 1
           mstop1(nmgrp) = itemp1 + mstop - mstrt + 1
           itemp1 = mstop1(nmgrp)
           !DCC+
           qswitr = scent .gt. c2onnb
           If (qswitr) then
              rijl  = c2onnb - scent
              riju  = c2ofnb - scent
              funct = riju * riju * (riju - three * rijl) &
                   * rul3 * 23.061d0
              dfn   = rijl * riju * rul12
              dfm   = dfn / mnum
              dfq   = dfn / qnum
           Endif
           ! . Fill the intermediate atom arrays.
           loop10:Do matm = mstrt,mstop
              !------------------------------------------------------------------
              !      QM-Link atom case: exclude the link atom from the MM list
              IF(QATLAB(MATM).GT.90) cycle loop10
              !------------------------------------------------------------------
              n = n + 1
              jnbl(n) = matm
              cgl(n)  = sfact * cg(matm)
              scale(n) = funct
           enddo loop10
        Endif
     enddo loop20
     ! . Loop over the QM atoms.
     !       including the QM-Link atoms, nqmlnk
     QQSTOP = QSTOP
     loop510:Do qatom = qstrt,QQSTOP
        numqm = numqm + 1
        If (n .gt. 0 .and. qatlab(qatom) .gt. 0) then
           !--------------------------------------------------------------------------
           !   One should rather delete more mm interaction than without
           !   including qm atoms (link atoms) in qm calculations.  It's
           !   not generating "balanced" charge distributions at the H(link)
           !   atom.   JG 12/96
           !   Not changed in standard CHARMM....JG 1/6/99
           !           If (n .gt. 0 .and. qatlab(qatom) .ge. 0) then
           !--------------------------------------------------------------------------
           ! . Define constants for the quantum mechanical atom.
           nqm    = nat(numqm)
           alpha  = alfa(nqm)
           cgqm   = core(nqm)
           n1 = n1st(numqm)
           n2 = nmidle(numqm)
           n3 = nlast(numqm)
           xq = coords(1,numqm)
           yq = coords(2,numqm)
           zq = coords(3,numqm)
           !
           rho0 = (half / bdd(nqm,1) + rho0mm) **2
           !
           ! . Determine some intermediate arrays.
           Do j = 1,n
              xqm(j) = xq - x(jnbl(j))
              yqm(j) = yq - y(jnbl(j))
              zqm(j) = zq - z(jnbl(j))
              rqm2(j) = xqm(j)*xqm(j)+yqm(j)*yqm(j)+zqm(j)*zqm(j)
              rqm(j)  = sqrt(rqm2(j))
              temp1   = one / rqm(j)
              xn(j)   = temp1 * xqm(j)
              yn(j)   = temp1 * yqm(j)
              zn(j)   = temp1 * zqm(j)
              rqmi(j) = temp1
           enddo
           ! . Calculate the integrals for atoms with one orbital.
           If (natorb(nqm) .eq. 1) then
              Do j = 1,n
                 trqm2b = atobh2 * rqm2(j)
                 fact0(j) = one / (trqm2b + rho0)
                 e1b(j,1) = ctoev * sqrt(fact0(j)) * cgl(j)
              enddo
              e1b(1:n,2:10) = zero
              !
              ! . Do the integrals for atoms with more than one orbital.
           Else
              rho1 = (half / bdd(nqm,2) + rho0mm) **2
              rho2 = (half / bdd(nqm,3) + rho0mm) **2
              !
              d1 = dd(nqm)
              d2 = two * qq(nqm)
              d4 = d2 * d2
              ! . Loop over the molecular mechanics atoms.
              Do j = 1,n
                 rqm2b(j)  = atobh2 * rqm2(j)
                 rqmb(j)   = atobhr * rqm(j)
                 work1(j)  = rqmb(j) - d1
                 work2(j)  = rqmb(j) + d1
                 work3(j)  = rqmb(j) - d2
                 work4(j)  = rqmb(j) + d2
              enddo
              !
              Do j = 1,n
                 fact0(j)  = one / (rqm2b(j) + rho0)
                 fact1m(j) = one / (work1(j) * work1(j) + rho1)
                 fact1p(j) = one / (work2(j) * work2(j) + rho1)
                 fact20(j) = one / (rqm2b(j) + rho2)
                 fact2m(j) = one / (work3(j) * work3(j) + rho2)
                 fact2p(j) = one / (work4(j) * work4(j) + rho2)
                 fact4p(j) = one / (rqm2b(j) + d4 + rho2)
                 sfct1m(j) = sqrt(fact1m(j))
                 sfct1p(j) = sqrt(fact1p(j))
                 sfct20(j) = sqrt(fact20(j))
                 sfct2m(j) = sqrt(fact2m(j))
                 sfct2p(j) = sqrt(fact2p(j))
                 sfct4p(j) = sqrt(fact4p(j))
              enddo
              !
              Do j = 1,n
                 ss(j)   = sqrt(fact0(j))
                 spz(j)  = half * (sfct1p(j) - sfct1m(j))
                 pzpz(j) = ss(j) + quartr*(sfct2p(j) + sfct2m(j)) - &
                      half * sfct20(j)
                 pppp(j) = ss(j) + half * (sfct4p(j) - sfct20(j))
              enddo
              ! . Define some variables needed for the transformation.
              xn2(1:n) = xn(1:n) * xn(1:n)
              yn2(1:n) = yn(1:n) * yn(1:n)
              zn2(1:n) = zn(1:n) * zn(1:n)
              ! . Fill arrays with the calculated integrals.
              Do j = 1,n
                 e1b(j,1)  = ss(j)
                 e1b(j,2)  = xn(j) * spz(j)
                 e1b(j,3)  = xn2(j) * pzpz(j) + &
                      (yn2(j) + zn2(j)) * pppp(j)
                 e1b(j,4)  = yn(j) * spz(j)
                 e1b(j,5)  = xn(j) * yn(j) * (pzpz(j) - pppp(j))
                 e1b(j,6)  = yn2(j) * pzpz(j) + &
                      (xn2(j) + zn2(j)) * pppp(j)
                 e1b(j,7)  = zn(j) * spz(j)
                 e1b(j,8)  = xn(j) * zn(j) * (pzpz(j) - pppp(j))
                 e1b(j,9)  = yn(j) * zn(j) * (pzpz(j) - pppp(j))
                 e1b(j,10) = zn2(j) * pzpz(j) + &
                      (xn2(j) + yn2(j)) * pppp(j)
              enddo
              !
              Do i = 1,10
                 Do j = 1,n
                    e1b(j,i) = ctoev * cgl(j) * e1b(j,i)
                 enddo
              enddo
           Endif
           !DCC-
           ! . If include MM-charge/QM-valence or QM electronic energy.
           !
           If (qeqtrm(qmee).and..not.qeqtrm(qqel)) then
              Call wrndie(-1,'<EAMPE2>', &
                   'CANNOT EVALUATE QMEE WITHOUT QQEL')
           Endif
           If (qeqtrm(qmee)) then
              !DCC+
              jj = 0
              Do i = n1,n2
                 ii = (i * (i - 1)) / 2 + n1 - 1
                 Do j = n1,i
                    ii = ii + 1
                    jj = jj + 1
                    DO K = 1,N

                       CHMAT=JNBL(K)
                       IF (I.EQ.J) THEN
                          FQCFOR(CHMAT)=FQCFOR(CHMAT)- &
                               E1B(K,JJ)*SCALE(K)*DENS(II)
                       ELSE
                          FQCFOR(CHMAT)=FQCFOR(CHMAT)- &
                               E1B(K,JJ)*SCALE(K)*DENS(II)*2.0d0
                       ENDIF
                    ENDDO
                 enddo
              enddo
              Do i = (n2+1),n3
                 ii = (i * (i + 1)) / 2
                 DO K = 1,N
                    CHMAT=JNBL(K)
                    !                       IF (I.EQ.J) THEN
                    FQCFOR(CHMAT)=FQCFOR(CHMAT)- &
                         E1B(K,1)*SCALE(K)*DENS(II)
                    !                       ELSE
                    !                          FQCFOR(CHMAT)=FQCFOR(CHMAT)-
                    !    &                         E1B(K,1)*SCALE(K)*DENS(II)*2.0d0
                    !                       ENDIF
                 ENDDO
              enddo
              !DCC-
           Endif
        Endif
     enddo loop510
     !
  enddo loop520
  Return
end SUBROUTINE FQQMM2

! GAMESS interface; GAMESS uses atomic units for its energies, so these
! must be converted to KCal/mol with the TOKCAL constant in consta.f90

#elif KEY_GAMESS==1 /*quantum_main*/

SUBROUTINE FQQCOR(CHMAT,VALUE)
  !
  !     Adds the contribution from the QM core repulsion energy VALUE to
  !     the charge force on atom CHMAT
  !
  !     Author: Ben Webb, 1999
  !
  use flucqm,only:fqcfor
  use chm_kinds
  use dimens_fcm
  use flucq
  use consta
  implicit none
  INTEGER CHMAT
  real(chm_real) VALUE
  IF (QFLUC) THEN
     FQCFOR(CHMAT) = FQCFOR(CHMAT) + VALUE * TOKCAL
  ENDIF
  RETURN
END SUBROUTINE FQQCOR

SUBROUTINE FQQMMM(DMAT)
  !
  !     Calculates the 1-electron contribution to FlucQ charge force, adds it to
  !     to the nuclear repulsion contribution, and converts from GAMESS units to
  !     CHARMM units.
  !     Called from GAMESS STVDER routine in grd1.src, after the SCF
  !
  !     DMAT = density matrix, passed after SCF
  !
  !     Author: Ben Webb, 2000
  !
  use flucqm,only:fqcfor
  use chm_kinds
  use dimens_fcm
  use psf
  use number
  use gamess_fcm
  use flucq
  implicit none
  REAL(CHM_REAL) DMAT(*)
  INTEGER I
  IF (QFLUC) THEN
     ! Pass required variables through to work routine, to avoid
     ! name clashes between fcm files and GAMESS code
     CALL FQQMM2(DMAT,FQCFOR,NATOM)
  ENDIF
  RETURN
END SUBROUTINE FQQMMM

SUBROUTINE FQQMM2(DMAT,FQCFOR,NATOM)
  !
  !     This routine is taken almost verbatim from gamint/blur.src;
  !     however, instead of adding the 1-electron terms in to the H matrix,
  !     it sums them into the charge force arrays. An automatic conversion
  !     from GAMESS to CHARMM numbering for the atoms is performed, and the
  !     relevant density matrix elements are multiplied in.
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use gamess_fcm
  use gmsblur
  implicit none
  !      IMPLICIT REAL(CHM_REAL) (A-H,O-Z)
  !      IMPLICIT INTEGER (I-N)

  real(chm_real) DMAT(*)
  real(chm_real) FQCFOR(*)
  !MH09: igmsel comes from use gamess!
  INTEGER NATOM

  INTEGER CHMAT

  ! TOKCAL taken from consta.f90
  real(chm_real) TOKCAL
  PARAMETER (TOKCAL = 627.5095D0 )
  !
  real(chm_real) MOROKM
  !
  LOGICAL IANDJ,NORM,DOUBLE,GOPARR,DSKWRK,MASWRK,QBARI,QSHORT
  !
  real(chm_real)  VBLK(225),FT(225),DIJ(225),zvblk(225)
  INTEGER IJX(225),IJY(225),IJZ(225)
  real(chm_real)  XIN(125),YIN(125),ZIN(125)
  INTEGER IX(35),IY(35),IZ(35),JX(35),JY(35),JZ(35)
  !
  ! namkh 09/10/04
  !      INTEGER MXSH,MXGTOT,MXATM,MXCHRM,KGUES,KHFDFT
  !     INTEGER MXSH,MXGTOT,MXATM,MXCHRM,KGUES
  !      PARAMETER (MXSH=1000, MXGTOT=5000, MXATM=500)
  !      PARAMETER (MXCHRM=25120)
  !
  ! namkh 09/10/04
  !      COMMON /CGAMES/ NGAMES,NQQCHG,QQCHG(MXATM),FQQCHG(MXATM)
  !      LOGICAL QGMREM,QGMEXG,QINIGM,QBLUCH,QNOGU,QFMO
  !      COMMON /GAMESL/QGMREM,QGMEXG,QINIGM,QBLUCH,QNOGU,QFMO
  !      INTEGER MAXBLU
  !      PARAMETER (MAXBLU=25120)
  !      INTEGER NBLUCH,IBLUCH
  !      real(chm_real) EBLUCH,CBLUCH
  !      COMMON /BLUR/EBLUCH(MAXBLU),CBLUCH(MAXBLU),IBLUCH(MAXBLU),NBLUCH
  !      real(chm_real) CGBLCH,SGBLCH
  !      COMMON /MBLUR/ CGBLCH(MAXBLU), SGBLCH(MAXBLU)
  !
  integer,parameter :: mxatm=2000, mxgtot=20000,mxsh=5000  ! from GAMESS
  real(chm_real) ak, c, cd, cdi, cdj, cf, cfi, cfj, ch, ci
  real(chm_real) ckn, cp, cpi, cpj, cg, cgi, cgj, cln
  real(chm_real) cs, csi, csj, cx, cy, cz, denmax
  real(chm_real) dum, dum1, dum2, ex, fac, exetyp
  real(chm_real) gpople, qq4, qx, qy, qz
  real(chm_real) rho, rr, runtyp, screen, taa, tol, tt
  real(chm_real) u, uu, vlamb, w, ww, x0, xi, xint, y0, yi, yint
  real(chm_real) xx, yj, z0, zi, zint,  zan, xj, zj, znuc,nrootsq
  !
  integer i, i1, i2, iblur, ibtyp, ic, ich, icut, idaf, iend
  integer iexch, ig, ij, ii, ijq, ijkl, ijqm, in, ioda
  integer ip, ipcount, iptim, ipk, ir, is, ish, istart, itol
  integer iw, j, j1, j2, jj, jg, jgmax, jn, jsh, k, kloc, klq
  integer kmax, kmin, kng, ksh, kstart, ktype, l1, l2, li, lit
  integer litq, lj, ljtq, lktq, isave, jstart, katom, ljt, llt
  integer loci, locij, lociq, locj, locjq, lockq, loclq, lsh
  integer master, me, max, maxi, maxiq, maxj, maxjq, maxkq, maxlq
  integer mini, miniq, minj, minjq, minkq, minlq, mm,nhlevl,nglevl
  integer mul, na, nat, nated, natst, nav, nb, ne, nevals
  integer ni, nijq, nint, nj, nn, nnp, nopk, norg, normf, normp
  integer nprint, nproc, nschwz, nshell, num, nx, ny, nz,ian
  !
  COMMON /CBARI/ QX,QY,QZ,CKN,CLN,AK
  !
  COMMON /INFOA / NAT,ICH,MUL,NUM,NNP,NE,NA,NB,ZAN(MXATM),&
       C(3,MXATM),ian(mxatm)
  COMMON /IOFILE/ IR,IW,IP,IS,IPK,IDAF,NAV,IODA(950)
  COMMON /NSHEL / EX(MXGTOT),CS(MXGTOT),CP(MXGTOT),CD(MXGTOT), &
       CF(MXGTOT),CG(MXGTOT),CH(MXGTOT),CI(MXGTOT), &
       KSTART(MXSH),KATOM(MXSH),KTYPE(MXSH),KNG(MXSH), &
       KLOC(MXSH),KMIN(MXSH),KMAX(MXSH),NSHELL
  COMMON /OUTPUT/ NPRINT,ITOL,ICUT,NORMF,NORMP,NOPK
  COMMON /PAR   / ME,MASTER,NPROC,IBTYP,IPTIM,GOPARR,DSKWRK,MASWRK
  COMMON /ROOT  / XX,U(13),W(13),NROOTSq
  COMMON /RUNOPT/ RUNTYP,EXETYP,NEVALS,NGLEVL,NHLEVL
  COMMON /SCINP / VLAMB,SCREEN
  COMMON /STV   / XINT,YINT,ZINT,TAA,X0,Y0,Z0, &
       XI,YI,ZI,XJ,YJ,ZJ,NI,NJ
  COMMON /SYMIND/ TOL,II,JJ,LIT,LJT,MINI,MINJ,MAXI,MAXJ,IANDJ
  !
  !     These are taken from int2a.src:
  !-----------------------------------
  !                                 NORG needed in SHELLS !
  COMMON /GOUT  / GPOPLE(768),NORG
  COMMON /SHLNOS/ QQ4,LITQ,LJTQ,LKTQ,LLT,LOCIQ,LOCJQ,LOCKQ,LOCLQ, &
       MINIQ,MINJQ,MINKQ,MINLQ,MAXIQ,MAXJQ,MAXKQ,MAXLQ, &
       NIJQ,IJQ,KLQ,IJKL
  INTEGER MXGSH, MXG2
  PARAMETER (MXGSH=30, MXG2=MXGSH*MXGSH)
  !
  real(chm_real) DDIJ(16*MXG2)
  !
  !-----------------------------------
  !
  real(chm_real) aa,aa1,aax,aay,aaz,ai,aj,ax,ay,az,axi,ayi,azi,arri
  real(chm_real) zero,pt5,one,two,three,five,seven,nine,eleven
  real(chm_real) e999, pi212,sqrt3,sqrt5,sqrt7,rln10,pi,pi32
  !
  PARAMETER (ZERO=0.0D+00, PT5=0.5D+00, ONE=1.0D+00, TWO=2.0D+00,&
       THREE=3.0D+00, FIVE=5.0D+00, SEVEN=7.0D+00, &
       NINE=9.0D+00, ELEVEN=11.0D+00, E999=999.0D+00, &
       PI212=1.1283791670955D+00, &
       SQRT3=1.73205080756888D+00, &
       SQRT5=2.23606797749979D+00, &
       SQRT7=2.64575131106459D+00, &
       RLN10=2.30258D+00,PI=3.14159265358979323844D+00, &
       PI32=5.56832799683170D+00)
  !
  DATA JX / 0, 1, 0, 0, 2, 0, 0, 1, 1, 0, &
       3, 0, 0, 2, 2, 1, 0, 1, 0, 1, &
       4, 0, 0, 3, 3, 1, 0, 1, 0, 2, &
       2, 0, 2, 1, 1/
  DATA IX / 1, 6, 1, 1,11, 1, 1, 6, 6, 1, &
       16, 1, 1,11,11, 6, 1, 6, 1, 6, &
       21, 1, 1,16,16, 6, 1, 6, 1,11, &
       11, 1,11, 6, 6/
  DATA JY / 0, 0, 1, 0, 0, 2, 0, 1, 0, 1, &
       0, 3, 0, 1, 0, 2, 2, 0, 1, 1, &
       0, 4, 0, 1, 0, 3, 3, 0, 1, 2, &
       0, 2, 1, 2, 1/
  DATA IY / 1, 1, 6, 1, 1,11, 1, 6, 1, 6, &
       1,16, 1, 6, 1,11,11, 1, 6, 6, &
       1,21, 1, 6, 1,16,16, 1, 6,11, &
       1,11, 6,11, 6/
  DATA JZ / 0, 0, 0, 1, 0, 0, 2, 0, 1, 1, &
       0, 0, 3, 0, 1, 0, 1, 2, 2, 1, &
       0, 0, 4, 0, 1, 0, 1, 3, 3, 0, &
       2, 2, 1, 1, 2/
  DATA IZ / 1, 1, 1, 6, 1, 1,11, 1, 6, 6, &
       1, 1,16, 1, 6, 1, 6,11,11, 6, &
       1, 1,21, 1, 6, 1, 6,16,16, 1, &
       11,11, 6, 6,11/
  DATA MOROKM/8HMOROKUMA/
  !
  !     ----- COMPUTE ADDITIONAL H INTEGRALS DUE TO CHARMM FIELD -----
  !
  TOL = RLN10*ITOL
  NORM = NORMF .NE. 1 .OR. NORMP .NE. 1
  !
  !     ----- RESET SOME PARAMETERS FOR MOROKUMA DECOMPOSITIONS -----
  !     ISAVE .EQ. 0 : SAVE S, H, AND T TO DAF 12, 11, AND 13
  !     ISAVE .EQ. 1 : SAVE S, H, AND T TO DAF 12, 11, AND 13
  !                    AND SAVE S AND H TO DAF 312 AND 311
  !     NOTE THAT LL2 IS ALWAYS (NUM*NUM+NUM)/2,
  !     L1,L2 MAY BE SMALLER THAN USUAL FOR A MONOMER IN A MOROKUMA RUN
  !
  IF (RUNTYP.EQ.MOROKM) THEN
     CALL STINT1(ISTART,IEND,JSTART,LOCIJ,NATST,NATED,ISAVE,L1,L2)
  ELSE
     ISTART = 1
     IEND   = NSHELL
     JSTART = 1
     LOCIJ  = 0
     NATST  = NAT+1
     NATED  = NAT+NCHMAT
     ISAVE  = 0
     L1 = NUM
     L2 = (NUM*(NUM+1))/2
  END IF
  !
  NINT  = 0
  NSCHWZ= 0
  DENMAX = ZERO
  !
  !     ----- INTIALIZE PARALLEL -----
  !
  IPCOUNT = ME - 1
  !
  !     This is the loop for external charges. Because
  !     the ``blur'' method needs data before the inner loop
  !     the loop over the charge
  !     centers has to be taken out from the inner loop.
  !
  !     -NCHMAT- IS NONZERO IF THERE ARE EXTERNAL CHARGES WHICH
  !     PERTURB THE SYSTEM, SUCH AS IF CHARMM IS IN USE.  NOTE
  !     THAT THERE IS ALSO A NUCLEAR REPULSION TERM WHICH IS NOT
  !     INCLUDED HERE, IT IS IN THE CHARMM INTERFACE CODE.
  !
  !
  IBLUR=1
  IC=NAT
  ! Loop over all CHARMM atoms, so that we know which CHARMM atom index
  ! corresponds to that used by GAMESS
  loop460: DO CHMAT=1,NATOM
     IF((IGMSEL(CHMAT).EQ.0).OR. &
          ((IGMSEL(CHMAT).EQ.5).AND.QBLUCH)) THEN

        IC=IC+1
        ZNUC = -QCHM(IC-NAT)
        CX = XCHM(IC-NAT)
        CY = YCHM(IC-NAT)
        CZ = ZCHM(IC-NAT)
        !
        QBARI=.FALSE.
        QSHORT=.FALSE.
        AK=ZERO
        IF(QBLUCH.AND.(IBLUCH(IBLUR).EQ.(IC-NAT))) THEN
           AK=PT5/SGBLCH(IBLUR)/SGBLCH(IBLUR)
           ZNUC=-CGBLCH(IBLUR)
           QBARI=.TRUE.
           QSHORT=.FALSE.
           IBLUR=IBLUR+1
        ENDIF
        !        This is the normalization factor
        CKN=TWO*AK/PI
        CKN=CKN*CKN*CKN
        CKN=SQRT(CKN)
        CKN=SQRT(CKN)
        CLN=ZNUC*CKN
        QX=CX
        QY=CY
        QZ=CZ
        !
        !     ----- I SHELL -----
        !
        loop720:DO II = ISTART,IEND
           I = KATOM(II)
           XI = C(1,I)
           YI = C(2,I)
           ZI = C(3,I)
           I1 = KSTART(II)
           I2 = I1+KNG(II)-1
           LIT = KTYPE(II)
           MINI = KMIN(II)
           MAXI = KMAX(II)
           LOCI = KLOC(II)-MINI-LOCIJ
           !
           !     ----- J SHELL -----
           !
           loop700:DO JJ = JSTART,II
              !
              !     ----- GO PARALLEL! -----
              !
              IF (GOPARR) THEN
                 IPCOUNT = IPCOUNT + 1
                 IF (MOD(IPCOUNT,NPROC).NE.0) cycle loop700
              END IF
              J = KATOM(JJ)
              XJ = C(1,J)
              YJ = C(2,J)
              ZJ = C(3,J)
              J1 = KSTART(JJ)
              J2 = J1+KNG(JJ)-1
              LJT = KTYPE(JJ)
              MINJ = KMIN(JJ)
              MAXJ = KMAX(JJ)
              LOCJ = KLOC(JJ)-MINJ-LOCIJ
              NROOTS = (LIT+LJT-2)/2+1
              RR = (XI-XJ)**2+(YI-YJ)**2+(ZI-ZJ)**2
              IANDJ = II .EQ. JJ
              !
              !     ----- PREPARE INDICES FOR PAIRS OF (I,J) FUNCTIONS
              !
              IJ = 0
              MAX = MAXJ
              DO I = MINI,MAXI
                 NX = IX(I)
                 NY = IY(I)
                 NZ = IZ(I)
                 IF (IANDJ) MAX = I
                 DO J = MINJ,MAX
                    IJ = IJ+1
                    IJX(IJ) = NX+JX(J)
                    IJY(IJ) = NY+JY(J)
                    IJZ(IJ) = NZ+JZ(J)
                    IF (J.LE.1) FT(IJ) = THREE
                    IF ((J.GT.1).AND.(J.LE.4)) FT(IJ) = FIVE
                    IF ((J.GT.4).AND.(J.LE.10)) FT(IJ) = SEVEN
                    IF ((J.GT.10).AND.(J.LE.20)) FT(IJ) = NINE
                    IF (J.GT.20) FT(IJ) = ELEVEN
                 enddo
              enddo
              !
              zVBLK(1:ij) = ZERO
              VBLK(1:ij) = ZERO
              !
              IF (QBARI.AND.(.NOT.QSHORT)) THEN
                 !
                 !     ----- K,L SHELL ----- For BARI ignore the loops and set
                 !                           the indeces to be always 1
                 !
                 IEXCH = 1
                 ISH = II
                 JSH = JJ
                 KSH = 1
                 LSH = 1
                 QQ4 = ONE
                 !
                 !        ----- COMPUTE TWO-ELECTRON INTEGRALS ----
                 !
                 !     USE HONDO RYS POLYNOMIAL CODE FOR OTHER BLOCKS
                 !
                 !        ----- GET INFORMATION ABOUT ISH AND JSH -----
                 !        ----- FORM PAIRS OF PRIMITIVES FROM ISH AND JSH -----
                 !        ----- GET INFORMATION ABOUT KSH AND LSH -----
                 !
                 NORG=0
                 CALL SHELLS(1,ISH,JSH,KSH,LSH)
                 CALL IJPRIM1(DDIJ)
                 !     Info which is needed, but is missing because call to
                 !     SHELLS(2,ISH,JSH,KSH,LSH) would be meaningless for BARI
                 !     Provide here:
                 IJKL=IJ
                 !
                 !        ----- DO INTEGRAL BATCH, SSSS IS A SPECIAL CASE -----
                 !                                 but general can handle them too!
                 !
                 CALL BARIGN(VBLK,DDIJ)

              else
                 !
                 !     ----- I PRIMITIVE
                 !
                 JGMAX = J2
                 loop520:DO IG = I1,I2
                    AI = EX(IG)
                    ARRI = AI*RR
                    AXI = AI*XI
                    AYI = AI*YI
                    AZI = AI*ZI
                    CSI = CS(IG)
                    CPI = CP(IG)
                    CDI = CD(IG)
                    CFI = CF(IG)
                    CGI = CG(IG)
                    !
                    !     ----- J PRIMITIVE
                    !
                    IF (IANDJ) JGMAX = IG
                    loop500:DO JG = J1,JGMAX
                       AJ = EX(JG)
                       AA = AI+AJ
                       AA1 = ONE/AA
                       DUM = AJ*ARRI*AA1
                       IF (QSHORT.AND.QBARI.AND.(AK.EQ.ZERO)) cycle loop500
                       IF (DUM .GT. TOL) cycle loop500
                       FAC = EXP(-DUM)
                       CSJ = CS(JG)
                       CPJ = CP(JG)
                       CDJ = CD(JG)
                       CFJ = CF(JG)
                       CGJ = CG(JG)
                       AX = (AXI+AJ*XJ)*AA1
                       AY = (AYI+AJ*YJ)*AA1
                       AZ = (AZI+AJ*ZJ)*AA1
                       !
                       !     ----- DENSITY FACTOR
                       !
                       DOUBLE=IANDJ.AND.IG.NE.JG
                       MAX = MAXJ
                       NN = 0
                       DUM1 = ZERO
                       DUM2 = ZERO
                       DO I = MINI,MAXI
                          IF (I.EQ.1) DUM1=CSI*FAC
                          IF (I.EQ.2) DUM1=CPI*FAC
                          IF (I.EQ.5) DUM1=CDI*FAC
                          IF ((I.EQ. 8).AND.NORM) DUM1=DUM1*SQRT3
                          IF (I.EQ.11) DUM1=CFI*FAC
                          IF ((I.EQ.14).AND.NORM) DUM1=DUM1*SQRT5
                          IF ((I.EQ.20).AND.NORM) DUM1=DUM1*SQRT3
                          IF (I.EQ.21) DUM1=CGI*FAC
                          IF ((I.EQ.24).AND.NORM) DUM1=DUM1*SQRT7
                          IF ((I.EQ.30).AND.NORM) DUM1=DUM1*SQRT5/SQRT3
                          IF ((I.EQ.33).AND.NORM) DUM1=DUM1*SQRT3
                          IF (IANDJ) MAX = I
                          DO J = MINJ,MAX
                             IF (J.EQ.1) THEN
                                DUM2=DUM1*CSJ
                                IF (DOUBLE) THEN
                                   IF (I.LE.1) THEN
                                      DUM2=DUM2+DUM2
                                   ELSE
                                      DUM2=DUM2+CSI*CPJ*FAC
                                   END IF
                                END IF
                             ELSE IF (J.EQ.2) THEN
                                DUM2=DUM1*CPJ
                                IF (DOUBLE) DUM2=DUM2+DUM2
                             ELSE IF (J.EQ.5) THEN
                                DUM2=DUM1*CDJ
                                IF (DOUBLE) DUM2=DUM2+DUM2
                             ELSE IF ((J.EQ.8).AND.NORM) THEN
                                DUM2=DUM2*SQRT3
                             ELSE IF (J.EQ.11) THEN
                                DUM2=DUM1*CFJ
                                IF (DOUBLE) DUM2=DUM2+DUM2
                             ELSE IF ((J.EQ.14).AND.NORM) THEN
                                DUM2=DUM2*SQRT5
                             ELSE IF ((J.EQ.20).AND.NORM) THEN
                                DUM2=DUM2*SQRT3
                             ELSE IF (J.EQ.21) THEN
                                DUM2=DUM1*CGJ
                                IF (DOUBLE) DUM2=DUM2+DUM2
                             ELSE IF ((J.EQ.24).AND.NORM) THEN
                                DUM2=DUM2*SQRT7
                             ELSE IF ((J.EQ.30).AND.NORM) THEN
                                DUM2=DUM2*SQRT5/SQRT3
                             ELSE IF ((J.EQ.33).AND.NORM) THEN
                                DUM2=DUM2*SQRT3
                             END IF
                             NN = NN+1
                             DIJ(NN) = DUM2
                          enddo
                       enddo
                       !
                       !     ----- NUCLEAR ATTRACTION
                       !
                       !                    PI212 = TWO/SQRT(PI), PI32 = PI**(3/2)
                       !
                       DUM = PI212*AA1
                       IF(QBARI.AND.QSHORT) &
                            DUM=DUM*PI32*CKN*CKN/(TWO*AK*SQRT(AA+TWO*AK))
                       DIJ(1:ij) = DIJ(1:ij)*DUM
                       !
                       AAX = AA*AX
                       AAY = AA*AY
                       AAZ = AA*AZ
                       !
                       IF (QBARI.AND.QSHORT) THEN
                          RHO=AA*TWO*AK/(AA+AK+AK)
                          XX = RHO*((AX-CX)**2+(AY-CY)**2+(AZ-CZ)**2)
                       ELSE
                          XX = AA*((AX-CX)**2+(AY-CY)**2+(AZ-CZ)**2)
                       ENDIF
                       IF (NROOTS.LE.3) CALL RT123
                       IF (NROOTS.EQ.4) CALL ROOT4
                       IF (NROOTS.EQ.5) CALL ROOT5
                       MM = 0
                       DO K = 1,NROOTS
                          IF(QBARI.AND.QSHORT) THEN
                             UU = RHO*U(K)
                          ELSE
                             UU = AA*U(K)
                          ENDIF
                          WW = W(K)*ZNUC
                          TT = ONE/(AA+UU)
                          TAA = SQRT(TT)
                          X0 = (AAX+UU*CX)*TT
                          Y0 = (AAY+UU*CY)*TT
                          Z0 = (AAZ+UU*CZ)*TT
                          IN = -5+MM
                          DO I = 1,LIT
                             IN = IN+5
                             NI = I
                             DO J = 1,LJT
                                JN = IN+J
                                NJ = J
                                CALL STVINT
                                XIN(JN) = XINT
                                YIN(JN) = YINT
                                ZIN(JN) = ZINT*WW
                             enddo
                          enddo
                          MM = MM+25
                       enddo
                       DO I = 1,IJ
                          NX = IJX(I)
                          NY = IJY(I)
                          NZ = IJZ(I)
                          DUM = ZERO
                          MM = 0
                          DO K = 1,NROOTS
                             DUM = DUM+XIN(NX+MM)*YIN(NY+MM)*ZIN(NZ+MM)
                             MM = MM+25
                          enddo
                          VBLK(I) = VBLK(I) + DUM*DIJ(I)
                       enddo
                       !
                       !     ----- END OF PRIMITIVE LOOPS -----
                       !
                    enddo loop500
                 enddo loop520
              endif
              !
              !     ----- COPY BLOCK INTO H-CORE, OVERLAP, AND KINETIC ENERGY MATRICES
              !
              MAX = MAXJ
              NN = 0
              DO I = MINI,MAXI
                 LI = LOCI+I
                 IN = (LI*(LI-1))/2
                 IF (IANDJ) MAX = I
                 DO J = MINJ,MAX
                    LJ = LOCJ+J
                    JN = LJ+IN
                    NN = NN+1
                    ! Off-diagonal elements should be counted twice
                    IF (LJ.EQ.LI) THEN
                       FQCFOR(CHMAT)=FQCFOR(CHMAT)+ &
                            DMAT(JN)*VBLK(NN)*TOKCAL
                    ELSE
                       FQCFOR(CHMAT)=FQCFOR(CHMAT)+ &
                            DMAT(JN)*VBLK(NN)*TWO*TOKCAL
                    ENDIF
                 enddo
              enddo
              !
              !     ----- END OF SHELL LOOPS -----
              !
           enddo loop700
        enddo loop720
     ENDIF
  enddo loop460
  !
  RETURN

  ! CADPAC interface; similar to GAMESS, this also uses atomic units,
  ! which must be converted.
end SUBROUTINE FQQMM2


#elif KEY_CADPAC==1 /*quantum_main*/

SUBROUTINE FQQCOR(CHMIN,VALUE)
  !
  !     Adds the contribution from the QM core repulsion energy VALUE to
  !     the charge force; CHMIN is the index of the previous CHARMM atom
  !     (this must be updated by the routine)
  !
  !     Author: Ben Webb, 1999
  !
  use flucqm, only: fqcfor
  use chm_kinds
  use dimens_fcm
  use consta
  use flucq
  use gamess_fcm
  use parallel,only:mynod

  implicit none
  INTEGER(chm_int4) CHMIN
  real(chm_real) VALUE
  !
  INTEGER IND
  IF (QFLUC) THEN
#if KEY_PARALLEL==1
     ! CADPAC computes these forces on all nodes; we only need them once!
     IF (MYNOD.EQ.0) THEN
#endif
10      CHMIN=CHMIN+1
        IF (IGMSEL(CHMIN).EQ.0) THEN
           IND=CHMIN
           FQCFOR(IND) = FQCFOR(IND) + VALUE * TOKCAL
        ELSE
           GOTO 10
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif
  ENDIF
  RETURN
END SUBROUTINE FQQCOR

SUBROUTINE FQQMMM(DMAT)
  !
  !     Calculates the 1-electron contribution to FlucQ charge force, adds it
  !     to the nuclear repulsion contribution, and converts from CADPAC units to
  !     CHARMM units.
  !     Called from CADPAC HFGRAD routine in main.F, after the SCF
  !
  !     Author: Ben Webb, 2000
  !
  use flucqm,only:fqcfor
  use chm_kinds
  use dimens_fcm
  use psf
  use number
  use gamess_fcm
  use flucq
  implicit none
  REAL(CHM_REAL) DMAT(*)
  INTEGER I
  IF (QFLUC) THEN
     ! Pass required variables through to work routine, to avoid
     ! name clashes between fcm files and CADPAC code
     CALL FQQMM2(DMAT,FQCFOR,NATOM,IGMSEL,QBLUCH)
  ENDIF
  RETURN
END SUBROUTINE FQQMMM

SUBROUTINE FQQMM2(DMAT,FQCFOR,NATOM,IGMSEL,QBLUCH)
  !
  !     This routine is taken almost verbatim from cadini/cadpac/standv.F;
  !     however, instead of adding the 1-electron terms in to the H matrix,
  !     it sums them into the charge force arrays. An automatic conversion
  !     from CADPAC to CHARMM numbering for the atoms is performed, and the
  !     relevant density matrix elements are multiplied in.
  !     N.B. Since CADPAC uses 4-byte integers under Alpha Linux, while
  !          CHARMM uses 8-byte integers, we have to be careful about
  !          declaring integer variables here.
  !
  !     Author: Ben Webb, 2000
  !
  use chm_kinds
  use parallel
  use consta
  IMPLICIT INTEGER(chm_int4) (I-N)
  IMPLICIT REAL(CHM_REAL)(A-H,O-Z)
  real(chm_real) FQCFOR(*)
  LOGICAL QBLUCH
  INTEGER NATOM,IGMSEL(*),CHMAT
  REAL(CHM_REAL) DMAT(*)

  INTEGER(chm_int4) ATII,ATJJ
  LOGICAL IANDJ,DOUBLE
  LOGICAL ARRAY,LARRAY,OUT,NORM,SKIPER,SKIPRA,LGHO1,LGHO2,LIM,LMKM

  !
  COMMON/ITEMP/TOLA,TOLI,CUTOFF,OUT,NORM,SKIPER,LGHO1,LGHO2,LIM, &
       SKIPRA,ARRAY,LARRAY,NATOM1,NATOM2,NSH1,NSH2, &
       NMAX1,NMAX2,NMAX12,LMKM
  !
  COMMON/RTWT/XX,U(12),W(12),NROOTS
  !
  PARAMETER (MAXCEN=100)
  COMMON/INFOA/NAT,ICH,MUL,NUM,NNP,NE,NA,NB,ZAN(MAXCEN),C(3,MAXCEN)
  !
  PARAMETER (MAXEXP=3000,MAXSHL=500)
  COMMON/NSHEL/EX(MAXEXP),CS(MAXEXP),CP(MAXEXP),CD(MAXEXP), &
       KSTART(MAXSHL),KATOM(MAXSHL),KTYPE(MAXSHL),KNG(MAXSHL), &
       KLOC(MAXSHL),KMIN(MAXSHL),KMAX(MAXSHL),NSHELL,NSEL2
  !
  PARAMETER (MAXLAT=25120)
  COMMON/LATTIC/CLAT(3,MAXLAT),ZLAT(MAXLAT),NLAT
  !
  PARAMETER (IDEP10=16,ICN=IDEP10*IDEP10)
  COMMON/BIG/XINT,YINT,ZINT,T,X0,Y0,Z0,XI,YI,ZI,XJ,YJ,ZJ,NI,NJ &
       ,S(ICN),G(ICN),FT(ICN),DIJ(ICN),XIN(ICN),YIN(ICN), &
       ZIN(ICN),IJX(ICN),IJY(ICN),IJZ(ICN) &
       ,GCOUL(ICN,MAXCEN)
  PARAMETER (MAXBFN=1000)
  COMMON/MAPPER/IA(MAXBFN+2)
  DATA PI212 /1.1283791670955D0/

  MOLI=1
  !-----------------------------------------------------
#if KEY_PARALLEL==1
  ncount=-1
#endif
  !
  !----------------START LOOPS OVER SHELLS-----------------
  !
  !------------SHELL II
  loop160:DO II=1,NSHELL
     IF(II.GT.NSH1) MOLI=2
     I=KATOM(II)
     XI=C(1,I)
     YI=C(2,I)
     ZI=C(3,I)
     I1=KSTART(II)
     I2=I1+KNG(II)-1
     LIT=KTYPE(II)
     MINI=KMIN(II)
     MAXI=KMAX(II)
     LOCI=KLOC(II)-MINI
     MOLJ=1
     !------------ SHELL  JJ
     loop160a: DO JJ=1,II
        IF(JJ.GT.NSH1) MOLJ=2
        TOL=TOLA
        IF(LIM.AND.MOLI.NE.MOLJ)TOL=TOLI
        !
        !     SKIP IF DO NOT WANT INTERMOLECULAR INTEGRALS
        !
        IF((LMKM.OR.SKIPER).AND.MOLI.NE.MOLJ) cycle loop160a
#if KEY_PARALLEL==1
        !mem divide work amongst nodes
        ncount= ncount+1
        if(MYNOD.eq.MOD(ncount,NUMNOD))then
           !{
#endif
           J=KATOM(JJ)
           XJ=C(1,J)
           YJ=C(2,J)
           ZJ=C(3,J)
           J1=KSTART(JJ)
           J2=J1+KNG(JJ)-1
           LJT=KTYPE(JJ)
           MINJ=KMIN(JJ)
           MAXJ=KMAX(JJ)
           LOCJ=KLOC(JJ)-MINJ
           NROOTS=(LIT+LJT)/2
           RR=(XI-XJ)**2+(YI-YJ)**2+(ZI-ZJ)**2
           IANDJ=II.EQ.JJ
           !----- PREPARE INDICES FOR PAIRS OF (I,J) FUNCTIONS
           !----- INDICES DEPEND ONLY ON TYPE OF SHELLS ( S,P,D ETC )
           CALL INDEXA(IJX,IJY,IJZ,IJ,MINI,MAXI,MINJ,MAXJ,IANDJ,4,1,1)
           !
           !----- SET SOME FACTORS USED IN KINETIC ENERGY
           !----- AND CLEAR ARRAYS USED TO HOLD RESULTS FOR PAIR
           !----- OF SHELLS
           IJ=0
           MAX=MAXJ
           DO I=MINI,MAXI
              IF(IANDJ) MAX=I
              DO J=MINJ,MAX
                 IJ=IJ+1
              enddo
           enddo
           !----- START LOOPS OVER PRIMITIVE GAUSSIANS
           !
           !----- I PRIMITIVE
           JGMAX=J2
           loop110: DO IG=I1,I2
              AI=EX(IG)
              ARRI=AI*RR
              AXI=AI*XI
              AYI=AI*YI
              AZI=AI*ZI
              CSI=CS(IG)
              CPI=CP(IG)
              CDI=CD(IG)
              ! ----- J PRIMTIVE
              IF(IANDJ) JGMAX=IG
              loop110a: DO JG=J1,JGMAX
                 AJ=EX(JG)
                 AA=AI+AJ
                 AAINV=1.0D0/AA
                 !----- TEST FOR SMALL EXPONENTIALS
                 DUM=AJ*ARRI*AAINV
                 IF(DUM.GT.TOL) cycle loop110a
                 FAC=DEXP(-DUM)
                 CSJ=CS(JG)*FAC
                 CPJ=CP(JG)*FAC
                 CDJ=CD(JG)*FAC
                 AX=(AXI+AJ*XJ)*AAINV
                 AY=(AYI+AJ*YJ)*AAINV
                 AZ=(AZI+AJ*ZJ)*AAINV
                 ! ----- DENSITY FACTOR IS APPROPRIATE COMBINATION OF
                 ! ----- CONTRACTION COEFFICIENTS
                 DOUBLE=IANDJ.AND.IG.NE.JG
                 CALL DENFAN(DIJ,CSI,CPI,CDI,CSJ,CPJ,CDJ,MINI,MAXI, &
                      MINJ,MAXJ,IANDJ,DOUBLE,NORM)
                 !     ----- NUCLEAR ATTRACTION
                 DUM=PI212/AA
                 DIJ(1:ij)=DIJ(1:ij)*DUM
                 NATLAT=NAT+NLAT
                 AAX=AA*AX
                 AAY=AA*AY
                 AAZ=AA*AZ
                 MOLIC=1
                 !---------------------------------------------------------
                 !
                 !--------------LOOP OVER NUMBER OF NUCLEI
                 !
                 IC=NAT
                 ! Loop over all CHARMM atoms, so that we know which CHARMM atom index
                 ! corresponds to that used by CADPAC
                 loop100: DO CHMAT=1,NATOM
                    IF((IGMSEL(CHMAT).EQ.0).OR. &
                         ((IGMSEL(CHMAT).EQ.5).AND.QBLUCH)) THEN

                       IC=IC+1

                       G(1:ij)=zero

                       !----------------THESE ARE LATTICE POINTS, NOT REAL NUCLEI
                       ZNUC=-ZLAT(IC-NAT)
                       CX=CLAT(1,IC-NAT)
                       CY=CLAT(2,IC-NAT)
                       CZ=CLAT(3,IC-NAT)
                       !----------------------------------------------------
                       XX=AA*((AX-CX)**2+(AY-CY)**2+(AZ-CZ)**2)
                       !
                       !     ROOTS AND WEIGHTS FOR RYS SCHEME
                       !
                       IF(NROOTS.LE.3)CALL RT123
                       IF(NROOTS.EQ.4)CALL ROOT4
                       IF(NROOTS.EQ.5)CALL ROOT5
                       IF(NROOTS.GT.5)CALL ROOTS
                       MM=0
                       DO K=1,NROOTS
                          UU=AA*U(K)
                          WW=W(K)*ZNUC
                          TT=AA+UU
                          T=DSQRT(TT)
                          TINV=1.0D0/TT
                          X0=(AAX+UU*CX)*TINV
                          Y0=(AAY+UU*CY)*TINV
                          Z0=(AAZ+UU*CZ)*TINV
                          IN=-4+MM
                          DO I=1,LIT
                             IN=IN+4
                             NI=I
                             DO J=1,LJT
                                JN=IN+J
                                NJ=J
                                CALL STVINT
                                XIN(JN)=XINT
                                YIN(JN)=YINT
                             enddo
                          enddo
                          MM=MM+16
                       enddo
                       DO I=1,IJ
                          NX=IJX(I)
                          NY=IJY(I)
                          NZ=IJZ(I)
                          DUM=0.0D0
                          MM=0
                          !
                          !     ASSEMBLE NUCLEAR ATTRACTION INTEGRALS
                          !
                          DO K=1,NROOTS
                             DUM=DUM+XIN(NX+MM)*YIN(NY+MM)*ZIN(NZ+MM)
                             MM=MM+16
                          enddo
                          FACTOR=DUM*DIJ(I)
                          G(I)=G(I)+FACTOR
                       enddo

                       ! Add 1-electron contribution to charge force
                       IDIM=MAXI-MINI+1
                       JDIM=MAXJ-MINJ+1
                       LOCI=KLOC(II)-1
                       LOCJ=KLOC(JJ)-1
                       NN=0
                       MAX=JDIM
                       DO I=1,IDIM
                          LI=LOCI+I
                          IF (IANDJ) MAX=I
                          JN=LI*(LI-1)/2+LOCJ
                          DO J=1,MAX
                             NN=NN+1
                             JN=JN+1
                             !        jn=ia(li)+locj+j
                             !        if (locj+j.gt.li) jn=ia(li)+li
                             IF (ABS(G(NN)).GT.CUTOFF) THEN
                                ! N.B. Off-diagonal elements contribute twice
                                IF (LI.EQ.(LOCJ+J)) THEN
                                   FQCFOR(CHMAT)=FQCFOR(CHMAT)+G(NN)*DMAT(JN)*TOKCAL
                                ELSE
                                   FQCFOR(CHMAT)=FQCFOR(CHMAT)+G(NN)* &
                                        DMAT(JN)*2.0d0*TOKCAL
                                ENDIF
                             ENDIF
                          ENDDO
                       ENDDO
                       !
                       !---------------------END OF LOOP OVER ALL NUCLEI
                       !---------------------END OF NUCLEAR REPULSION INTEGRALS
                       !---------------------FOR THIS PAIR OF SHELLS
                    ENDIF
                 enddo loop100
              enddo loop110a
           enddo loop110
           !
#if KEY_PARALLEL==1
           !}
        ENDIF
#endif
     enddo loop160a
  enddo loop160
  !
  !
  !--------------------------------------------------------------------
  !
  RETURN
end SUBROUTINE FQQMM2

#elif KEY_GAMESSUK==1 /*quantum_main*/

SUBROUTINE FQQCST
  use chm_kinds
  use dimens_fcm
  use flucq
  implicit none
  FQNUCD=.FALSE.
  RETURN
END SUBROUTINE FQQCST

SUBROUTINE FQQCDN
  use chm_kinds
  use dimens_fcm
  use flucq
  implicit none
  FQNUCD=.TRUE.
  RETURN
END SUBROUTINE FQQCDN

SUBROUTINE FQQMMM(Q,ISO,NSHELS)
  use flucqm,only:fqcfor
  use chm_kinds
  use dimens_fcm
  use flucq
  use consta
  use gamess_fcm
  implicit none
  INTEGER Q,ISO,NSHELS

  !
  ! This routine is implemented in GAMESS-UK (version 6.3.1
  ! onwards)
  !
  IF (QFLUC) CALL FQQMM2(Q,ISO,NSHELS,FQCFOR,GMSMAP, TOKCAL)
  RETURN
END SUBROUTINE FQQMMM

SUBROUTINE FQQCOR(I,J,VALUE)
  !
  !     Adds the contribution from the QM core repulsion energy VALUE
  !     between atoms I and J, if either atom is a CHARMM atom
  !
  !     Author: Ben Webb, 2000
  !
  use flucqm,only:fqcfor
  use chm_kinds
  use dimens_fcm
  use flucq
  use consta
  use gamess_fcm
#if KEY_PARALLEL==1
  use parallel
#endif
  implicit none
  INTEGER I,J,II,JJ
  real(chm_real) VALUE

#if KEY_PARALLEL==1
  ! GAMESS-UK computes these forces on all nodes; we only need them once!
  IF (MYNOD.EQ.0) THEN
#endif

     II = GMSMAP(I)
     JJ = GMSMAP(J)

     IF (QFLUC.AND.(.NOT.FQNUCD)) THEN
        ! I is QM, J is MM
        IF ((IGMSEL(II) == 1 .OR. IGMSEL(II) == 2) &
             .AND. IGMSEL(JJ) == 0) THEN
           FQCFOR(JJ) = FQCFOR(JJ) + VALUE * TOKCAL

           ! J is QM, I is MM
        ELSE IF ((IGMSEL(JJ) == 1 .OR. IGMSEL(JJ) == 2) &
             .AND. IGMSEL(II) == 0) THEN
           FQCFOR(II) = FQCFOR(II) + VALUE * TOKCAL
        ENDIF
     ENDIF

#if KEY_PARALLEL==1
  ENDIF
#endif
  RETURN
end SUBROUTINE FQQCOR
#endif /* (quantum_main)*/
#else /* (flucq)*/
#endif /* (flucq)*/
