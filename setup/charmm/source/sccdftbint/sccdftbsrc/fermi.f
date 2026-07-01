! Output from Public domain Ratfor, version 1.0
! MG_QC_UW1207: note change of arguments
!               occperorb: maximal occupation number per orbital =1 for spinpolarized/ =2 for spinunpolarized calculations
      subroutine fermi(nel,telec,ndim,ev,dacc,occ,efermi,occperorb)
      use sccdftbsrc, only: mdim,racc 
      implicit real*8 (a-h,o-z)
!     integer nndim
!     parameter( nndim= 650)
!     integer maxint
!     parameter( maxint= 160)
!     integer mdim
!     parameter( mdim= 1650)
!     integer maxtyp
!     parameter( maxtyp= 6)
!     integer maxtab
!     parameter( maxtab= 600)
!     integer ldim
!     parameter( ldim= 9)
!     integer maxopt
!     parameter(maxopt=3*nndim)
!     integer maxsiz
!     parameter(maxsiz=mdim)
!     integer maxtp
!     parameter(maxtp=maxtyp)

      integer ndim
      real*8 nel,telec,efermi,dacc,ev(mdim),occ(mdim),occperorb !MG_QC_UW1207
      parameter(ckbol = 3.16679d-6)
      parameter(degtol = 1.0d-4)
      real*8 beta,etol,occdg,chleft,ef1,ef2,ceps,eeps,charge,fac
      integer i,nef1,nef2,nup,ndown,nocc2,ndeg,istart,iend,hlp !MG_QC_UW1207
      logical tzero
!     common /machine/ racc

      do23000 i = 1,ndim 
      occ(i) = 0.0
23000 continue
23001 continue
      if(nel .lt. 1.0d-5)then
      goto 10
      endif
      if(nel .gt. int(occperorb)*ndim)then !MG_QC_UW1207
! MG_QC_UW1207: exit gracefully
      print *,'# of electrons expected: ', nel
      print *,'# of electrons calculated: ', int(occperorb)*ndim
      call WRNDIE(-5,'<FERMI>','SCCDFTB calculates too many electrons')
      !print *,'too many electrons'
      !stop
      endif
      if(telec .gt. 5.0)then
      beta = 1.0/(ckbol*telec)
      etol = ckbol*telec*(log(beta)-log(racc))
      tzero = .false.
      else
      etol = degtol
      tzero = .true.
      endif
      hlp = int(occperorb) !MG_QC_UW1207
      if(nel.gt.int(nel))then
      nef1 = (nel+hlp)/hlp !MG_QC_UW1207
      nef2 = (nel+hlp)/hlp !MG_QC_UW1207
      else
      nef1 = (nel+hlp-1)/hlp !MG_QC_UW1207
      nef2 = (nel+hlp)/hlp   !MG_QC_UW1207
      !nef1 = (nel+1)/2
      !nef2 = (nel+2)/2
      endif
      efermi = 0.5*(ev(nef1)+ev(nef2))
      nup = nef1
      ndown = nef1
23010 if(nup .lt. ndim)then
      if(abs(ev(nup+1)-efermi) .gt. etol)then
      goto 23011
      endif
      nup = nup+1
      goto 23010
      endif
23011 continue
23014 if(ndown .gt. 0)then
      if(abs(ev(ndown)-efermi) .gt. etol)then
      goto 23015
      endif
      ndown = ndown-1
      goto 23014
      endif
23015 continue
      ndeg = nup-ndown
      nocc2 = ndown
      do23018 i = 1,nocc2 
      occ(i) = occperorb !MG_QC_UW1207
23018 continue
23019 continue
      if(ndeg .eq. 0)then
      goto 10
      endif
      if(tzero)then
      occdg = ndeg
      occdg = (nel-occperorb*nocc2)/occdg !MG_QC_UW1207
      do23024 i = nocc2+1,nocc2+ndeg 
      occ(i) = occdg
23024 continue
23025 continue
      else
      chleft = nel-occperorb*nocc2 !MG_QC_UW1207
      istart = nocc2+1
      iend = istart+ndeg-1
      if(ndeg .eq. 1)then
      occ(istart) = chleft
      goto 10
      endif
      ef1 = efermi-etol-degtol
      ef2 = efermi+etol+degtol
      ceps = dacc*chleft
      eeps = dacc*max(abs(ef1),abs(ef2))
23028 continue
      efermi = 0.5*(ef1+ef2)
      charge = 0.0
      do23031 i = istart,iend 
      occ(i) = occperorb/(1.0+exp(beta*(ev(i)-efermi))) !MG_QC_UW1207
      charge = charge+occ(i)
23031 continue
23032 continue
      if(charge .gt. chleft)then
      ef2 = efermi
      else
      ef1 = efermi
      endif
23029 if(.not.((abs(charge-chleft) .lt. ceps) .or. (abs(ef1-ef2) .lt. ee
     *ps)))goto 23028
23030 continue
      if(abs(charge-chleft) .gt. ceps)then
      fac = chleft/charge
      do23037 i = istart,iend 
      occ(i) = occ(i)*fac
23037 continue
23038 continue
      endif
      endif
10    continue
      end

