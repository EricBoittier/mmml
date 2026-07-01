! Output from Public domain Ratfor, version 1.0
      subroutine dylcao
      use dimens_fcm
      use coord
      use sccdftbsrc
      use sccdftb, izp => scctyp, xe => cptc, ze => zptc, 
     $             ne =>nptc, ext=>extflag!,qnuc=>scczin
      use blockscc_fcm
      use dftd3, only: init_dftd3
      use cpe, only: init_cpepar
      implicit real*8 (a-h,o-z)

!     ----------------------- Basic data structure -----------------

      real*8 slkcutoff

      external skhpar, skspar
!     ----------------------- File control ---------------------
      character*65 infile, outfile, tmpchr
!     character*60 smode(9)

!     ---------------------- Mostly Useless control para -----------
      logical atomic, evector, scfhelp
      logical converge, mdcorr, dorelax, doecalc
      logical chrr,constr2
      logical lbfgs,lzmat,lgrad
!     common /concgr/ convec, constr2, conat

      character*8 dummy ! MG_QC_UW1206

!     QC_UW031205: change to xscc to avoid overlap with common block
      real*8 xscc(3*nndim)

!     MG_QC_UW1207
      integer lcount ! counter for filling l-dependent arrays

      integer :: scczin_int(maxcen)

!     integer nscctc,nptc,ichscc,izp
!     real*8 scczin
!     character*10 sccatm
!     common /sccinf/ nscctc,ichscc,scczin(nndim*3),izp(nndim*3),sccatm(
!    *nndim*3)
!     izp =>scctyp

!     integer mxchmpt
!     parameter(mxchmpt=25120)
!     real*8 xe(3,mxchmpt), ze(mxchmpt)
!     integer ne
!     character*2 ext
!     integer mysccid
!     common/extchr/ xe, ze, ne, ext,mysccid
!     xe =>cptc; ze=> zptc; ne=>nptc; ext=>extflag

!     integer mxitscf,lprvec
!     common/sccopt/qmat(nndim), telec,scftol,mxitscf,lprvec

      integer iunit

!cccccccccccccccccccccccccccccccccccccccccccccccccccccc
!     QC: Simplify GL's common block
!     integer izp2a(nndim)
!     integer izp2b(nndim)
!     real*8 nela,nelb  
!     logical qaflag,qbflag
!     common /sccbloka/izp2a,nela,qaflag
!     common /sccblokb/izp2b,nelb,qbflag
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc
!     QC_UW031205: Add common blocks from fcm to allow dispersion
!     Also the QM Lst for dispersion_read
!     INTEGER NSCCRP
!     LOGICAL LSCCRP
!     INTEGER MXQM2,MXRP,MX_STORE
!     PARAMETER (MXQM2=100)
!     PARAMETER (MXRP=50)
!     INTEGER IQMLST(MXQM2,MXRP)
!     COMMON/SCCRPL/ NSCCRP,IQMLST,LSCCRP
!ccccccccccccccccccccccccccccccccccccccccccccccccccccc

!     data smode /'MD relaxation with heat reservoir   ', 'free MD witho
!    *ut external constraints', 'steepest descent relaxation         ', 
!    *'conjugate gradient relaxation       ', 'constraint conjugate grad
!    *ient relaxation', 'Mulliken charge and atomic energy calculation',
!    * 'Option number 6 + prints out the eigenvectors', 'BFGS relaxation
!    * in internal coordinates', 'Option number 6 + Z-matrix input'/

      if(mysccid .eq.0)then
      print *,'** dftb **'
      endif
      racc = 1.0
23002 if((1.0+racc) .gt. 1.0)then
      racc = racc*0.5
      goto 23002
      endif
23003 continue
      racc = 2*racc
      atomic = .false. 
      evector = .false.
      constr2 =.false.
      lgrad = .true.
      if(mysccid .eq.0)then
      print *,'enter mode, fmax, scf, scftol, read charge,dispers,EXT'
      endif
      if(ext.eq.'CH')then
        iunit=4
        scfhelp= .true.
        chrr= .false.
        mode= 3
        open(iunit,file='sccdftb.dat',status='old')
        rewind iunit
        writeout = 0
        nn=nscctc
        write(*,*) "Number of QM atoms in dylcao:",nn
        if (mysccid.eq.0) 
     $  print *,'skip struc. file,atoms from charmm:',nn,ne
      else
        write(*,*) "UNSUPPORTED IN CHARMM"
        goto 9
      endif

      if(scfhelp)then
      scftol=abs(scftol)
      else
      scftol=-abs(scftol)
      endif

      nbeweg=nn
      n3 = 3*nn

      ntype = 1
      do23052 n = 1,nn 
      if(izp(n) .gt. ntype)then
      ntype = izp(n)
      endif
      izp2(n) = izp(n)
23052 continue

23053 continue
      if(ntype .gt. maxtyp)then
      print *,'dftb: ntype >', maxtyp
      goto 9
      endif
      if(mysccid .eq.0)then
      write (*,*) 'enter filename for output structure'
      endif

! Guanhua_puja_QC_UW1212: Mixer options
      if (sccmix .eq. 0) then
        write (*,*) 'DYLCAO> Mixer used is: simple mixer'
      elseif (sccmix .eq. 1) then
        write (*,*) 'DYLCAO> Mixer used is: Anderson mixer'
        write (*,*) 'DYLCAO> maximum generations is: ',gens
        write (*,*) 'DYLCAO> symmetry breaking number is: ',wsy
      elseif (sccmix .eq. 2) then
        write (*,*) 'DYLCAO> Mixer used is: Broyden mixer'
      elseif (sccmix .eq. 3) then
        write (*,*) 'DYLCAO> Mixer used is: DIIS mixer'
      else
        write (*,*) 'DYLCAO> Wrong mixer option!'
        stop
      endif
!

!     call gettab(ntype)
      call gettab
      nel=0.0d0
      do23060 i = 1,nn 
      nel = nel+qzeroscc(izp(i),4)
23060 continue
23061 continue
      nel = nel - dble(ichscc)
      if(mysccid .eq.0)then
      write(*,*) 'Total charge of SCCDFTB molecule:',ichscc
      endif
      if(mysccid .eq.0)then
      write(*,*) 'Total number of electrons:', nel
      endif

      if(scfhelp)then
      if(.not.chrr)then
      ! MG_QC_UW1207: l-shell dependent charges (here for lcolspin necessary); note change in loop
      ! MG_UW1210: ldep
      lcount=0
      do i = 1,nn 
        qmat(i) = qzeroscc(izp(i),4)
        do j=1,lmax(izp(i))
          lcount=lcount+1
          ql(lcount)     = qzeroscc(izp(i),j)
          qlup(lcount)   = qzeroscc(izp(i),j)/2.0d0
          qldown(lcount) = qzeroscc(izp(i),j)/2.0d0
        enddo
      enddo     
      ! end MG_QC_UW1207
      else
      ! MG_UW1210: ldep
      if(lldep) then 
         call wrndie(-5,'<DYLCAO>','LDEP not implemented in 
     &   connection with reading charges from CHR.DAT.')
      endif
      open(37,file="CHR.DAT",status="unknown")
      do23072 i=1,5 
      read(37,*) tmpchr
23072 continue
23073 continue
      do23074 i=1,nn 
      read(37,*) j,qmat(i)
23074 continue
23075 continue
      close(37)
      endif
      endif

!     QC_UW031205: Dispersion re-activated
      if(dispers)then
      do23080 i=1,nn 
        xscc(3*i-2)=x(iqmlst(i,1))/0.52917
        xscc(3*i-1)=y(iqmlst(i,1))/0.52917
        xscc(3*i  )=z(iqmlst(i,1))/0.52917
23080 continue
      write(*,*) "DYLCAO> Reading Dispersion parameters"
!     call dispersionread(nn,ntype,izp,xscc)
      call dispersionread(nn,xscc)
      endif
!     QC_UW031205: Dispersion re-activated

      if(ext.eq.'EF')then
      open(98,file='EXTFIELD.DAT')
      read(98,*) ne
      do23084 i=1,ne 
      read(98,*) xe(1,i),ze(i)
      xe(1,i)=xe(1,i)/0.529177
23084 continue
23085 continue
      endif
!     ---------------------- QC_UW04: for ewald --------------------
      if(period)then
      slkcutoff=0.0d0
      do23100 i = 1,ntype 
      do23102 j = 1,ntype 
      yhlp=sr(i,j)*dim(i,j)+0.3
      if(yhlp.gt.slkcutoff)then
      slkcutoff=yhlp
      endif
23102 continue
23103 continue
23100 continue
23101 continue
      call gamma_summind(slkcutoff,boxsiz,nlat)
      if (mysccid.eq.0) 
     $write(*,*) 'Number of lattice sums for Gammapoint matrix:', nlat(1
     *), nlat(2), nlat(3)
!TMP  QC: SET nlat to be zero
      nlat(1) = 0
      nlat(2) = 0
      nlat(3) = 0
      if (mysccid.eq.0) 
     $write(*,*) 'Number of lattice sums for Gammapoint matrix:', nlat(1
     *), nlat(2), nlat(3)
      endif
!-----------------------------------------------------------------
! MG_QC_UW1207: read in atomic spin constants
! Read wspin matrix for each atom type (sorted as above): Wss, Wsp, Wps, Wpp, Wsd, Wpd, Wdd, Wds, Wdp
      if (lcolspin) then
        write(*,*) "Reading in atomic spin constants: "
        write(*,*) "\'spin\'-dummy, Wss, Wsp, Wps, Wpp, Wsd, Wpd, Wdd, 
     &Wds, Wdp"
        do i=1,ntype 
          read(4,*) dummy,                    wspin(i,1,1),wspin(i,1,2),
     &              wspin(i,2,1),wspin(i,2,2),wspin(i,1,3),wspin(i,2,3),
     &              wspin(i,3,3),wspin(i,3,1),wspin(i,3,2)
          !write(*,*)"spin '",dummy,"' ",      wspin(i,1,1),wspin(i,1,2),
          write(*,*)dummy,                    wspin(i,1,1),wspin(i,1,2),
     &              wspin(i,2,1),wspin(i,2,2),wspin(i,1,3),wspin(i,2,3),
     &              wspin(i,3,3),wspin(i,3,1),wspin(i,3,2)
        enddo
        nelup   = 0.5d0 * (nel + unpe)
        neldown = 0.5d0 * (nel - unpe)
        write(*,*) 'total number of up-electrons:', nelup
        write(*,*) 'total number of down-electrons:', neldown
      endif
!     QC: If read in derivative of the Hubbard parameters 
! MG+Guanhua_QC_UW1205
      if (luhder .or. lscc3rd) then
        write(*,*) "Reading in derivative of Hubbard para"
        i=0
        do while (i.lt.ntype)
          i=i+1
          !MG_UW1210 ldep
          if (lldep) then
            read(4,*) dummy,(uhder(i,4-j),j=1,3)
          else       
            read(4,*) dummy,uhder(i,1)
          endif
          if (dummy(1:4)=="spin") then
            i=i-1
          else
            if (lldep) then
              write(6,*) "Hubbard derivative (d,p,s): ",
     &                            i,dummy,(uhder(i,4-j),j=1,3)
            else       
              write(6,*) "Hubbard derivative: ",i,dummy,uhder(i,1)
            endif
          endif
          if (i.gt.MAXTYP) then
            call wrndie(-5,'<DYLCAO>','erroneous sccdftb.dat file')
          endif
        enddo
      endif
      if (luhgau) then
        write(*,*) "Reading in Gaussian para in Hubbard deri"
        read(4,*) v0_hgau,alp_hgau,q0_hgau
          write(6,*) "Gaussian para for Hubbard derivative: ",
     $            v0_hgau,alp_hgau,q0_hgau
      endif
!     QC: If read in the hbond correction terms
      if (lscchb) then
        write(*,*) "Reading in hbond correction parameter: kl1"
        read(4,*) kl1
        write(6,*) "Hbond correction parameter: kl1: ", kl1
        write(6,*) "Hbond is invoked for atom pairs where at least one",
     & "atom is of nuclear charge equals one"
      endif

      if (lsccdftd2.or.lsccdftd3) then

          ! Convert charges to integers
          do i=1,nn
              ! write (*,*), "ASC - typ", i, scczin(i), nn
              scczin_int(i) = int(scczin(i))
          enddo

          ! Initialize everythign for DFT-D3 dispersion correction
          call init_dftd3(scczin_int, nn, lscc3rd, lsccdftd3, lcpe)
      endif

#if KEY_DFTBMKL==1
      if (lcpe) call init_cpepar(nn)
#endif

!-----------------------------------------------------------------
! MG_QC_UW1205
!      do i=1,ntype
!        if (lscchb.and.(uhubb(i).eq.0.4195d0)) then
!          izpxh(i)=.true.
!        else
!          izpxh(i)=.false.
!        endif
!      enddo
! MG: scczin (nuclear charges) is an array of all qm-atoms, thus we have to run over all atoms instead of only the qm atom types (ntype)
      do i=1,nn
        if (lscchb.and.(scczin(i).lt.1.5)) then
          izpxh(izp(i))=.true.
        else
          izpxh(izp(i))=.false.
        endif
      enddo

!ccccccccccccccccccccccccccccccccccccccccccc
!     QC: Simplify GL's data copy
!     MG_UW1211: extension to lcolspin for Free Energy Perturbation
      if((qaflag) .and. (.not.qbflag))then
        nela=nel
        do i=1,nn
          izp2a(i)=izp2(i)
        enddo 
        if (lldep) then
          do i=1,3*nn
            qla(i)=ql(i)
          enddo
        else
          do i=1,nn
            qmata(i)=qmat(i)
          enddo
        endif
        if (lcolspin) then
          nelupa=nelup
          neldowna=neldown
          do i=1,3*nn
            qlupa(i)=qlup(i)
            qldowna(i)=qldown(i)
          enddo
        endif

      elseif((qbflag) .and. (.not.qaflag))then
        nelb=nel
        do i=1,nn
          izp2b(i)=izp2(i)
        enddo 
        if (lldep) then
          do i=1,3*nn
            qlb(i)=ql(i)
          enddo
        else
          do i=1,nn
            qmatb(i)=qmat(i)
          enddo
        endif
        if (lcolspin) then
          nelupb=nelup
          neldownb=neldown
          do i=1,3*nn
            qlupb(i)=qlup(i)
            qldownb(i)=qldown(i)
          enddo
        endif
      endif
!ccccccccccccccccccccccccccccccccccccccc




      if(ext.eq.'CH')then
      close (4)
      return
      endif

9     continue
12    format(/2x,4x,'iter  free',6x,'e(total)',5x,'e(bandstr)', 5x,'e(bi
     *nding)',6x,'deltax(max)',9x,'f(max)'/,1x,95('='))
13    format(1x,i5,1x,'/',1x,i2,1x,i3,3(1x,f14.6),1x,f14.6,2(1x,f15.6))
      print *,' '
      print *,'***** end of dftb *****'
      return
      end

