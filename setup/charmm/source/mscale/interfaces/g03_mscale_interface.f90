Program interface_gaussian
  !        use nwchem_int
  !----------------------------------------------------------------------
  !
  !     This is the template program for MSCAle command in CHARMM
  !     It is for programs needed to communicate with the CHARMM
  !     but which are not part of the CHARMM. We run these programs
  !     through system call.
  !
  !     NOTE: this could be folded into CHARMM, but then it might be a
  !     problem with licensing issues. This way it is easy to publish
  !     just this program and the developers of the programs in SYSTEM()
  !     call can improve on it, eg make faster communication than with
  !     files using MPI, plugins, sockets, etc
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !   Make a separate file with the communication routines
  !   and the rest of utilities that are always needed!
  !
  !   How to include this into CHARMM tree. Maybe install.com gnu xxlarge mscale ???
  !   this would compile all the interfaces in source/mscale directory and put them to
  !   exec/gnu directory
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !
  use mpi
  implicit none
  !
  CHARACTER(LEN=250) :: PROGRAM
  INTEGER(4) IERR,COMPARENT,MEROOT,IONE,MPIINT,MPIDP,N,MYNOD
  INTEGER(4) MPICOMW,LENENT,NP
  INTEGER ARGC,LL,LNWCPATH,NATOM,ASTAT,I,LFN,LETERM
  CHARACTER(LEN=100) :: ARG,FN,FOUT
  REAL(8) QMEL
  REAL(8), allocatable,dimension(:) :: AN,X,Y,Z,ETERM
  !
  CALL MPI_INIT(IERR)

  ARGC = COMMAND_ARGUMENT_COUNT()
  !      write(*,*)'argc=',argc
  !
  LNWCPATH=42 ! full PATH?!
  program(1:LNWCPATH)='/v/estor1/home/hlwood/bin/gaussian/run_g03'
  !
  if (argc.eq.0) then
     write(*,*)'No parameters.'
     call exit
  endif
  do i = 1, argc
     call get_command_argument(i,arg)
     ll=len_trim(arg)
     !        write(*,*)'arg=',arg(1:ll)
     if(arg.eq.'-output')then
        call get_command_argument(i+1,arg)
        ll=len_trim(arg)
        FOUT(1:ll)=arg(1:ll)
        !            open(unit=6,file=arg(1:ll),status='unknown')
        program(LNWCPATH+1:LNWCPATH+3)='   '
        program(LNWCPATH+4:LNWCPATH+3+ll)=arg(1:ll)
        LNWCPATH=LNWCPATH+3+LL
     endif
     if(arg.eq.'-input')then
        call get_command_argument(i+1,arg)
        lfn=len_trim(arg)
        fn(1:lfn)=arg(1:lfn)
        program(lnwcpath+1:lnwcpath+1)=' '
        program(LNWCPATH+2:LNWCPATH+1+lfn)=arg(1:lfn)
        program(LNWCPATH+2+lfn:LNWCPATH+4+lfn)='.in'
        lnwcpath=lnwcpath+4+lfn
     endif
  enddo

  !      write(*,*)'interface>program=',program(1:LNWCPATH)

  CALL MPI_COMM_GET_PARENT(COMPARENT,IERR)

  MPICOMW=MPI_COMM_WORLD
  MPIINT=MPI_INTEGER
  MPIDP=MPI_DOUBLE_PRECISION
  LENENT=128
  LETERM=LENENT
  IONE=1
  CALL MPI_COMM_RANK(MPICOMW,MYNOD,IERR)

100 continue

  !     First get the atomic data
  !
  MEROOT=0
  !
  CALL MPI_BCAST(N,IONE,MPIINT,MEROOT,COMPARENT,IERR)
  NATOM=N
  !     NP is number of processes for this job:
  !     to be put into program string
  CALL MPI_BCAST(NP,IONE,MPIINT,MEROOT,COMPARENT,IERR)


  ALLOCATE(AN(NATOM),X(NATOM),Y(NATOM),Z(NATOM),ETERM(LETERM),STAT=ASTAT)

  !
  CALL MPI_BCAST(AN,N,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(X,N,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(Y,N,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(Z,N,MPIDP,MEROOT,COMPARENT,IERR)

  CALL PREPRUN(NATOM,FN(1:lfn),AN,X,Y,Z)

  !      This can be in the input script itself: task shell ""
  !      call system('/bin/rm -f sys1.b sys1.b^-1 sys1.c sys1.db')
  !      call system('/bin/rm -f sys1.gridpts.0 sys1.grinfo.0')
  !      call system('/bin/rm -f sys1.movecs sys1.p sys1.zmat')
  CALL SYSTEM(program(1:LNWCPATH))

  CALL read_en_n_force(NATOM,FOUT(1:ll),QMEL,X,Y,Z)

  do i = 1, lenent
     eterm(i)=0
  enddo
  ! the number 26 is the right place for the QM energy in CHARMM
  eterm(26) = qmel
  !
  !      write(*,*)'qmel=',qmel
  !      write(*,*)'dx,dy,dz=',x(1),y(1),z(1),x(2),y(2),z(2)

  IF(MYNOD.EQ.0)THEN
     MEROOT=MPI_ROOT
  ELSE
     MEROOT=MPI_PROC_NULL
  ENDIF
  !
  CALL MPI_BCAST(X,NATOM,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(Y,NATOM,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(Z,NATOM,MPIDP,MEROOT,COMPARENT,IERR)
  CALL MPI_BCAST(ETERM,LENENT,MPIDP,MEROOT,COMPARENT,IERR)
  !
  DEALLOCATE(AN,X,Y,Z,ETERM,STAT=ASTAT)

  GOTO 100
  !
  CALL MPI_FINALIZE(IERR)
  !C
END program interface_gaussian

!      module nwchem_int

!      contains

subroutine PREPRUN(natom,gau_ctrl,AN,X,Y,Z)
  implicit none

  character(len=*) gau_ctrl
  character(len=120) line
  character(len=2)atn
  real(8) AN(*),X(*),Y(*),Z(*)
  integer ll,natom,i,ifound

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !Test initialization parameters!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !       gau_ctrl="gau_ctrl"
  !       open(unit=4,file='xyztest',status='old')
  !       read(4,'(a120)',end=101)line 
  !       ifound=index(line,'natoms=')
  !        if(ifound.gt.0)then
  !           read(line,*)tmp1,natom
  !        endif
  !
  !         do i = 1, natom
  !           read(4,*)an(i),X(i),Y(i),Z(i)
  !           write(*,*)an(i),X(i),Y(i),Z(i)
  !         enddo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ll=len_trim(gau_ctrl)

  open(unit=1,file=gau_ctrl(1:ll),status='old')
  gau_ctrl(ll+1:ll+3)='.in'

  open(unit=2,file=gau_ctrl(1:ll+3),status='unknown')

100 continue
  read(1,'(a120)', end=105)line
  write(2,'(a120)')line
  goto 100


105 continue 
  do i = 1, natom
     call atomname(an(i),atn)
     write(2,'('' '',a2,3f20.10)')atn,x(i),y(i),z(i)            
  enddo
  write(2,'(a1)')" "


101 continue
  close(unit=1)
  close(unit=2)
  return
end subroutine PREPRUN



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                           !
!   Subroutine to read gaussian energy and forces           !
!                                                           !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


subroutine read_en_n_force(natom,fname,ener,dx,dy,dz)

  implicit none

  integer natom,ifound,jfound,kfound,i,ll
  character(len=14) tmp1,tmp2,tmp3,tmp4
  character(len=120) rname,rname1,line
  character(len=*) fname

  real(8) scf_e, mp2_cor, mp2tot, ccsd_e, ener 
  real(8) dx(*), dy(*), dz(*), tokcal, bohrr, tmp5, tmp6

  tokcal = 627.5095D0
  bohrr =  0.529177249D0

  ll=len_trim(fname)
  rname(1:ll)=fname(1:ll)

  open(unit=3,file=rname(1:ll),status='unknown')

  !     open(unit=3,file='output.dat',status='old')
2 read(3,'(a120)',end=101)line

  ifound=index(line,'NAtoms=')
  if (ifound.gt.0)then
     read(line,*)tmp1,natom
     !print(*,*) natom
  endif

  ifound=index(line,'SCF Done:') 
  if (ifound.gt.0)then
     read(line,*)tmp1,tmp2,tmp3,tmp4,scf_e
     !print(*,*) scf_e
     ener=scf_e
  endif

  ifound=index(line,'E2 =')
  if (ifound.gt.0)then
     read(line,*)tmp1,tmp2,mp2_cor
     !print(*,*) mp2_cor 
     ener=scf_e+mp2_cor
  endif

  ifound=index(line,'DE(Corr)=')
  if (ifound.gt.0)then
     read(line,*)tmp1,tmp2,tmp3,ccsd_e
     !print(*,*) ccsd_e
     ener=ccsd_e
  endif

  ifound = index(line, &
       'Center     Atomic                   Forces (Hartrees/Bohr)')

  if(ifound.gt.0)then
     read(3,'(a120)')rname
     read(3,'(a120)')rname1
     do i = 1, natom
        read(3,*)tmp1,tmp2,dx(i),dy(i),dz(i)
        dx(i)=-1.0D0*dx(i)*tokcal/bohrr
        dy(i)=-1.0D0*dy(i)*tokcal/bohrr
        dz(i)=-1.0D0*dz(i)*tokcal/bohrr
     enddo
  endif

  goto 2

  close(unit=3)

101 continue

  mp2tot = scf_e + mp2_cor
  !print(*,*) mp2tot
  ener = ener*tokcal

end subroutine read_en_n_force


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                            !
!  Subroutine to get atomic masses                           !
!                                                            !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



subroutine atomname(an,name)

  implicit none
  real(8) an
  integer ind
  character(len=2) name
  character(len=2) :: element(111) =                      &
       ['H ','He','Li','Be','B ','C ','N ','O ','F ','Ne',&
       'Na','Mg','Al','Si','P ','S ','Cl','Ar',           &
       'K ','Ca','Sc','Ti','V ','Cr','Mn','Fe','Co','Ni', &
       'Cu','Zn','Ga','Ge','As','Se','Br','Kr',           &
       'Rb','Sr','Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd', &
       'Ag','Cd','In','Sn','Sb','Te','I ','Xe',           &
       'Cs','Ba','La',                                    &
       'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho', &
       'Er','Tm','Yb','Lu',                               &
       'Hf','Ta','W ','Re','Os','Ir','Pt',                &
       'Au','Hg','Tl','Pb','Bi','Po','At','Rn',           &
       'Fr','Ra','Ac',                                    &
       'Th','Pa','U ','Np','Pu','Am','Cm','Bk','Cf','Es', &
       'Fm','Md','No','Lr',                               &
       'Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg'            &      
       ]
  ind=an
  if((ind.gt.111).or.(ind.le.0))write(*,*)'atomname>No such element.'
  name=element(ind)
  return

end subroutine atomname



