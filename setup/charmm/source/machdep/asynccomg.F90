 module asynccomg
! general async communication routines  --RJP/Milan Hodoscek ~June 2011
#if KEY_PARALLEL==1 /*asyncpar*/
  use dimens_fcm
  use chm_kinds
  use chm_types,only: AROFAR,AROFAR_I4,AROFAR_I8
  use nbndcc_utilb, only: parstoperr
  implicit none
!  real(chm_real),parameter :: large_real=1.0D30

contains
!--------------------------------------------------------------------------------
!******************************************************************************************************
!******************************************************************************************************
     SUBROUTINE ASYNCLST_RG(NROUND,NODARR,RECVMNY,  &
      RHANDLES,PRIBUF,PRI8BUF,PRRBUF,PCCATOR,PNMAP,PTYPE,PDATAWID,plabint)
! general routine for posting lists of asynchronous receives; no sends or waits. --rjp/mh, June 2011
     use memory
!     use spacdat_mod,only: AROFAR,AROFAR_I4,AROFAR_I8
     use parallel, only: MYNODP 
#if KEY_PARINFNTY
     use parallel,only: qinfinity
#endif
     implicit none
     integer, intent(in) :: NROUND  !number of rounds of receives
     integer,intent(in),dimension(:) :: NODARR,RECVMNY  ! NODARR array of "nodes", indexed by round
                                                        ! RECVMNY size of receives, indexed by "nodes" --i.e. values of nodarr
     integer(chm_int4),allocatable,dimension(:),intent(inout) :: RHANDLES
     type(AROFAR),allocatable,dimension(:),optional,intent(inout) :: PRRBUF
     type(AROFAR_I4),allocatable,dimension(:),optional,intent(inout) :: PRIBUF
     type(AROFAR_I8),allocatable,dimension(:),optional,intent(inout) :: PRI8BUF
     integer(chm_int4),intent(in),optional :: PCCATOR  !communicator
     integer,intent(in),dimension(:),optional :: PNMAP   !maps round to rank of "from" process in current communicator
     integer(chm_int4),intent(in),optional :: PTYPE  !optional   !passed additional mpi tag. sends and receives must match; def = 1
     integer, intent(in),optional :: PDATAWID   !passed width of incoming data (# columns); deflt is 1
     integer, intent(in), optional :: plabint
!     logical :: qmemerror
!
     include 'mpif.h'

! local
     integer(chm_int4) :: CCATOR  !communicator
     integer(chm_int4),dimension(:),allocatable :: NMAP   !maps round to rank of "from" process in current communicator
     integer(chm_int4) :: TYPE  ! type (mpi tag)
     integer :: DATAWID    ! width of data (number of columns)
     integer(chm_int4) :: statigno(MPI_STATUS_SIZE)
     integer :: RECVND,IBEGIN,IEND,RECVLEN,COUNT,III,ATOM,ROUND
     integer(chm_int4) :: RSIZE,FROM,IE,MPIDOUB,MPINT,MPINT8,myid
     integer :: R8SIZE
     integer(chm_int4) :: ihit  !temporary
     logical :: QINTBUF,QRELBUF,QINT8BUF
     integer :: labint  !integer label
     integer(chm_int4) :: ROUNDI4
!  end of declarations
!------------------------------------------------------------------------------------------------------------------
#if KEY_PARINFNTY==1
     if(qinfinity) return 
#endif
     MPINT=MPI_INTEGER
     MPINT8=MPI_INTEGER8  
     MPIDOUB=MPI_DOUBLE_PRECISION
     STATIGNO(:)=MPI_STATUS_IGNORE(:)
!

! Process optional arguments
! label
     labint = 0
     if(present(plabint)) labint = plabint

     QINTBUF = .false.
     QRELBUF = .false.
     QINT8BUF = .false.
     if(present(PRIBUF)) then
       QINTBUF = .true.
       if(allocated(PRIBUF)) deallocate(PRIBUF)
       allocate(PRIBUF(NROUND))
     endif
     if(present(PRI8BUF)) then
       if(QINTBUF) then
         WRITE(6,*) '<ASYNCLST_RG> ERROR: cannot pass both i4 and i8 buffers'
         STOP
       else 
        QINTBUF = .true.
        QINT8BUF = .true.
       endif 
       if(allocated(PRI8BUF)) deallocate(PRI8BUF)
       allocate(PRI8BUF(100*NROUND))
     endif
     if(present(PRRBUF)) then
       if(QINTBUF) then
         WRITE(6,*) '<ASYNCLST_RG> ERROR: cannot pass both real and integer buffers'
         STOP
       else
        QRELBUF = .true.
        if(allocated(PRRBUF)) deallocate(PRRBUF)
        allocate(PRRBUF(NROUND))
       endif
     endif 
     if((.not.QINTBUF).and.(.not.QRELBUF)) then
       WRITE(6,*) '<ASYNCLST_RG> ERROR: neither real nor integer buffer passed'
         STOP
     endif
     
     TYPE = 1
     if(present(PTYPE)) TYPE = PTYPE

     if(allocated(NMAP)) call chmdealloc('asynccomg.src','ASYNCLST_RG','NMAP',NROUND,ci4=NMAP)
!deallocate(NMAP)
     call chmalloc('asynccomg.src','ASYNCLST_RG','NMAP',NROUND,ci4=NMAP)
!allocate(NMAP(NROUND))
     NMAP(1:NROUND) = NODARR(1:NROUND) - 1  !default
     if(present(PNMAP)) NMAP = PNMAP

     CCATOR = MPI_COMM_WORLD
     if(present(PCCATOR)) CCATOR = PCCATOR
     DATAWID = 1
     if(present(PDATAWID)) DATAWID = PDATAWID

!deallocate(RHANDLES)
     if(allocated(RHANDLES)) call chmdealloc('asynccomg.src','ASYNCLST_RG','RHANDLES',NROUND,ci4=RHANDLES)
!allocate(RHANDLES(NROUND))
     call chmalloc('asynccomg.src','ASYNCLST_RG','RHANDLES',NROUND,ci4=RHANDLES)

     RHANDLES = 0
!
     DO ROUND = 1,NROUND
       ROUNDI4 = ROUND
       RECVND = NODARR(ROUND)   !node pointing into recvlst

       IF (RECVND==0) THEN
        call parstoperr('<ASYNCLST_RG>','RANKP = 0, RECEIVE NODE MUST EXIST')
       ENDIF
       RECVLEN = RECVMNY(RECVND)
       IF(RECVLEN.LE.0) THEN
         WRITE(6,*) 'ROUND AT ZERO LENGTH IS ',ROUND,' RECVND ',RECVND
         WRITE(6,*) 'MYNODP ',MYNODP,' RECEIVE LENGTH ZERO, label ',labint
         call parstoperr('<ASYNCLST_RG>',' RECEIVE LENGTH = 0, NOT ALLOWED ')
!         CALL WRNDIE(-5,'<ASYNCLST_RG>','RECEIVE LENGTH = 0, NOT ALLOWED')
       ENDIF
!
       RSIZE = RECVLEN*DATAWID
       R8SIZE = RSIZE
       from=NMAP(ROUND) !rank of from node in current communicator
! post integer receives
       if(QINTBUF) then
        if(QINT8BUF) then
         allocate(PRI8BUF(ROUND)%b(R8SIZE))
          PRI8BUF(ROUND)%b = 0
         call mpi_irecv(PRI8BUF(ROUND)%b,RSIZE,MPINT8,FROM,TYPE,CCATOR,RHANDLES(ROUNDI4),IE)
        else
         allocate(PRIBUF(ROUND)%b(RSIZE))
         PRIBUF(ROUND)%b= 0
         call mpi_irecv(PRIBUF(ROUND)%b,RSIZE,MPINT,FROM,TYPE,CCATOR,RHANDLES(ROUNDI4),IE)
        endif
! else post real receives
       else if(QRELBUF) then
        allocate(PRRBUF(ROUND)%b(RSIZE))
        call mpi_irecv(PRRBUF(ROUND)%b,RSIZE,MPIDOUB,FROM,TYPE,CCATOR,RHANDLES(ROUNDI4),IE)
       else
        call parstoperr('<ASYNCLST_RG>',' bad buffer data type ',pqstop=.false.)
       endif
!
     ENDDO  !loop over rounds

    END SUBROUTINE ASYNCLST_RG
!******************************************************************************************************
!******************************************************************************************************
     SUBROUTINE ASYNCLST_SG(NROUND,NODARR,SENDLST,SENDHI,SENDMNY,  &
     SHANDLES,PSIBUF,PSI8BUF,PSRBUF,PCCATOR,PNMAP,PTYPE,PDATAWID,PRDATA,PIDATA,PXX,PYY,PZZ,plabint)
! posts list of asynchronous sends; no waits or receives; fairly general routine   -rjp/mh June 2011
! By default, SENDLST contains the (integer) data to be sent.  If optional data array
! is (are) passed, SENDLST becomes an array of indexes for the data array.
! The 2D data arrays (PRDATA, PIDATA) are traversed across the first index (columns) first,
! where the number of columns=datawid. One of the buffers PSIBUF or PSRBUF must be
! passed.
     use memory
!     use spacdat_mod, only: AROFAR,AROFAR_I4,AROFAR_I8
     use parallel, only: MYNODP !temporary
     implicit none

     integer, intent(in) :: NROUND     ! number of sends
     integer,intent(in),dimension(:) :: NODARR  ! maps round to corresponding part of sendlst (e.g. CPU)
     integer,intent(in),dimension(:),target :: SENDLST  ! send list
     integer,intent(in),dimension(:) :: SENDHI,SENDMNY   ! marker arrays for sendlist
     integer(chm_int4),allocatable,dimension(:),intent(out) :: SHANDLES   !mpi handles for posted sends
     type(AROFAR_I4),allocatable,dimension(:),optional,intent(inout) :: PSIBUF   !send data buffer (passed in/out b/c deallocated later)
     type(AROFAR_I8),allocatable,dimension(:),optional,intent(inout) :: PSI8BUF   !send data buffer (passed in/out b/c deallocated later)
     type(AROFAR),allocatable,dimension(:),optional,intent(inout) :: PSRBUF   !send data buffer
     integer(chm_int4),intent(in),optional :: PCCATOR  !(passed) communicator; default is MPI_COMM_WORLD
     integer,intent(in),optional,dimension(:) :: PNMAP   !maps round to rank of "to" process in current communicator; def is nodarr
     integer(chm_int4),intent(in),optional :: PTYPE  !optional   !passed additional mpi tag. sends and receives must match; def = 1
     integer, intent(in),optional :: PDATAWID   !passed width of data (# columns); deflts o/w taken fr size of array
     real(chm_real),intent(in),dimension(:,:),optional,target :: PRDATA
     integer,intent(in),dimension(:,:),optional,target :: PIDATA  !data arrays using SENDLST as index
     real(chm_real),intent(in),dimension(:),optional,target :: PXX,PYY,PZZ 
     integer,intent(in),optional :: plabint
!
     include 'mpif.h'
! local variables/arrays
     real(chm_real),dimension(:,:),pointer :: RARDATA   !array holding real data to be sent  (e.g. coordinates); default is null
     integer,dimension(:,:),pointer :: IARDATA   !array holding integer data to be sent  (e.g. atom numbers); default is SENDLST
     integer :: DATAWID    ! width of data (number of columns)
     integer(chm_int4) :: TYPE  ! type (mpi tag)
     integer(chm_int4),dimension(:),allocatable :: NMAP   !maps round to rank of "to" process in current communicator; def is nodarr -1
     integer(chm_int4) :: CCATOR  !communicator; default is MPI_COMM_WORLD
     integer(chm_int4) :: statigno(MPI_STATUS_SIZE)
     integer :: SENDND,IBEGIN,IEND,SENDLEN,COUNT,III,ROWIND,ROUND,PX,PY,PZ,PT,COLIND
     integer(chm_int4) :: TO,SSIZE,IE,MPIDOUB,STATUS,IHIT,MPINT,MPINT8
     integer :: S8SIZE
     real(chm_real),target,dimension(1) :: RNULL
     logical :: QINTDATA,QRELDATA,QXYZDATA,QDEFDATA,QPSIBUF,QPSI8BUF
     integer(chm_int4) :: myid !temp
     integer :: labint
     integer :: ii !temp
     integer :: pass=0
!--------------------------------------------------------------------------------------------------------------------------
!  end of declarations
!--------------------------------------------------------------------------------------------------------------------------
     MPIDOUB=mpi_double_precision
     MPINT=MPI_INTEGER
     MPINT8=MPI_INTEGER8  
     statigno(:)=MPI_STATUS_IGNORE(:)

      pass = pass + 1
!deallocate(SHANDLES)
     if(allocated(SHANDLES)) call chmdealloc('asynccomg.src','ASYNCLST_SG','SHANDLES',NROUND,ci4=SHANDLES)
!allocate(SHANDLES(NROUND))
     call chmalloc('asynccomg.src','ASYNCLST_SG','SHANDLES',NROUND,ci4=SHANDLES)

     SHANDLES = 0
!
! process optional arguments and perform internal checks
! label
     labint = 0
     if(present(plabint)) labint = plabint  

!  process data arrays
     QDEFDATA = .false.
     QINTDATA = .false.
     QRELDATA = .false.
     QXYZDATA = .false.
     QPSIBUF = .false.
     QPSI8BUF = .false.
! if passing real data
     if(present(PRDATA)) then
      QRELDATA = .true.
      DATAWID = size(PRDATA,1)
!      WRITE(6,*) 'DATAWID is ',DATAWID
     endif
! if passing integer data
     if(present(PIDATA)) then
      if(QRELDATA) then
       WRITE(6,*) '<ASYNCLST_SG> ERROR: cannot pass real and integer data simultaneously' 
       STOP
      else
       QINTDATA = .true.
       DATAWID = size(PIDATA,1)
      endif
     endif
!process x,y,z
! if passing x, y, z data
     PX = 0
     if(present(PXX)) PX = 1
     PY = 0
     if(present(PYY)) PY = 1
     PZ = 0
     if(present(PZZ)) PZ = 1
     PT = PX+PY+PZ
     if(PT.GT.0) then
       if((QRELDATA).OR.(QINTDATA)) then
        WRITE(6,*) '<ASYNCLST_SG> ERROR: passing multiple data arrays simultaneously'
        STOP       
       else if (PT.LT.3) then
        WRITE(6,*) '<ASYNCLST_SG> ERROR: if passing X,Y,Z, must pass all three arrays'
        STOP
       else 
        QXYZDATA = .true.
        DATAWID = 3
       endif
     endif
     if((.not.QXYZDATA).and.(.not.QINTDATA).and.(.not.QRELDATA)) then
       QDEFDATA = .true. 
       DATAWID = 1
     endif
     if(QRELDATA.OR.QXYZDATA) then
      if(.not.present(PSRBUF)) then
        WRITE(6,*) '<ASYNCLST_SG> ERROR: must pass real buffer if passing real data'
        STOP
      endif
      if(allocated(PSRBUF)) deallocate(PSRBUF)
      allocate(PSRBUF(NROUND))
     else if (QINTDATA.OR.QDEFDATA) then
      if((.not.present(PSIBUF)).and.(.not.present(PSI8BUF))) then
        call parstoperr('<ASYNCLST_SG>',' must pass integer buffer if passing integer data')
      endif
      if(present(PSIBUF)) then
       QPSIBUF=.true.
       if(allocated(PSIBUF)) deallocate(PSIBUF)
       allocate(PSIBUF(NROUND))
      endif
      if(present(PSI8BUF)) then
       if(QPSIBUF) then
         call parstoperr('<ASYNCLST_SG>',' cannot pass both I4 and I8 data')
       else
        QPSI8BUF = .true.
        if(allocated(PSI8BUF)) deallocate(PSI8BUF)
        allocate(PSI8BUF(100*NROUND))
       endif
      endif
     else
      call parstoperr('<ASYNCLST_SG>',' bad data type specification')
     endif
     if(QINTDATA) then
       if(QPSIBUF) then
        if(kind(PSIBUF(1)%b).ne.kind(PIDATA)) then
           call parstoperr('<ASYNCLST_SG>',' type of integer data does not match type of integer(4) buffer')
        endif 
       elseif(QPSI8BUF) then
         if(kind(PSI8BUF(1)%b).ne.kind(PIDATA)) then
           call parstoperr('<ASYNCLST_SG>',' type of integer data does not match type of integer(8) buffer')
        endif
       endif
     endif
     if(QDEFDATA) then
       if(QPSIBUF) then
        if(kind(PSIBUF(1)%B).ne.kind(SENDLST)) then
           call parstoperr('<ASYNCLST_SG>',' type of integer data does not match type of integer(4) buffer')
        endif
       elseif(QPSI8BUF) then
         if(kind(PSI8BUF(1)%B).ne.kind(SENDLST)) then
           call parstoperr('<ASYNCLST_SG>',' type of integer data does not match type of integer(8) buffer')
           STOP
        endif
       endif
     endif
!
     if(present(PDATAWID)) then
       if(PDATAWID.gt.DATAWID) then
         call parstoperr('<ASYNCLST_SG>',' specified data width is greater than actual')
       else 
        DATAWID = PDATAWID
       endif
     endif
     if(DATAWID.le.0) then
       call parstoperr('<ASYNCLST_SG>','data width cannot be <=0')
     endif

     TYPE = 1
     if(present(PTYPE)) TYPE = PTYPE

     if(allocated(NMAP)) call chmdealloc('asynccomg.src','ASYNCLST_SG','NMAP',NROUND,ci4=NMAP)
! deallocate(NMAP) 
     call chmalloc('asynccomg.src','ASYNCLST_SG','NMAP',NROUND,ci4=NMAP)
!allocate(NMAP(NROUND))
     NMAP(1:NROUND) = NODARR(1:NROUND) - 1
     if(present(PNMAP)) NMAP = PNMAP
     
     CCATOR = MPI_COMM_WORLD
     if(present(PCCATOR)) CCATOR = PCCATOR

!---------------------------------------------------------------------------------------------------
! main loop over rounds begins
!---------------------------------------------------------------------------------------------------
     do ROUND = 1,NROUND
        SENDND = NODARR(ROUND)
        if (SENDND==0) THEN
         call parstoperr('<ASYNCLST_SG>','RANKP = 0, EMPTY COMMUNICATION ROUNDS NOT ALLOWED')
        endif
        if(SENDND.lt.0) then
         WRITE(6,*) 'BAD SENDNODE ON CPU ',MYNODP,' ROUND ',ROUND,' SENDND ',SENDND
         call parstoperr('<ASYNCLST_SG>','BAD SENDNODE')
        endif
        IEND = SENDHI(SENDND)    
        SENDLEN = SENDMNY(SENDND)
!        if(MYNODP.eq.1) then
!         WRITE(6,*) 'in _SG NODE 1 is sending ',SENDLEN,' to node ',SENDND,' pass ',pass,' lab ',labint
!        endif
        IF(SENDLEN.LE.0) THEN
         call parstoperr('<ASYNCLST_SG>','LENGTH OF SEND ARRAY = 0, NOT ALLOWED')
        ENDIF
        
        IBEGIN = IEND - SENDLEN + 1
!
        SSIZE = DATAWID*SENDLEN
        S8SIZE = SSIZE
        if(QRELDATA.OR.QXYZDATA) then
         allocate(PSRBUF(ROUND)%B(S8SIZE))
        else if(QINTDATA.OR.QDEFDATA) then
         if(QPSIBUF) then
          allocate(PSIBUF(ROUND)%B(SSIZE))      
         elseif(QPSI8BUF) then
          allocate(PSI8BUF(ROUND)%B(S8SIZE))      
         else
          call parstoperr('<ASYNCLST_SG>','BAD INTEGER DATA TYPE')
         endif
        else
          call parstoperr('<ASYNCLST_SG>','data type specification error')
        endif
!
        COUNT=0
        if(QRELDATA) then
         do III = IBEGIN,IEND
          ROWIND = SENDLST(III)
          COLIND = 1 
          do while(COLIND.le.DATAWID)
           COUNT=COUNT+1
           PSRBUF(ROUND)%B(COUNT) = PRDATA(COLIND,ROWIND)
!           WRITE(6,*) 'PSRBUF at round ',ROUND,' is ',PSRBUF(ROUND)%B(COUNT)
           COLIND = COLIND + 1
          enddo
         enddo
        else if(QINTDATA) then
         do III = IBEGIN,IEND
          ROWIND = SENDLST(III)
          COLIND = 1
          do while(COLIND.le.DATAWID)
           COUNT=COUNT+1
           if(QPSIBUF) then
            PSIBUF(ROUND)%B(COUNT) = PIDATA(COLIND,ROWIND)
           elseif(QPSI8BUF) then
            PSI8BUF(ROUND)%B(COUNT) = PIDATA(COLIND,ROWIND)
           endif
           COLIND = COLIND + 1
          enddo
         enddo
        else if(QXYZDATA) then
         do III = IBEGIN,IEND
          ROWIND = SENDLST(III)
          COUNT = COUNT + 1
          PSRBUF(ROUND)%B(COUNT) = PXX(ROWIND)
          COUNT = COUNT + 1
          PSRBUF(ROUND)%B(COUNT) = PYY(ROWIND)
          COUNT = COUNT + 1
          PSRBUF(ROUND)%B(COUNT) = PZZ(ROWIND)
         enddo
        else   !default sendlst data array
         do III = IBEGIN,IEND
          COUNT = COUNT + 1
          if(QPSIBUF) then
           PSIBUF(ROUND)%B(COUNT) = SENDLST(III)
          elseif(QPSI8BUF) then
            if(COUNT.GT.SIZE(PSI8BUF(ROUND)%B)) then
              WRITE(6,*) 'COUNT/MEM ERROR:  MYNODP',MYNODP,' SIZE ',SIZE(PSI8BUF(ROUND)%B),' COUNT ',COUNT 
            endif
           PSI8BUF(ROUND)%B(COUNT) = SENDLST(III)
          endif
         enddo
        endif  !what kind of data is it

        TO=NMAP(ROUND)
        if(QRELDATA.OR.QXYZDATA) then
          call mpi_isend(PSRBUF(ROUND)%B,SSIZE,MPIDOUB,TO,TYPE,CCATOR,SHANDLES(ROUND),IE)
        else if(QINTDATA.OR.QDEFDATA) then
          if(QPSIBUF) then
!           if(MYNODP.eq.1) then
!         WRITE(6,*) 'in _SG right before send, NODE 1 is sending ',SENDLEN,' to node ',SENDND
!        endif
           call mpi_isend(PSIBUF(ROUND)%B,SSIZE,MPINT,TO,TYPE,CCATOR,SHANDLES(ROUND),IE)
          elseif(QPSI8BUF) then
           call mpi_isend(PSI8BUF(ROUND)%B,SSIZE,MPINT8,TO,TYPE,CCATOR,SHANDLES(ROUND),IE)
          endif
        endif
     enddo  !loop over rounds
!---------------------------------------------------------------------------------------------------------------
! end of main loop over rounds
!---------------------------------------------------------------------------------------------------------------
     if(allocated(NMAP)) call chmdealloc('asynccomg.src','ASYNCLST_SG','NMAP',NROUND,ci4=NMAP)

 !SHANDLES and data buffers are not deallocated here  since needed by waits
   END SUBROUTINE ASYNCLST_SG
!******************************************************************************************************
!******************************************************************************************************
     SUBROUTINE ASYNCLST_WRG(NROUND,NODARR,RECVLST,RECVHI,RECVMNY,  &
       RHANDLES,PRIBUF,PRI8BUF,PRRBUF,PDATAWID,PRDATA,PIDATA,PXX,PYY,PZZ,padd,plabint)
! general routine, waits for list of posted receives only.  --rjp/mh, June 2011
     use memory
     use parallel
!     use spacdat_mod,only: AROFAR,AROFAR_I4,AROFAR_I8
     implicit none
     integer,intent(in) :: NROUND
     integer,intent(inout),dimension(:) :: RECVLST   !by default, received data (integers); otherwise points to rows of received data
     integer,intent(in),dimension(:) :: RECVHI,RECVMNY,NODARR
     integer*4,allocatable,dimension(:),intent(inout) :: RHANDLES
     type(AROFAR),allocatable,dimension(:),intent(inout),optional :: PRRBUF
     type(AROFAR_I4),allocatable,dimension(:),intent(inout),optional :: PRIBUF
     type(AROFAR_I8),allocatable,dimension(:),intent(inout),optional :: PRI8BUF
     integer,intent(in),optional :: PDATAWID   !passed width of data (# columns); deflts o/w taken fr size of array
     real(chm_real),intent(inout),dimension(:,:),optional,target :: PRDATA
     integer,intent(inout),dimension(:,:),optional,target :: PIDATA  !data arrays using RECVLST as index
     real(chm_real),intent(inout),dimension(:),optional,target :: PXX,PYY,PZZ  !should intent be just out
     logical,intent(in),optional :: PADD  !true if accumulating data, def = .false.
     integer, intent(in), optional :: plabint
     integer :: pass=0
     include 'mpif.h'

! local
     integer(chm_int4) :: NROUNDI4
     integer(chm_int4) :: statigno(MPI_STATUS_SIZE)
     integer*4 to,from,sizes,ie,IHIT
     integer :: NODE2,IBEGIN,IEND,RECVLEN,COUNT,III,ROWIND,ROUND,PX,PY,PZ,PT,COLIND
     logical :: QINTDATA,QRELDATA,QXYZDATA,QDEFDATA,QPRIBUF,QPRI8BUF
     integer :: DATAWID    ! width of data (number of columns)
     logical :: LADD
     integer :: labint, sizetemp
     integer :: I8HIT
     integer(chm_int4) :: recvd_count,ierr,STATUS(MPI_STATUS_SIZE),datatype !for debug
     integer(chm_int4) :: MPIDOUB,MPINT,MPINT8  !for debug
     logical :: QASYNCGDEB !for debug
!  end of declarations
!-------------------------------------------------------------------------------------------------------------------

     if(present(PRDATA)) then
!     WRITE(6,*) 'MYNODP ',MYNODP,' very top sizes PRDATA', size(PRDATA,1),size(PRDATA,2),' plabint ', &
!      plabint
     endif
     pass = pass + 1
     QASYNCGDEB = .true.  !temporary
     NROUNDI4 = NROUND
     if(QASYNCGDEB) then
      MPIDOUB=mpi_double_precision
      MPINT=MPI_INTEGER
      MPINT8=MPI_INTEGER8
     endif
!     if(present(PRDATA)) then
!     WRITE(6,*) 'MYNODP ',MYNODP,' 2nd top sizes PRDATA', size(PRDATA,1),size(PRDATA,2),' plabint ', &
!      plabint
!     endif

!process optional arguments
     LADD = .false.
     if(present(PADD)) LADD = PADD
!  labeling integer
    labint = 0
    if(present(plabint)) labint = plabint

! process optional arguments
! first process data arrays
     QDEFDATA = .false.
     QINTDATA = .false.
     QRELDATA = .false.
     QXYZDATA = .false.
     QPRIBUF = .false.
     QPRI8BUF = .false.
     if(present(PRDATA)) then
      QRELDATA = .true.
      DATAWID = size(PRDATA,1)
     endif
!     if(present(PRDATA)) then
!     WRITE(6,*) 'MYNODP ',MYNODP,' 3rd top sizes PRDATA', size(PRDATA,1),size(PRDATA,2),' plabint ', &
!      plabint
!     endif
     if(present(PIDATA)) then
      if(QRELDATA) then
       WRITE(6,*) '<ASYNCLST_WRG> ERROR: cannot pass real and integer data simultaneously' 
       STOP
      else
       QINTDATA = .true.
       DATAWID = size(PIDATA,1)
      endif
     endif
!process x,y,z
     PX = 0
     if(present(PXX)) PX = 1
     PY = 0
     if(present(PYY)) PY = 1
     PZ = 0
     if(present(PZZ)) PZ = 1
     PT = PX+PY+PZ
     if(PT.GT.0) then
       if((QRELDATA).OR.(QINTDATA)) then
        WRITE(6,*) '<ASYNCLST_WRG> ERROR: passing multiple data arrays simultaneously'
        STOP       
       else if (PT.LT.3) then
        WRITE(6,*) '<ASYNCLST_WRG> ERROR: if passing X,Y,Z, must pass all three arrays'
        STOP
       else 
        QXYZDATA = .true.
        DATAWID = 3
       endif
     endif
     if((.not.QXYZDATA).and.(.not.QINTDATA).and.(.not.QRELDATA)) then
       QDEFDATA = .true. 
       DATAWID = 1
     endif
     if(QRELDATA.OR.QXYZDATA) then
      if(.not.present(PRRBUF)) then
        WRITE(6,*) '<ASYNCLST_WRG> ERROR: must pass real buffer if passing real data'
        STOP
      endif
     else if (QINTDATA.OR.QDEFDATA) then
      if((.not.present(PRIBUF)).and.(.not.present(PRI8BUF))) then
        WRITE(6,*) '<ASYNCLST_WRG> ERROR: must pass either I4 or I8 buffer if passing integer data'
        STOP
      endif
      if(present(PRIBUF)) then
        QPRIBUF = .true.
      endif
      if(present(PRI8BUF)) then
        if(QPRIBUF) then
         WRITE(6,*) '<ASYNCLST_WRG> ERROR: cannot pass both I4 or I8 buffers'
         STOP
        else
         QPRI8BUF = .true.
        endif
      endif
     else
      WRITE(6,*) '<ASYNCLST_WRG> ERROR: bad data type specification'
      STOP
     endif

     if(QINTDATA) then
       if(QPRIBUF) then
        if(kind(PRIBUF(1)%b).ne.kind(PIDATA)) then
           WRITE(6,*) '<ASYNCLST_WRG> type of integer data does not match type of integer(4) buffer'
           STOP
        endif
       elseif(QPRI8BUF) then
         if(kind(PRI8BUF(1)%b).ne.kind(PIDATA)) then
           WRITE(6,*) '<ASYNCLST_WRG> type of integer data does not match type of integer (8) buffer '
           STOP
        endif
       endif
     endif
     if(QDEFDATA) then
       if(QPRIBUF) then
        if(kind(PRIBUF(1)%B).ne.kind(RECVLST)) then
           WRITE(6,*) '<ASYNCLST_WRG> type of integer data does not match type of integer(4) buffer'
           STOP
        endif
       elseif(QPRI8BUF) then
         if(kind(PRI8BUF(1)%B).ne.kind(RECVLST)) then
           WRITE(6,*) '<ASYNCLST_WRG> type of integer data does not match type of integer (8) buffer '
           STOP
        endif
       endif
     endif
!
     if(present(PDATAWID)) then
       if(PDATAWID.gt.DATAWID) then
         WRITE(6,*) '<ASYNCLST_WRG> ERROR: specified data width is greater than actual'
         STOP
       else 
        DATAWID = PDATAWID
       endif
     endif
     if(DATAWID.le.0) then
       WRITE(6,*) '<ASYNCLST_WRG> ERROR: data width cannot be <= 0'
     endif

#if KEY_PARINFNTY==1
     if(qinfinity) return 
#endif
     statigno(:)=MPI_STATUS_IGNORE(:)
     
!
     DO ROUND=1,NROUND
       if(.not.QASYNCGDEB) then
         call mpi_waitany(NROUNDI4,RHANDLES,IHIT,STATIGNO,IE)
       else
         datatype = MPIDOUB
         if(QPRIBUF) datatype = MPINT 
         if(QPRI8BUF) datatype = MPINT8 
         call mpi_waitany(NROUNDI4,RHANDLES,IHIT,STATUS,IE)
         call mpi_get_count(STATUS,datatype,recvd_count,ierr)
       endif
       IF(IHIT.LE.0) then
          call parstoperr('<ASYNCLST_WRG>','RECEIVE ARRAY HANDLE = 0, MISSING HANDLE')
       ENDIF
       NODE2 = NODARR(IHIT)  !maps original round to rankp of node pointing into recvlst (IHIT corresponds to round, since index of rhandles)
!        
       RECVLEN = RECVMNY(NODE2)

       IF(RECVLEN.LE.0) then
          call parstoperr('<ASYNCLST_WRG>','RECEIVE ARRAY LENGTH = 0, NOT ALLOWED')
       ENDIF
      
       if(QASYNCGDEB) then
         if(RECVLEN*DATAWID.NE.recvd_count) then
           WRITE(6,*) 'MPI ARRAY SIZE MISMATCH ON NODE ',MYNODP,' RECEIVED ',recvd_count, &
           ' EXPECTED ',RECVLEN*DATAWID,' rec-fr node ',NODE2,' label ',labint    
           call parstoperr('<ASYNCLST_WRG>','MPI ARRAY SIZE MISMATCH')
         endif
       endif

       IEND = RECVHI(NODE2)
       IF(IEND.GT.SIZE(RECVLST)) THEN
          WRITE(6,*) 'MYNODP ',MYNODP,' IEND ',IEND,' > SIZE RECVLST ',size(RECVLST),' in WRG, NODE2 = ',NODE2
          call parstoperr('<ASYNCLST_WRG>','DATA RECEIVE LIST OVERFLOW')
       ENDIF
       IBEGIN = IEND - RECVLEN + 1
       COUNT = 0
! empty receive buffers
       if(QRELDATA) then
         do III = IBEGIN,IEND
          ROWIND = RECVLST(III)
          COLIND = 1
          do while(COLIND.le.DATAWID)
           COUNT=COUNT+1
           if(.not.LADD) THEN
            PRDATA(COLIND,ROWIND) = PRRBUF(IHIT)%B(COUNT) 
!            WRITE(6,*) 'MYNODP ',MYNODP,' PRDATA at round ',ROUND,' is ',PRDATA(COLIND,ROWIND), &
!       ' COL ',COLIND,' ROW ',ROWIND
           else
            PRDATA(COLIND,ROWIND) = PRDATA(COLIND,ROWIND) + PRRBUF(IHIT)%B(COUNT)
           endif
           COLIND = COLIND + 1
          enddo
         enddo
       else if(QINTDATA) then
         do III = IBEGIN,IEND
          ROWIND = RECVLST(III)
          COLIND = 1
          do while(COLIND.le.DATAWID)
           COUNT=COUNT+1
           if(.not.LADD) THEN
            if(QPRIBUF) then
             PIDATA(COLIND,ROWIND) = PRIBUF(IHIT)%B(COUNT)
            elseif(QPRI8BUF) then
             PIDATA(COLIND,ROWIND) = PRI8BUF(IHIT)%B(COUNT)
            endif
           else
            if(QPRIBUF) then
             PIDATA(COLIND,ROWIND) = PIDATA(COLIND,ROWIND) + PRIBUF(IHIT)%B(COUNT)
            elseif(QPRI8BUF) then
             PIDATA(COLIND,ROWIND) = PIDATA(COLIND,ROWIND) + PRI8BUF(IHIT)%B(COUNT)
            endif 
           endif
           COLIND = COLIND + 1
          enddo
         enddo
       else if(QXYZDATA) then
        sizetemp = size(PRRBUF(IHIT)%b)
        if(.not.LADD) then
         DO III = IBEGIN,IEND
           ROWIND = RECVLST(III)
           COUNT = COUNT + 1
!           if(COUNT.gt.sizetemp) then
!              WRITE(6,*) 'WE HAVE AN OVERFLOW IN WRG, labint is ',labint 
!           endif
           PXX(ROWIND) = PRRBUF(IHIT)%b(COUNT)
           COUNT = COUNT + 1
           PYY(ROWIND) = PRRBUF(IHIT)%b(COUNT)
           COUNT = COUNT + 1
           PZZ(ROWIND) = PRRBUF(IHIT)%b(COUNT)
         ENDDO
        else
          DO III = IBEGIN,IEND
!           IF(III.LT.1) then
!          WRITE(6,*) 'MYNODP ',MYNODP,' INDEX LT 1, NODE2 = ',NODE2,' IHIT ',IHIT,' IND ',III
!          WRITE(6,*) 'MYNODP ',MYNODP,' IEND ',IEND,' RECVLEN ',RECVLEN
!            ENDIF
           ROWIND = RECVLST(III)
           COUNT = COUNT + 1
           PXX(ROWIND) = PXX(ROWIND) + PRRBUF(IHIT)%b(COUNT)
           COUNT = COUNT + 1
           PYY(ROWIND) = PYY(ROWIND) + PRRBUF(IHIT)%b(COUNT)
           COUNT = COUNT + 1
           PZZ(ROWIND) = PZZ(ROWIND) + PRRBUF(IHIT)%b(COUNT)
         ENDDO
        endif !if not ladd
       else  !default receive 
        do III = IBEGIN,IEND
          COUNT = COUNT + 1
          if(.not.LADD) then
           if(QPRIBUF) then
            RECVLST(III) = PRIBUF(IHIT)%B(COUNT)
           elseif(QPRI8BUF) then
            RECVLST(III) = PRI8BUF(IHIT)%B(COUNT)
           endif 
          else
           if(QPRIBUF) then
            RECVLST(III) = RECVLST(III) + PRIBUF(IHIT)%B(COUNT) 
           elseif(QPRI8BUF) then
            RECVLST(III) = RECVLST(III) + PRI8BUF(IHIT)%B(COUNT) 
           endif
          endif
        enddo
       endif  !what type of data in receive buffers
     ENDDO !loop over rounds
!
!     if(present(PRDATA)) then
!     WRITE(6,*)'MYNODP ',MYNODP,'PRDATA(1,1) ',PRDATA(1,1)
!     WRITE(6,*) 'MYNODP ',MYNODP,' PRDATA(1,20) ',PRDATA(1,20)
!     WRITE(6,*) 'MYNODP ',MYNODP,' PRDATA SIZE 1 ',size(PRDATA,1),' SIZE 2 ',size(PRDATA,2)
!     DO III = 1,10
!     WRITE(6,*) 'MYNODP ',MYNODP,' bottom: PRDATA at rowind ',III,' is ',PRDATA(1,III)
!     ENDDO
!     endif
! use NROUNDS for chmdealloc here
     if(QRELDATA.OR.QXYZDATA) then
      deallocate(PRRBUF)
     else if (QINTDATA.OR.QDEFDATA) then
      do ROUND = 1,NROUND
       if(QPRIBUF) then
        deallocate(PRIBUF(ROUND)%b)
       elseif(QPRI8BUF) then
        deallocate(PRI8BUF(ROUND)%b)
       endif
      enddo
      if(QPRIBUF) then
       deallocate(PRIBUF)
      elseif(QPRI8BUF) then
       deallocate(PRI8BUF)
      endif
     endif
     call chmdealloc('asynccomg.src','ASYNCLST_WRG','RHANDLES',NROUND,ci4=RHANDLES)

!
    END SUBROUTINE ASYNCLST_WRG
!******************************************************************************************************
!******************************************************************************************************
    SUBROUTINE ASYNCLST_WSG(NROUND,NODARR,SENDMNY,SHANDLES,PSIBUF,PSI8BUF,PSRBUF)
! general routine, waits for the list of posted sends  -rjp/mh, June 2011
     use memory
     use parallel,only: MYNODP
#if KEY_PARINFNTY
     use parallel,only: qinfinity
#endif
!     use spacdat_mod,only: AROFAR,AROFAR_I4,AROFAR_I8
     implicit none
     integer,intent(in),dimension(:) :: NODARR  !maps round to corresponding part of sendlst (e.g. CPU)
     integer,intent(in),dimension(:) :: SENDMNY   !marker arrays for sendlist
     integer*4,allocatable,dimension(:),intent(inout) :: SHANDLES
     type(AROFAR),allocatable,dimension(:),optional :: PSRBUF
     type(AROFAR_I4),allocatable,dimension(:),optional :: PSIBUF
     type(AROFAR_I8),allocatable,dimension(:),optional :: PSI8BUF
     integer,intent(in) :: NROUND

     include 'mpif.h'

! local
     integer(chm_int4) :: NROUNDI4
     integer(chm_int4) :: statigno(MPI_STATUS_SIZE)
     integer*4 ie,STATUS,IHIT
     integer :: ROUND,SENDLEN,NODE
     logical :: QINTBUF,QRELBUF
!  end of declarations
!--------------------------------------------------------------------
     QINTBUF = .false.
     QRELBUF = .false.
     NROUNDI4 = NROUND
#if KEY_PARINFNTY==1
     if(qinfinity) return 
#endif
     statigno(:)=MPI_STATUS_IGNORE(:)
! process optionals
     if((present(PSIBUF)).or.(present(PSI8BUF))) then
      QINTBUF = .true.
     endif
     if((present(PSIBUF)).and.(present(PSI8BUF))) then
      call parstoperr('<ASYNCLST_WSG>','I4 and I8 buffers cannot be passed simultaneously')
     endif
     if(present(PSRBUF)) then
      if(QINTBUF) then
       call parstoperr('<ASYNCLST_WSG>','integer and real buffers cannot be passed simultaneously')
      else
       QRELBUF = .true.
      endif
     endif
     if((.not.QINTBUF).and.(.not.QRELBUF)) then
       call parstoperr('<ASYNCLST_WSG>','neither integer nor real buffer passed')
     endif
!
!call mpi_waitall(ix,rq,ms,ie) ! this is also not so bad ???
     do ROUND=1,NROUND
        call mpi_waitany(NROUNDI4,SHANDLES,IHIT,STATIGNO,IE)
        IF(IHIT.LE.0) then
          call parstoperr('<ASYNCLST_WSG>','SEND ARRAY HANDLE = 0, MISSING HANDLE')
        ENDIF
 99 continue
     enddo

! use bufsize in calls to chmdealloc here
     if(QINTBUF) then
       if(present(PSIBUF)) then
        deallocate(PSIBUF)
       elseif(present(PSI8BUF)) then
        deallocate(PSI8BUF)
       endif
     else
        deallocate(PSRBUF) 
     endif
     call chmdealloc('asynccomg.src','ASYNCLST_WSG','SHANDLES',NROUND,ci4=SHANDLES)

    END SUBROUTINE ASYNCLST_WSG

!******************************************************************************************************
!******************************************************************************************************

#endif /* (asyncpar)*/
!-----------------------------------------------------------------------------------------------------------------
end module asynccomg
