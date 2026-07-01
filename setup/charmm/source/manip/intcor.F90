module intcor_module

  use chm_kinds
  use chm_types
  implicit none

  !Describes the relationship between four points in space
  ! ic == InternalCoordinate

  ! Create two instances of InternalCoordinate type
  ! This is carried around when "use intcor_module" is defined

  type(internalCoordinate),save :: icr_struct,ics_struct
  ! icr_struct == InternalCoordinate runtime
  ! ics_struct == InternalCoordinate saved
  logical,private :: verbose = .false. !--.true.




  !================================
contains
  !================================

  subroutine initialize_icr_structs()
    !if(verbose)print *,"initialize_icr_structs: ",allocated(icr_struct%jar)

    call deallocateic(icr_struct)
    call deallocateic(ics_struct)
    icr_struct%lenic=0
    icr_struct%intlen=0
    ics_struct%lenic=0
    ics_struct%intlen=0

    return
  end subroutine initialize_icr_structs



  !-----------------------------------------------------------
  !           REINTC_NEW
  !-----------------------------------------------------------
  subroutine reintc_new(newlen,lic)
    !     TODO rename to reintc when fully tested
    !     Resizes the internal coordinate data structure
    !     with the help of switch_ic()


    use exfunc
    integer,intent(in)                      :: newlen
    type(InternalCoordinate),intent(inout)  :: lic
    type(InternalCoordinate)                :: tmpic

    if(verbose)print *,"REINTC_NEW: newlen, oldlen, oldlenic",newlen, lic%intlen,lic%lenic
    if(verbose)print *,"REINTC_NEW: lic%jar allocated:  ",allocated(lic%jar)

    if (allocated(lic%jar)) then
       if (verbose) then
          print *,"REINTC_NEW: Found allocated lic%jar size, intlen, lenic", &
               size(lic%jar),lic%intlen,lic%lenic
          if (allocated(lic%iar)) then
             print *, "REINTC_NEW: size iar ", size(lic%iar)
          else
             print *, "REINTC_NEW: iar not allocated"
          endif
          if (allocated(lic%kar)) then
             print *, "REINTC_NEW: size kar ", size(lic%kar)
          else
             print *, "REINTC_NEW: kar not allocated"
          endif
          if (allocated(lic%lar)) then
             print *, "REINTC_NEW: size lar ", size(lic%lar)
          else
             print *, "REINTC_NEW: lar not allocated"
          endif
       endif

       if(newlen <= 0) then
          if(verbose)print *,"REINTC_NEW: Deallocating since newlen <= 0"
          call deallocateic(lic)
          !          call chmdealloc('intcor.src','reintc_new','lic',size(lic),???=lic)
          return

       elseif(lic%intlen < 0) then
          if(verbose)print *,"REINTC_NEW: Deallocating since intlen < 0"
          call deallocateic(lic)

       else  !do the resize
          !TODO return if they are the same size..

          !Create a temp internal coordinate type to hold the 
          !original data in prior to resizing

          if(verbose)print *,"REINTC_NEW: allocateic tmpic",lic%intlen
          call allocateic(tmpic,lic%intlen)
          if(verbose)print *,"REINTC_NEW: allocateic tmpic done"
          if( .not. allocated(tmpic%iar)) then
             call wrndie(-4,"<intcor.src>reintc_new", &
                  "REINTC_NEW: Allocation of tmpic failed")
          endif

          !Assign the contents of lic to tmpic
          if(verbose)print *, "REINTC_NEW: At switch_ic", &
               tmpic%lenic, tmpic%intlen, lic%lenic, lic%intlen
          if ( lic%intlen > 0) then
             if(verbose)print *,"REINTC_NEW: copying..."
             call switch_ic(tmpic,lic)
          endif
          if(verbose)print *, "REINTC_NEW: After copy lenic is", tmpic%lenic

          !deallocate the original internal coordinate type
          call deallocateic(lic)
          !and then recreate an empty one with the new length
          call allocateic(lic,newlen)

          !Now populate the new with the data in tmpic
          if(verbose)print *, "REINTC_NEW: At 2nd switchic", &
               lic%intlen, tmpic%intlen, lic%lenic, tmpic%lenic
          call switch_ic(lic,tmpic)

          if(verbose)print *, "REINTC_NEW: After 2nd switchic", lic%lenic, lic%intlen
          !Free up tmp array; it has served it purpose
          call deallocateic(tmpic)

       endif

    else !if (allocated(lic%jar))
       if(verbose)print *,"REINTC_NEW: reintc_new: NOT found allocated lic%jar"
       if(newlen <= 0) return

       call allocateic(lic,newlen)
       !First time, hence set lenic to zero
       !         lic%lenic=0

    endif
    return
  end subroutine reintc_new




  !======== Copy the contents of an InternalCoordinate type to a new one
  subroutine switch_ic(new,old)
    !      type(internalcoordinate),intent(out) :: new
    !      type(internalcoordinate),intent(inout)  :: old
    type(internalcoordinate) :: new
    type(internalcoordinate) :: old
    integer copylen, i
    real(chm_real) :: v



    ! if(verbose)print *, "new, old", new%lenic, old%lenic
    ! if(verbose)print *, "intlen: new, old", new%intlen, old%intlen
    !if((new%lenic) > (old%lenic)) then
    !   copylen = old%lenic
    !else
    copylen = old%lenic
    !endif
    ! if(verbose)print *,"switch_ic: copylen",copylen
    ! if(verbose)print *,"size of array",size(new%b1ic)
    if(copylen == 0) return
    new%b1ic(1:copylen)=old%b1ic(1:copylen)
    new%b2ic(1:copylen)=old%b2ic(1:copylen)
    new%t1ic(1:copylen)=old%t1ic(1:copylen)
    new%t2ic(1:copylen)=old%t2ic(1:copylen)
    new%pic(1:copylen) =old%pic(1:copylen)
    new%iar(1:copylen)=old%iar(1:copylen)
    new%jar(1:copylen)=old%jar(1:copylen)
    new%kar(1:copylen)=old%kar(1:copylen)
    new%lar(1:copylen)=old%lar(1:copylen)
    new%tar(1:copylen)=old%tar(1:copylen)
    new%lenic = old%lenic

    return

  end subroutine switch_ic





  subroutine AllocateIC(lic,len)
    !Allocate the sub arrays within the specified instance
    ! lic
    !  |
    !   \-- iar(lenPlusBuffer)
    !  |
    !   \-- jar(lenPlusBuffer)
    !
    !  etc
    !       
    type(internalCoordinate) :: lic
    integer :: len !Length of InternalCoordinate table
    integer :: lenPlusBuffer !The actual size of the array


    if(verbose)print *,"AllocateIC: allocating space for ",len," internalCoordinate types"
    lenPlusBuffer = len + 100
    if(verbose)print *,"giving a total internalCoordinate size of ",lenPlusBuffer

    allocate(lic%iar(lenPlusBuffer))
    allocate(lic%jar(lenPlusBuffer))
    allocate(lic%kar(lenPlusBuffer))
    allocate(lic%lar(lenPlusBuffer))
    allocate(lic%b1ic(lenPlusBuffer))
    allocate(lic%b2ic(lenPlusBuffer))
    allocate(lic%t1ic(lenPlusBuffer))
    allocate(lic%t2ic(lenPlusBuffer))
    allocate(lic%pic(lenPlusBuffer))
    allocate(lic%tar(lenPlusBuffer))

    ! initialize ! hwm220723
    lic%iar=0
    lic%jar=0
    lic%kar=0
    lic%lar=0
    lic%b1ic=0
    lic%b2ic=0
    lic%t1ic=0
    lic%t2ic=0
    lic%pic=0
    lic%tar=.false.

    !       lic%lenic = 0 !Set to zero on create
    lic%intlen = lenPlusBuffer
    lic%lenic = 0

  end subroutine AllocateIC



  !Deallocate the above      
  subroutine DeallocateIC(lic)
    type(internalcoordinate) :: lic

    if(verbose)print *,"Freeing iar type"
    if(allocated(lic%iar))deallocate(lic%iar)
    if(verbose)print *,"Freeing jar type"
    if(allocated(lic%jar))deallocate(lic%jar)
    if(verbose)print *,"Freeing kar type"
    if(allocated(lic%kar))deallocate(lic%kar)
    if(verbose)print *,"Freeing lar type"
    if(allocated(lic%lar))deallocate(lic%lar)
    if(verbose)print *,"Freeing b1ic type"
    if(allocated(lic%b1ic))deallocate(lic%b1ic)
    if(verbose)print *,"Freeing b2ic type"
    if(allocated(lic%b2ic))deallocate(lic%b2ic)
    if(verbose)print *,"Freeing t2ic type"
    if(allocated(lic%t1ic))deallocate(lic%t1ic)
    if(verbose)print *,"Freeing t2ic type"
    if(allocated(lic%t2ic))deallocate(lic%t2ic)
    if(verbose)print *,"Freeing pic type"
    if(allocated(lic%pic))deallocate(lic%pic)
    if(verbose)print *,"Freeing tar type"
    if(allocated(lic%tar))deallocate(lic%tar)
    lic%lenic = 0 !Set to zero on destroy
    lic%intlen = -1

  end subroutine DeallocateIC



  !Test routine to intercept the some of the contents that 
  !would be passed from the HEAP to AUTGENIC() and feed them
  ! into an internalCoordinate type
  !----------------------------------------------
  !       COPY_TO_IC
  !----------------------------------------------
  subroutine copy_to_ic(lic,len, &
       INTB1 ,INTB2 , &
       INTT1 ,INTT2 , &
       INTPIC,INTIAR, &
       INTJAR,INTKAR, &
       INTLAR,INTTAR)
    use chm_kinds
    implicit none

    type(InternalCoordinate) :: lic
    integer len
    real(chm_real),dimension(len) :: INTB1 ,INTB2 ,INTT1 ,INTT2 , &
         INTPIC

    integer,dimension(len) :: INTIAR,INTJAR,INTKAR,INTLAR
    logical,dimension(len) :: INTTAR

    if (len > lic%intlen) then
       call wrndie(-4,"<intcor.src>copy_to_ic", &
            "InternalCoordinate structure too small")
    endif

    lic%iar  = intiar
    lic%jar  = intjar
    lic%kar  = intkar
    lic%lar  = intlar
    lic%b1ic = intb1
    lic%b2ic = intb2
    lic%t1ic = intt1
    lic%t2ic = intt2
    lic%tar  = inttar
    lic%pic  = intpic

    !Set the length of known internalcoordinates
    lic%lenic = len
    return

  end subroutine copy_to_ic

  !----------------------------------------------
  !       COPY_FROM_IC
  !----------------------------------------------
  !Test routine to exact relevant internCoordinate properties from a precreated
  !instance of internCoordinate BACK to the HEAP
  subroutine copy_from_ic(lic,lenmax, &
       INTB1 ,INTB2 , &
       INTT1 ,INTT2 , &
       INTPIC,INTIAR, &
       INTJAR,INTKAR, &
       INTLAR,INTTAR)
    use chm_kinds

    implicit none
    type(InternalCoordinate) :: lic
    integer lenmax
    real(chm_real),dimension(lenmax) :: INTB1 ,INTB2 ,INTT1 ,INTT2 , &
         INTPIC

    integer,dimension(lenmax) :: INTIAR,INTJAR,INTKAR,INTLAR
    logical,dimension(lenmax) :: INTTAR

    if (lic%lenic > lenmax) then
       call wrndie(-4,"<intcor.src>copy_to_ic", &
            "InternalCoordinate structure too big to copy out")
    endif


    intiar = lic%iar
    intjar = lic%jar
    intkar = lic%kar
    intlar = lic%lar
    intb1  = lic%b1ic
    intb2  = lic%b2ic
    intt1  = lic%t1ic
    intt2  = lic%t2ic
    intpic = lic%pic
    inttar = lic%tar

    return

  end subroutine copy_from_ic


  !----------------------------------------------
  !       INTCOR
  !----------------------------------------------
  SUBROUTINE INTCOR(COMLYN,COMLEN)
    !
    !     Process the internal coordinate command.
    !
    use dimens_fcm
    use exfunc
    !
    use coord
    use coordc
    use param
    use psf
    use stream
    use string
    use memory

    implicit none
    ! . Passed variables.
    CHARACTER(len=*) COMLYN
    INTEGER   COMLEN
    ! . Local variables.
    INTEGER,allocatable,dimension(:) ::   ISLCT
    LOGICAL   QCOMP
    !
    QCOMP = INDXA(COMLYN, COMLEN, 'COMP')  >  0
    !
    call chmalloc('intcor.src','intcor','islct',natom,intg=islct)
    !
    IF(QCOMP) THEN
       IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
            ' INTCOR> The comparison coordinates will be used.'
       CALL INTCR2(XCOMP, YCOMP, ZCOMP, WCOMP, X,Y,Z, WMAIN, &
            COMLYN, COMLEN, ISLCT)
    ELSE
       CALL INTCR2(X,Y,Z, WMAIN, XCOMP, YCOMP, ZCOMP, WCOMP, &
            COMLYN, COMLEN, ISLCT)
    ENDIF
    call chmdealloc('intcor.src','intcor','islct',natom,intg=islct)
    !
    RETURN
  END SUBROUTINE INTCOR

  SUBROUTINE INTCR2(X,Y,Z,WMAIN,XCOMP,YCOMP,ZCOMP, &
       WCOMP,COMLYN,COMLEN,ISLCT)
    !
    !   THIS ROUTINE CONTROLS THE GENERATION AND MODIFICATION OF INTERNAL
    !     COORDINATES AND THEIR USE FOR CONSTRUCTING CARTESIAN COORDINATES
    !
    !     By Bernard R. Brooks    1982
    !
    use chm_kinds
    use dimens_fcm
    use select
    use number
    use consta
    use psf
    use bases_fcm
    use clcg_mod,only:rngmodseeds
    use ctitla
    use rndnum
    use stream
    use string
    use parallel
#if KEY_ZEROM==1
    use zdata_mod
#endif
    use memory
    use vector
    use intcor2,only:fillic,writic,purgic,diffic,seed,bildc,intder,autgenic
    use wrgaus_mod
    use param_store, only: set_param

    implicit none
    !
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    real(chm_real) XCOMP(*),YCOMP(*),ZCOMP(*),WCOMP(*)
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN
    INTEGER ISLCT(*)

    integer,allocatable,dimension(:) :: natbon,idone,icount
    integer,allocatable,dimension(:,:) :: iatbon
    integer,allocatable,dimension(:) :: lrngseeds
    logical qpresent
    !
    INTEGER IUNIC,ICARD,ISTART
    INTEGER I,J,K,ISPACE,IPA(5),NIPA
    INTEGER IUNIT,NUNIT,NBEGN,NSTOP,NSKIP
    INTEGER IX,IY,IZ
    INTEGER B1SUM,B2SUM,T1SUM,T2SUM,PSUM
    INTEGER B1REF,B2REF,T1REF,T2REF,PREF
    real(chm_real),allocatable,dimension(:) :: IRVAL,IAVAL,IDVAL
    integer,allocatable,dimension(:) :: IUSED
    real(chm_real) DELTA,KBOND,KANGL,KDIHE,KIMPR
    CHARACTER(len=4) WRD
    LOGICAL LAPPE,LPRES,LOVER,LDIHE,LIMPR
    LOGICAL QFLUCT,ERROR
    INTEGER :: ISEED=314159
    !***CLBIII modification 11/91********************************
    LOGICAL QTRJ
    !***end of CLBIII modification 11/91*************************
    LOGICAL QTHREE,QRTF

    LOGICAL NEWZMT
    !
    !-----------------------------------------------------------------------
    !
    ! (flag for obvious coding errors - no reason to issue a message)
    !      IF(LENIC > 0  .AND.  LENIC > INTLEN)  CALL DIEWRN(-4)
    !      IF(LENICS > 0 .AND. LENICS > INTLENS) CALL DIEWRN(-4)
    !
    WRD=NEXTA4(COMLYN,COMLEN)
    !
    !-----------------------------------------------------------------------
    IF(WRD == 'READ') THEN
       !  READ-INTERNAL-COORDINATE-FILE
       IUNIC=GTRMI(COMLYN,COMLEN,'UNIT',-1)
       LAPPE=INDXA(COMLYN,COMLEN,'APPE') /= 0
       ICARD=1
       IF(INDXA(COMLYN,COMLEN,'FILE') /= 0) ICARD=0
       IF(INDXA(COMLYN,COMLEN,'CARD') /= 0) ICARD=1
       ISTART=1
       IF(ICARD == 1 .AND. IUNIC < 0) IUNIC=ISTRM
       IF(INDXA(COMLYN,COMLEN,'SAVE') /= 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be filled.'
          IF(LAPPE) ISTART=ics_struct%lenic+1
          CALL READIC(ISTART,ICARD,IUNIC,ics_struct)
       ELSE
          IF(LAPPE) ISTART=icr_struct%lenic+1
          CALL READIC(ISTART,ICARD,IUNIC,icr_struct)
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'WRIT') THEN
       !  WRITE-INTERNAL-COORDINATE-FILE
       IUNIC=GTRMI(COMLYN,COMLEN,'UNIT',-1)
       ICARD=1
       IF(INDXA(COMLYN,COMLEN,'FILE') /= 0) ICARD=0
       IF(INDXA(COMLYN,COMLEN,'CARD') /= 0) ICARD=1
       IF(INDXA(COMLYN,COMLEN,'RESI') /= 0) ICARD=2
       IF(INDXA(COMLYN,COMLEN,'RTF') > 0) ICARD=3
       CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
       IF(INDXA(COMLYN,COMLEN,'SAVE') /= 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be used.'
               CALL WRITIC(1,ics_struct%lenic,0,ICARD,IUNIC, &
               ics_struct%B1ic,ics_struct%B2ic, &
               ics_struct%T1ic,ics_struct%T2ic, &
               ics_struct%PIC, ics_struct%IAR, &
               ics_struct%JAR, ics_struct%KAR, &
               ics_struct%LAR, ics_struct%TAR)    
       ELSE
               CALL WRITIC(1,icr_struct%lenic,0,ICARD,IUNIC, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR)
       ENDIF
       IF(IUNIC /= OUTU) CALL VCLOSE(IUNIC,'KEEP',ERROR)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'PRIN') THEN
       !  PRINT-INTERNAL-COORDINATE-FILE
       IUNIC=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
       ICARD=1
       IF(INDXA(COMLYN,COMLEN,'RTF') > 0) ICARD=3
       !
       IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be used.'
          CALL WRITIC(1,ics_struct%lenic,1,ICARD,IUNIC, &
               ics_struct%B1ic,ics_struct%B2ic, &
               ics_struct%T1ic,ics_struct%T2ic, &
               ics_struct%PIC, ics_struct%IAR, &
               ics_struct%JAR, ics_struct%KAR, &
               ics_struct%LAR, ics_struct%TAR)
       ELSE
          CALL WRITIC(1,icr_struct%lenic,1,ICARD,IUNIC, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR)
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'FILL') THEN
       !  PROCESS-FILL-COMMAND
       LAPPE=INDXA(COMLYN,COMLEN,'APPE') /= 0
       LPRES=INDXA(COMLYN,COMLEN,'PRES') /= 0
       IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be filled.'

          CALL FILLIC(ics_struct%lenic,LAPPE,LPRES,X,Y,Z, &
               ics_struct%B1ic,ics_struct%B2ic, &
               ics_struct%T1ic,ics_struct%T2ic, &
               ics_struct%PIC, ics_struct%IAR, &
               ics_struct%JAR, ics_struct%KAR, &
               ics_struct%LAR, ics_struct%TAR)
       ELSE
          CALL FILLIC(icr_struct%lenic,LAPPE,LPRES,X,Y,Z, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR)
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'DYNA') THEN
       !  PROCESS-DYNAMICS-COMMAND
       QFLUCT=.FALSE.
       IF(INDXA(COMLYN,COMLEN,'AVER') <= 0) &
            QFLUCT=(INDXA(COMLYN,COMLEN,'FLUC') > 0)
       !***CLBIII modification 11/91******************************
       QTRJ=(INDXA(COMLYN,COMLEN,'TRAJ') > 0)
       IUNIC=GTRMI(COMLYN,COMLEN,'UNIT',71)
       !***end of CLBIII modification 11/91***********************
       IUNIT=GTRMI(COMLYN,COMLEN,'FIRS',51)
       NUNIT=GTRMI(COMLYN,COMLEN,'NUNI',1)
       NBEGN=GTRMI(COMLYN,COMLEN,'BEGI',0)
       NSTOP=GTRMI(COMLYN,COMLEN,'STOP',0)
       NSKIP=GTRMI(COMLYN,COMLEN,'NSKI',1)
       !
       CALL ICDYNA(qfluct, &
            icr_struct%lenic, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            icr_struct%PIC, icr_struct%IAR, &
            icr_struct%JAR, icr_struct%KAR, &
            icr_struct%LAR, icr_struct%TAR, &
            NATOM,NUNIT,IUNIT, &
            NSKIP,NBEGN,NSTOP,QTRJ,IUNIC)

       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'EDIT') THEN
       !  PROCESS-EDIT-COMMAND
       IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be edited.'

          IF(200+ics_struct%lenic > ics_struct%intlen) THEN
             I=ics_struct%lenic+200  ! allocate extra space for new IC entries
             CALL REINTC_NEW(I,ics_struct)
          endif
          CALL EDITIC(ics_struct%lenic,ics_struct%intlen,&
               ISTRM,ATYPE,IBASE, &
               SEGID,RESID,NICTOT,NSEGT,RES,NATOMT,ISLCT, &
               ics_struct%B1ic,ics_struct%B2ic, &
               ics_struct%T1ic,ics_struct%T2ic, &
               ics_struct%PIC, ics_struct%IAR, &
               ics_struct%JAR, ics_struct%KAR, &
               ics_struct%LAR, ics_struct%TAR)
       ELSE
          IF(200+icr_struct%lenic > icr_struct%intlen) THEN
             I=icr_struct%intlen+200  ! allocate extra space for new IC entries
             CALL REINTC_NEW(I,icr_struct)
          endif
          CALL EDITIC(icr_struct%lenic,icr_struct%intlen,&
               ISTRM,ATYPE,IBASE, &
               SEGID,RESID,NICTOT,NSEGT,RES,NATOMT,ISLCT, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR)
       endif
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'BILD'.OR.WRD == 'BUIL') THEN
       !  process-build-command
#if KEY_ZEROM==1
       IF(INDXA(COMLYN,COMLEN,'REVE') > 0) QICBREV = .TRUE.
#endif 
       IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be used.'
          CALL BILDC(1,ics_struct%lenic,X,Y,Z, &
               ics_struct%B1ic,ics_struct%B2ic, &
               ics_struct%T1ic,ics_struct%T2ic, &
               ics_struct%PIC, ics_struct%IAR, &
               ics_struct%JAR, ics_struct%KAR, &
               ics_struct%LAR, ics_struct%TAR, &
               NATOM)
       ELSE
          CALL BILDC(1,icr_struct%lenic,X,Y,Z, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR, &
               NATOM)
       ENDIF
#if KEY_ZEROM==1
       QICBREV = .FALSE.
#endif 
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'SEED') THEN
       !  PROCESS-SEED-COMMAND
       IPA(1)=0
       IPA(2)=0
       IPA(3)=0
       CALL NXTATM(IPA,NIPA,3,COMLYN,COMLEN,ISLCT, &
            SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
       !
       I=IPA(1)
       J=IPA(2)
       K=IPA(3)
       IF(I <= 0.OR.J <= 0.OR.K <= 0) THEN
          CALL WRNDIE(0,'<INTCOR>','ATOM OF SEED DOES NOT EXIST.')
       ELSE

          CALL SEED(I,J,K,X,Y,Z,icr_struct%lenic, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR)
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'DELE') THEN
       IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be modified.'
          CALL KEEPIC(.FALSE.,ISLCT,X,Y,Z,WMAIN, &
               ics_struct%lenic, &
               ics_struct%B1ic,ics_struct%B2ic, &
               ics_struct%T1ic,ics_struct%T2ic, &
               ics_struct%PIC, ics_struct%IAR, &
               ics_struct%JAR, ics_struct%KAR, &
               ics_struct%LAR, ics_struct%TAR)
          IF(ics_struct%lenic+500 > ics_struct%intlen) THEN
             I=ics_struct%lenic+100  ! free excess space
             CALL REINTC_NEW(I,ics_struct)
          endif
       ELSE
          CALL KEEPIC(.FALSE.,ISLCT,X,Y,Z,WMAIN,&
               icr_struct%lenic, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR)
          IF(icr_struct%lenic+500 > icr_struct%intlen) THEN
             I=icr_struct%lenic+100  ! free excess space
             CALL REINTC_NEW(I,icr_struct)
          endif
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'KEEP') THEN
       IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be modified.'
          CALL KEEPIC(.TRUE.,ISLCT,X,Y,Z,WMAIN, &
               ics_struct%lenic, &
               ics_struct%B1ic,ics_struct%B2ic, &
               ics_struct%T1ic,ics_struct%T2ic, &
               ics_struct%PIC, ics_struct%IAR, &
               ics_struct%JAR, ics_struct%KAR, &
               ics_struct%LAR, ics_struct%TAR)
          IF(ics_struct%lenic+500 > ics_struct%intlen) THEN
             I=ics_struct%lenic+100  ! free excess space
             CALL REINTC_NEW(I,ics_struct)
          endif
       ELSE
          CALL KEEPIC(.TRUE.,ISLCT,X,Y,Z,WMAIN, &
               icr_struct%lenic, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR)
          IF(icr_struct%lenic+500 > icr_struct%intlen) THEN
             I=icr_struct%lenic+100  ! free excess space
             CALL REINTC_NEW(I,icr_struct)
          endif
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'PURG') THEN
       !  PROCESS-PURGE-COMMAND
       IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be modified.'

          CALL PURGIC(ics_struct%lenic, &
               ics_struct%B1ic,ics_struct%B2ic, &
               ics_struct%T1ic,ics_struct%T2ic, &
               ics_struct%PIC, ics_struct%IAR, &
               ics_struct%JAR, ics_struct%KAR, &
               ics_struct%LAR, ics_struct%TAR, &
               .FALSE.,.FALSE.,.FALSE.,.FALSE.,.TRUE., &
               NATOM,IMOVE,IAC)
          IF(ics_struct%lenic+500 > ics_struct%intlen) THEN
             I=ics_struct%lenic+100  ! free excess space
             CALL REINTC_NEW(I,ics_struct)
          endif
       ELSE
          CALL PURGIC(icr_struct%lenic, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR, &
               .FALSE.,.FALSE.,.FALSE.,.FALSE.,.TRUE., &
               NATOM,IMOVE,IAC)
          IF(icr_struct%lenic+500 > icr_struct%intlen) THEN
             I=icr_struct%lenic+100  ! free excess space
             CALL REINTC_NEW(I,icr_struct)
          endif
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'PARA') THEN
       !  PROCESS-PARAMETER-COMMAND
       LAPPE=INDXA(COMLYN,COMLEN,'ALL') /= 0
       LDIHE=INDXA(COMLYN,COMLEN,'DIHE') /= 0
       LIMPR=INDXA(COMLYN,COMLEN,'IMPR') /= 0
       CALL PURGIC(icr_struct%lenic, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            icr_struct%PIC, icr_struct%IAR, &
            icr_struct%JAR, icr_struct%KAR, &
            icr_struct%LAR, icr_struct%TAR, &
            .TRUE.,LAPPE,LDIHE,LIMPR,.FALSE., &
            NATOM,IMOVE,IAC)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'DIFF') THEN
       !  PROCESS-DIFFERENCE-COMMAND
       LAPPE=INDXA(COMLYN,COMLEN,'APPE') /= 0
       DELTA=GTRMF(COMLYN,COMLEN,'SCAL',ONE)
       CALL DIFFIC(DELTA,X,Y,Z,XCOMP,YCOMP,ZCOMP,&
            icr_struct%lenic,LAPPE, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            icr_struct%PIC, icr_struct%IAR, &
            icr_struct%JAR, icr_struct%KAR, &
            icr_struct%LAR, icr_struct%TAR)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'DERI') THEN
       !  PROCESS-DERIVATIVE-COMMAND
       LAPPE=INDXA(COMLYN,COMLEN,'APPE') /= 0
       DELTA=GTRMF(COMLYN,COMLEN,'DELT',ONE)
       CALL INTDER(DELTA,X,Y,Z,XCOMP,YCOMP,ZCOMP,&
            icr_struct%lenic,LAPPE, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            icr_struct%PIC, icr_struct%IAR, &
            icr_struct%JAR, icr_struct%KAR, &
            icr_struct%LAR, icr_struct%TAR)

       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'SCAL') THEN
       !  PROCESS-SCALE-COMMAND
       KBOND=GTRMF(COMLYN,COMLEN,'BOND',ONE)
       KANGL=GTRMF(COMLYN,COMLEN,'ANGL',ONE)
       KDIHE=GTRMF(COMLYN,COMLEN,'DIHE',ONE)
       KIMPR=GTRMF(COMLYN,COMLEN,'IMPR',ONE)
       IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be scaled.'
          CALL SCALR8(ics_struct%B1ic,ics_struct%lenic,KBOND)
          CALL SCALR8(ics_struct%B2ic,ics_struct%lenic,KBOND)
          CALL SCALR8(ics_struct%T1ic,ics_struct%lenic,KANGL)
          CALL SCALR8(ics_struct%T2ic,ics_struct%lenic,KANGL)
          CALL SCALICD(ics_struct%PIC,ics_struct%TAR, &
               ics_struct%lenic,KDIHE,.FALSE.)
          CALL SCALICD(ics_struct%PIC,ics_struct%TAR, &
               ics_struct%lenic,KIMPR,.TRUE.)
       ELSE
          CALL SCALR8(icr_struct%B1ic,icr_struct%lenic,KBOND)
          CALL SCALR8(icr_struct%B2ic,icr_struct%lenic,KBOND)
          CALL SCALR8(icr_struct%T1ic,icr_struct%lenic,KANGL)
          CALL SCALR8(icr_struct%T2ic,icr_struct%lenic,KANGL)
          CALL SCALICD(icr_struct%PIC,icr_struct%TAR, &
               icr_struct%lenic,KDIHE,.FALSE.)
          CALL SCALICD(icr_struct%PIC,icr_struct%TAR, &
               icr_struct%lenic,KIMPR,.TRUE.)
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'ADD' .OR. WRD == 'SUBT') THEN
       !  PROCESS-ADD/SUBTRACT-COMMAND

       IF(icr_struct%lenic == ics_struct%lenic) THEN
          DELTA=ONE
          IF(WRD == 'SUBT') DELTA=MINONE
          IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
             IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
                  ' INTCOR> The saved IC table will be modified.'

             CALL INTCPY(ics_struct%lenic, ics_struct%B1ic,ics_struct%B2ic, &
                  ics_struct%T1ic,ics_struct%T2ic, &
                  ics_struct%PIC, ics_struct%IAR, &
                  ics_struct%JAR, ics_struct%KAR, &
                  ics_struct%LAR, ics_struct%TAR, &
                  icr_struct%lenic, &
                  icr_struct%B1ic,icr_struct%B2ic, &
                  icr_struct%T1ic,icr_struct%T2ic, &
                  icr_struct%PIC, icr_struct%IAR, &
                  icr_struct%JAR, icr_struct%KAR, &
                  icr_struct%LAR, icr_struct%TAR, &
                  .FALSE.,.FALSE.,.TRUE.,DELTA)
          ELSE
             CALL INTCPY(icr_struct%lenic, icr_struct%B1ic,icr_struct%B2ic, &
                  icr_struct%T1ic,icr_struct%T2ic, &
                  icr_struct%PIC, icr_struct%IAR, &
                  icr_struct%JAR, icr_struct%KAR, &
                  icr_struct%LAR, icr_struct%TAR, &
                  ics_struct%lenic, &
                  ics_struct%B1ic,ics_struct%B2ic, &
                  ics_struct%T1ic,ics_struct%T2ic, &
                  ics_struct%PIC, ics_struct%IAR, &
                  ics_struct%JAR, ics_struct%KAR, &
                  ics_struct%LAR, ics_struct%TAR, &
                  .FALSE.,.FALSE.,.TRUE.,DELTA)
          ENDIF
       ELSE
          CALL WRNDIE(0,'<INTCOR>', &
               'Length of the IC table does not match the saved length')
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'REST') THEN
       !  PROCESS-RESTORE-COMMAND
       LPRES=INDXA(COMLYN,COMLEN,'PRES') /= 0
       LOVER=INDXA(COMLYN,COMLEN,'OVER') /= 0

       IF(ics_struct%lenic > 0) THEN
          I=ics_struct%lenic
          IF(LPRES .OR. LOVER) I=I+icr_struct%lenic
          IF(icr_struct%intlen < I) CALL REINTC_NEW(I,icr_struct)
          CALL INTCPY(icr_struct%lenic,&
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR, &
               ics_struct%lenic, &
               ics_struct%B1ic,ics_struct%B2ic, &
               ics_struct%T1ic,ics_struct%T2ic, &
               ics_struct%PIC, ics_struct%IAR, &
               ics_struct%JAR, ics_struct%KAR, &
               ics_struct%LAR, ics_struct%TAR, &
               LPRES,LOVER,.FALSE.,zero)
          IF(icr_struct%lenic+500 > icr_struct%intlen) THEN
             I=ics_struct%lenic+100
             CALL REINTC_NEW(I,icr_struct)
          endif
       ELSE
          CALL WRNDIE(0,'<INTCOR>','NO IC HAS BEEN SAVED.')
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'SAVE') THEN
       !  PROCESS-SAVE-COMMAND
       LPRES=INDXA(COMLYN,COMLEN,'PRES') /= 0
       LOVER=INDXA(COMLYN,COMLEN,'OVER') /= 0

       IF(icr_struct%lenic > 0) THEN
          I=icr_struct%lenic
          IF(LPRES .OR. LOVER) I=I+ics_struct%lenic
          IF(ics_struct%intlen < I) CALL REINTC_NEW(I,ics_struct)
          CALL INTCPY(ics_struct%lenic, &
               ics_struct%B1ic,ics_struct%B2ic, &
               ics_struct%T1ic,ics_struct%T2ic, &
               ics_struct%PIC, ics_struct%IAR, &
               ics_struct%JAR, ics_struct%KAR, &
               ics_struct%LAR, ics_struct%TAR, &
               icr_struct%lenic, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR, &
               LPRES,LOVER,.FALSE.,zero)
          IF(ics_struct%lenic+500 > ics_struct%intlen) THEN
             I=ics_struct%lenic+100
             CALL REINTC_NEW(I,ics_struct)
          endif
       ELSE
          CALL WRNDIE(0,'<INTCOR>','THERE IS NO IC TO SAVE.')
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'RAND') THEN
       !  PROCESS-RANDOM-COMMAND
       !
       call chmalloc('intcor.src','INTCR2','lrngseeds',Nrand,intg=lrngseeds)
       lrngseeds(1:nrand)=rngseeds(1:nrand)
       if(qoldrandom.or.qbrokenclcg)lrngseeds(1:nrand)=iseed
       call gtrmim(nrand,comlyn,comlen,'ISEE',lrngseeds,rngseeds,qpresent)
       call rngmodseeds(qpresent,iseed)
       call chmdealloc('intcor.src','INTCR2','lrngseeds',Nrand,intg=lrngseeds)
       !
       !      ISEED=GTRMI(COMLYN,COMLEN,'ISEE',ISEED)
       !     if (.not.qoldrng) then     !yw 05-Aug-2008
       !        CALL CLCGINIT(ISEED)
       !        ISEED=1
       !     endif

       IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
          IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
               ' INTCOR> The saved IC table will be randomized.'

          CALL INTRND(ics_struct%lenic,ics_struct%PIC,ISEED)
       ELSE
          CALL INTRND(icr_struct%lenic,icr_struct%PIC,ISEED)
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'GAUS') THEN
       !  PROCESS-GAUSSIAN-COMMAND
       IUNIC=GTRMI(COMLYN,COMLEN,'UNIT',-1)
       NEWZMT=(INDXA(COMLYN,COMLEN,'ZMIX') > 0)
       IPA(1)=0
       IPA(2)=0
       IPA(3)=0
       CALL NXTATM(IPA,NIPA,3,COMLYN,COMLEN,ISLCT, &
            SEGID,RESID,ATYPE,IBASE,NICTOT,NSEGT,RES,NATOMT)
       !
       I=IPA(1)
       J=IPA(2)
       K=IPA(3)
       IF(I <= 0.OR.J <= 0.OR.K <= 0) THEN
          CALL WRNDIE(0,'<INTCOR>', &
               'ATOM OF GAUSSIAN SEED DOES NOT EXIST.')
       ELSE
          !
          call chmalloc('intcor.src','intcr2','irval',natom,crl=irval)
          call chmalloc('intcor.src','intcr2','iaval',natom,crl=iaval)
          call chmalloc('intcor.src','intcr2','idval',natom,crl=idval)
          call chmalloc('intcor.src','intcr2','iused',natom,intg=iused)

          CALL WRGAUS(IUNIC,I,J,K,X,Y,Z,&
               icr_struct%lenic, &
               icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR, &
               IRVAL,IAVAL,IDVAL,IUSED,NEWZMT)
          call chmdealloc('intcor.src','intcr2','irval',natom,crl=irval)
          call chmdealloc('intcor.src','intcr2','iaval',natom,crl=iaval)
          call chmdealloc('intcor.src','intcr2','idval',natom,crl=idval)
          call chmdealloc('intcor.src','intcr2','iused',natom,intg=iused)

       ENDIF
       IF(IUNIC /= OUTU) CALL VCLOSE(IUNIC,'KEEP',ERROR)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'PUCK') THEN
       !  PROCESS-PUCKER-COMMAND
       CALL NXTATM(IPA,NIPA,5,COMLYN,COMLEN,ISLCT, &
            SEGID,RESID,ATYPE,IBASE,NICTOT,NSEGT,RES,NATOMT)
       IF(NIPA /= 5) THEN
          CALL WRNDIE(0,'<INTCOR>', &
               'ATOM OF PUCKER RING DOES NOT EXIST.')
       ELSE
          !
          KBOND=GTRMF(COMLYN,COMLEN,'AMPL',ZERO)
          KANGL=GTRMF(COMLYN,COMLEN,'ANGL',ZERO)

          CALL ICPUCK(IPA,KBOND,KANGL,&
               icr_struct%lenic,icr_struct%intlen, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC,icr_struct%IAR, &
               icr_struct%JAR,icr_struct%KAR, &
               icr_struct%LAR,icr_struct%TAR)
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'GENE') THEN
       !  PROCESS-GENERATE-COMMAND
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       !!
       I=NSELCT(NATOM,ISLCT)

       IF(I+icr_struct%lenic > icr_struct%intlen) THEN
          I=I+icr_struct%lenic+100 ! allocate extra space to prevent frequent resizing 
          CALL REINTC_NEW(I,icr_struct)
       ENDIF

       QTHREE=(INDXA(COMLYN,COMLEN,'THRE') > 0)
       QRTF=(INDXA(COMLYN,COMLEN,'RTF') > 0)
       

       call chmalloc('intcor.src','intcr2','natbon',natom,intg=natbon)
       call chmalloc('intcor.src','intcr2','idone ',natom,intg=idone)
       call chmalloc('intcor.src','intcr2','icount',natom,intg=icount)
       call chmalloc('intcor.src','intcr2','iatbon',iatbmx,natom,intg=iatbon)

       CALL AUTGENIC(NATOM,NBOND,NATBON,IATBON, &
            ISLCT,IDONE,ICOUNT, &
            AMASS,IB,JB,QTHREE,QRTF,icr_struct%intlen,&
            icr_struct%lenic, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            icr_struct%PIC, icr_struct%IAR, &
            icr_struct%JAR, icr_struct%KAR, &
            icr_struct%LAR, icr_struct%TAR)


       call chmdealloc('intcor.src','intcr2','natbon',natom,intg=natbon)
       call chmdealloc('intcor.src','intcr2','idone ',natom,intg=idone)
       call chmdealloc('intcor.src','intcr2','icount',natom,intg=icount)
       call chmdealloc('intcor.src','intcr2','iatbon',iatbmx,natom,intg=iatbon)

       !-----------------------------------------------------------------------
    ELSE
       CALL WRNDIE(0,'<INTCOR>','Unrecognized IC command or option')
       !-----------------------------------------------------------------------
    ENDIF
    !

    CALL set_param('NIC',icr_struct%LENIC)

    RETURN
  END SUBROUTINE INTCR2

  SUBROUTINE EDITIC(LENIC,INTLEN,IUNIC,ATYPE,IBASE, &
       SEGID,RESID,NICTOT,NSEG,RES,NATOM,ISLCT, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    !     EDIT INTERNAL COORDINATES
    !
    !     By Bernard R. Brooks    1982
    !
    use chm_kinds
    use dimens_fcm
    use comand
    use stream
    use string
    use intcor2,only:writic
    use select

    implicit none
    !C  use number
    !
    INTEGER LENIC,INTLEN,IUNIC
    character(len=*) SEGID(*),RESID(*),ATYPE(*),RES(*)
    INTEGER IBASE(*),NICTOT(*)
    INTEGER NSEG,NATOM,ISLCT(*)
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    real(chm_real) ACOM
    INTEGER NOLD,NNEW,I,J,IIC,IVL,K,L
    INTEGER QAT(4),NQAT
    INTEGER II,INC,ISTART,MARK
    !
    character(len=4) WRD
    LOGICAL T
    LOGICAL FOUND,KILL,EOF,OK,LPOS,LNEG,DONE
    LOGICAL QADD
    !
    !
    EOF=.FALSE.
    NOLD=LENIC
    NNEW=0
    KILL=.FALSE.
    DONE=.FALSE.
    !
    DO WHILE(.NOT.DONE)
       CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIC,EOF,.TRUE., &
            .TRUE.,'EDITIC> ')
       DO I=1,4
          QAT(I)=0
       ENDDO
       WRD=NEXTA4(COMLYN,COMLEN)
       !-----------------------------------------------------------------------
       IF(WRD == '    ') THEN
          CONTINUE
          !-----------------------------------------------------------------------
       ELSE IF(WRD == 'DIST' .OR. WRD == 'BOND') THEN
          ! process-distance-edit
          CALL NXTATM(QAT,NQAT,2,COMLYN,COMLEN,ISLCT, &
               SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
          I=QAT(1)
          J=QAT(2)
          QADD=(INDXA(COMLYN,COMLEN,'ADD') > 0)
          ACOM=NEXTF(COMLYN,COMLEN)
          !
          IF(I <= 0.OR.J <= 0) THEN
             !              atom-doesnt-exist
             IF(WRNLEV >= 2) WRITE(OUTU,925)
             CALL DIEWRN(0)
          ELSE
             FOUND=.FALSE.
             DO IIC=1,LENIC
                IVL=0
                IF(I == IAR(IIC).OR.J == IAR(IIC)) IVL=IVL+1
                IF(I == JAR(IIC).OR.J == JAR(IIC)) IVL=IVL+2
                IF(I == KAR(IIC).OR.J == KAR(IIC)) IVL=IVL+4
                IF(I == LAR(IIC).OR.J == LAR(IIC)) IVL=IVL+8
                IF((IVL == 5.AND.TAR(IIC)).OR. &
                     (IVL == 3.AND..NOT.TAR(IIC))) THEN
                   IF(QADD) THEN
                      B2IC(IIC)=B2IC(IIC)+ACOM
                   ELSE
                      B2IC(IIC)=ACOM
                   ENDIF
                   FOUND=.TRUE.
                   IF(PRNLEV >= 2) WRITE(OUTU,115) IIC
115                FORMAT(15X,'FOUND IN IC',I5,' ON LEFT  SIDE')
                ENDIF
                IF(IVL == 12) THEN
                   IF(QADD) THEN
                      B1IC(IIC)=B1IC(IIC)+ACOM
                   ELSE
                      B1IC(IIC)=ACOM
                   ENDIF
                   FOUND=.TRUE.
                   IF(PRNLEV >= 2) WRITE(OUTU,125) IIC
125                FORMAT(15X,'FOUND IN IC',I5,' ON RIGHT SIDE')
                ENDIF
             ENDDO
             !
             IF(.NOT.FOUND) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,175)
175             FORMAT(/10X,'ERROR IN EDITIC. INTERNAL DISTANCE ', &
                     'COORDINATE CANT BE FOUND. ADDING ****'/)
                IF(QADD) CALL WRNDIE(-1,'<EDITIC>', &
                     'Existing bond not found with ADD option')
                !                 add-new-ic-element
                LENIC=LENIC+1
                NNEW=NNEW+1
                IF(PRNLEV >= 2) WRITE(OUTU,375) LENIC
                IAR(LENIC)=I
                JAR(LENIC)=J
                KAR(LENIC)=-99
                LAR(LENIC)=-99
                TAR(LENIC)=.FALSE.
                B1IC(LENIC)=0.0
                B2IC(LENIC)=ACOM
                T1IC(LENIC)=0.0
                T2IC(LENIC)=0.0
                PIC(LENIC)=0.0
             ENDIF
          ENDIF
          !-----------------------------------------------------------------------
       ELSE IF(WRD == 'ANGL' .OR. WRD == 'THET') THEN
          ! PROCESS-ANGLE-EDIT
          CALL NXTATM(QAT,NQAT,3,COMLYN,COMLEN,ISLCT, &
               SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
          I=QAT(1)
          J=QAT(2)
          K=QAT(3)
          QADD=(INDXA(COMLYN,COMLEN,'ADD') > 0)
          ACOM=NEXTF(COMLYN,COMLEN)
          !
          IF(I <= 0.OR.J <= 0.OR.K <= 0) THEN
             ! ATOM-DOESNT-EXIST
             IF(WRNLEV >= 2) WRITE(OUTU,925)
             CALL DIEWRN(0)
          ELSE
             FOUND=.FALSE.
             DO IIC=1,LENIC
                IVL=0
                IF(I == IAR(IIC).OR.K == IAR(IIC)) IVL=IVL+1
                IF(I == JAR(IIC).OR.K == JAR(IIC)) IVL=IVL+2
                IF(I == KAR(IIC).OR.K == KAR(IIC)) IVL=IVL+4
                IF(I == LAR(IIC).OR.K == LAR(IIC)) IVL=IVL+8
                IF(.NOT.((IVL /= 3.OR.J /= KAR(IIC).OR..NOT.TAR(IIC)) &
                     .AND.(IVL /= 5.OR.J /= JAR(IIC).OR.TAR(IIC)))) THEN
                   IF(QADD) THEN
                      T2IC(IIC)=T2IC(IIC)+ACOM
                   ELSE
                      T2IC(IIC)=ACOM
                   ENDIF
                   FOUND=.TRUE.
                   IF(PRNLEV >= 2) WRITE(OUTU,215) IIC
215                FORMAT(15X,'FOUND IN IC',I5,' ON LEFT  SIDE')
                ENDIF
                IF(IVL == 10.AND.J == KAR(IIC)) THEN
                   IF(QADD) THEN
                      T1IC(IIC)=T1IC(IIC)+ACOM
                   ELSE
                      T1IC(IIC)=ACOM
                   ENDIF
                   FOUND=.TRUE.
                   IF(PRNLEV >= 2) WRITE(OUTU,225) IIC
225                FORMAT(15X,'FOUND IN IC',I5,' ON RIGHT SIDE')
                ENDIF
             ENDDO
             !
             IF(.NOT.FOUND) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,275)
275             FORMAT(/10X,'ERROR IN EDITIC. INTERNAL ANGLE ', &
                     'COORDINATE CANT BE FOUND. ADDING ****'/)
                IF(QADD) CALL WRNDIE(-1,'<EDITIC>', &
                     'Existing angle not found with ADD option')
                !                 add-new-ic-element
                LENIC=LENIC+1
                NNEW=NNEW+1
                IF(PRNLEV >= 2) WRITE(OUTU,375) LENIC
                IAR(LENIC)=I
                JAR(LENIC)=K
                KAR(LENIC)=J
                LAR(LENIC)=-99
                TAR(LENIC)=.TRUE.
                B1IC(LENIC)=0.0
                B2IC(LENIC)=0.0
                T1IC(LENIC)=0.0
                T2IC(LENIC)=ACOM
                PIC(LENIC)=0.0
             ENDIF
          ENDIF
          !-----------------------------------------------------------------------
       ELSE IF(WRD == 'DIHE' .OR. WRD == 'PHI ') THEN
          ! PROCESS-DIHEDRAL-EDIT
          MARK=-999
          IIC=GTRMI(COMLYN,COMLEN,'ICNU',MARK)
          IF(IIC /= MARK) THEN
             IF(IIC <= 0 .OR. IIC > LENIC) THEN
                CALL WRNDIE(-3,'<EDITIC>','Invalid ICNUmber specified')
                RETURN
             ENDIF
             I=IAR(IIC)
             J=JAR(IIC)
             K=KAR(IIC)
             L=LAR(IIC)
             QADD=(INDXA(COMLYN,COMLEN,'ADD') > 0)
             ACOM=NEXTF(COMLYN,COMLEN)
             !
             IF(I <= 0.OR.J <= 0.OR.K <= 0.OR.L <= 0) THEN
                !             one or more atoms do not exist
                IF(WRNLEV >= 2) WRITE(OUTU,925)
                CALL DIEWRN(0)
             ELSE
                IF(PRNLEV >= 2) WRITE(OUTU,325) IIC
                IF(QADD) THEN
                   PIC(IIC)=PIC(IIC)+ACOM
                ELSE
                   PIC(IIC)=ACOM
                ENDIF
                IF(PIC(IIC) >  180.0) PIC(IIC)=PIC(IIC)-360.0
                IF(PIC(IIC) < -180.0) PIC(IIC)=PIC(IIC)+360.0
             ENDIF
          ELSE
             CALL NXTATM(QAT,NQAT,4,COMLYN,COMLEN,ISLCT, &
                  SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
             I=QAT(1)
             J=QAT(2)
             K=QAT(3)
             T=(K < 0)
             IF(T) K=-K
             L=QAT(4)
             QADD=(INDXA(COMLYN,COMLEN,'ADD') > 0)
             ACOM=NEXTF(COMLYN,COMLEN)
             !
             IF(I <= 0.OR.J <= 0.OR.K <= 0.OR.L <= 0) THEN
                ! ATOM-DOESNT-EXIST
                IF(WRNLEV >= 2) WRITE(OUTU,925)
                CALL DIEWRN(0)
             ELSE
                !
                FOUND=.FALSE.
                DO IIC=1,LENIC
                   IF(I == IAR(IIC).AND.L == LAR(IIC)) THEN
                      LPOS=(J == JAR(IIC).AND.K == KAR(IIC))
                      LNEG=(J == KAR(IIC).AND.K == JAR(IIC))
                   ELSE
                      IF(I == LAR(IIC).AND.L == IAR(IIC)) THEN
                         LNEG=(J == JAR(IIC).AND.K == KAR(IIC))
                         LPOS=(J == KAR(IIC).AND.K == JAR(IIC))
                      ELSE
                         LNEG=.FALSE.
                         LPOS=.FALSE.
                      ENDIF
                   ENDIF
                   !
                   IF(LNEG) THEN
                      IF(PRNLEV >= 2) WRITE(OUTU,315) IIC
315                   FORMAT(15X,'FOUND DIHEDRAL IN IC',I5,' AS OPPOSITE')
                      IF(T.NEQV.TAR(IIC).AND.WRNLEV >= 2) WRITE(OUTU,316)
316                   FORMAT(20X,'BUT TYPE VALUES DONT MATCH')
                      IF(T.EQV.TAR(IIC)) FOUND=.TRUE.
                      IF(QADD) THEN
                         PIC(IIC)=PIC(IIC)-ACOM
                      ELSE
                         PIC(IIC)=-ACOM
                      ENDIF
                      IF(PIC(IIC) >  180.0) PIC(IIC)=PIC(IIC)-360.0
                      IF(PIC(IIC) < -180.0) PIC(IIC)=PIC(IIC)+360.0
                   ENDIF
                   IF(LPOS) THEN
                      IF(PRNLEV >= 2) WRITE(OUTU,325) IIC
                      IF(T.NEQV.TAR(IIC) .AND. WRNLEV >= 2) &
                           WRITE(OUTU,316)
                      IF(QADD) THEN
                         PIC(IIC)=PIC(IIC)+ACOM
                      ELSE
                         PIC(IIC)=ACOM
                      ENDIF
                      IF(PIC(IIC) >  180.0) PIC(IIC)=PIC(IIC)-360.0
                      IF(PIC(IIC) < -180.0) PIC(IIC)=PIC(IIC)+360.0
                      IF(T.EQV.TAR(IIC)) FOUND=.TRUE.
325                   FORMAT(15X,'FOUND IN IC',I5,' AS POSITIVE')
                   ENDIF
                ENDDO
                !
                IF(.NOT.FOUND) THEN
                   !                 add-new-ic-element
                   IF(QADD) CALL WRNDIE(-1,'<EDITIC>', &
                        'Existing dihedral not found with ADD option')
                   !TODO CHECK if LENIC is equal to INTLENX
                   LENIC=LENIC+1
                   NNEW=NNEW+1
                   IF(PRNLEV >= 2) WRITE(OUTU,375) LENIC
                   IAR(LENIC)=I
                   JAR(LENIC)=J
                   KAR(LENIC)=K
                   LAR(LENIC)=L
                   TAR(LENIC)=T
                   B1IC(LENIC)=0.0
                   B2IC(LENIC)=0.0
                   T1IC(LENIC)=0.0
                   T2IC(LENIC)=0.0
                   PIC(LENIC)=ACOM
                ENDIF
             ENDIF
          ENDIF
          !-----------------------------------------------------------------------
       ELSE IF(WRD == 'END ') THEN
          DONE=.TRUE.
          !
          !-----------------------------------------------------------------------
       ELSE
          IF(WRNLEV >= 2) WRITE(OUTU,39) WRD
39        FORMAT(' ** UNRECOGNIZED OPERATION "',A4,'" IN EDITIC  **'/)
          CALL DIEWRN(0)
          !-----------------------------------------------------------------------
       ENDIF
       !
       IF(LENIC >= INTLEN) THEN
          CALL WRNDIE(-4,'<EDITIC>', &
               'Too many IC edits (out of memory), split the IC EDIT command')
          DONE=.TRUE.
       ENDIF
       !
    ENDDO
    !
    ! process-end-edit
    DO II=1,NNEW
       INC=II+NOLD
       I=IAR(INC)
       J=JAR(INC)
       IF(TAR(INC)) J=KAR(INC)
       IF(B2IC(INC)< 0.001) then
          loop420: DO IIC=1,LENIC
             IF(IIC == INC) cycle loop420
             IF(IAR(IIC) == I.AND.JAR(IIC) == J.AND..NOT.TAR(IIC)) &
                  GOTO 415
             IF(IAR(IIC) == J.AND.JAR(IIC) == I.AND..NOT.TAR(IIC)) &
                  GOTO 415
             IF(IAR(IIC) == I.AND.KAR(IIC) == J.AND.TAR(IIC)) GOTO 415
             IF(IAR(IIC) == J.AND.KAR(IIC) == I.AND.TAR(IIC)) GOTO 415
             IF(LAR(IIC) == I.AND.KAR(IIC) == J) GOTO 417
             IF(LAR(IIC) == J.AND.KAR(IIC) == I) GOTO 417
             cycle loop420
415          IF(B2IC(IIC) <= 0.001) cycle loop420
             B2IC(INC)=B2IC(IIC)
             cycle loop420
417          IF(B1IC(IIC) <= 0.001) cycle loop420
             B2IC(INC)=B1IC(IIC)
          enddo loop420
          IF(B2IC(INC) <= 0.001) KILL=.TRUE.
       endif
       !
       K=KAR(INC)
       L=LAR(INC)
       IF(B1IC(INC) < 0.001) then   !GOTO 460
          loop450: DO IIC=1,LENIC
             IF(IIC == INC) cycle loop450
             IF(IAR(IIC) == K.AND.JAR(IIC) == L.AND..NOT.TAR(IIC)) &
                  GOTO 445
             IF(IAR(IIC) == L.AND.JAR(IIC) == K.AND..NOT.TAR(IIC)) &
                  GOTO 445
             IF(IAR(IIC) == K.AND.KAR(IIC) == L.AND.TAR(IIC)) GOTO 445
             IF(IAR(IIC) == L.AND.KAR(IIC) == K.AND.TAR(IIC)) GOTO 445
             IF(LAR(IIC) == K.AND.KAR(IIC) == L) GOTO 447
             IF(LAR(IIC) == L.AND.KAR(IIC) == K) GOTO 447
             cycle loop450
445          IF(B2IC(IIC) <= 0.001) cycle loop450
             B1IC(INC)=B2IC(IIC)
             cycle loop450
447          IF(B1IC(IIC) <= 0.001) cycle loop450
             B1IC(INC)=B1IC(IIC)
          enddo loop450
          IF(B1IC(INC) <= 0.001) KILL=.TRUE.
       endif   !460  CONTINUE
       !
       IF(TAR(INC)) K=JAR(INC)
       IF(T2IC(INC) < 0.001) then !GOTO 500
          loop490: DO IIC=1,LENIC
             IF(IIC == INC) cycle loop490
             !
             IF(KAR(IIC) /= J) GOTO 470
             IF(LAR(IIC) == I.AND.JAR(IIC) == K) GOTO 487
             IF(LAR(IIC) == K.AND.JAR(IIC) == I) GOTO 487
             IF(IAR(IIC) == I.AND.JAR(IIC) == K.AND.TAR(IIC)) GOTO 485
             IF(IAR(IIC) == K.AND.JAR(IIC) == I.AND.TAR(IIC)) GOTO 485
             cycle loop490
470          IF(JAR(IIC) /= J.OR..NOT.TAR(IIC)) cycle loop490
             IF(IAR(IIC) == I.AND.KAR(IIC) == K) GOTO 485
             IF(IAR(IIC) == K.AND.KAR(IIC) == I) GOTO 485
             cycle loop490
485          IF(T2IC(IIC) <= 0.001) cycle loop490
             T2IC(INC)=T2IC(IIC)
             cycle loop490
487          IF(T1IC(IIC) <= 0.001) cycle loop490
             T2IC(INC)=T1IC(IIC)
          enddo loop490
          IF(T2IC(INC) <= 0.001) KILL=.TRUE.
       endif  !500  CONTINUE
       !
       I=LAR(INC)
       J=KAR(INC)
       K=JAR(INC)
       IF(T1IC(INC) < 0.001) then !GOTO 540
          loop530: DO IIC=1,LENIC
             IF(IIC == INC) cycle loop530
             !
             IF(KAR(IIC) /= J) GOTO 510
             IF(LAR(IIC) == I.AND.JAR(IIC) == K) GOTO 527
             IF(LAR(IIC) == K.AND.JAR(IIC) == I) GOTO 527
             IF(IAR(IIC) == I.AND.JAR(IIC) == K.AND.TAR(IIC)) GOTO 525
             IF(IAR(IIC) == K.AND.JAR(IIC) == I.AND.TAR(IIC)) GOTO 525
             cycle loop530
510          IF(JAR(IIC) /= J.OR..NOT.TAR(IIC)) cycle loop530
             IF(IAR(IIC) == I.AND.KAR(IIC) == K) GOTO 525
             IF(IAR(IIC) == K.AND.KAR(IIC) == I) GOTO 525
             cycle loop530
525          IF(T2IC(IIC) <= 0.001) cycle loop530
             T1IC(INC)=T2IC(IIC)
             cycle loop530
527          IF(T1IC(IIC) <= 0.001) cycle loop530
             T1IC(INC)=T1IC(IIC)
          enddo loop530
          IF(T1IC(INC) <= 0.001) KILL=.TRUE.
       endif !  540        CONTINUE
    ENDDO
    !
    IF(KILL .AND. WRNLEV >= 2) WRITE(OUTU,905)
905 FORMAT(/,15X,'ERROR IN EDIT TERMINATION. SOME NEW INTERNAL', &
         ' COORDINATES UNDEFINED. CONTINUING'/)
    ISTART=NOLD+1
    IF(ISTART > LENIC) RETURN
    IF(PRNLEV >= 2) WRITE(OUTU,605)
605 FORMAT(/15X,'EDIT COMPLETE. NEW COORDINATES ARE GIVEN.'/)
    !
    CALL WRITIC(ISTART,LENIC,-1,0,OUTU, &
         B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    RETURN
    !yw
375 FORMAT(15X,'DIHEDRAL NOT FOUND. ADDING NEW IC',I5)
925 FORMAT(/15X,'**** ERROR. ATOM OF INTERNAL COORDINATE DOES NOT', &
         ' EXIST. IGNORING ****'/)
    !
  END SUBROUTINE EDITIC

  SUBROUTINE READIC(ISTART,ICARD,IUNIT,icx)

    !     THIS ROUTINE READS AN INTERNAL COORDINATE FILE MODULE
    !
    !     By Bernard R. Brooks    1982

    use chm_kinds
    use dimens_fcm
    use exfunc

    use ctitla

    use stream

    use parallel,only: psnd4,psnd8

    implicit none

    type(InternalCoordinate) :: icx   
    INTEGER ISTART,ICARD,IUNIT

    INTEGER ICNTRL(2),N,I
    CHARACTER(len=4) HDR
    logical :: verbose=.false.  !.true.
    integer lenic_new

    IF(IUNIT < 0) THEN
       CALL WRNDIE(0,'<READIC>','No unit specified')
       RETURN
    ENDIF
    IF(IUNIT /= 5 .AND. PRNLEV >= 2) WRITE(OUTU,822) IUNIT
822 FORMAT(' INTERNAL COORDINATES READ FROM UNIT',I3)
    !

    if(verbose)print *,"READIC: iolev,icard",iolev,icard
    IF(IOLEV > 0) THEN
       IF(ICARD == 0) THEN
          ! read-binary-file
          READ(IUNIT) HDR,ICNTRL
          CALL RDTITL(TITLEB,NTITLB,IUNIT,-1)
          CALL WRTITL(TITLEB,NTITLB,OUTU,1)
          IF(ICNTRL(1) < 20) THEN
             CALL WRNDIE(-1,'<READIC>','WRONG VERSION NUMBER')
          ENDIF
          READ(IUNIT) N
       ELSE
          ! read-card-file
          CALL RDTITL(TITLEB,NTITLB,IUNIT,0)
          CALL WRTITL(TITLEB,NTITLB,OUTU,1)
          READ(IUNIT,27) ICNTRL
27        FORMAT(20I4)
          READ(IUNIT,28) N
28        FORMAT(I5)
       ENDIF
       if(verbose)print *,"READIC: N",n
       lenic_new=ISTART+N-1
       IF(lenic_new > icx%intlen) THEN
          I=lenic_new+100  ! allocate extra space to prevent frequent resizing
          if(verbose)print *, "READIC: Allocating new icx of length",I
          CALL REINTC_NEW(I,icx)
          if(verbose)print *, "READIC:New data structure icx: lenic, intlen", icx%lenic, icx%intlen

       ENDIF
       icx%lenic=lenic_new

       CALL READIC2(ISTART,icx%lenic,ICARD,ICNTRL,IUNIT, &
            icx%B1ic,icx%B2ic, &
            icx%T1ic,icx%T2ic, &
            icx%PIC, icx%IAR, &
            icx%JAR, icx%KAR, &
            icx%LAR, icx%TAR)
       if(verbose)print *, "READIC: Done readic2: lenic, intlen", icx%lenic, icx%intlen
    ENDIF

#if KEY_PARALLEL==1

    I=icx%intlen
    CALL PSND4(I,1)
    IF(I /= icx%intlen) CALL reintc_new(I,icx)

    CALL PSND4(icx%intlen,1)
    CALL PSND4(icx%lenic,1)
    CALL PSND4(icx%IAR,icx%intlen)
    CALL PSND4(icx%JAR,icx%intlen)
    CALL PSND4(icx%KAR,icx%intlen)
    CALL PSND4(icx%LAR,icx%intlen)
    CALL PSND4(icx%TAR,icx%intlen)
    CALL PSND8(icx%B1ic,icx%intlen)
    CALL PSND8(icx%B2ic,icx%intlen)
    CALL PSND8(icx%T1ic,icx%intlen)
    CALL PSND8(icx%T2ic,icx%intlen)
    CALL PSND8(icx%PIC,icx%intlen)
#endif 
    !
    RETURN
  END SUBROUTINE READIC

  SUBROUTINE READIC2(ISTART,LENIC,ICARD,ICNTRL,IUNIT, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
    !
    !     THIS ROUTINE READS AN INTERNAL COORDINATE FILE MODULE
    !
    !     By Bernard R. Brooks    1982
    !
    use chm_kinds
    use dimens_fcm
    use exfunc
    use chutil,only:matom,getatn
    use cvio
    use psf
    use stream
    use string
    use memory
    !
    implicit none
    !
    INTEGER ISTART,LENIC,ICARD,ICNTRL(*),IUNIT
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    INTEGER N,I,IRESI,IIRES,JJRES,KKRES,LLRES
    character(len=8) AKKTMP
    INTEGER :: MARK=-99999999
    !
    character(len=8) AII,AJJ,AKK,ALL,AIS,AIR,AJS,AJR,AKS,AKR,ALS,ALR
    character(len=40) fmt3
    character(len=60) fmt4
    !
    IRESI=ICNTRL(2)
    if (icntrl(1) >= 30) then ! extened format in use
       fmt3='(10X,4(I5,1X,A8),F9.4,3F8.2,F9.4)'
       fmt4='(10X,4(1X,A8,1X,A8,1X,A8,1X),F12.6,3F12.4,F12.6)'
    else
       fmt3='(6X,4(I3,1X,A4),F9.4,3F8.2,F9.4)'
       fmt4='(5X,4(1X,A4,1X,A4,1X,A4,1X),F12.6,3F12.4,F12.6)'
    endif
    !  3  FORMAT(6X,4(I3,1X,A4),F9.4,3F8.2,F9.4)
    ! 34  FORMAT(5X,4(1X,A4,1X,A4,1X,A4,1X),F12.6,3F12.4,F12.6)

    IF(ICARD == 0) THEN
       ! read-binary-file
       READ(IUNIT) (IAR(I),I=ISTART,LENIC)
       READ(IUNIT) (JAR(I),I=ISTART,LENIC)
       READ(IUNIT) (KAR(I),I=ISTART,LENIC)
       READ(IUNIT) (LAR(I),I=ISTART,LENIC)
       READ(IUNIT) (TAR(I),I=ISTART,LENIC)
       READ(IUNIT) (B1IC(I),I=ISTART,LENIC)
       READ(IUNIT) (B2IC(I),I=ISTART,LENIC)
       READ(IUNIT) (T1IC(I),I=ISTART,LENIC)
       READ(IUNIT) (T2IC(I),I=ISTART,LENIC)
       READ(IUNIT) (PIC(I),I=ISTART,LENIC)
       DO I=ISTART,LENIC
          IF(IAR(I) <= 0) IAR(I)=MARK
          IF(JAR(I) <= 0) JAR(I)=MARK
          IF(KAR(I) <= 0) KAR(I)=MARK
          IF(LAR(I) <= 0) LAR(I)=MARK
       ENDDO
    ELSE
       ! read-card-file
       IF(IRESI < 2) THEN
          DO I=ISTART,LENIC
             READ(IUNIT,fmt3,ERR=24,END=24) IIRES,AII,JJRES,AJJ, &
                  KKRES,AKK,LLRES, &
                  ALL,B2IC(I),T2IC(I), &
                  PIC(I),T1IC(I),B1IC(I)
             IAR(I)=MATOM(IIRES,AII,ATYPE,IBASE,1,NREST,.TRUE.)
             JAR(I)=MATOM(JJRES,AJJ,ATYPE,IBASE,1,NREST,.TRUE.)
             KAR(I)=MATOM(KKRES,AKK,ATYPE,IBASE,1,NREST,.TRUE.)
             LAR(I)=MATOM(LLRES,ALL,ATYPE,IBASE,1,NREST,.TRUE.)
             TAR(I)=(EQSTA(AKK,1,'*'))
          ENDDO
       ELSE
          DO I=ISTART,LENIC
             READ(IUNIT,fmt4,ERR=24,END=24) AIS,AIR,AII,AJS,AJR,AJJ, &
                  AKS,AKR,AKK,ALS,ALR,ALL, &
                  B2IC(I),T2IC(I),PIC(I), &
                  T1IC(I),B1IC(I)
             IAR(I)=GETATN(AIS,AIR,AII,SEGID,RESID,ATYPE,IBASE, &
                  NICTOT,NSEGT)
             JAR(I)=GETATN(AJS,AJR,AJJ,SEGID,RESID,ATYPE,IBASE, &
                  NICTOT,NSEGT)
             TAR(I)=(EQSTA(AKK,1,'*'))
             IF(TAR(I)) THEN
                AKKTMP=AKK
                AKK=AKKTMP(2:4)
             ENDIF
             KAR(I)=GETATN(AKS,AKR,AKK,SEGID,RESID,ATYPE,IBASE, &
                  NICTOT,NSEGT)
             LAR(I)=GETATN(ALS,ALR,ALL,SEGID,RESID,ATYPE,IBASE, &
                  NICTOT,NSEGT)
          ENDDO
       ENDIF
       !
    ENDIF
    !
    RETURN
24  LENIC=I-1
    CALL WRNDIE(2,'<READIC>','INPUT ERROR OR EOF ENCOUNTERED')
    IF(PRNLEV >= 2) WRITE(OUTU,44) LENIC
44  FORMAT(I5,' INTERNAL COORDINATES WERE FOUND')
    RETURN
  END SUBROUTINE READIC2

  SUBROUTINE ICDYNA(QFLUCT, &
       LENIC,B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR, &
       NATOM,NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP,QTRJ,IUNIC)

    !     THIS ROUTINE GETS THE AVERAGE COORDINATES FROM A DYNAMICS RUN

    !     By Bernard R. Brooks   9/22/1983

    use cheq,only:qcg

    use chm_kinds
    use dimens_fcm
    use exfunc
    use number
    use ctitla
    use stream
    use cvio
    use memory
    use intcor2,only:fillic
    implicit none

    LOGICAL QFLUCT
    INTEGER LENIC
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    real(chm_real),allocatable,dimension(:) :: x,y,z, &
         B1SUM,B2SUM,T1SUM,T2SUM,PSUM,B1REF,B2REF,T1REF,T2REF,PREF
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    INTEGER NATOM,NUNIT,FIRSTU,NSKIP,NBEGN,NSTOP
    !
    real(chm_real) B1S,B2S,T1S,T2S,PS
    INTEGER NCOORD,ISTATS,I,IUNIT
    INTEGER IFILE,NFREAT,ISTEP,NDEGF,NSAVV
    integer,allocatable,dimension(:) :: ifreat
    real(chm_real4),allocatable,dimension(:) :: temp
#if KEY_CHEQ==1
    real(chm_real),allocatable,dimension(:) :: CCG        
#endif
    !

    real(chm_real) DELTA
    character(len=4) :: HDR1='COOR',HDR2='CORD'

    INTEGER ICNTRL(20), IUNIC
    LOGICAL QTRJ, ERROR
    character(len=4) :: HDR='IC  '
    real(chm_real) pref_i

    call chmalloc('intcor.src','icdyna','B1SUM',lenic,crl=B1SUM)
    call chmalloc('intcor.src','icdyna','B2SUM',lenic,crl=B2SUM)
    call chmalloc('intcor.src','icdyna','T1SUM',lenic,crl=T1SUM)
    call chmalloc('intcor.src','icdyna','T2SUM',lenic,crl=T2SUM)
    call chmalloc('intcor.src','icdyna','PSUM' ,lenic,crl=PSUM )
    call chmalloc('intcor.src','icdyna','B1REF',lenic,crl=B1REF)
    call chmalloc('intcor.src','icdyna','B2REF',lenic,crl=B2REF)
    call chmalloc('intcor.src','icdyna','T1REF',lenic,crl=T1REF)
    call chmalloc('intcor.src','icdyna','T2REF',lenic,crl=T2REF)
    ! APH: This is a work around for a gfortran ver 4.8.0 compiler bug.
    !      If pref is allocated using chmalloc, the compiler produces an
    !      infinite loop in the "DO WHILE(PREF..." below with -O2 optimization.
    !      GCC Bugzilla Bug 57296
!    call chmalloc('intcor.src','icdyna','PREF' ,lenic,crl=PREF )
    allocate(pref(lenic))
    !                           
    call chmalloc('intcor.src','icdyna','x' ,natom,crl=x )
    call chmalloc('intcor.src','icdyna','y' ,natom,crl=y )
    call chmalloc('intcor.src','icdyna','z' ,natom,crl=z )

    DO I=1,LENIC
       B1SUM(I)=0.0
       B2SUM(I)=0.0
       T1SUM(I)=0.0
       T2SUM(I)=0.0
       PSUM(I)=0.0
       B1REF(I)=B1IC(I)
       B2REF(I)=B2IC(I)
       T1REF(I)=T1IC(I)
       T2REF(I)=T2IC(I)
       PREF(I)=PIC(I)
       DO WHILE(PREF(I) > 180.0d0)
          PREF(I)=PREF(I)-360.0d0
       ENDDO
       DO WHILE(PREF(I) < -180.0d0)
          PREF(I)=PREF(I)+360.0d0
       ENDDO
    ENDDO

    NCOORD=0

    call chmalloc('intcor.src','icdyna','ifreat',natom,intg=ifreat)
    call chmalloc('intcor.src','icdyna','temp',natom,cr4=temp)
#if KEY_CHEQ==1
    call chmalloc('intcor.src','icdyna','ccg',natom,crl=ccg)         
#endif

    IUNIT=FIRSTU
    ISTATS=1
    DO WHILE(ISTATS >= 0)
       CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
            CCG,QCG,                & 
#endif
            TEMP,NATOM,IFREAT,NFREAT, &
            FIRSTU,NUNIT,IUNIT,IFILE, &
            ISTEP,ISTATS,NDEGF,DELTA, &
            NBEGN,NSTOP,NSKIP,NSAVV,HDR1,HDR2, &
            TITLEB,NTITLB,.FALSE., (/ ZERO /), .true.)
       NCOORD=NCOORD+1
       !
       CALL FILLIC(LENIC,.FALSE.,.FALSE.,X,Y,Z, &
            B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)

       IF(QTRJ .AND. IOLEV > 0) THEN

          IF(NCOORD == 1) THEN
             DO I=1,20
                ICNTRL(I)=0
             ENDDO
             ICNTRL(1) = (NSTOP - NBEGN)/NSKIP
             ICNTRL(2) = NBEGN
             ICNTRL(3) = NSKIP
             ICNTRL(4) = NSTOP - NBEGN
             ICNTRL(5) = NSAVV
             ICNTRL(8) = NDEGF
             ICNTRL(9) = NATOM - NFREAT
             ! store bytes of nskip * delta in icntrl(10)
             icntrl(10) = transfer(nskip * delta, icntrl(10))
             WRITE(IUNIC) HDR,ICNTRL
             CALL WRTITL(TITLEA,NTITLA,IUNIC,-1)
          ENDIF

          WRITE(IUNIC) LENIC
          WRITE(IUNIC) (IAR(I),I=1,LENIC)
          WRITE(IUNIC) (JAR(I),I=1,LENIC)
          WRITE(IUNIC) (KAR(I),I=1,LENIC)
          WRITE(IUNIC) (LAR(I),I=1,LENIC)
          WRITE(IUNIC) (TAR(I),I=1,LENIC)
          WRITE(IUNIC) (B1IC(I),I=1,LENIC)
          WRITE(IUNIC) (B2IC(I),I=1,LENIC)
          WRITE(IUNIC) (T1IC(I),I=1,LENIC)
          WRITE(IUNIC) (T2IC(I),I=1,LENIC)
          WRITE(IUNIC) (PIC(I),I=1,LENIC)
       ENDIF

       !     NOW ADD IN THE CONTRIBUTION FOR THIS COORDINATE SET
       DO I=1,LENIC
          B1S=B1IC(I)-B1REF(I)
          B2S=B2IC(I)-B2REF(I)
          T1S=T1IC(I)-T1REF(I)
          T2S=T2IC(I)-T2REF(I)
          PS=PIC(I)-PREF(I)
          IF(PS > 180.0) PS=PS-360.0
          IF(PS < -180.0) PS=PS+360.0
          IF(QFLUCT) THEN
             B1SUM(I)=B1S*B1S+B1SUM(I)
             B2SUM(I)=B2S*B2S+B2SUM(I)
             T1SUM(I)=T1S*T1S+T1SUM(I)
             T2SUM(I)=T2S*T2S+T2SUM(I)
             PSUM(I)=PS*PS+PSUM(I)
          ELSE
             B1SUM(I)=B1S+B1SUM(I)
             B2SUM(I)=B2S+B2SUM(I)
             T1SUM(I)=T1S+T1SUM(I)
             T2SUM(I)=T2S+T2SUM(I)
             PSUM(I)=PS+PSUM(I)
          ENDIF
       ENDDO

    ENDDO

    call chmdealloc('intcor.src','icdyna','ifreat',natom,intg=ifreat)
    call chmdealloc('intcor.src','icdyna','temp',natom,cr4=temp)
#if KEY_CHEQ==1
    call chmdealloc('intcor.src','icdyna','ccg',natom,crl=ccg)         
#endif

    !     NOW SCALE THE SUM APPROPRIATELY

    DO I=1,LENIC
       B1S=B1SUM(I)/NCOORD
       B2S=B2SUM(I)/NCOORD
       T1S=T1SUM(I)/NCOORD
       T2S=T2SUM(I)/NCOORD
       PS=PSUM(I)/NCOORD
       IF(QFLUCT) THEN
          B1IC(I)=SQRT(B1S)
          B2IC(I)=SQRT(B2S)
          T1IC(I)=SQRT(T1S)
          T2IC(I)=SQRT(T2S)
          PIC(I)=SQRT(PS)
       ELSE
          B1IC(I)=B1S+B1REF(I)
          B2IC(I)=B2S+B2REF(I)
          T1IC(I)=T1S+T1REF(I)
          T2IC(I)=T2S+T2REF(I)
          PIC(I)=PS+PREF(I)
          IF(PIC(I) > 180.0) PIC(I)=PIC(I)-360.0
          IF(PIC(I) < -180.0) PIC(I)=PIC(I)+360.0
       ENDIF
    ENDDO
    !
    IF(QTRJ .AND. IOLEV > 0) THEN
       IF(PRNLEV >= 2) WRITE(OUTU,101) &
            ICNTRL(1)+1,(ICNTRL(I),I=2,3),IUNIC
101    FORMAT(/2X,I5,'   IC COORDINATE SETS STARTING FROM',/, &
            5X,'STEP NO ',I8,'   FOR EVERY ',I5,'  STEPS',/, &
            5X,'WRITTEN ON UNIT',I5,/)
       CALL VCLOSE(IUNIC,'KEEP',ERROR)
    ENDIF

    call chmdealloc('intcor.src','icdyna','x' ,natom,crl=x )
    call chmdealloc('intcor.src','icdyna','y' ,natom,crl=y )
    call chmdealloc('intcor.src','icdyna','z' ,natom,crl=z )
    call chmdealloc('intcor.src','icdyna','B1SUM',lenic,crl=B1SUM)
    call chmdealloc('intcor.src','icdyna','B2SUM',lenic,crl=B2SUM)
    call chmdealloc('intcor.src','icdyna','T1SUM',lenic,crl=T1SUM)
    call chmdealloc('intcor.src','icdyna','T2SUM',lenic,crl=T2SUM)
    call chmdealloc('intcor.src','icdyna','PSUM' ,lenic,crl=PSUM )
    call chmdealloc('intcor.src','icdyna','B1REF',lenic,crl=B1REF)
    call chmdealloc('intcor.src','icdyna','B2REF',lenic,crl=B2REF)
    call chmdealloc('intcor.src','icdyna','T1REF',lenic,crl=T1REF)
    call chmdealloc('intcor.src','icdyna','T2REF',lenic,crl=T2REF)
    call chmdealloc('intcor.src','icdyna','PREF' ,lenic,crl=PREF )

    RETURN
  END SUBROUTINE ICDYNA

  SUBROUTINE INTCPY(LENIC, &
       B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR, &
       LENICX,INTB1,INTB2,INTT1,INTT2,INTPIC,INTIAR,INTJAR, &
       INTKAR,INTLAR,INTTAR,LPRES,LOVER,QADD,VADD)
    !
    ! THIS ROUTINE COPIES THE INTERNAL COORDINATE TABLE
    ! FROM ONE SET TO THE OTHER.  THE PRESERVE OPTION WILL
    ! RETAIN ALL CURRENT VALUES INT THE MODIFIED SET.
    !
    use chm_kinds
    implicit none
    !
    INTEGER LENIC
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    INTEGER LENICX
    real(chm_real) INTB1(*),INTB2(*),INTT1(*),INTT2(*),INTPIC(*)
    INTEGER INTIAR(*),INTJAR(*),INTKAR(*),INTLAR(*)
    LOGICAL INTTAR(*),LPRES,LOVER,QADD
    real(chm_real) VADD
    !
    INTEGER I,J,N,K
    !
    ! process preserve mode
    IF(LPRES) THEN
       N=LENIC
       DO I=1,LENICX
          DO J=1,LENIC
             K=J
             IF(TAR(J).NEQV.INTTAR(I)) GOTO 100
             IF(KAR(J) /= INTKAR(I)) GOTO 40
             IF(JAR(J) /= INTJAR(I)) GOTO 100
             IF(IAR(J) /= INTIAR(I)) GOTO 60
             IF(LAR(J) /= INTLAR(I)) GOTO 100
             GOTO 200
40           CONTINUE
             IF(TAR(J)) GOTO 100
             IF(JAR(J) /= INTKAR(I)) GOTO 100
             IF(IAR(J) /= INTLAR(I)) GOTO 100
             IF(LAR(J) /= INTIAR(I)) GOTO 100
             GOTO 200
60           CONTINUE
             IF(.NOT.TAR(J)) GOTO 100
             IF(IAR(J) /= INTLAR(I)) GOTO 100
             IF(LAR(J) /= INTIAR(I)) GOTO 100
             GOTO 200
100          CONTINUE
          ENDDO
          N=N+1
          K=N
          B1IC(K)=INTB1(I)
          B2IC(K)=INTB2(I)
          T1IC(K)=INTT1(I)
          T2IC(K)=INTT2(I)
          PIC(K)=INTPIC(I)
          IAR(K)=INTIAR(I)
          JAR(K)=INTJAR(I)
          KAR(K)=INTKAR(I)
          LAR(K)=INTLAR(I)
          TAR(K)=INTTAR(I)
200       CONTINUE
       ENDDO
       LENIC=N
       !
       ! process overwrite mode
    ELSE IF(LOVER) THEN
       N=LENIC
       DO I=1,LENICX
          DO J=1,LENIC
             K=J
             IF(TAR(J).NEQV.INTTAR(I)) GOTO 300
             IF(KAR(J) /= INTKAR(I)) GOTO 240
             IF(JAR(J) /= INTJAR(I)) GOTO 300
             IF(IAR(J) /= INTIAR(I)) GOTO 260
             IF(LAR(J) /= INTLAR(I)) GOTO 300
             GOTO 400
240          CONTINUE
             IF(TAR(J)) GOTO 300
             IF(JAR(J) /= INTKAR(I)) GOTO 300
             IF(IAR(J) /= INTLAR(I)) GOTO 300
             IF(LAR(J) /= INTIAR(I)) GOTO 300
             GOTO 400
260          CONTINUE
             IF(.NOT.TAR(J)) GOTO 300
             IF(IAR(J) /= INTLAR(I)) GOTO 300
             IF(LAR(J) /= INTIAR(I)) GOTO 300
             GOTO 400
300          CONTINUE
          ENDDO
          N=N+1
          K=N
400       CONTINUE
          B1IC(K)=INTB1(I)
          B2IC(K)=INTB2(I)
          T1IC(K)=INTT1(I)
          T2IC(K)=INTT2(I)
          PIC(K)=INTPIC(I)
          IAR(K)=INTIAR(I)
          JAR(K)=INTJAR(I)
          KAR(K)=INTKAR(I)
          LAR(K)=INTLAR(I)
          TAR(K)=INTTAR(I)
       ENDDO
       LENIC=N
       !
       ! process add/sum mode
    ELSE IF(QADD) THEN
       DO I=1,LENIC
          B1IC(I)=B1IC(I)+VADD*INTB1(I)
          B2IC(I)=B2IC(I)+VADD*INTB2(I)
          T1IC(I)=T1IC(I)+VADD*INTT1(I)
          T2IC(I)=T2IC(I)+VADD*INTT2(I)
          PIC(I) =PIC(I) +VADD*INTPIC(I)
          IF(PIC(I) >  180.0) PIC(I)=PIC(I)-360.0
          IF(PIC(I) < -180.0) PIC(I)=PIC(I)+360.0
       ENDDO
    ELSE
       LENIC=LENICX
       DO I=1,LENIC
          B1IC(I)=INTB1(I)
          B2IC(I)=INTB2(I)
          T1IC(I)=INTT1(I)
          T2IC(I)=INTT2(I)
          PIC(I)=INTPIC(I)
          IF(PIC(I) >  180.0) PIC(I)=PIC(I)-360.0
          IF(PIC(I) < -180.0) PIC(I)=PIC(I)+360.0
          IAR(I)=INTIAR(I)
          JAR(I)=INTJAR(I)
          KAR(I)=INTKAR(I)
          LAR(I)=INTLAR(I)
          TAR(I)=INTTAR(I)
       ENDDO
    ENDIF
    RETURN
  END SUBROUTINE INTCPY

  SUBROUTINE INTRND(LENIC,PIC,ISEED)
    !
    ! THIS ROUTINE RANDOMIZES ALL OF THE DIHEDRAL VALUES
    ! IN THE INTERNAL COORDINATE TABLE
    !
    use clcg_mod,only:random
    use chm_kinds
    !  use exfunc
    implicit none
    !
    INTEGER LENIC
    real(chm_real) PIC(*)
    INTEGER ISEED
    !
    INTEGER I
    !
    DO I=1,LENIC
       PIC(I)=(360.0*RANDOM(ISEED)-180.0)
    ENDDO
    RETURN
  END SUBROUTINE INTRND

  SUBROUTINE ICPUCK(IPA,MAGN,ANGL,LENIC,INTLEN, &
       B1IC,B2IC,T1IC,T2IC,PIC, &
       IAR,JAR,KAR,LAR,TAR)

    ! This routine sets that value of the specified ring IC elements
    ! to a particular pseudorotational angle and magnitude.

    use chm_kinds

    use number
    use consta
    use stream
    use intcor2,only:writic
    implicit none
    !
    INTEGER IPA(5)
    real(chm_real) MAGN,ANGL
    INTEGER LENIC,INTLEN
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)

    INTEGER I,J,IFOUND(5),IATOM(10)

    ifound(1:5) = 0
    iatom(1:5)  = ipa(1:5)
    iatom(6:10) = ipa(1:5)

    DO I=1,LENIC
       DO J=1,5
          IF(IAR(I) /= IATOM(J)) cycle
          IF(JAR(I) /= IATOM(J+1)) exit
          IF(KAR(I) /= IATOM(J+2)) exit
          IF(LAR(I) /= IATOM(J+3)) exit

          PIC(I)=MAGN*COS(DEGRAD*(ANGL+TWELVE*TWELVE*(J-3)))
          CALL WRITIC(I,I,-1,0,OUTU, &
               B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
          IFOUND(J)=IFOUND(J)+1
       ENDDO
    ENDDO

    I=0
    DO J=1,5
       IF(IFOUND(J) == 0) I=I+1
    ENDDO
    IF(I > 0) CALL WRNDIE(0,'<ICPUCK>','Some pucker IC not found.')

    RETURN
  END SUBROUTINE ICPUCK

  SUBROUTINE SCALICD(PIC,TAR,LENIC,KDIHE,QTAR)

    ! This routine scales dihedral or improper dihedral values

    use chm_kinds
    use number
    use consta
    use stream

    implicit none
    real(chm_real) PIC(*)    ! dihedral value array
    LOGICAL TAR(*)   ! improper flag array
    INTEGER LENIC    ! numer of IC talbe elements
    real(chm_real) KDIHE     ! scale factor
    LOGICAL QTAR     ! which set to scale
    INTEGER I

    IF(QTAR) THEN
       ! only scale improper dihedrals
       DO I=1,LENIC
          IF(TAR(I)) PIC(I)=PIC(I)*KDIHE
       ENDDO
    ELSE
       ! only scale regular dihedrals
       DO I=1,LENIC
          IF(.NOT.TAR(I)) PIC(I)=PIC(I)*KDIHE
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE SCALICD

  SUBROUTINE KEEPIC(QKEEP,ISLCT,X,Y,Z,WMAIN, &
       LENIC,B1IC,B2IC,T1IC,T2IC,PIC, &
       IAR,JAR,KAR,LAR,TAR)
    !
    !    Process the IC KEEP and IC DELEte commands
    !
    !     By Bernard R. Brooks    1998
    !
    use comand
    use consta
    use select
    use string
    use stream
    use intcor2,only:purgic

    !
    implicit none
    !
    LOGICAL QKEEP
    INTEGER ISLCT(*)
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    INTEGER LENIC
    real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
    INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
    LOGICAL TAR(*)
    !
    LOGICAL LKEEP,IDELE
    INTEGER :: I,J,ISD,JSD,MARK2=-99
    LOGICAL QBYNUM,QIAR,QJAR,QKAR,QLAR,QTAR,QNTAR
    integer,dimension(1) :: idum
    !
    QBYNUM=(INDX(COMLYN,COMLEN,'BYNU',4) > 0) .AND. &
         (INDX(COMLYN,COMLEN,'SELE',4) <= 0)
    IF(QBYNUM) THEN
       QBYNUM=(INDXA(COMLYN,COMLEN,'BYNU') > 0)
       ISD=NEXTI(COMLYN,COMLEN)
       JSD=NEXTI(COMLYN,COMLEN)
       IF(JSD == 0) JSD=ISD
       IF(ISD < 1) ISD=1
       IF(JSD > LENIC) JSD=LENIC

       IF(QKEEP) THEN
          DO I=1,ISD-1
             IAR(I)=MARK2
          ENDDO
          DO I=JSD+1,LENIC
             IAR(I)=MARK2
          ENDDO
       ELSE
          DO I=ISD,JSD
             IAR(I)=MARK2
          ENDDO
       ENDIF
    ELSE
       CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
       QIAR=(INDXA(COMLYN,COMLEN,'FIRS') > 0)
       QJAR=(INDXA(COMLYN,COMLEN,'SECO') > 0)
       QKAR=(INDXA(COMLYN,COMLEN,'THIR') > 0)
       QLAR=(INDXA(COMLYN,COMLEN,'FOUR') > 0)
       QTAR=(INDXA(COMLYN,COMLEN,'IMPR') > 0)
       QNTAR=(INDXA(COMLYN,COMLEN,'DIHE') > 0)
       IF(.NOT.(QIAR.OR.QJAR.OR.QKAR.OR.QLAR)) THEN
          QIAR=.TRUE.
          QJAR=.TRUE.
          QKAR=.TRUE.
          QLAR=.TRUE.
       ENDIF
       !
       LKEEP=QKEEP
       DO I=1,LENIC
          IDELE=LKEEP
          IF(QIAR) THEN
             J=IAR(I)
             IF(J > 0) THEN
                IF(ISLCT(J) > 0) IDELE=.NOT.LKEEP
             ENDIF
          ENDIF
          IF(QJAR) THEN
             J=JAR(I)
             IF(J > 0) THEN
                IF(ISLCT(J) > 0) IDELE=.NOT.LKEEP
             ENDIF
          ENDIF
          IF(QKAR) THEN
             J=KAR(I)
             IF(J > 0) THEN
                IF(ISLCT(J) > 0) IDELE=.NOT.LKEEP
             ENDIF
          ENDIF
          IF(QLAR) THEN
             J=LAR(I)
             IF(J > 0) THEN
                IF(ISLCT(J) > 0) IDELE=.NOT.LKEEP
             ENDIF
          ENDIF
          IF(LKEEP) THEN
             IF(QTAR .AND. TAR(I)) IDELE=.FALSE.
             IF(QNTAR .AND. .NOT.TAR(I)) IDELE=.FALSE.
          ELSE
             IF(QNTAR .AND. .NOT.QTAR .AND. TAR(I)) IDELE=.FALSE.
             IF(.NOT.QNTAR .AND. QTAR .AND. .NOT.TAR(I)) IDELE=.FALSE.
          ENDIF
          IF(IDELE) IAR(I)=MARK2
       ENDDO
    ENDIF
    ! Now get rid of the unwanted ones...
    CALL PURGIC(icr_struct%lenic,&
         icr_struct%B1ic,icr_struct%B2ic, &
         icr_struct%T1ic,icr_struct%T2ic, &
         icr_struct%PIC, icr_struct%IAR, &
         icr_struct%JAR, icr_struct%KAR, &
         icr_struct%LAR, icr_struct%TAR, &
         .FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.,0,idum,idum)
    !
    RETURN
  END SUBROUTINE KEEPIC
  SUBROUTINE REMAPIC(MAP,MARK)
    !
    !     REMAPS AND COMPRESS INTERNAL COORDINATES
    !
    !     By Bernard R. Brooks    1998
    !
    use intcor2,only:remapic2
    implicit none
    !
    INTEGER MAP(*),MARK
    !
    !
    ! Remap the main IC table
    IF(icr_struct%lenic > 0) THEN
       CALL REMAPIC2(MAP,MARK,&
            icr_struct%lenic, &
            icr_struct%B1ic,icr_struct%B2ic, &
            icr_struct%T1ic,icr_struct%T2ic, &
            icr_struct%PIC, icr_struct%IAR, &
            icr_struct%JAR, icr_struct%KAR, &
            icr_struct%LAR, icr_struct%TAR)
    endif
    !
    ! Remap the saved IC table
    IF(ics_struct%lenic > 0) THEN
       CALL REMAPIC2(MAP,MARK,&
            ics_struct%lenic, &
            ics_struct%B1ic,ics_struct%B2ic, &
            ics_struct%T1ic,ics_struct%T2ic, &
            ics_struct%PIC, ics_struct%IAR, &
            ics_struct%JAR, ics_struct%KAR, &
            ics_struct%LAR, ics_struct%TAR)
    endif
    !
    RETURN
  END SUBROUTINE REMAPIC


end module intcor_module

