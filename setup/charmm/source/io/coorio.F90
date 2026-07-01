module coorio_mod
contains
  subroutine coorio(iomode,iunit,comlyn,comlen, &
       titleb,ntitlb,icntrl,natom,x,y,z,wmain, &
       atype,resid,res,nres,ibase,segid,nictot,nseg,lcheck)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE PARSES THE COMMAND LINE FOR OPTIONS AND ATOM
    !     SPECIFICATION IN READING AND WRITING COORDINATES.
    !
    !     SOME OF THE CALLING SEQUENCE;
    !       IOMODE - INTEGER - NEGATIVE READ, ZERO WRITE, POSITIVE PRINT
    !       IUNIT - INTEGER - FORTRAN UNIT NUMBER FOR IO
    !       ISLCT - I*2     - LIST OF SELECTED ATOMS RETURNED
    !       LMUST - LOGICAL - IF TRUE, WILL CHECK FOR ALL ATOMS PLACED
    !
    !     Overhauled by Bernard R. Brooks   1983
    !

#if KEY_CHEQ==1
    use cheq,only:qcg    
#endif
    use chm_kinds
    use number
#if KEY_CHEQ==1
    use dimens_fcm      
#endif
#if KEY_PIPF==1
    use pipfm           
#endif
    use univ
    !
    !++LNI add for reading dynamics RESTART FILE
    use dynio
    use memory
    use select
    use stream
    use string
    use parallel  ! mh050712
    use chutil,only:initia
    !
#if KEY_STRINGM==1 /*   VO string method */
    use machio, only: ifreeu
    use multicom_aux
#endif
    !
    implicit none
    !
    real(chm_real),allocatable,dimension(:),target :: space_rl1
    real(chm_real),pointer,dimension(:) :: IX1,IY1,IZ1,IX2,IY2,IZ2,idm
    !
#if KEY_CHEQ==1
    real(chm_real),allocatable,dimension(:),target :: space_cg     
    real(chm_real),pointer,dimension(:) :: icgtmp,icgold,ivcg      
#endif
    !
#if KEY_PIPF==1
    real(chm_real),allocatable,dimension(:),target :: space_pipf   
    real(chm_real),pointer,dimension(:) :: iuind,iuindo,ivuind     
#endif

    integer,allocatable,dimension(:) :: ifreea,islct
    real(chm_real) x(*),y(*),z(*),wmain(*)
    integer  ibase(*)
    character(len=*) atype(*),resid(*),res(*),segid(*)
    integer nictot(*)
    logical lcheck,linit,lrsid,lappe,lfree
    integer comlen
    character(len=*) comlyn

#if KEY_CHEQ==1
    logical   qcheqrdprm     
#endif
#if KEY_PIPF==1
    integer   n3uind         
#endif

    !
    integer natom,iomode,ifile,ioffs,ninput,imode,nseg,nchain
    integer iunit,iresm,lenap,ires,nres,iresc,i
    character(len=*) titleb(*)
    integer ntitlb,icntrl(20),modecw,model,modfl
    !
    logical error
    character(len=80) iline
    logical official
    !
    !++lni add for reading dynamics restart file
    integer jdum,ldyna
    real(chm_real) dum

    integer deci
    !
#if KEY_STRINGM==1 /*  VO stringm v */
    integer :: oldiol
    logical :: qstr
    common /replicaio/ qstr ! need global variable
    qstr=iunit.le.size(ifreeu) ! protect against OOB on ifreeu
    if (qstr) qstr=((MOD(IFREEU(IUNIT),8).eq.0).and.(IFREEU(IUNIT).ne.0).and.(ME_LOCAL.eq.0))
    if (qstr) then
     oldiol=iolev
     iolev=1
    endif
#endif
    !
    !--LNI add for reading dynamics RESTART FILE
    official = .false.
    call chmalloc('coorio.src','coorio','islct',natom,intg=islct)

    if (iomode.lt.0) then
       ifile=gtrmi(comlyn,comlen,'IFIL',1)
       ioffs=gtrmi(comlyn,comlen,'OFFS',0)
       ninput=0
       if (indxa(comlyn,comlen,'FILE').gt.0) then
          ninput=0
       else if (indxa(comlyn,comlen,'CARD').gt.0) then
          ninput=1
       else if (indxa(comlyn,comlen,'IGNO').gt.0) then
          ninput=3
       else if (indxa(comlyn,comlen,'PDB').gt.0) then
          ninput=-1
          nchain=gtrmi(comlyn,comlen, 'NCHA',0)
          official = indxa(comlyn,comlen,'OFFI').gt.0
          if(official .and. prnlev >= 2 )then
             write(outu,'(A,/,A)') ' Read official pdb format.', &
                  ' Note that the segid (chain id) must be limited to one character.'
          else
             if (prnlev >= 2) write(outu,*) ' read CHARMM-pdb format'
          endif
          ! LNI Check if PDB NMR MODEL is to be read
          model=gtrmi(comlyn,comlen,'MODE',0)
       else if (indxa(comlyn,comlen,'UNIV').gt.0) then
          ninput=-2
          !**clbiii add for reading lattice coordinates
       else if (indxa(comlyn,comlen,'LATT').gt.0) then
          ninput=-4
          !END OF**clbiii add for reading lattice coordinates
          !++LNI add for reading dynamics RESTART FILE
       else if (indxa(comlyn,comlen,'DYNR').gt.0) then
          ninput=-5
          !--LNI add for reading dynamics RESTART FILE
#if KEY_TMD==1
       else if (indxa(comlyn,comlen,'TARG').gt.0) then 
          ninput=1
#endif
       endif
       !++
       ! LN ADD /APR 90: HANDLE BINARY INPUT W/O REWINDING THE FILE ALL THE TIME
       if(ninput .eq. 0) then
          if (reallow) then   
             if(indxa(comlyn,comlen,'CONT').gt.0) ninput=-3
          else                
             ninput=-3         
          endif               
       endif
       !--
       LRSID=(INDXA(COMLYN,COMLEN,'RESI').GT.0)
       LINIT=(INDXA(COMLYN,COMLEN,'INIT').GT.0).OR. &
            (INDXA(COMLYN,COMLEN,'REST').GT.0)
       LAPPE=(INDXA(COMLYN,COMLEN,'APPE').GT.0)
       LFREE=(INDXA(COMLYN,COMLEN,'FREE').GT.0)
       !
       IMODE=0
       CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0)  THEN
          !         Begin Procedure CRAP-OUT
          CALL WRNDIE(0,'<COORIO>','Atom selection parsing error')
#if KEY_STRINGM==0 /*  VO stringm */
          call chmdealloc('coorio.src','COORIO','ISLCT',NATOM,intg=ISLCT)
          RETURN
#else
          goto 999
#endif
          !         End Procedure CRAP-OUT
       ENDIF
       !
       IF(IUNIT.LT.0 .AND. NINPUT.NE.0) IUNIT=ISTRM
       IF(PRNLEV.GE.2) WRITE(OUTU,430) IUNIT
430    FORMAT(10X,'SPATIAL COORDINATES BEING READ FROM UNIT',I3)
       IF(IUNIT.LT.0) THEN
          CALL WRNDIE(0,'<COORIO>','INVALID UNIT NUMBER')
#if KEY_STRINGM==0 /*  VO stringm */
          call chmdealloc('coorio.src','COORIO','ISLCT',NATOM,intg=ISLCT)
          RETURN
#else
          goto 999
#endif
       ENDIF
       !
       !**clbiii add for reading lattice coordinates
       IF( NINPUT .EQ. -4 ) THEN
          if ( iomode .lt. 0 ) CALL RLATT(IUNIT,ISLCT)
          !          if ( iomode .eq. 0 ) CALL WLATT0
          !          if ( iomode .gt. 0 ) CALL PLATT0
#if KEY_STRINGM==0 /*  VO stringm */
          call chmdealloc('coorio.src','COORIO','ISLCT',NATOM,intg=ISLCT)
          RETURN
#else
          goto 999
#endif
       ENDIF
       !END of**clbiii add for reading lattice coordinates
       !++LNI add for reading dynamics RESTART FILE
       IF(NINPUT .EQ. -5)THEN
          ! Just try to keep track of what we actually need from the file
          ! LDYNA may need to be set properly to interpret old files?
          ! RESTART file contents depend on the integrator used.
          !
          ! LDYNA=1 means we behave as if this was being read into the
          ! leap frog integrator.
          ! The idea is that CURR returns coordinates at CURRENT step
          !                  DELT returns coordinate displacement FROM CURRENT
          !                  VEL  returns current VELOCITIES
          ! Note that the restart file written after a crash may be different!
          !
          LDYNA=1
          IF(INDXA(COMLYN,COMLEN,'VERL').GT.0) LDYNA=-1
          IF(INDXA(COMLYN,COMLEN,'LEAP').GT.0) LDYNA=1
          call chmalloc('coorio.src','COORIO','space_rl1',7*NATOM,crl=space_rl1)
          IX1 => space_rl1(0*natom+1 : 1*natom)
          IY1 => space_rl1(1*natom+1 : 2*natom)
          IZ1 => space_rl1(2*natom+1 : 3*natom)
          IX2 => space_rl1(3*natom+1 : 4*natom)
          IY2 => space_rl1(4*natom+1 : 5*natom)
          IZ2 => space_rl1(5*natom+1 : 6*natom)
          IDM => space_rl1(6*natom+1 : 7*natom)
#if KEY_CHEQ==1
          call chmalloc('coorio.src','COORIO','space_cg',3*NATOM,crl=space_cg)
          ICGTMP   => space_cg(0*natom+1 : 1*natom)
          ICGOLD   => space_cg(1*natom+1 : 2*natom)
          IVCG     => space_cg(2*natom+1 : 3*natom)
#endif 
          ! PJ 06/2005
#if KEY_PIPF==1
          IF (QPIPF .AND. QPFDYN) THEN
             N3UIND = 3 * NATOM
          ELSE
             N3UIND = 3
          ENDIF
          call chmalloc('coorio.src','COORIO','space_pipf',3*N3uind,crl=space_pipf)
          IUIND  => space_pipf(0*n3uind+1 : 1*n3uind)
          IUINDO => space_pipf(1*n3uind+1 : 2*n3uind)
          IVUIND => space_pipf(2*n3uind+1 : 3*n3uind)
#endif 
          IF(INDXA(COMLYN,COMLEN,'DELT').GT.0)THEN
             CALL READYN(IUNIT,NATOM, &
                  IX1,IY1,IZ1, &
                  X,Y,Z,IX2,IY2,IZ2, &
#if KEY_CHEQ==1
                  ICGTMP,ICGOLD,IVCG,QCG,  & 
#endif
#if KEY_PIPF==1
                  IUIND,IUINDO,IVUIND,QPFDYN,  & 
#endif
#if KEY_PIPF==1
                  NPFBATHS,PFNHSBATH,PFNHSOBATH,                & 
#endif
#if KEY_DYNVV2==1
                  .FALSE.,IX2,IY2,IZ2,        & 
#endif
                  JDUM,JDUM, &
                  JDUM,JDUM,JDUM,JDUM,DUM,DUM,JDUM,LDYNA &
#if KEY_BLOCK==1
                  ,.FALSE.,.FALSE.,JDUM,IDM,IDM,IDM,JDUM &    /*ldm*/
#endif
#if KEY_FOURD==1
                  ,IDM,IDM             & 
#endif
#if KEY_SCCDFTB==1
                  ,.FALSE.,.FALSE.,.FALSE.,0,0,ZERO,ZERO,ZERO   & 
#endif
                  )
          ELSEIF(INDXA(COMLYN,COMLEN,'CURR').GT.0)THEN
             CALL READYN(IUNIT,NATOM,X,Y,Z, &
                  IX1,IY1,IZ1, &
                  IX2,IY2,IZ2, &
#if KEY_CHEQ==1
                  ICGTMP,ICGOLD,IVCG,QCG,  & 
#endif
#if KEY_PIPF==1
                  IUIND,IUINDO,IVUIND,QPFDYN,  & 
#endif
#if KEY_PIPF==1
                  NPFBATHS,PFNHSBATH,PFNHSOBATH,                & 
#endif
#if KEY_DYNVV2==1
                  .FALSE.,IX2,IY2,IZ2,        & 
#endif
                  JDUM,JDUM, &
                  JDUM,JDUM,JDUM,JDUM,DUM,DUM,JDUM,LDYNA &
#if KEY_BLOCK==1
                  ,.FALSE.,.FALSE.,JDUM,IDM,IDM,IDM,JDUM &   /*ldm*/
#endif
#if KEY_FOURD==1
                  ,IDM,IDM             & 
#endif
#if KEY_SCCDFTB==1
                  ,.FALSE.,.FALSE.,.FALSE.,0,0,ZERO,ZERO,ZERO   & 
#endif
                  )
          ELSEIF(INDXA(COMLYN,COMLEN,'VEL').GT.0)THEN
             CALL READYN(IUNIT,NATOM, &
                  IX1,IY1,IZ1, &
                  IX2,IY2,IZ2,X,Y,Z, &
#if KEY_CHEQ==1
                  ICGTMP,ICGOLD,IVCG,QCG,  & 
#endif
#if KEY_PIPF==1
                  IUIND, IUINDO,IVUIND,QPFDYN,  & 
#endif
#if KEY_PIPF==1
                  NPFBATHS,PFNHSBATH,PFNHSOBATH,                & 
#endif
#if KEY_DYNVV2==1
                  .FALSE.,IX2,IY2,IZ2,        & 
#endif
                  JDUM,JDUM, &
                  JDUM,JDUM,JDUM,JDUM,DUM,DUM,JDUM,LDYNA &
#if KEY_BLOCK==1
                  ,.FALSE.,.FALSE.,JDUM,IDM,IDM,IDM,JDUM &  /*ldm*/
#endif
#if KEY_FOURD==1
                  ,IDM,IDM             & 
#endif
#if KEY_SCCDFTB==1
                  ,.FALSE.,.FALSE.,.FALSE.,0,0,ZERO,ZERO,ZERO   & 
#endif
                  )
          ELSE
             CALL WRNDIE(0,'<COORIO>','Unknown READ COOR DYNR option')
          ENDIF
#if KEY_STRINGM==0 /* (string)  VO stringm v */
          if(allocated(space_rl1)) &
               call chmdealloc('coorio.src','COORIO','space_rl1',7*NATOM,crl=space_rl1)
#if KEY_CHEQ==1
          if(allocated(space_cg)) &             
#endif
#if KEY_CHEQ==1
               call chmdealloc('coorio.src','COORIO','space_cg',3*NATOM,crl=space_cg)  
#endif
#if KEY_PIPF==1
          if(allocated(space_pipf)) &          
#endif
#if KEY_PIPF==1
               call chmdealloc('coorio.src','COORIO','space_pipf', &   
#endif
#if KEY_PIPF==1
               3*N3uind,crl=space_pipf)   
#endif
          call chmdealloc('coorio.src','COORIO','ISLCT',NATOM,intg=ISLCT)          
          RETURN
#else /* (string) */
          goto 999
#endif /*(string) */
       ENDIF
       !--LNI add for reading dynamics RESTART FILE
       !
       call chmalloc('coorio.src','COORIO','IFREEA',NATOM,intg=IFREEA)
       if( &
            (IOLEV.GT.0) .AND. & 
            IUNIT.NE.ISTRM) THEN
          IF (reallow) THEN      
             IF(IFILE.EQ.1 .AND. NINPUT.NE.-3) REWIND IUNIT
          ENDIF                  
       ENDIF
       !
       IRESM=0
       LENAP=0
       IF (LAPPE) THEN
          DO IRES=1,NRES
             IRESC=0
             DO I=IBASE(IRES)+1,IBASE(IRES+1)
                IF(INITIA(I,X,Y,Z)) IRESC=1
             ENDDO
             IF(IRESC.EQ.1) IRESM=IRES
          ENDDO
       ENDIF
       IF(IRESM.GT.0) LENAP=IBASE(IRESM+1)
       IF(IRESM.EQ.NRES) THEN
          CALL WRNDIE(0,'<COORIO>','Cannot append to full set')
#if KEY_STRINGM==0 /*  VO */
          if(allocated(space_rl1)) &
               call chmdealloc('coorio.src','COORIO','space_rl1',7*NATOM,crl=space_rl1)
#if KEY_CHEQ==1
          if(allocated(space_cg)) &             
#endif
#if KEY_CHEQ==1
               call chmdealloc('coorio.src','COORIO','space_cg',3*NATOM,crl=space_cg)    
#endif
#if KEY_PIPF==1
          if(allocated(space_pipf)) &          
#endif
#if KEY_PIPF==1
               call chmdealloc('coorio.src','COORIO','space_pipf', &     
#endif
#if KEY_PIPF==1
               3*N3uind,crl=space_pipf)   
#endif
          if(allocated(ifreea)) &
               call chmdealloc('coorio.src','COORIO','IFREEA',NATOM,intg=IFREEA)
          RETURN
#else
          goto 999
#endif
       ENDIF
       !
       IOFFS=IOFFS+IRESM
       IF(IOFFS.NE.0) THEN
          IF (LRSID) THEN
             CALL WRNDIE(0,'<COORIO>', &
                  'APPEnd and OFFSet options not allowed with RESI option')
             IOFFS=0
          ELSE
             IF(PRNLEV.GE.2) WRITE(OUTU,129) IOFFS
129          FORMAT(' A RESIDUE OFFSET OF',I4,' WILL BE USED.')
          ENDIF
       ENDIF
       !
       ISLCT(1:LENAP) = 0
       IF(LINIT) THEN
          DO I=1,NATOM
             IF(ISLCT(I).EQ.1) THEN
                X(I)=ANUM
                Y(I)=ANUM
                Z(I)=ANUM
             ENDIF
          ENDDO
       ENDIF
       !
       IF (NINPUT.NE.-2) THEN
          CALL CREAD(IUNIT,TITLEB,NTITLB,ICNTRL,X,Y,Z,WMAIN,NATOM, &
               NINPUT,ISLCT,IOFFS, &
               RES,NRES,ATYPE,IBASE,IFILE,IFREEA, &
               SEGID,RESID,NICTOT,NSEG,LRSID,LFREE,ILINE,80,MODEL,OFFICIAL,NCHAIN)
       ELSE
          CALL CREADU(IUNIT,X,Y,Z,WMAIN,NATOM,ISLCT, &
               RES,NRES,ATYPE,IBASE,SEGID,RESID,NICTOT,NSEG, &
               LRSID,LFREE,IOFFS)
       ENDIF
       IF (LCHECK) THEN
          DO I=1,NATOM
             IF(.NOT.INITIA(I,X,Y,Z)) THEN
                CALL WRNDIE(1,'<COORIO>', &
                     'The coordinates for some atoms were not read')
                GOTO 30
             ENDIF
          ENDDO
       ENDIF
30     CONTINUE
       !
       !       Write out coordinates.
    ELSE
       IF(IUNIT.LT.0 .AND. IOMODE.GT.0) IUNIT=OUTU
       IF(IUNIT.LT.0 .AND. IOLEV.GT.0) THEN
          CALL WRNDIE(0,'<COORIO>','INVALID UNIT NUMBER')
#if KEY_STRINGM==0 /*  VO stringm v */
          if(allocated(space_rl1)) &
               call chmdealloc('coorio.src','COORIO','space_rl1',7*NATOM,crl=space_rl1)
#if KEY_CHEQ==1
          if(allocated(space_cg)) &             
#endif
#if KEY_CHEQ==1
               call chmdealloc('coorio.src','COORIO','space_cg',3*NATOM,crl=space_cg)   
#endif
#if KEY_PIPF==1
          if(allocated(space_pipf)) &          
#endif
#if KEY_PIPF==1
               call chmdealloc('coorio.src','COORIO','space_pipf', &  
#endif
#if KEY_PIPF==1
               3*N3uind,crl=space_pipf)   
#endif
          if(allocated(ifreea)) &
               call chmdealloc('coorio.src','COORIO','IFREEA',NATOM,intg=IFREEA)
          RETURN
#else
          goto 999
#endif
       ENDIF
       MODECW=1
       IF(INDXA(COMLYN,COMLEN,'FILE').NE.0) MODECW=1
       IF(INDXA(COMLYN,COMLEN,'CARD').NE.0) MODECW=2
       IF(INDXA(COMLYN,COMLEN,'PDB').NE.0) THEN
          MODECW=4
          OFFICIAL = INDXA(COMLYN,COMLEN,'OFFI').GT.0
          IF(OFFICIAL .AND. PRNLEV >= 2)THEN
             WRITE(OUTU,'(A,/,A)') ' Write official pdb format.', &
                  ' Note that the segid (chain id) will be truncated to only one character.'
          ELSE IF(PRNLEV >= 2)THEN
             WRITE(OUTU,*) ' Write CHARMM-pdb format'
          ENDIF
       ENDIF
       IF(INDXA(COMLYN,COMLEN,'DUMB').NE.0) MODECW=5
       IF(INDXA(COMLYN,COMLEN,'XYZ').NE.0) MODECW=6
       IF(IOMODE.GT.0) MODECW=3
       IOFFS=GTRMI(COMLYN,COMLEN,'OFFS',0)
       IMODE=0
       ! LNI Check if NMR model is to be written to PDB file
       IF(MODECW.EQ.4)THEN
          MODEL=GTRMI(COMLYN,COMLEN,'MODE',0)
          MODFL=0
          ! MODFL= 0 don't force header or END line writing
          !        1 force header, 2 force END, 3 force both header and END
          IF(INDXA(COMLYN,COMLEN,'FIRS').NE.0) MODFL=1
          IF(INDXA(COMLYN,COMLEN,'LAST').NE.0) MODFL=MODFL+2
       ENDIF
       CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0)  THEN
          !         Begin Procedure CRAP-OUT
          CALL WRNDIE(0,'<COORIO>','Atom selection parsing error')
#if KEY_STRINGM==0 /*  VO stringm */
          if(allocated(space_rl1)) &
               call chmdealloc('coorio.src','COORIO','space_rl1',7*NATOM,crl=space_rl1)
#if KEY_CHEQ==1
          if(allocated(space_cg)) &             
#endif
#if KEY_CHEQ==1
               call chmdealloc('coorio.src','COORIO','space_cg',3*NATOM,crl=space_cg)   
#endif
#if KEY_PIPF==1
          if(allocated(space_pipf)) &          
#endif
#if KEY_PIPF==1
               call chmdealloc('coorio.src','COORIO','space_pipf', &  
#endif
#if KEY_PIPF==1
               3*N3uind,crl=space_pipf)   
#endif
          if(allocated(ifreea)) &
               call chmdealloc('coorio.src','COORIO','IFREEA',NATOM,intg=IFREEA)
          RETURN
#else
          goto 999
#endif
          !         End Procedure CRAP-OUT
       ENDIF

       deci = GTRMI(COMLYN,COMLEN,'DECI',-1)

       CALL CWRITE(IUNIT,TITLEB,NTITLB,ICNTRL,X,Y,Z,WMAIN, &
            RES,ATYPE,IBASE,NRES,NATOM,ISLCT,MODECW,MODEL, &
            MODFL,OFFICIAL,deci=deci)
       IF(MODECW.LT.3) CALL VCLOSE(IUNIT,'KEEP',ERROR)
    ENDIF
    !
#if KEY_STRINGM==1 /*  VO stringm */
 999  continue
    !
      if (qstr) iolev=oldiol ! VO : restore iolev
    !
#endif
    !
    if(allocated(space_rl1)) &
         call chmdealloc('coorio.src','COORIO','space_rl1',7*NATOM,crl=space_rl1)
#if KEY_CHEQ==1
    if(allocated(space_cg)) &          
#endif
#if KEY_CHEQ==1
         call chmdealloc('coorio.src','COORIO','space_cg',3*NATOM,crl=space_cg)   
#endif
#if KEY_PIPF==1
    if(allocated(space_pipf)) &          
#endif
#if KEY_PIPF==1
         call chmdealloc('coorio.src','COORIO','space_pipf',3*N3uind,crl=space_pipf)   
#endif
    if(allocated(ifreea)) &
         call chmdealloc('coorio.src','COORIO','IFREEA',NATOM,intg=IFREEA)
    call chmdealloc('coorio.src','COORIO','ISLCT',NATOM,intg=ISLCT)
    RETURN
  END SUBROUTINE COORIO


  SUBROUTINE CWRITE(IUNIT,TITLE,NTITL,ICNTRL,X,Y,Z,WMAIN, &
       RES,ATYPE,IBASE,NRES,NATOM,ISLCT,MODE,MODEL,MODFL, &
       OFFICIAL,deci)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE WRITES COORDINATE MODULES OR CARD FILES.
    !
    !     MODE = 1 FOR BINARY MODULES
    !     MODE = 2 FOR CARD FILES
    !     MODE = 3 FOR PRINT OUT
    !     MODE = 4 FOR PDB FORMAT  AB/LN JAN-85
    !              option to write nmr-style file with multiple models added
    !              MODEL=0 Standard PDBfile. MODEL=1 write header and first model
    !              MODEL>1 write one model. MODEL=N (N<0) write MODEN |N| and END line
    !                  FIRST|LAST keyword forces writing of header|END. Oct-03 (c31a1). L.Nilsson
    !     MODE = 5 FOR DUMB CARD OUTPUT
    !     MODE = 6 FOR .XYZ OUTPUT sept 2016  rick venable
    !
    !     Overhauled by Bernard R. Brooks   1983
    !
    use chm_kinds
    use dimens_fcm
    use fourdm
    use stream
    use string
    use image
    use number
    use parallel  ! mh050712
    use chutil,only:atomid
    use memory
    use machutil,only:die

    implicit none

    integer,optional :: deci
    INTEGER IUNIT,NTITL
    integer*4 ntitl4
    CHARACTER(len=*) TITLE(*)
    INTEGER ICNTRL(20)
    integer*4 icntrl4(20)
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    CHARACTER(len=*) RES(*),ATYPE(*)
    INTEGER IBASE(*),ISLCT(*)
    INTEGER NRES,NATOM,MODE,MODEL,MODFL
    real(chm_real4),allocatable,dimension(:) :: WW
    real(chm_real),allocatable,dimension(:,:) :: xyz
    LOGICAL OFFICIAL
    !
    CHARACTER(len=4) :: HDR='COOR'
    CHARACTER(len=8) SID,RID,REN,AC,ARID,ATYPEI
    character(len=60) fm2                          ! yw
    INTEGER NSLCT,IRES,IPT,I,L
    integer*4 nslct4
    LOGICAL QCRYS
    integer :: decip
    !
#if KEY_FOURD==1 /*4dsetw*/
    !     If a four-D minimization is requested then their coordinates
    !     are placed into WMAIN here to print in .crd file.
    !
    IF (DIM4) THEN
       DO I=1,NATOM
          WMAIN(I)=FDIM(I)
       ENDDO
    ENDIF
#endif /* (4dsetw)*/
    !
    QCRYS=(XTLTYP.NE.'    ')
    !
    IF(IOLEV.LT.0) RETURN
    !
    NSLCT=0
    DO I=1,NATOM
       IF(ISLCT(I).EQ.1) NSLCT=NSLCT+1
    ENDDO
    IF(NSLCT.EQ.0) THEN
       CALL WRNDIE(2,'<COORIO>', &
            'ZERO ATOMS SPECIFIED TO WRITE. NO FILE CREATED')
       RETURN
    ENDIF
    IF(NSLCT.LT.NATOM .AND. PRNLEV.GE.3) WRITE(OUTU,127)
127 FORMAT(' NOTE: A SELECTED SUBSET OF ATOMS WILL BE USED'/)
    !
    call chmalloc("coorio.src","cwrite","ww",natom,cr4=ww)

    IF (MODE.EQ.1) THEN
       !       Begin Procedure WRITE-BINARY-FILE
       IF(NSLCT.LT.NATOM .AND. PRNLEV.GE.2) WRITE(OUTU,135) NSLCT, &
            NATOM
135    FORMAT(/' **** INFO ***** IN CWRITE. BINARY MODULE WRITTEN ', &
            'WITH ONLY A PARTIAL SET OF ATOMS.'/,' NSLCT=',I5,' NATOM=' &
            ,I5)
       DO I=1,20
          ICNTRL(I)=0
       ENDDO
       ICNTRL(1)=1
       IF(QCRYS) ICNTRL(11)=1
       WRITE(IUNIT) HDR,ICNTRL
       CALL WRTITL(TITLE,NTITL,IUNIT,-1)
       WRITE(IUNIT) NSLCT
       !
       IF(QCRYS) WRITE(IUNIT) XTLABC
       !
       !       FILL W ARRAYS BASED ON SELECTED ATOMS ONLY
       !
       IPT=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) THEN
             IPT=IPT+1
             WW(IPT)=X(I)
          ENDIF
       ENDDO
       WRITE(IUNIT) (WW(I),I=1,NSLCT)
       !
       IPT=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) THEN
             IPT=IPT+1
             WW(IPT)=Y(I)
          ENDIF
       ENDDO
       WRITE(IUNIT) (WW(I),I=1,NSLCT)
       !
       IPT=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) THEN
             IPT=IPT+1
             WW(IPT)=Z(I)
          ENDIF
       ENDDO
       WRITE(IUNIT) (WW(I),I=1,NSLCT)
       !
       IPT=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) THEN
             IPT=IPT+1
             WW(IPT)=WMAIN(I)
          ENDIF
       ENDDO
       WRITE(IUNIT) (WW(I),I=1,NSLCT)
       !
       !       End Procedure WRITE-BINARY-FILE
    ELSE IF (MODE.EQ.2 .OR. MODE.EQ.3) THEN
       IF(MODE.EQ.3) THEN
          ! print to the output
          WRITE(IUNIT,'(/10X,A)') 'COORDINATE FILE MODULE'
          CALL WRTITL(TITLE,NTITL,IUNIT,1)
       ELSEIF (MODE.EQ.2) THEN
          ! write to the file
          CALL WRTITL(TITLE,NTITL,IUNIT,0)
       ENDIF
       !       Begin Procedure WRITE-CARD-FILE
       !yw++
       qextfmt=qxform()
       if(qextfmt) then
          write(iunit,'(i10,2x,a)') nslct,'EXT'
          fm2='(2I10,2X,A8,2X,A8,3F20.10,2X,A8,2X,A8,F20.10)'
       else
          write(iunit,'(i5)') nslct
          fm2='(2I5,1X,A4,1X,A4,3F10.5,1X,A4,1X,A4,F10.5)'
       endif

       decip = -1
       if (present(deci)) decip = deci
       if (decip >= 0) then
          call chmalloc("coorio.src","cwrite","xyz",3,natom,xyz)
          call fillxyz()
          DO IRES=1,NRES
             DO I=IBASE(IRES)+1,IBASE(IRES+1)
                IF(ISLCT(I).EQ.1) THEN
                   CALL ATOMID(I,SID,RID,REN,AC)
                   WRITE(IUNIT,fm2) &
                        I,IRES,RES(IRES),ATYPE(I),Xyz(1,I),xYz(2,I),xyZ(3,I),SID,RID,WMAIN(I)
                ENDIF
             ENDDO
          ENDDO
          call chmdealloc("coorio.src","cwrite","xyz",3,natom,xyz)
       else
          DO IRES=1,NRES
             DO I=IBASE(IRES)+1,IBASE(IRES+1)
                IF(ISLCT(I).EQ.1) THEN
                   CALL ATOMID(I,SID,RID,REN,AC)
                   WRITE(IUNIT,fm2) &
                        I,IRES,RES(IRES),ATYPE(I),X(I),Y(I),Z(I),SID,RID,WMAIN(I)
                ENDIF
             ENDDO
          ENDDO
       endif
       !       End Procedure WRITE-CARD-FILE
    ELSE IF (MODE.EQ.4) THEN
       !
       ! write PDB title
       !
       IF(MODEL.EQ.0 .OR. MODEL .EQ.1   &
            .OR. MODFL.EQ.1 .OR. MODFL.EQ.3)THEN
          CALL WRTITL(TITLE,NTITL,0,2)
          WRITE(IUNIT,'(A,A)') ('REMARK ',TITLE(I)(2:),I=1,NTITL)
       ENDIF
       !       Begin Procedure WRITE-PDB-FILE
       !
       ! use Brookhaven PDB format
       !
       !TOM   1223  O   GLY   153     -11.704  -9.200    .489  1.00  0.80
       !     ccccc ''''Iyyy O,,,,L   ........>>>>>>>>////////ppppppiiiiii iii
       !                         ^ insertion character
       !                    ^ chain identifier
       !           ^ additional character for some atom names (mostly h's)
       !
       ! adjust resid's so that they are as close as possible to the original
       ! PDB format
       !
       IF(MODEL .NE. 0) WRITE(IUNIT,'(A,I9)') 'MODEL',IABS(MODEL)
       DO IRES=1,NRES
          DO I=IBASE(IRES)+1,IBASE(IRES+1)
             IF(ISLCT(I).EQ.1) THEN
                CALL ATOMID(I,SID,RID,REN,AC)
                !              L=4
                !              CALL TRIME(RID,L)
                !              ARID='    '
                !              IF (L.EQ.4.OR.RID(L:L).GE.'A') THEN
                !                ARID(4-L+1:4)=RID(1:L)
                !              ELSE
                !                ARID(4-L:3)=RID(1:L)
                !              ENDIF
                ! Allow 5 character RESID (4 char resSeq + 1 char insertion code)
                L=5
                CALL TRIME(RID,L)
                ARID='    '
                IF (L.EQ.5.OR.RID(L:L).GE.'A') THEN
                   ARID(5-L+1:5)=RID(1:L)
                ELSE
                   ARID(5-L:4)=RID(1:L)
                ENDIF
                ! shift atom names when they exceed 3 characters
                IF (ATYPE(I)(4:4).EQ.' ') THEN
                   ATYPEI=' '//ATYPE(I)(1:3)
                ELSE
                   ATYPEI=ATYPE(I)
                ENDIF
                ! the SEGID is written to the last four characters of the line
                !brb..07-FEB-99 Change default occupancy from zero to one
                ! Format correction. L. Nilsson, November 07
                ! Previous format:  (A,I5,1X,A4,1X,A3,1X,A1,1X,A3,4X,3F8.3,6X,F6.2,6X,A4)
                IF(OFFICIAL)THEN
                   WRITE(IUNIT, &
                        '(A6,I5,1X,A4,1X,A3,1X,A1,A5,3X,3F8.3,2F6.2,6X,A4)') &
                        'ATOM  ',I,ATYPEI,REN,SID,ARID,X(I),Y(I),Z(I),1.0,WMAIN(I),SID
                ELSE
                   WRITE(IUNIT, &
                        '(A6,I5,1X,A4,1X,A4,1X,   A5,3X,3F8.3,2F6.2,6X,A4)') &
                        'ATOM  ',I,ATYPEI,REN,ARID,X(I),Y(I),Z(I),1.0,WMAIN(I) &
                        ,SID
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       WRITE(IUNIT,'(A3,I8,6X,A4,2X,A4)') 'TER',NATOM+1,REN,ARID
       ! write END statement for PDB file
       IF(MODEL .NE. 0) WRITE(IUNIT,'(A)') 'ENDMDL'
       IF(MODEL.EQ.0 .or. MODEL .LE. 0 .OR. MODFL.GE.2)  &
            WRITE(IUNIT,'(A)') 'END'
       !       End Procedure WRITE-PDB-FILE
    ELSE IF (MODE.EQ.5) THEN
       !       DUMB CARD OUTPUT
       DO I=1,NATOM
          IF(ISLCT(I).NE.0) THEN
             WRITE(IUNIT,27) X(I),Y(I),Z(I)
27           FORMAT(4F12.6)
          ENDIF
       ENDDO
    ELSE IF (MODE.EQ.6) THEN
       !       .xyz output
       WRITE(IUNIT,'(I6)') NSLCT   ! atom count
       WRITE(IUNIT,'(A)') TITLE(1) ! first title line
       DO I=1,NATOM
          IF(ISLCT(I).NE.0) THEN
             WRITE(IUNIT,28) ATYPE(I),X(I),Y(I),Z(I)
28           FORMAT(A8,1X,3F11.5)
          ENDIF
       ENDDO
    ELSE
       CALL DIE
    ENDIF
    !
    call chmdealloc("coorio.src","cwrite","ww",natom,cr4=ww)
    RETURN

  contains
    subroutine fillxyz
      real(chm_real) :: x0,y0,z0,fac

      fac = ten**deci
      do i=1,natom
         x0 = anint(x(i)*fac)
         y0 = anint(y(i)*fac)
         z0 = anint(z(i)*fac)
         xyz(1,i)=x0/fac
         xyz(2,i)=y0/fac
         xyz(3,i)=z0/fac
      enddo
      return
    end subroutine fillxyz
  END SUBROUTINE CWRITE

  SUBROUTINE CWRITE2(IUNIT,TITLE,NTITL,ICNTRL,X,Y,Z,WMAIN, &
       SEGID,RES,RESID,ATYPE,NICTOT,IBASE,NSEG,NRES,NATOM,ISLCT, &
       MODE,MODEL,MODFL,OFFICIAL,deci)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE WRITES COORDINATE MODULES OR CARD FILES.
    !     DIFFERENCE WITH CWRITE IS THE ADDITION OF NEW ARGUMENTS,
    !     SO IT WON'T USE ANY INFORMATION FROM THE PSF MODULE.
    !
    !     MODE = 1 FOR BINARY MODULES
    !     MODE = 2 FOR CARD FILES
    !     MODE = 3 FOR PRINT OUT
    !     MODE = 4 FOR PDB FORMAT  AB/LN JAN-85
    !              option to write nmr-style file with multiple models added
    !              MODEL=0 Standard PDBfile. MODEL=1 write header and first model
    !              MODEL>1 write one model. MODEL=N (N<0) write MODEN |N| and END line
    !                  FIRST|LAST keyword forces writing of header|END. Oct-03 (c31a1). L.Nilsson
    !     MODE = 5 FOR DUMB CARD OUTPUT
    !
    !    Nathan Desdouits & Arnaud Blondel 4/2012 
    !
    use chm_kinds
    use dimens_fcm
    use fourdm
    use stream
    use string
    use image
    use number
    use parallel  ! mh050712
    use chutil
    use memory
    use machutil,only:die

    implicit none

    integer,optional :: deci
    INTEGER IUNIT,NTITL
    integer*4 ntitl4
    CHARACTER(len=*) TITLE(*)
    INTEGER ICNTRL(20)
    integer*4 icntrl4(20)
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    CHARACTER(len=*) SEGID(*),RES(*),RESID(*),ATYPE(*)
    INTEGER IBASE(*),ISLCT(*),NICTOT(*)
    INTEGER NSEG,NRES,NATOM,MODE,MODEL,MODFL
    real(chm_real4),allocatable,dimension(:) :: WW
    real(chm_real),allocatable,dimension(:,:) :: xyz
    LOGICAL OFFICIAL
    !
    CHARACTER(len=4) :: HDR='COOR'
    CHARACTER(len=8) SID,RID,REN,AC,ARID,ATYPEI
    character(len=60) fm2                          ! yw
    INTEGER NSLCT,IRES,IPT,I,L
    integer*4 nslct4
    LOGICAL QCRYS
    integer :: decip
    !
#if KEY_FOURD==1 /*4dsetw*/
    !     If a four-D minimization is requested then their coordinates
    !     are placed into WMAIN here to print in .crd file.
    !
    IF (DIM4) THEN
       DO I=1,NATOM
          WMAIN(I)=FDIM(I)
       ENDDO
    ENDIF
#endif /* (4dsetw)*/
    !
    QCRYS=(XTLTYP.NE.'    ')
    !
#if KEY_ENSEMBLE==0
    IF(IOLEV.LT.0) RETURN
#endif 
    !
    NSLCT=0
    DO I=1,NATOM
       IF(ISLCT(I).EQ.1) NSLCT=NSLCT+1
    ENDDO
    IF(NSLCT.EQ.0) THEN
       CALL WRNDIE(2,'<COORIO>', &
            'ZERO ATOMS SPECIFIED TO WRITE. NO FILE CREATED')
       RETURN
    ENDIF
    IF(NSLCT.LT.NATOM .AND. PRNLEV.GE.3) WRITE(OUTU,127)
127 FORMAT(' NOTE: A SELECTED SUBSET OF ATOMS WILL BE USED'/)
    !
    call chmalloc("coorio.src","cwrite","ww",natom,cr4=ww)

    IF (MODE.EQ.1) THEN
       !       Begin Procedure WRITE-BINARY-FILE
       IF(NSLCT.LT.NATOM .AND. PRNLEV.GE.2) WRITE(OUTU,135) NSLCT, &
            NATOM
135    FORMAT(/' **** INFO ***** IN CWRITE2. BINARY MODULE WRITTEN ', &
            'WITH ONLY A PARTIAL SET OF ATOMS.'/,' NSLCT=',I5,' NATOM=' &
            ,I5)
       DO I=1,20
          ICNTRL(I)=0
       ENDDO
       ICNTRL(1)=1
       IF(QCRYS) ICNTRL(11)=1
       WRITE(IUNIT) HDR,ICNTRL
       CALL WRTITL(TITLE,NTITL,IUNIT,-1)
       WRITE(IUNIT) NSLCT
       !
       IF(QCRYS) WRITE(IUNIT) XTLABC
       !
       !       FILL W ARRAYS BASED ON SELECTED ATOMS ONLY
       !
       IPT=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) THEN
             IPT=IPT+1
             WW(IPT)=X(I)
          ENDIF
       ENDDO
       WRITE(IUNIT) (WW(I),I=1,NSLCT)
       !
       IPT=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) THEN
             IPT=IPT+1
             WW(IPT)=Y(I)
          ENDIF
       ENDDO
       WRITE(IUNIT) (WW(I),I=1,NSLCT)
       !
       IPT=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) THEN
             IPT=IPT+1
             WW(IPT)=Z(I)
          ENDIF
       ENDDO
       WRITE(IUNIT) (WW(I),I=1,NSLCT)
       !
       IPT=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1) THEN
             IPT=IPT+1
             WW(IPT)=WMAIN(I)
          ENDIF
       ENDDO
       WRITE(IUNIT) (WW(I),I=1,NSLCT)
       !
       !       End Procedure WRITE-BINARY-FILE
    ELSE IF (MODE.EQ.2 .OR. MODE.EQ.3) THEN
       IF(MODE.EQ.3) THEN
          ! print to the output
          WRITE(IUNIT,'(/10X,A)') 'COORDINATE FILE MODULE'
          CALL WRTITL(TITLE,NTITL,IUNIT,1)
       ELSEIF (MODE.EQ.2) THEN
          ! write to the file
          CALL WRTITL(TITLE,NTITL,IUNIT,0)
       ENDIF
       !       Begin Procedure WRITE-CARD-FILE
       !yw++
       qextfmt=qxform()
       if(qextfmt) then
          write(iunit,'(i10,2x,a)') nslct,'EXT'
          fm2='(2I10,2X,A8,2X,A8,3F20.10,2X,A8,2X,A8,F20.10)'
       else
          write(iunit,'(i5)') nslct
          fm2='(2I5,1X,A4,1X,A4,3F10.5,1X,A4,1X,A4,F10.5)'
       endif

       decip = -1
       if (present(deci)) decip = deci
       if (decip >= 0) then
          call chmalloc("coorio.src","cwrite","xyz",3,natom,xyz)
          call fillxyz()
          DO IRES=1,NRES
             DO I=IBASE(IRES)+1,IBASE(IRES+1)
                IF(ISLCT(I).EQ.1) THEN
                   CALL ATOMID2(I,SID,RID,REN,AC, &
                        SEGID,RES,RESID,ATYPE,NICTOT,IBASE,NSEG,NRES)
                   WRITE(IUNIT,fm2) &
                        I,IRES,RES(IRES),ATYPE(I),Xyz(1,I),xYz(2,I),xyZ(3,I),SID,RID,WMAIN(I)
                ENDIF
             ENDDO
          ENDDO
          call chmdealloc("coorio.src","cwrite","xyz",3,natom,xyz)
       else
          DO IRES=1,NRES
             DO I=IBASE(IRES)+1,IBASE(IRES+1)
                IF(ISLCT(I).EQ.1) THEN
                   CALL ATOMID2(I,SID,RID,REN,AC, &
                        SEGID,RES,RESID,ATYPE,NICTOT,IBASE,NSEG,NRES)
                   WRITE(IUNIT,fm2) &
                        I,IRES,RES(IRES),ATYPE(I),X(I),Y(I),Z(I),SID,RID,WMAIN(I)
                ENDIF
             ENDDO
          ENDDO
       endif
       !       End Procedure WRITE-CARD-FILE
    ELSE IF (MODE.EQ.4) THEN
       !
       ! write PDB title
       !
       IF(MODEL.EQ.0 .OR. MODEL .EQ.1   &
            .OR. MODFL.EQ.1 .OR. MODFL.EQ.3)THEN
          CALL WRTITL(TITLE,NTITL,0,2)
          WRITE(IUNIT,'(A,A)') ('REMARK ',TITLE(I)(2:),I=1,NTITL)
       ENDIF
       !       Begin Procedure WRITE-PDB-FILE
       !
       ! use Brookhaven PDB format
       !
       !TOM   1223  O   GLY   153     -11.704  -9.200    .489  1.00  0.80
       !     ccccc ''''Iyyy O,,,,L   ........>>>>>>>>////////ppppppiiiiii iii
       !                         ^ insertion character
       !                    ^ chain identifier
       !           ^ additional character for some atom names (mostly h's)
       !
       ! adjust resid's so that they are as close as possible to the original
       ! PDB format
       !
       IF(MODEL .NE. 0) WRITE(IUNIT,'(A,I9)') 'MODEL',IABS(MODEL)
       DO IRES=1,NRES
          DO I=IBASE(IRES)+1,IBASE(IRES+1)
             IF(ISLCT(I).EQ.1) THEN
                CALL ATOMID2(I,SID,RID,REN,AC, &
                     SEGID,RES,RESID,ATYPE,NICTOT,IBASE,NSEG,NRES)
                !              L=4
                !              CALL TRIME(RID,L)
                !              ARID='    '
                !              IF (L.EQ.4.OR.RID(L:L).GE.'A') THEN
                !                ARID(4-L+1:4)=RID(1:L)
                !              ELSE
                !                ARID(4-L:3)=RID(1:L)
                !              ENDIF
                ! Allow 5 character RESID (4 char resSeq + 1 char insertion code)
                L=5
                CALL TRIME(RID,L)
                ARID='    '
                IF (L.EQ.5.OR.RID(L:L).GE.'A') THEN
                   ARID(5-L+1:5)=RID(1:L)
                ELSE
                   ARID(5-L:4)=RID(1:L)
                ENDIF
                ! shift atom names when they exceed 3 characters
                IF (ATYPE(I)(4:4).EQ.' ') THEN
                   ATYPEI=' '//ATYPE(I)(1:3)
                ELSE
                   ATYPEI=ATYPE(I)
                ENDIF
                ! the SEGID is written to the last four characters of the line
                !brb..07-FEB-99 Change default occupancy from zero to one
                ! Format correction. L. Nilsson, November 07
                ! Previous format:  (A,I5,1X,A4,1X,A3,1X,A1,1X,A3,4X,3F8.3,6X,F6.2,6X,A4)
                IF(OFFICIAL)THEN
                   WRITE(IUNIT, &
                        '(A6,I5,1X,A4,1X,A3,1X,A1,A5,3X,3F8.3,2F6.2,6X,A4)') &
                        'ATOM  ',I,ATYPEI,REN,SID,ARID,X(I),Y(I),Z(I),1.0,WMAIN(I),SID
                ELSE
                   WRITE(IUNIT, &
                        '(A6,I5,1X,A4,1X,A4,1X,   A5,3X,3F8.3,2F6.2,6X,A4)') &
                        'ATOM  ',I,ATYPEI,REN,ARID,X(I),Y(I),Z(I),1.0,WMAIN(I) &
                        ,SID
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       WRITE(IUNIT,'(A3,I8,6X,A4,2X,A4)') 'TER',NATOM+1,REN,ARID
       ! write END statement for PDB file
       IF(MODEL .NE. 0) WRITE(IUNIT,'(A)') 'ENDMDL'
       IF(MODEL.EQ.0 .or. MODEL .LE. 0 .OR. MODFL.GE.2)  &
            WRITE(IUNIT,'(A)') 'END'
       !       End Procedure WRITE-PDB-FILE
    ELSE IF (MODE.EQ.5) THEN
       !       DUMB CARD OUTPUT
       DO I=1,NATOM
          IF(ISLCT(I).NE.0) THEN
             WRITE(IUNIT,27) X(I),Y(I),Z(I)
27           FORMAT(4F12.6)
          ENDIF
       ENDDO
    ELSE
       CALL DIE
    ENDIF
    !
    call chmdealloc("coorio.src","cwrite","ww",natom,cr4=ww)
    RETURN

  contains
    subroutine fillxyz
      real(chm_real) :: x0,y0,z0,fac

      fac = ten**deci
      do i=1,natom
         x0 = anint(x(i)*fac)
         y0 = anint(y(i)*fac)
         z0 = anint(z(i)*fac)
         xyz(1,i)=x0/fac
         xyz(2,i)=y0/fac
         xyz(3,i)=z0/fac
      enddo
      return
    end subroutine fillxyz
  END SUBROUTINE CWRITE2

  subroutine cread(iunit,title,ntitl,icntrl,x,y,z,wmain,natom, &
       ninput,islct,ioffs,res,nres,atype,ibase, &
       ifile,freeat,segid,resid,nictot,nseg,lrsid,lfree,lyn,mxlen, &
       model,official,nchain_)
    !-----------------------------------------------------------------------
    !     COORDINATE READING ROUTINES CARD READING SECTION MODIFIED TO
    !     MAP COORDINATES BY THE SEQUENCE NUMBER, RESIDUE TYPE, AND ATOM
    !     TYPE IN THE INPUT FILE, AND TO CHECK FOR SEQUENCE CONSISTENCY:
    !
    !             SEQUENCE ERRORS ARE FATAL
    !             MISSING COORDINATES RESULT IN WARNINGS
    !             MULTIPLE COORIDNATES RESULT IN WARNINGS
    !             UNFOUND ATOMS ARE IGNORED
    !            Residues out of range result in warnings
    !
    !
    !     MODIFIED TO INCLUDE THE DUMB CARD READING OPTION
    !
    !
    !     FREEAT IS USED TO STORE THE FREEAT ARRAY THAT IS READ FROM A
    !     DYNAMICS TRAJECTORY FILE. IT IS A WORK ARRAY ONLY.
    !
    !     A SIMPLE PDB FORMAT READ OPTION (NINPUT=-1) ADDED AB/LN JAN-85
    !
    !     Modified to allow continued reading of selected coordinate
    !     sets from a trajectory w/o getting the file rewound all the time.
    !     This case is signalled by NINPUT=-3. April 1987 /LN
    !

    !     Overhauled by Bernard R. Brooks   1983
    !
    use chm_kinds
    use dimens_fcm
    use fourdm
    use number
    use memory
    use stream
    use string
    use image
    use parallel  ! mh050712
    use chutil,only:initia,matom

    implicit none
    INTEGER IUNIT,NTITL
    CHARACTER(len=*) TITLE(*)
    INTEGER ICNTRL(20)
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
    INTEGER NATOM,NINPUT,IFILE
    real(chm_real4),allocatable,dimension(:),target :: space_w
    real(chm_real4),pointer,dimension(:) :: WX,WY,WZ,WW
    INTEGER IBASE(*),ISLCT(*)
    INTEGER IOFFS,NRES,NSEG
    CHARACTER(len=*) RESID(*),SEGID(*),RES(*),ATYPE(*)
    CHARACTER(len=8) SID,RID,RESIN,ATOMIN
    INTEGER FREEAT(*),NICTOT(*)
    LOGICAL LRSID,LFREE
    CHARACTER(len=*) LYN
    !
    ! VO 9/2014 :  make nchain optional for compatibility with older code
    integer, optional, intent(in) :: nchain_
    !
    INTEGER MXLEN,MODEL,NCHAIN
    LOGICAL OFFICIAL

    !
    CHARACTER(len=4) HDR
    CHARACTER(len=140) PDBLIN
    character(len=40)  fmt40,fmt50
    character(len=10)  fmt10
    logical lextfmt
    integer ilen

    INTEGER ERRCNT,NFREAT
    real(chm_real)  XIN,YIN,ZIN,WIN
    CHARACTER(len=20) CXIN,CYIN,CZIN,CWIN
    LOGICAL EOF, QATOM
    LOGICAL DYN, QBIN, QCONT, QCRYS
    INTEGER SLEN, IJ
    INTEGER NSLCT,IATOM,NMULT,ISRES,NRNG,NSEQM,NDESL,ISEQ,IRES,ISEG
    integer*4 iatom4
    INTEGER ISTP,IPOINT,NMISS,IPT,I,II
    INTEGER MDL
    real(chm_real),allocatable,dimension(:) :: transf
    integer,allocatable,dimension(:) :: itmpint4

    integer rifile, nfile
    save    rifile, nfile, qcrys
    !
    ! VO 9/2014 :  make nchain optional for compatibility with older code
    if (present(nchain_)) then ; nchain=nchain_ ; else ; nchain=0 ; endif
    !
    eof=.false.
    nslct=0
    do i=1,natom
       if(islct(i).eq.1) nslct=nslct+1
    enddo
    if(nslct.eq.0) then
       call wrndie(1,'<COORIO>','ZERO ATOMS SPECIFIED IN SELECTION')
       return
    endif
    if(nslct.lt.natom .and. prnlev.ge.2) write(outu,127)
127 format(' INFO: A subset of total atoms will be read.'/)
    errcnt=0
    !++
    ! LN ADD /APR 90: FLAG BINARY AND BINARY W/O REWIND
    !
    qbin=.false.
    qcont=.false.
    if(ninput .eq. 0) qbin=.true.
    if(ninput .eq. -3) then
       qbin=.true.
       qcont=.true.
    endif
    !

    call chmalloc('coorio.src','COORIO','space_W',4*NATOM+4,cr4=space_W)
    WX => space_w(0*natom+1 : 1*natom+1)
    WY => space_w(1*(natom+1) : 2*(natom+1))
    WZ => space_w(2*(natom+1) : 3*(natom+1))
    WW => space_w(3*(natom+1) : 4*(natom+1))

    if( (iolev.gt.0) ) then


       if( qbin ) then
          !
          !     READ COORDINATES FROM A COORDINATE FILE MODULE
          !     OR THE 'IFILE' COORDINATE SET FROM A DYNAMICS 'CORD' FILE
          !
          if (ifile.lt.0) then
             iatom=natom
             if(qcrys) read(iunit) xtlabc
             read(iunit) (wx(i),i=1,iatom)
             read(iunit) (wy(i),i=1,iatom)
             read(iunit) (wz(i),i=1,iatom)
             do i=1,iatom
                ww(i)=0.0
             enddo
             goto 138
          endif
          if( .not. qcont) then
             !
             call tryoro(iunit,'UNFORMATTED')
             read(iunit) hdr,icntrl
             if (hdr.eq.'COOR') then
                DYN=.FALSE.
             ELSE IF (HDR.EQ.'CORD') THEN
                DYN=.TRUE.
             ELSE IF (HDR.EQ.'VELD') THEN
                DYN=.TRUE.
             ELSE
                DYN=.FALSE.
                CALL WRNDIE(-1,'<CREAD>','HEADERS DONT MATCH')
             ENDIF
             CALL RDTITL(TITLE,NTITL,IUNIT,-1)
             CALL WRTITL(TITLE,NTITL,OUTU,1)
             READ(IUNIT) IATOM
             NFREAT=IATOM-ICNTRL(9)
             QCRYS=(ICNTRL(11).EQ.1)
             IF(IATOM.NE.NSLCT) THEN
                IF(WRNLEV.GE.2) WRITE(OUTU,135) IATOM,NSLCT
135             FORMAT(/' ** WARNING ** Number of atoms in binary', &
                     ' file does not match the number of selected atoms.'/, &
                     ' IATOM=',I5,' NSLCT=',I5)
                CALL DIEWRN(0)
             ENDIF
             IF(IATOM.GT.NATOM) IATOM=NATOM
             IF(NFREAT.GT.NATOM) NFREAT=NATOM
          ENDIF
          IF (.NOT. DYN) THEN
             IF(QCRYS) READ(IUNIT) XTLABC
             READ(IUNIT) (WX(I),I=1,IATOM)
             READ(IUNIT) (WY(I),I=1,IATOM)
             READ(IUNIT) (WZ(I),I=1,IATOM)
             READ(IUNIT,ERR=137,END=137) (WW(I),I=1,IATOM)
             GOTO 136
137          CONTINUE
             DO I=1,IATOM
                WW(I)=0.0
             ENDDO
136          CONTINUE
          ELSE
             IF (QCONT) THEN
                !
                ! This option does not work when there are fixed atoms
                !
                IF(NFREAT .NE. IATOM) THEN
                   CALL WRNDIE(0,'<CREAD>', &
                        'Cannot CONTinue reading trajectory w/ fixed atoms')
                   GOTO 900
                ENDIF
                IF (IFILE .LT. 1) IFILE = 1
                NFILE=NFILE+IFILE
                IF(NFILE .GT. RIFILE) THEN
                   IF(WRNLEV.GE.2) WRITE(OUTU,1000) RIFILE,IFILE
                   CALL DIEWRN(1)
                   NFILE=NFILE-IFILE
                   IFILE=RIFILE-NFILE
                   NFILE=RIFILE
                ENDIF
                IF(PRNLEV.GE.2) WRITE (OUTU,200) NFILE
                !
                ! ln MOD TO GET THIS TO WORK ON MACHINES W/O RECORD CONCEPT:
                !
                DO I=2,IFILE
                   IF(QCRYS) READ(IUNIT)
                   READ(IUNIT)
                   READ(IUNIT)
                   READ(IUNIT)
                ENDDO
                IF(QCRYS) READ(IUNIT) XTLABC
                READ(IUNIT) (WX(I),I=1,IATOM)
                READ(IUNIT) (WY(I),I=1,IATOM)
                READ(IUNIT) (WZ(I),I=1,IATOM)
                DO I=1,IATOM
                   WW(I)=0.0
                ENDDO
             ELSE
                RIFILE=ICNTRL(1)
                IF(RIFILE.LT.IFILE) THEN
                   IF(WRNLEV.GE.2) WRITE(OUTU,1000) RIFILE,IFILE
1000               FORMAT(/' ** WARNING ** IFILE is too big,  File has ',i9, &
                        '  IFILE = ',I9/' IFILE will be set to last set on file.')
                   CALL DIEWRN(1)
                   IFILE=RIFILE
                ENDIF
                IF(IFILE.LT.1) IFILE=1
                NFILE=IFILE
                IF(PRNLEV.GE.2) WRITE(OUTU,200) IFILE
200             FORMAT(' Reading from coordinate trajectory, IFILE = ',I9)
                IF (NFREAT.EQ.IATOM) THEN
                   DO I=2,IFILE
                      IF(QCRYS) READ(IUNIT)
                      READ(IUNIT)
                      READ(IUNIT)
                      READ(IUNIT)
                   ENDDO
                   IF(QCRYS) READ(IUNIT) XTLABC
                   READ(IUNIT) (WX(I),I=1,IATOM)
                   READ(IUNIT) (WY(I),I=1,IATOM)
                   READ(IUNIT) (WZ(I),I=1,IATOM)
                   DO I=1,IATOM
                      WW(I)=0.0
                   ENDDO
                ELSE
                   READ(IUNIT) (FREEAT(I),I=1,NFREAT)
                   IF(QCRYS) READ(IUNIT) XTLABC
                   READ(IUNIT) (WX(I),I=1,IATOM)
                   READ(IUNIT) (WY(I),I=1,IATOM)
                   READ(IUNIT) (WZ(I),I=1,IATOM)
                   DO I=1,IATOM
                      WW(I)=1.0
                   ENDDO
                   IF (IFILE.GT.1) THEN
                      DO I=1,NFREAT
                         IF(FREEAT(I).GT.NATOM) FREEAT(I)=NATOM+1
                      ENDDO
                      DO I=3,IFILE
                         IF(QCRYS) READ(IUNIT)
                         READ(IUNIT)
                         READ(IUNIT)
                         READ(IUNIT)
                      ENDDO
                      IF(QCRYS) READ(IUNIT) XTLABC
                      READ(IUNIT) (WX(FREEAT(I)),I=1,NFREAT)
                      READ(IUNIT) (WY(FREEAT(I)),I=1,NFREAT)
                      READ(IUNIT) (WZ(FREEAT(I)),I=1,NFREAT)
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
138       CONTINUE
          IPT=0
          NMULT=0
          DO I=1,NATOM
             IF(ISLCT(I).EQ.1 .AND. IPT.LT.IATOM) THEN
                IPT=IPT+1
                IF (INITIA(I,X,Y,Z)) NMULT=NMULT+1
                X(I)=WX(IPT)
                Y(I)=WY(IPT)
                Z(I)=WZ(IPT)
                WMAIN(I)=WW(IPT)
#if KEY_FOURD==1 /*4dread*/
                IF (DIM4) THEN
                   FDIM(I)=WW(IPT)
                   WMAIN(I)=0.0
                ENDIF
#endif /* (4dread)*/
             ENDIF
          ENDDO
          IF (NMULT.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,55) NMULT
          GOTO 900
       ENDIF
       !
       CALL TRYORO(IUNIT,'FORMATTED')
       IF(NINPUT.GT.1) GOTO 90
       !
       ! READ COORDINATES FROM CARDS
       !
       ! CHARMM FORMAT OR PDB FORMAT
       ! NINPUT=-1 MEANS PDB FORMAT.
       !
       IF (NINPUT.EQ.-1) THEN
          !
          ! read PDB title
          !
          NTITL=0
99963     CONTINUE
          READ(IUNIT,'(A)',END=61,ERR=61) PDBLIN
          !ln...05-Jan-95, convert the string to upper case
          SLEN=LEN(PDBLIN)
          CALL CNVTUC(PDBLIN,SLEN)
          !ln...(1)
          IF (PDBLIN(1:6).EQ.'REMARK') THEN
             NTITL=NTITL+1
             TITLE(NTITL)=PDBLIN(8:80)
             GOTO 99963
          ENDIF
          CALL WRTITL(TITLE,NTITL,OUTU,+1)
          ! LNI look for NMR MODEL
          IF(MODEL.GT.0)THEN
700          CONTINUE
             IF(PDBLIN(1:5).EQ. 'MODEL') THEN
                READ(PDBLIN(6:14),'(I9)') MDL
                IF(MDL.EQ.MODEL)THEN
                   IF(PRNLEV.GE.2) WRITE(OUTU,'(/A,I8/)')  &
                        'Found model #',MODEL
                   GOTO 702
                ELSE
                   READ(IUNIT,'(A)',END=61,ERR=61) PDBLIN
                   SLEN=LEN(PDBLIN)
                   CALL CNVTUC(PDBLIN,SLEN)
                   GOTO 700
                ENDIF
             ELSE
                READ(IUNIT,'(A)',END=61,ERR=61) PDBLIN
                SLEN=LEN(PDBLIN)
                CALL CNVTUC(PDBLIN,SLEN)
                GOTO 700
             ENDIF
          ENDIF
702       CONTINUE
          !!!! Now look for the specified chain number NCHAIN
          IF(NCHAIN > 1) THEN
            I=1  ! indicates chain currently being read
            DO WHILE (I < NCHAIN)
              DO WHILE(PDBLIN(1:3) /='TER')
                READ(IUNIT,'(A)',END=61,ERR=61) PDBLIN
                SLEN=LEN(PDBLIN)
                CALL CNVTUC(PDBLIN,SLEN)
              ENDDO
              I=I+1
            ENDDO
            READ(IUNIT,'(A)',END=61,ERR=61) PDBLIN
            SLEN=LEN(PDBLIN)
            CALL CNVTUC(PDBLIN,SLEN)
            IF(PRNLEV.GE.2) WRITE(OUTU,'(/A,I8/)')  &
                        'Found chain #',NCHAIN
            IF(PDBLIN(1:3) == 'TER' .OR. PDBLIN(1:3) == 'END') THEN
              CALL WRNDIE(-2,'<CREAD>','Chain seems to be empty')
            ENDIF
          ENDIF
       ELSE
          CALL RDTITL(TITLE,NTITL,IUNIT,0)
       ENDIF
       !
       lextfmt=.false.
       IF (NINPUT .LT. 0) THEN
          IATOM=0
       ELSE
          !yw++ 28-Jan-2003 use PDBLIN to process a line
          !yw       READ(IUNIT,30) IATOM
          read(iunit,'(a)') pdblin
          slen=len(pdblin)
          call cnvtuc(pdblin,slen)
          iatom=nexti(pdblin,slen)
          lextfmt=indxa(pdblin,slen,'EXT').gt.0
          if (iatom.ge.100000) lextfmt=.true.
          !yw--
       ENDIF
       IF(IATOM.EQ.0) THEN
          IATOM=99999999
          IF (NINPUT.NE.-1 .AND. WRNLEV.GE.2) WRITE(OUTU,32)
       ENDIF
32     FORMAT(' ** No atom count specified in card file.', &
            ' Will read atoms until EOF is reached. **')

       if (lextfmt) then
          fmt40='(2I10,2(2X,A8),3A20,2X,A8,2X,A8,A20)'
          fmt10='(F20.10)'
          ilen=8
       else
          fmt40='(2I5,2(1X,A4),3A10,1X,A4,1X,A4,A10)'
          fmt10='(F10.5)'
          ilen=4
       endif

       ISRES=IOFFS+1
       !
       NRNG=0
       NSEQM=0
       NMULT=0
       NDESL=0
       loop60: DO  I=1,IATOM
          !
          IF (LFREE) THEN
             !
             ! FREE FIELD INPUT
             !
             CALL RDCMND(LYN,MXLEN,SLEN,IUNIT,EOF,.FALSE.,.FALSE.,' ')
             ISEQ=0
             IRES=0
             RESIN='    '
             ATOMIN='    '
             XIN=0.0
             YIN=0.0
             ZIN=0.0
             SID='    '
             RID='    '
             WIN=0.0
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) ISEQ=NEXTI(LYN,SLEN)
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) IRES=NEXTI(LYN,SLEN)
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) RESIN=NEXTA8(LYN,SLEN)
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) ATOMIN=NEXTA8(LYN,SLEN)
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) XIN=NEXTF(LYN,SLEN)
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) YIN=NEXTF(LYN,SLEN)
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) ZIN=NEXTF(LYN,SLEN)
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) SID=NEXTA8(LYN,SLEN)
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) RID=NEXTA8(LYN,SLEN)
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) WIN=NEXTF(LYN,SLEN)
             CALL TRIME(LYN,SLEN)
             IF(SLEN.GT.0) CALL XTRANE(LYN,SLEN,'CREAD')
          ELSE
             !
             ! FIXED FIELD INPUT
             !
             IF (NINPUT .LT. 0) THEN
                !
                ! PDB format
                !
                QATOM=.FALSE.
                IF (PDBLIN(1:3).EQ.'END') THEN
                   GOTO 61
                ELSE IF (PDBLIN(1:4).EQ.'ATOM') THEN
                   QATOM=.TRUE.
                ELSE IF (PDBLIN(1:4).EQ.'HETA') THEN
                   QATOM=.TRUE.
                ELSE
                   !
                   ! keep reading until reaching ATOM or END
                   !
99962              CONTINUE
                   READ(IUNIT,'(A)',END=61,ERR=61) PDBLIN
                   !ln...05-Jan-95, convert the string to upper case
                   SLEN=LEN(PDBLIN)
                   CALL CNVTUC(PDBLIN,SLEN)
                   !ln...(2)
                   QATOM=(PDBLIN(1:4).EQ.'ATOM'.OR.PDBLIN(1:4).EQ.'HETA')
                   IF (PDBLIN(1:3).EQ.'END') GOTO 61
                   IF (.NOT.QATOM) GOTO 99962
                ENDIF
                IF (QATOM) THEN
                   !
                   !               process ATOM line
                   IF (OFFICIAL) THEN
                      ! Read with the official PDB format :
                      ! ATOM      2  CA  GLN A  12      50.249   6.624   3.918  1.00151.29
                      ! Format correction, and allow insertion code. L.Nilsson, November 07
                      ! Previous format:
                      !               (6X,I5,1X,A4,1X,A3,1X,A1,1X,A3,4X,3F8.3,6X,F6.2,6X,A4)
                      !
                      READ(PDBLIN, &
                           '(6X,I5,1X,A4,1X,A3,1X,A1,A5,3X,3F8.3,6X,F6.2,6X,A4)') &
                           ISEQ,ATOMIN,RESIN,SID,RID,XIN,YIN,ZIN,WIN
                   ELSE
                      ! The CHARMM PDB format is:
                      ! ATOM     69  HA  THR     5      10.000   0.000   0.000  1.00  0.00      A
                      !
                      READ(PDBLIN, &
                           '(6X,I5,1X,A4,1X,A4,1X,A5,3X,3F8.3,6X,F6.2,6X,A4)') &
                           ISEQ,ATOMIN,RESIN,RID,XIN,YIN,ZIN,WIN,SID

                   ENDIF
                   !
                   ! make ATOMIN left-justified
                   !
                   IJ=8
                   CALL TRIMA(ATOMIN,IJ)
                   !
                   ! make RESIN left-justified
                   !
                   IJ=8
                   CALL TRIMA(RESIN,IJ)
                   !
                   ! make RID left-justified
                   !
                   IF(.NOT.LRSID) THEN
                      IRES=-99999999
                      READ(RID(1:4),'(I4)',ERR=37) IRES
37                    CONTINUE
                   ENDIF
                   IJ=5
                   CALL TRIMA(RID,IJ)
                   !
                   ! read next PDB line
                   !
                   READ(IUNIT,'(A)',ERR=61,END=61) PDBLIN
                   !ln...05-Jan-95, convert the string to upper case
                   SLEN=LEN(PDBLIN)
                   CALL CNVTUC(PDBLIN,SLEN)
                   !ln...(3)
                ENDIF
             ELSE
                !ln...05-Jan-95, convert the string to upper case
                READ(IUNIT,'(A)',ERR=61,END=61) PDBLIN
                SLEN=LEN(PDBLIN)
                CALL CNVTUC(PDBLIN,SLEN)
                READ(PDBLIN,fmt40) ISEQ,IRES,RESIN,ATOMIN, &
                     CXIN,CYIN,CZIN,SID,RID,CWIN
             ENDIF
          ENDIF
          !
          !CC       IF(LRSID.OR.NINPUT.EQ.-1) THEN
          !CC - brb -- reinstate LRES option for PDB (change was not in io.doc)
          IF(LRSID) THEN
             !           GET ATOM FROM RESID AND SEGID FIELDS
             ISEG=1
99961        IF (SEGID(ISEG).NE.SID) THEN
                ISEG=ISEG+1
                IF(ISEG.GT.NSEG) GOTO 888
                GOTO 99961
             ENDIF
             IRES=NICTOT(ISEG)+1
             ISTP=NICTOT(ISEG+1)
             IF(IRES.GT.ISTP) GOTO 888
99960        IF (RESID(IRES).NE.RID) THEN
                IRES=IRES+1
                IF(IRES.GT.ISTP) GOTO 888
                GOTO 99960
             ENDIF
             GOTO 889
888          IRES=-99999999
          ENDIF
          !
          IRES=IRES+ISRES-1
889       CONTINUE
          !
          !     CHECK THE INPUT TO SEE THAT:
          !
          !             THE SEQUENCE MATCHES THE PSF
          !             THE ATOM TYPE IS LOCATED PROPERLY
          !             NO COORDINATES ARE MULTIPLY DEFINED
          !             THAT THE COORDINATES ARE WITHIN THE DESIRED INTERVAL
          !
          IF (IRES.LT.1.OR.IRES.GT.NRES) THEN
             NRNG=NRNG+1
             IF (NRNG.LT.5) THEN
                IF (LRSID) THEN !!!.OR.NINPUT.EQ.-1) THEN
                   IF(WRNLEV.GE.2) WRITE(OUTU,82) SID(1:ilen), &
                        RID(1:ilen),RESIN(1:ilen),ATOMIN(1:ilen)
82                 FORMAT(/' ** WARNING ** For atom in coordinate file,' &
                        ,' could not find residue in PSF,', &
                        ' and is thus ignored:',/ &
                        /'  SEGID=',A,' RESID=',A,' RESNAME= ',A, &
                        ' TYPE= ',A)
                ELSE
                   IF(WRNLEV.GE.2) WRITE(OUTU,83) IRES, &
                        RESIN(1:ilen),ATOMIN(1:ilen)
83                 FORMAT(/' ** WARNING ** For atom in coordinate file,' &
                        ,' the residue number is out of range,', &
                        ' and is thus ignored:',/ &
                        /'  IRES=',I5,' RESNAME= ',A,' TYPE= ',A)
                ENDIF
                CALL DIEWRN(1)
             ENDIF
          ELSE
             IF(RES(IRES).NE.RESIN) THEN
                NSEQM=NSEQM+1
                IF(NSEQM.LE.5) THEN
                   IF(WRNLEV.GE.2) WRITE(OUTU,85) IRES, &
                        RES(IRES)(1:ilen),RESIN(1:ilen)
85                 FORMAT(/' ** WARNING ** For atom in coordinate file,' &
                        ,' the residue type does not match', &
                        ' that (RESN) in the PSF:', &
                        I5,' PSF= ',A,' INPUT= ',A)
                ENDIF
             ENDIF
             IPOINT=MATOM(IRES,ATOMIN,ATYPE,IBASE,IRES,IRES,.FALSE.)
             IF (IPOINT.LT.0) THEN
                ERRCNT=ERRCNT+1
                IF (ERRCNT.LT.20) THEN
                   IF(WRNLEV.GE.2) WRITE(OUTU,45) ISEQ,IRES, &
                        RESID(IRES)(1:ilen),RESIN(1:ilen),ATOMIN(1:ilen)
45                 FORMAT(/' ** WARNING ** For atom in coordinate file, the', &
                        ' corresponding residue in the PSF lacks that atom:',/ &
                        ' INDEX=',I5,' IRES=',I5,' RESID=',A,' RES=',A,' ATOM=',A)
                ENDIF
             ELSE IF (IPOINT.LT.1.OR.IPOINT.GT.NATOM) THEN
                NRNG=NRNG+1
                IF(NRNG.LE.5 .AND. WRNLEV.GE.2) WRITE(OUTU,50) &
                     ISEQ,IRES,RESIN(1:ilen),ATOMIN(1:ilen),IPOINT
50              FORMAT(/' ** WARNING ** For atom in coordinate file, the', &
                     ' corresponding atom is out of range, and thus ignored:',/ &
                     '  ISEQ=',I6,' IRES=',I6,' RESIN=',A,' ATOMIN=',A,' INDEX=',I6)
             ELSE IF (ISLCT(IPOINT).EQ.1) THEN
                IF (INITIA(IPOINT,X,Y,Z)) NMULT=NMULT+1
                IF (LFREE .OR. NINPUT.LT.0) THEN
                   X(IPOINT)=XIN
                   Y(IPOINT)=YIN
                   Z(IPOINT)=ZIN
                   WMAIN(IPOINT)=WIN
#if KEY_FOURD==1 /*4dread*/
                   IF (DIM4) THEN
                      FDIM(IPOINT)=WIN
                      WMAIN(IPOINT)=0.0
                   ENDIF
#endif /* (4dread)*/
                ELSE
                   READ(CXIN,fmt10,ERR=58) X(IPOINT)
                   READ(CYIN,fmt10,ERR=58) Y(IPOINT)
                   READ(CZIN,fmt10,ERR=58) Z(IPOINT)
                   READ(CWIN,fmt10,ERR=58) WMAIN(IPOINT)
                   GOTO 59
58                 CONTINUE
                   CALL WRNDIE(1,'<CREAD>', &
                        'Bad characters in coordinate field: Initialized')
                   X(IPOINT)=ANUM
                   Y(IPOINT)=ANUM
                   Z(IPOINT)=ANUM
                   WMAIN(IPOINT)=ZERO
59                 CONTINUE
#if KEY_FOURD==1 /*4dread*/
                   IF (DIM4) THEN
                      FDIM(IPOINT)=WMAIN(IPOINT)
                      WMAIN(IPOINT)=0.0
                   ENDIF
#endif /* (4dread)*/
                ENDIF
             ELSE
                NDESL=NDESL+1
             ENDIF
          ENDIF
       enddo loop60
       !
       GOTO 63
61     I=I-1
       IF(IATOM.NE.99999999 .AND. WRNLEV.GE.2) &
            WRITE(OUTU,62) I,IATOM
62     FORMAT(' ** WARNING ** Error or EOF on input file.', &
            I10,' coordinates read.',I10,' expected.')
63     CONTINUE
       !
       !     CHECK TO SEE THAT ALL THE DESIRED COORDINATES WERE FOUND
       !
       IRES=1
       NMISS=0
       DO I=1,NATOM
          IF (I.GT.IBASE(IRES+1)) IRES=IRES+1
          IF(ISLCT(I).EQ.1) THEN
             IF (INITIA(I,X,Y,Z)) GOTO 70
             NMISS=NMISS+1
             IF (NMISS.GT.10) GOTO 70
             IF(WRNLEV.GE.2) WRITE(OUTU,65) I,IRES, &
                  RES(IRES)(1:ilen),ATYPE(I)(1:ilen)
65           FORMAT( &
                  ' ** WARNING ** After reading, there are no coordinates', &
                  ' for selected atom:',2I6,2(1X,A))
70           CONTINUE
          ENDIF
       ENDDO
       IF(NMISS.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,74) NMISS
74     FORMAT(/' ** A total of',I6,' selected atoms have no coordinates')
       IF (ERRCNT.GT.5 .AND. WRNLEV.GE.2) WRITE(OUTU,75) ERRCNT
75     FORMAT(/' ** A total of',I5,' warnings were encountered during', &
            ' coordinate reading **')
       IF (NMULT.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,55) NMULT
55     FORMAT(/' ** WARNING ** Coordinates were overwritten for',i6, &
            ' atoms.')
       IF(NDESL.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,77) NDESL
77     FORMAT(/' ** MESSAGE **',I6,' atoms in coordinate file were', &
            ' ignored because of the specified atom selection.')
       IF (NRNG.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,78) NRNG
78     FORMAT(/' ** MESSAGE **',I6,' atoms in coordinate file', &
            ' were outside the specified sequence range.')
       IF(NMISS+ERRCNT+NMULT.GT.0) CALL DIEWRN(2)
       IF (NSEQM.GT.0 .AND. WRNLEV.GE.2) THEN
          WRITE(OUTU,79) NSEQM
79        FORMAT(/' ** WARNING **',I6,' atoms in coordinates file had a', &
               ' sequence mismatch.')
          CALL DIEWRN(0)
       ENDIF
       GOTO 900
       !
       !
90     CONTINUE
       !
       !       READ CARDS IGNORING ATOM NAMES AND SEQUENCE INFO
       !
       IF(LFREE) CALL WRNDIE(0,'<CREAD> ', &
            'Cannot use both FREE and IGNOre options')
       !
       CALL RDTITL(TITLE,NTITL,IUNIT,0)
       !yw++ 28-Jan-2003 use PDBLIN to process a line
       !yw       READ(IUNIT,'(I5)') IATOM
       read(iunit,'(a)') pdblin
       slen=len(pdblin)
       call cnvtuc(pdblin,slen)
       iatom=nexti(pdblin,slen)
       lextfmt=indxa(pdblin,slen,'EXT').gt.0
       if (iatom.ge.100000) lextfmt=.true.
       if (lextfmt) then
          fmt50='(40X,3F20.10,20X,F20.10)'
       else
          fmt50='(20X,3F10.5,10X,F10.5)'
       endif
       !yw--
       IF(IATOM.EQ.0) THEN
          IATOM=99999999
          IF(WRNLEV.GE.2) WRITE(OUTU,32)
       ENDIF
       IPT=0
       DO I=1,NATOM
          IF(ISLCT(I).EQ.1 .AND. IPT.LT.IATOM) THEN
             IPT=IPT+1
             READ(IUNIT,fmt50,ERR=184,END=184) &
                  X(I),Y(I),Z(I),WMAIN(I)
          ENDIF
       enddo
       GOTO 900
       !
184    IPT=IPT-1
       IF(IATOM.NE.99999999) WRITE(OUTU,62) IPT,IATOM
       !
       !
900    CONTINUE
       !       Done reading coordinates.
    ENDIF
    !
!MFC-- This code is probably unnecessary now, will delete when ensemble conversion done
!-- ##IF ENSEMBLE
!--     IF (QENS) IOLEV=OLDIOL
!--     IF (.NOT.QENS) THEN
!--        CALL PSND8_ENS(X, NATOM)
!--        CALL PSND8_ENS(Y, NATOM)
!--        CALL PSND8_ENS(Z, NATOM)
!--        CALL PSND8_ENS(WMAIN, NATOM)
!--        CALL PSND4_ENS(QCRYS,1)
!--        CALL PSND8_ENS(XTLABC,6)
!--     ENDIF
!-- ##ENDIF
#if KEY_PARALLEL==1
    CALL PSND8(X, NATOM)
    CALL PSND8(Y, NATOM)
    CALL PSND8(Z, NATOM)
    CALL PSND8(WMAIN, NATOM)
    CALL PSND4(QCRYS,1)
    CALL PSND8(XTLABC,6)
#endif 
    IF(QCRYS .AND. XDIM.GT.0) THEN
       xucold(1:6) = xucell(1:6)
       CALL XTLLAT(XUCELL,XTLABC)
       CALL XTLMSR(XUCELL)
       !     Recompute the images from the crystal transformations.
       call chmalloc('coorio.src','CREAD','TRANSF',12*XNSYMM,crl=TRANSF)
       CALL IMFILL(TRANSF,.FALSE.)
       call chmdealloc('coorio.src','CREAD','transf',12*XNSYMM,transf)
    ENDIF
    !
    call chmdealloc('coorio.src','COORIO','space_W',4*NATOM+4,cr4=space_W)
    RETURN
  END SUBROUTINE CREAD
end module coorio_mod

