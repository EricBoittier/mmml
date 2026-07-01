module stream
  use chm_kinds
  !     This is the STREAM data block.
  !     It contains information abount the current runstream.
  !
  !     MXSTRM - Maximum number of active stream files.
  !     POUTU  - Default output unit number.
  !     NSTRM  - Number of active input streams.
  !     ISTRM = JSTRM(NSTRM) - Current input unit number
  !     JSTRM(*) - stack of input streams numbers.
  !     OUTU   - Unit number for all standard CHARMM output.
  !     PRNLEV - Print level control for all writing to OUTU
  !     IOLEV  - -1 to 1  -1=write no files.  1= write all files.
  !moved to chm_kinds-mfc     WRNLEV - -5 TO 10  0=SEVERE ONLY, 10=LIST ALL WARNINGS
  !     LOWER  - if .true. all files with names not in double quotes
  !              will be opened in lower case for write. For read
  !              UPPER case will be tried first and if not succesful
  !              file name will be converted to lower case.
  !     QLONGL - Use long lines in the output where appropriate.
  !              (Otherwise, restrict output lines to 80 characters)
  !     IECHO -  Unit number for output from ECHO command
  !yw++
  !     qoldfmt  enforce old-format i/o, .false. unless set by IOFO NOEX
  !     qnewfmt  enforce new-format i/o, .false. unless set by IOFO EXTE
  !     qextfmt  new-format i/o if natom>100000 or character(len=8) is used
  !     idleng   character ID length (4 for old format and 8 for extended)

  logical qoldfmt,qnewfmt,qextfmt,qcmdin,qcmdout
  integer idleng
  logical lower,qlongl
  character(len=200) :: cinfile,coutfile
  integer lcinfil,lcoutfil
  integer,parameter :: mxstrm=20,poutu=6
  integer   nstrm,istrm,jstrm(mxstrm),outu,prnlev,iolev,iecho
  !
#if KEY_MULTICOM==1
  integer :: comm_strm(mxstrm)
  ! VO Added ^ MULTICOM functionality: in addition to the streams, we store the communicator
  ! over which to broadcast the input line at each level
  ! This provides a clean way to split inputs
#endif
  !
  ! MF flag for allowing/disallowing rewinds, save icontrol in READCV
  logical reallow
  INTEGER icntsv(20)

contains

  subroutine stream_iniall()
#if KEY_LONGLINE==1 /*longline_init*/
    qlongl=.true.
#else /* (longline_init)*/
    qlongl=.false.
#endif /* (longline_init)*/
  iecho=poutu
  qoldfmt=.false.
  qnewfmt=.false.
  qextfmt=.false.
  idleng=4
  reallow=.true.             ! mf050629
    return
  end subroutine stream_iniall
end module stream

