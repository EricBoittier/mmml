! 
! MSES module
! Description: Energy = E_AT + E_CG + E_MSES
!
! Author: 
! Xiping Gong (2020, xipinggong@umass.edu)
! Jianhan Chen (jianhanc@umass.edu)
! 

MODULE omm_mses  ! AN EXAMPLE OPENMM PLUGIN
use omm_nbopts, only: omm_nbopts_t
use inbnd, only: qmses_omm
use chm_kinds, only: chm_real
use string, only: GTRMF, GTRMI, INDXA
use psf, only: SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG, nrest, nsegt
use chutil, only: GETATN, getres, getseg
use stream, only: OUTU,prnlev
#if KEY_OPENMM==1
use OpenMM
use OpenMMMSES_Types
use OpenMMMSES
#endif    
implicit none

! define variables
logical :: &
QMSES = .false., &
qmses_import_omm = .false.

integer :: &
fileID = -1

real(chm_real) :: &
scal = 1.D0

#if KEY_OPENMM==1
type(OpenMMMSES_MSESForce), save :: msesforce
#endif
    
! The following data and subrouties are
! public to use, otherwise private.
private
public &
QMSES, &
QMSES_import_omm, &
MSES_SET, &
getSegid
#if KEY_OPENMM==1 
public setup_mses, msesforce, update_mses
#endif

contains

      subroutine MSES_SET(comlyn, comlen)
      !----------------------------------
      ! This routine is to find out the contact file
      ! for the MSES simulations from the "MSES ..." 
      ! in the input file.
      !----------------------------------
      implicit none
      character(len=*) :: comlyn
      integer comlen

      ! FILE -> MSES contacts list force fields
      fileID = GTRMI(COMLYN,COMLEN,'IUNLIST',-1)

      ! SCAL
      SCAL = GTRMF(COMLYN,COMLEN,'SCAL',SCAL)

      IF(prnlev>0 .and. fileID>0) THEN
         qmses=.true.
         qmses_import_omm = .false.
         write(outu,*) "MSES> save MSES parameters into OMM_MSES module"
         write(outu,*) "MSES> setup MSES successfully"
      ENDIF

      !
      END subroutine mses_set

      subroutine getSegid(iatom,segid)
      implicit none
      integer iatom, segid
      segid = getres(iatom,ibase,nrest)
      segid = getseg(segid,nictot,nsegt)
      END subroutine getSegid

#if KEY_OPENMM==1 

      subroutine setup_mses(system, group, nbopts)
      implicit none
      
      type(OpenMM_System), intent(inout) :: system
      integer*4, intent(in) :: group
      type(omm_nbopts_t), intent(in) :: nbopts
      
      integer :: nb_method, ijunk
      
      call OpenMMMSES_MSESForce_create(msesforce)
      call OpenMM_Force_setForceGroup(   &
          transfer(msesforce, OpenMM_Force(0)),group)
      ijunk = OpenMM_System_addForce(system, transfer(msesforce, OpenMM_Force(0)))

      call mses_chm2omm('import') 
       
      end subroutine setup_mses

      subroutine update_mses(context)
      implicit none
      
      type(OpenMM_Context), intent(inout) :: context

      call mses_chm2omm('update') 
      call OpenMMMSES_MSESForce_updateParametersInContext(msesforce,context)

      end subroutine update_mses

      subroutine mses_chm2omm(str)
      implicit none

      character(len=6), intent(in) :: str
      character(len=132) tline
      character(len=1) :: commentChar = '!'
      character(len=4) MSESTYPE
      character(len=4) SEG1, RES1, ATO1, SEG2, RES2, ATO2
      character(len=4) SEG3, RES3, ATO3, SEG4, RES4, ATO4
      INTEGER p1,p2,p3,p4
      integer iDistPair, ijunk
      real*8 c1,c2,c3,c4,c5
      logical yes

      ! import MSES contact file into MSES OpenMM plugin
      IF(prnlev>0) write(outu,*) "MSES> importing contact list file data into OpenMM MSES plugin"
      IF(prnlev>=5) THEN
         write(outu,*) "MSES> current prnlev is", prnlev
         write(outu,*) "MSES> using prnlev < 5 to avoid the printing"
      ELSEIF(prnlev>0) THEN
         write(outu,*) "MSES> current prnlev is", prnlev
         write(outu,*) "MSES> using prnlev >= 5 to print out the details"
      ENDIF
      INQUIRE(UNIT=fileID,OPENED=yes)
      IF(.not. yes) THEN
         write(outu,*) "MSES> do not manually close contact list file, otherwise the data cannot be imported"
         STOP
      ENDIF
      REWIND(fileID)
      iDistPair = 0
      DO WHILE (.TRUE.) 
         ! read line by line
         READ(fileID,"(a)",iostat=ijunk) tline
         ! at the end of file?
         IF (ijunk/=0) EXIT
         ! delete the comments
         ijunk = index(tline,commentChar)
         if(ijunk>0) tline = tline(1:ijunk-1)
         ! upper tline
         ijunk = len_trim(tline)
         DO p1 = 1,ijunk
            if(tline(p1:p1).ge."a" .and. tline(p1:p1).le."z") then
               tline(p1:p1) =  char(ichar(tline(p1:p1))-32)
            endif
         ENDDO
         ! adjust tline to the left
         tline = adjustL(tline)
         ! DIST KEYWORD?
         IF(tline(1:4)=='DIST') THEN
            READ(tline,*) MSESTYPE, &
                          SEG1, RES1, ATO1, SEG2, RES2, ATO2, &
                          SEG3, RES3, ATO3, SEG4, RES4, ATO4, &
                          c1,c2,c3,c4,c5
            p1=GETATN(SEG1,RES1,ATO1,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
            p2=GETATN(SEG2,RES2,ATO2,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
            p3=GETATN(SEG3,RES3,ATO3,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
            p4=GETATN(SEG4,RES4,ATO4,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
            c1 = SCAL * c5 * c1 * OpenMM_AngstromsPerNm**2 * OpenMM_KJPerKcal;
            c2 = c2 * OpenMM_AngstromsPerNm * OpenMM_KJPerKcal;
            c3 = c3 / OpenMM_AngstromsPerNm;
            c4 = c4;
            if(str=='import') THEN
               IF(prnlev>=5) THEN
                  write(outu,100) MSESTYPE, p1,p2,p3,p4,c1,c2,c3,c4
               ENDIF
               ijunk=OpenMMMSES_MSESForce_addDistPair( msesforce,p1-1,p2-1,p3-1,p4-1,c1,c2,c3,c4)
            elseif(str=='update') THEN
               IF(prnlev>=5) THEN
                  write(outu,101) MSESTYPE, p1,p2,p3,p4,c1,c2,c3,c4
               ENDIF
               call OpenMMMSES_MSESForce_setDistPairParameters( msesforce,iDistPair,p1-1,p2-1,p3-1,p4-1,c1,c2,c3,c4) 
               iDistPair = iDistPair + 1
            endif
         ENDIF
      END DO
       
      qmses_import_omm=.true.
      IF(prnlev>0) write(outu,*) "MSES> complete importing contact file data"

100 FORMAT(1X,'MSES> Importing',1X, (A4,4X), 'AT1 =',(I7,4X),'AT2 =',(I7,4X),'AT3 =',(I7,4X),'AT4 =',(I7,4X), &
           'FORC =',(F11.6,2X),'FMAX =',(F11.6,2X), 'RSWI =',(F11.6,2X), 'SEXP =',(F11.6,2X) )
101 FORMAT(1X,'MSES> Updating',1X, (A4,4X), 'AT1 =',(I7,4X),'AT2 =',(I7,4X),'AT3 =',(I7,4X),'AT4 =',(I7,4X), &
           'FORC =',(F11.6,2X),'FMAX =',(F11.6,2X), 'RSWI =',(F11.6,2X), 'SEXP =',(F11.6,2X) )

      end subroutine mses_chm2omm

#endif /* KEY_OPENMM */

! 
END MODULE omm_mses

