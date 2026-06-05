#if KEY_FMA==1 /*fma_main*/
SUBROUTINE FMA(Elec, Enb, Level, Terms)
  !-----------------------------------------------------------------------
  ! To calculate the electrostatics using the fast multipole expansion.
  ! This fortran version is based heavily on John Board's c code. Some
  ! optimisations have been made. The code conforms to the algorithm
  ! outlined in:
  !   The Parallel Fast Multipole Algorithm In Three Dimensions (Leathrum
  !   And Board) Technical Report, Duke University, 1992
  ! Differences to John Board's code:
  !   Calculations independent of molecules position relative to origin
  !   linked-list for box hierarchy, all empty boxes deleted
  !   atoms with zero charge excluded for (electrostatic) calculations
  !
  ! Robert Nagle/Paul Adams
  !
  !
  ! includes
  !
  use chm_kinds
  use dimens_fcm
  use coord
  use deriv
  use psf
  use exfunc
  use mfmacons
  use memory
  implicit none
  !
  ! passed
  !
  real(chm_real) Elec, Enb
  integer Level,Terms
  !
  !
  ! local
  !
  integer NBoxes, MBoxes, ActAt, Dim, I
  real(chm_real) BoxDim

  real(chm_real),allocatable,dimension(:) :: XL,YL,ZL,DXL,DYL,DZL,CGL, &
       BoxX,BoxY,BoxZ,S1C,S2C,S3C,S4C,LEG,PhiX,PhiY,PsiX,PsiY
  integer,allocatable,dimension(:) :: ATID,Child,ChildNO,Parent,NPoint, &
       TEMP,IntList,IntPoint,Neighbour,BoxLOOK
  !
  !---------------------------- begin ----------------------
  !
  ! First initialise the box pointers
  ! Note the user gives Levels ranging from 0 upwards, internally the code
  !    uses Levels from 1 upwards (ie. 1 is added to the Level when the
  !    fma code is  called)
  call ARRAYINIT(Level)
  !
  ! Set the number of boxes to the maximum
  Nboxes = Finish(Level)
  !
  ! Allocate space for the arrays
  call chmalloc('fma.src','FMA','XL',NAtom,crl=XL)
  call chmalloc('fma.src','FMA','YL',NAtom,crl=YL)
  call chmalloc('fma.src','FMA','ZL',NAtom,crl=ZL)
  call chmalloc('fma.src','FMA','DXL',NAtom,crl=DXL)
  call chmalloc('fma.src','FMA','DYL',NAtom,crl=DYL)
  call chmalloc('fma.src','FMA','DZL',NAtom,crl=DZL)
  call chmalloc('fma.src','FMA','CGL',NAtom,crl=CGL)
  call chmalloc('fma.src','FMA','BoxX',NBoxes,crl=BoxX)
  call chmalloc('fma.src','FMA','BoxY',NBoxes,crl=BoxY)
  call chmalloc('fma.src','FMA','BoxZ',NBoxes,crl=BoxZ)
  call chmalloc('fma.src','FMA','ATID',NAtom,intg=ATID)
  call chmalloc('fma.src','FMA','Child',NBoxes,intg=Child)
  call chmalloc('fma.src','FMA','ChildNO',NBoxes,intg=ChildNO)
  call chmalloc('fma.src','FMA','Parent',NBoxes,intg=Parent)
  call chmalloc('fma.src','FMA','NPoint',NBoxes+1,intg=NPoint)
  call chmalloc('fma.src','FMA','S1C',2*2*Terms*Terms,crl=S1C)
  call chmalloc('fma.src','FMA','S2C',Terms*Terms*Terms*Terms*2,crl=S2C)
  call chmalloc('fma.src','FMA','S3C',Terms*Terms*Terms*Terms*2,crl=S3C)
  call chmalloc('fma.src','FMA','S4C',Terms*Terms*Terms*Terms*2,crl=S4C)
  call chmalloc('fma.src','FMA','LEG',Terms*Terms,crl=LEG)
  call chmalloc('fma.src','FMA','TEMP',NAtom,intg=TEMP)
  !
  call FMAInit(NAtom, NBoxes, Level, Terms, X, Y, Z, CG, &
       XL, YL, ZL, &
       DXL, DYL, DZL, CGL, &
       BoxX, BoxY, BoxZ, BoxDim, &
       ATID,Child,ChildNO,Parent, &
       NPoint,Temp, &
       S1C, S2C, S3C, S4C, &
       LEG, ActAt)
  !
  ! Empty boxes have been culled so only allocate appropriately
  MBoxes = Finish (Level)
  DIM = 0
  do I = 2,Level
     DIM = DIM+(2**(I-1))
  end do
  call chmalloc('fma.src','FMA','PhiX',MBoxes*Terms*Terms,crl=PhiX)
  call chmalloc('fma.src','FMA','PhiY',MBoxes*Terms*Terms,crl=PhiY)
  call chmalloc('fma.src','FMA','PsiX',MBoxes*Terms*Terms,crl=PsiX)
  call chmalloc('fma.src','FMA','PsiY',MBoxes*Terms*Terms,crl=PsiY)
  call chmalloc('fma.src','FMA','IntList',MBoxes*AvgIntrct,intg=IntList)
  call chmalloc('fma.src','FMA','IntPoint',MBoxes+1,intg=IntPoint)
  call chmalloc('fma.src','FMA','Neighbour',MBoxes*AvgNbr,intg=Neighbour)
  call chmalloc('fma.src','FMA','BoxLOOK',DIM*DIM*DIM,intg=BoxLOOK)
  !
  ! Compute the Interaction and Neighbour lists and
  !     complete initialization
  call FMAList(NAtom,Dim, MBoxes,Level,Terms, BoxDim, &
       BoxX, BoxY, BoxZ, &
       PhiX, PhiY, PsiX, PsiY, &
       Child, ChildNO, &
       IntList,IntPoint,NPoint, &
       Neighbour, Parent, BoxLook, &
       Temp)
  !
  !
  !
  ! This is the meat of the FMA method; compute all the expansions
  !
  !     If there are enough Levels to bother using the fma approximation
  if (Level  >  2) then
     call FMAExpan(MBoxes, Level, Terms, &
          XL,YL,ZL,CGL, &
          BoxX,BoxY,BoxZ, &
          PhiX,PhiY,PsiX,PsiY, &
          Child,ChildNO, &
          IntList,IntPoint,NPoint, &
          Neighbour,S1C,S2C,S3C, &
          S4C,LEG)
  endif
  !
  ! Finally, compute the potentials and forces
  call FMAEval(Elec, ENB, NAtom,MBoxes,Level,Terms, &
       XL,YL,ZL, &
       DXL,DYL,DZL,CGL, &
       BoxX,BoxY,BoxZ, &
       PhiX,PhiY,PsiX,PsiY, &
       ActAt, ATID,Child,ChildNO, &
       NPoint, Neighbour,S1C, LEG)

  call chmdealloc('fma.src','FMA','BoxLOOK',DIM*DIM*DIM,intg=BoxLOOK)
  call chmdealloc('fma.src','FMA','Neighbour',MBoxes*AvgNbr,intg=Neighbour)
  call chmdealloc('fma.src','FMA','IntPoint',MBoxes+1,intg=IntPoint)
  call chmdealloc('fma.src','FMA','IntList',MBoxes*AvgIntrct,intg=IntList)
  call chmdealloc('fma.src','FMA','PsiY',MBoxes*Terms*Terms,crl=PsiY)
  call chmdealloc('fma.src','FMA','PsiX',MBoxes*Terms*Terms,crl=PsiX)
  call chmdealloc('fma.src','FMA','PhiY',MBoxes*Terms*Terms,crl=PhiY)
  call chmdealloc('fma.src','FMA','PhiX',MBoxes*Terms*Terms,crl=PhiX)
  call chmdealloc('fma.src','FMA','TEMP',NAtom,intg=TEMP)
  call chmdealloc('fma.src','FMA','LEG',Terms*Terms,crl=LEG)
  call chmdealloc('fma.src','FMA','S4C',Terms*Terms*Terms*Terms*2,crl=S4C)
  call chmdealloc('fma.src','FMA','S3C',Terms*Terms*Terms*Terms*2,crl=S3C)
  call chmdealloc('fma.src','FMA','S2C',Terms*Terms*Terms*Terms*2,crl=S2C)
  call chmdealloc('fma.src','FMA','S1C',2*2*Terms*Terms,crl=S1C)
  call chmdealloc('fma.src','FMA','NPoint',NBoxes+1,intg=NPoint)
  call chmdealloc('fma.src','FMA','Parent',NBoxes,intg=Parent)
  call chmdealloc('fma.src','FMA','ChildNO',NBoxes,intg=ChildNO)
  call chmdealloc('fma.src','FMA','Child',NBoxes,intg=Child)
  call chmdealloc('fma.src','FMA','ATID',NAtom,intg=ATID)
  call chmdealloc('fma.src','FMA','BoxZ',NBoxes,crl=BoxZ)
  call chmdealloc('fma.src','FMA','BoxY',NBoxes,crl=BoxY)
  call chmdealloc('fma.src','FMA','BoxX',NBoxes,crl=BoxX)
  call chmdealloc('fma.src','FMA','CGL',NAtom,crl=CGL)
  call chmdealloc('fma.src','FMA','DZL',NAtom,crl=DZL)
  call chmdealloc('fma.src','FMA','DYL',NAtom,crl=DYL)
  call chmdealloc('fma.src','FMA','DXL',NAtom,crl=DXL)
  call chmdealloc('fma.src','FMA','ZL',NAtom,crl=ZL)
  call chmdealloc('fma.src','FMA','YL',NAtom,crl=YL)
  call chmdealloc('fma.src','FMA','XL',NAtom,crl=XL)
  !
  RETURN
END SUBROUTINE FMA

SUBROUTINE FMAInit(NAtom,NBoxes,Level,Terms, &
     X,Y,Z,CG,XL,YL,ZL,DXL,DYL,DZL,CGL, &
     BoxX, BoxY, BoxZ, Boxdim, &
     ATID, Child, ChildNO, Parent, NPoint, Temp, &
     S1C,S2C,S3C,S4C,LEG, ActAt)
  !-----------------------------------------------------------------------
  !     Description: see above
  !     Robert Nagle
  !
  ! includes
  !
  use chm_kinds
  use dimens_fcm
  use timerm
  use stream
  use exfunc
  use mfmacons
  use deriv
  implicit none
  !
  ! passed
  !
  integer NAtom,NBoxes,Level,Terms, ActAt
  integer ATID(*),Child(*),ChildNO(*), Parent(*)
  integer NPoint(*), Temp(*)
  real(chm_real) X(*),Y(*),Z(*),CG(*)
  real(chm_real) XL(*),YL(*),ZL(*),DXL(*),DYL(*),DZL(*),CGL(*)
  real(chm_real) BoxX(*),BoxY(*),BoxZ(*)
  real(chm_real) S1C(2*Terms,2*Terms),LEG(Terms,Terms)
  real(chm_real) S2C(Terms,Terms,Terms,Terms,2)
  real(chm_real) S3C(Terms,Terms,Terms,Terms,2)
  real(chm_real) S4C(Terms,Terms,Terms,Terms,2)
  real(chm_real) BoxDim
  !
  ! local
  !
  !---------------------------- begin ----------------------
  !
  !     Initialise the constants and temporary arrays
  call FMACONS(NAtom, NBoxes, Level, Terms, &
       DXL, DYL, DZL, S1C, S2C, S3C, S4C)
#if KEY_DEBUG==1
  if (Timer  >  1) call WRTTIM &
       (' FASTMA: Initialization Complete')
#endif 
  !
  !     Get the size of the system
  call BoxSIZ(NAtom, X, Y, Z, BoxDIM, BoxX, BoxY, BoxZ)
  !
  !
  call DVIDE(NAtom,X,Y,Z,CG,BoxDIM,Level,XL,YL,ZL,CGL, &
       BoxX,BoxY,BoxZ, ATID, Child, ChildNO, PARENT, NPoint, &
       Temp, ACTAT)
#if KEY_DEBUG==1
  if (Timer  >  1) call WRTTIM &
       (' FASTMA: Grid Assignment Complete')
#endif 
  !
  RETURN
END SUBROUTINE FMAInit

SUBROUTINE FMAList(NAtom, Dim, NBoxes, Level, &
     Terms, BoxDim, BoxX, BoxY, BoxZ, PhiX, PhiY, &
     PsiX, PsiY, Child, ChildNO, IntList, &
     IntPoint, NPoint, Neighbour, Parent, BoxLook, &
     Temp)
  !-----------------------------------------------------------------------
  !     Description: see above
  !     Robert Nagle
  !
  ! includes
  !
  use chm_kinds
  use dimens_fcm
  use number
  use timerm
  use stream
  use exfunc
  use mfmacons
  use deriv
  implicit none
  !
  ! passed
  !
  integer NAtom,NBoxes,Level,Terms,Dim
  integer Child(*),ChildNO(*)
  integer IntList(*),IntPoint(*),NPoint(*),Neighbour(*)
  integer BoxLOOK(DIM,DIM,DIM),PARENT(*),TEMP(*)
  real(chm_real) BoxX(*),BoxY(*),BoxZ(*)
  real(chm_real) PhiX(Terms,Terms,*),PhiY(Terms,Terms,*)
  real(chm_real) PsiX(Terms,Terms,*),PsiY(Terms,Terms,*)
  real(chm_real) BoxDim
  !
  ! local
  !
  integer I,J,K,L,M,CNUM,NTEMP,NNUM,LevelDIM
  integer StartX,StartY,StartZ,FinishX,FinishY,FinishZ
  integer Box,TEMPS,TREE,PAR,NP,NEAR,NEARKID,TNNUM
  integer LOCAL(8),WIDTH,POINTER,PUTP,LEV,SIDE
  integer XPOS,YPOS,ZPOS,ChildTEMP,OFFSET
  real(chm_real) LENGTH,ORIGX,ORIGY,ORIGZ
  LOGICAL PUTFLG,PARFLG
  integer KID,DISTANT,NBox,BoxP,N,I2,N2,NN,IN
  !
  !---------------------------- begin ----------------------
  !
  ! Zero Phi and Psi arrays
  do I = 1,NBoxes
     do J = 1,Terms
        do K = 1,Terms
           PhiX(K,J,I) = ZERO
           PhiY(K,J,I) = ZERO
           PsiX(K,J,I) = ZERO
           PsiY(K,J,I) = ZERO
        end do
     end do
  end do
  !
  ORIGX = BoxX(1)-BoxDIM/TWO
  ORIGY = BoxY(1)-BoxDIM/TWO
  ORIGZ = BoxZ(1)-BoxDIM/TWO
  !
  ! Construct a lookup so we can calculate the neighbour list
  OFFSET = 0
  do I = 2,Level
     SIDE = 2**(I-1)
     do J = (1+OFFSET),(SIDE+OFFSET)
        do K = (1+OFFSET),(SIDE+OFFSET)
           do M = (1+OFFSET),(SIDE+OFFSET)
              BoxLOOK(J,K,M) = 0
           end do
        end do
     end do
     LENGTH = BoxDIM/SIDE
     do J = Start(I),Finish(I)
        XPOS = 1+INT((BoxX(J)-ORIGX)/LENGTH)+OFFSET
        YPOS = 1+INT((BoxY(J)-ORIGY)/LENGTH)+OFFSET
        ZPOS = 1+INT((BoxZ(J)-ORIGZ)/LENGTH)+OFFSET
        BoxLOOK(XPOS,YPOS,ZPOS) = J
     end do
     OFFSET = OFFSET+SIDE
  end do
  !
  ! Calculate the neighbour list for each box
  WIDTH = 2
  OFFSET = 0
  NNUM = 0
  NPoint(Start(2)) = 1
  do I = 2,Level
     LENGTH = BoxDIM/(2**(I-1))
     LevelDIM = 2**(I-1)
     do J = Start(I),Finish(I)
        XPOS = 1+INT((BoxX(J)-ORIGX)/LENGTH)
        YPOS = 1+INT((BoxY(J)-ORIGY)/LENGTH)
        ZPOS = 1+INT((BoxZ(J)-ORIGZ)/LENGTH)
        StartX = MAX(1,(XPOS-WIDTH))+OFFSET
        StartY = MAX(1,(YPOS-WIDTH))+OFFSET
        StartZ = MAX(1,(ZPOS-WIDTH))+OFFSET
        FinishX = MIN(LevelDIM,(XPOS+WIDTH))+OFFSET
        FinishY = MIN(LevelDIM,(YPOS+WIDTH))+OFFSET
        FinishZ = MIN(LevelDIM,(ZPOS+WIDTH))+OFFSET
        do M = StartZ,FinishZ
           do L = StartY,FinishY
              do K = StartX,FinishX
                 NTEMP = BoxLOOK(K,L,M)
                 if (NTEMP  >  0) then
                    if (NTEMP /= J) then
                       NNUM = NNUM+1
                       Neighbour(NNUM) = NTEMP
                    end if
                 end if
              end do
           end do
        end do
        if (NNum  >  AvgNbr*NBoxes) then
           CALL WRNDIE (-1, 'FMA', 'Too many neighbors')
        endif
        NPoint(J+1) = NNUM+1
     end do
     OFFSET = OFFSET+(2**(I-1))
  end do
  !
  ! Calculate the interaction list, those distant boxes which interact
  !   with a box
  NNUM = 0
  IntPoint(Start(3)) = 1
  do TREE = 3,Level
     do Box = Start(TREE),Finish(TREE)
        PAR = PARENT(Box)
        do NP = NPoint(PAR),(NPoint(PAR+1)-1)
           NEAR = Neighbour(NP)
           PARFLG = .TRUE.
           TNNUM = 0
           do NEARKID = Child(NEAR),(Child(NEAR)+ChildNO(NEAR)-1)
              PUTFLG = .TRUE.
              do M = NPoint(Box),(NPoint(Box+1)-1)
                 if (NEARKID == Neighbour(M)) then
                    PUTFLG = .FALSE.
                    PARFLG = .FALSE.
                 end if
              end do
              if (PUTFLG) then
                 TNNUM = TNNUM+1
                 LOCAL(TNNUM) = NEARKID
              end if
           end do
           if (PARFLG.AND.(TREE  >  3)) then
              NNUM = NNUM+1
              IntList(NNUM) = NEAR
           else
              do I = 1,TNNUM
                 NNUM = NNUM+1
                 IntList(NNUM) = LOCAL(I)
              end do
           end if
        end do
        if (NNum  >  AvgIntrct*NBoxes) then
           CALL WRNDIE (-1, 'FMA', 'Too many interacting boxes')
        endif
        IntPoint(Box+1) = NNUM+1
     end do
  end do
  !
  RETURN
END SUBROUTINE FMAList

SUBROUTINE FMAExpan(NBoxes,Level,Terms, &
     XL,YL,ZL,CGL, &
     BoxX,BoxY,BoxZ,PhiX,PhiY,PsiX,PsiY, &
     Child,ChildNO,IntList,IntPoint, &
     NPoint,Neighbour,S1C,S2C,S3C,S4C,LEG)
  !-----------------------------------------------------------------------
  !     Description: see above
  !     Robert Nagle
  !
  ! includes
  !
  use chm_kinds
  use dimens_fcm
  use timerm
  use stream
  use exfunc
  use mfmacons
  use deriv
  implicit none
  !
  ! passed
  !
  integer NBoxes,Level,Terms
  integer Child(*),ChildNO(*)
  integer IntList(*),IntPoint(*),NPoint(*),Neighbour(*)
  real(chm_real) XL(*),YL(*),ZL(*),CGL(*)
  real(chm_real) BoxX(*),BoxY(*),BoxZ(*)
  real(chm_real) PhiX(Terms,Terms,*),PhiY(Terms,Terms,*)
  real(chm_real) PsiX(Terms,Terms,*),PsiY(Terms,Terms,*)
  real(chm_real) S1C(2*Terms,2*Terms),LEG(Terms,Terms)
  real(chm_real) S2C(Terms,Terms,Terms,Terms,2)
  real(chm_real) S3C(Terms,Terms,Terms,Terms,2)
  real(chm_real) S4C(Terms,Terms,Terms,Terms,2)
  !
  ! local
  !
  integer TREE,Box,KID,DISTANT,NBox,BoxP,I,N,I2,N2,NN,IN
  integer POINTER
  real(chm_real) R,A,B,ZR
  !
  !---------------------------- begin ----------------------
  !
  !     Calculate expansion for each box
  do Box = Start(Level),Finish(Level)
     I = Child(Box)
     N = ChildNO(Box)
     CALL MEXPAN(Terms,XL(I),YL(I),ZL(I),CGL(I),N, &
          BoxX(Box),BoxY(Box),BoxZ(Box), &
          PhiX(1,1,Box),PhiY(1,1,Box),S1C,LEG)
  end do
#if KEY_DEBUG==1
  if (Timer  >  1) call WRTTIM &
       (' FASTMA: Box expansions calculated')
#endif 
  !     translate the expansion to the parent box, do for each Level
  do TREE = (Level-1),3,-1
     do Box = Start(TREE),Finish(TREE)
        do KID = Child(Box),(Child(Box)+ChildNO(Box)-1)
           CALL ORTOSP(BoxX(Box),BoxY(Box), &
                BoxZ(Box),BoxX(KID),BoxY(KID), &
                BoxZ(KID),R,A,B,ZR)
           CALL BTRANS(R,A,B,ZR,PhiX(1,1,Box), &
                PhiY(1,1,Box),PhiX(1,1,KID), &
                PhiY(1,1,KID),Terms,S2C,LEG)
        end do
     end do
  end do
#if KEY_DEBUG==1
  if (Timer  >  1) call WRTTIM &
       (' FASTMA: Expansions translated to parent boxes')
#endif 
  !
  !     Calculate the local expansion due to distant boxes - boxes to be
  !     considered are given in the interaction list
  !     local expansions are passed down from parent to Child
  do TREE = 3,(Level-1)
     do Box = Start(TREE),Finish(TREE)
        do POINTER = IntPoint(Box),(IntPoint(Box+1)-1)
           DISTANT = IntList(POINTER)
           CALL ORTOSP(BoxX(Box),BoxY(Box), &
                BoxZ(Box),BoxX(DISTANT),BoxY(DISTANT), &
                BoxZ(DISTANT),R,A,B,ZR)
           CALL LOCEXP(Terms,R,A,B,ZR, &
                PhiX(1,1,DISTANT),PhiY(1,1,DISTANT), &
                PsiX(1,1,Box),PsiY(1,1,Box),S3C,LEG)
        end do
        do KID = Child(Box),(Child(Box)+ChildNO(Box)-1)
           CALL ORTOSP(BoxX(KID),BoxY(KID), &
                BoxZ(KID),BoxX(Box),BoxY(Box), &
                BoxZ(Box),R,A,B,ZR)
           CALL LETRAN(Terms,R,A,B,ZR, &
                PsiX(1,1,KID),PsiY(1,1,KID), &
                PsiX(1,1,Box),PsiY(1,1,Box),S4C,LEG)
        end do
     end do
  end do
#if KEY_DEBUG==1
  if (Timer  >  1) call WRTTIM &
       (' FASTMA: After local expansions from distant boxes')
#endif 
  !
  !     Calculate the local expansion at the lowest Level
  do Box = Start(Level),Finish(Level)
     do POINTER = IntPoint(Box),(IntPoint(Box+1)-1)
        DISTANT = IntList(POINTER)
        call ORTOSP(BoxX(Box),BoxY(Box), &
             BoxZ(Box),BoxX(DISTANT),BoxY(DISTANT), &
             BoxZ(DISTANT),R,A,B,ZR)
        call LOCEXP(Terms,R,A,B,ZR, &
             PhiX(1,1,DISTANT),PhiY(1,1,DISTANT), &
             PsiX(1,1,Box),PsiY(1,1,Box),S3C,LEG)
     end do
  end do
#if KEY_DEBUG==1
  if (Timer  >  1) call WRTTIM &
       (' FASTMA: After Local Expansions at lowest Level')
#endif 
  RETURN
END SUBROUTINE FMAExpan

SUBROUTINE FMAEval(ELEC,Enb,NAtom,NBoxes,Level,Terms, &
     XL, YL, ZL, DXL, DYL, DZL, CGL, &
     BoxX, BoxY, BoxZ, PhiX, PhiY, PsiX, PsiY, ActAt, &
     ATID, Child, ChildNo, NPoint, Neighbour, S1C, Leg)
  !-----------------------------------------------------------------------
  !     Description: see above
  !     Robert Nagle
  !
  ! includes
  !
  use chm_kinds
  use dimens_fcm
  use timerm
  use stream
  use exfunc
  use mfmacons
  use deriv
  implicit none
  !
  ! passed
  !
  integer NAtom,NBoxes,Level,Terms, ActAt
  integer ATID(*),Child(*),ChildNO(*)
  integer NPoint(*),Neighbour(*)
  real(chm_real) Elec, Enb
  real(chm_real) XL(*),YL(*),ZL(*),DXL(*),DYL(*),DZL(*),CGL(*)
  real(chm_real) BoxX(*),BoxY(*),BoxZ(*)
  real(chm_real) S1C(2*Terms,2*Terms),LEG(Terms,Terms)
  real(chm_real) PhiX(Terms,Terms,*),PhiY(Terms,Terms,*)
  real(chm_real) PsiX(Terms,Terms,*),PsiY(Terms,Terms,*)
  !
  ! local
  !
  integer Box,KID,DISTANT,NBox,BoxP,I,N,I2,N2,NN,IN
  !
  !---------------------------- begin ----------------------
  !
  if (Level  >  2) then
     !     Calculate the potential and forces due to the expansion
     do Box = Start(Level),Finish(Level)
        I = Child(Box)
        N = ChildNO(Box)
        call FARPOT(Terms,N,XL(I),YL(I),ZL(I),CGL(I), &
             BoxX(Box),BoxY(Box),BoxZ(Box), &
             PsiX(1,1,Box),PsiY(1,1,Box), &
             DXL(I),DYL(I),DZL(I),ELEC,S1C,LEG)
     end do
  end if
  !
  !     Calculate the direct potential and forces due to close atoms
#if KEY_DEBUG==1
  if (Timer  >  1) call WRTTIM &
       (' FASTMA: After Far Potential/Forces calculation')
#endif 
  do Box = Start(Level),Finish(Level)
     I = Child(Box)
     N = ChildNO(Box)
     call DIRECT(N,XL(I),YL(I),ZL(I),CGL(I),DXL(I),DYL(I), &
          DZL(I), ATID(I), Elec, ENB)
     IN = NPoint(Box)
     NN = NPoint(Box+1)-1
     do BoxP = IN,NN
        NBox = Neighbour(BoxP)
        if (NBox  >  Box) then
           I2 = Child(NBox)
           N2 = ChildNO(NBox)
           call DIRECT2(N,XL(I),YL(I),ZL(I),CGL(I),DXL(I), &
                DYL(I),DZL(I),N2,XL(I2),YL(I2),ZL(I2), &
                CGL(I2),DXL(I2),DYL(I2),DZL(I2), ATID(I2), &
                Elec, Enb)
        end if
     end do
  end do
#if KEY_DEBUG==1
  if (Timer  >  1) call WRTTIM &
       (' FASTMA: After all direct computations')
#endif 
  ! Transfer the temporary force accumulators to the proper place
  do I = 1,ACTAT
     DX(ATID(I)) = DX(ATID(I))+DXL(I)
     DY(ATID(I)) = DY(ATID(I))+DYL(I)
     DZ(ATID(I)) = DZ(ATID(I))+DZL(I)
  end do
  !
  RETURN
END SUBROUTINE FMAEval

SUBROUTINE FMACons(NAtom,NBoxes,Level,Terms,DXL,DYL,DZL, &
     S1C,S2C,S3C,S4C)
  !-----------------------------------------------------------------------
  !     This routine initialises the spherical harmonic constants which
  !     will be used later. They are taken from john board's code with
  !     some modifications to decrease array usage.
  !     Robert Nagle/Paul Adams
  !
  ! includes
  !
  use chm_kinds
  use number
  use mfmacons
  implicit none
  !
  ! passed
  !
  integer NAtom,NBoxes,Level,Terms
  real(chm_real) DXL(*),DYL(*),DZL(*)
  real(chm_real) S1C(2*Terms,2*Terms)
  real(chm_real) S2C(Terms,Terms,Terms,Terms,2)
  real(chm_real) S3C(Terms,Terms,Terms,Terms,2)
  real(chm_real) S4C(Terms,Terms,Terms,Terms,2)
  !
  ! local
  !
  integer I,J,K,L,NP,MP,IK,LJ,JL
  real(chm_real) TEMP,BESSEL,RI
  real(chm_real) FACT(4*MXTERM),A(2*MXTERM,2*MXTERM)
  !
  !---------------------------- begin ----------------------
  !
  ! Zero local force accumulators
  do I = 1,NAtom
     DXL(I) = ZERO
     DYL(I) = ZERO
     DZL(I) = ZERO
  end do
  !
  ! Temporary array used for constant calculation
  TEMP = ONE
  do I = 1,(4*(Terms-1)+1)
     FACT(I) = TEMP
     RI=I
     TEMP = TEMP*SQRT(RI)
  end do
  ! Temporary array used for constant calculation
  TEMP = ONE
  do I = 1,(2*(Terms-1)+1)
     do J = 1,I
        A(I,J) = TEMP/(FACT(I-J+1)*FACT(I+J-1))
     end do
     TEMP = -TEMP
  end do
  ! Spherical harmonic constant array 1
  do I = 1,(2*(Terms-1)+1)
     do J = 1,I
        S1C(I,J) = (FACT(I-J+1)/FACT(I+J-1))
     end do
  end do
  ! Spherical harmonic constant array 2
  do I = 1,Terms
     do J = 1,I
        do K = 1,I
           do L = 1,K
              JL = J-L
              IK = I-K
              if (ABS(JL)  <=  IK) then
                 IF ((L-1)*JL  <  0) then
                    if (MOD(MIN(ABS(JL),(L-1)),2)  /=  0) then
                       BESSEL = -ONE
                    else
                       BESSEL = ONE
                    end if
                 else
                    BESSEL = ONE
                 end if
                 S2C(I,J,K,L,1) = &
                      BESSEL*A(K,L)*A((IK+1),(ABS(JL)+1)) &
                      *S1C(K,L)/A(I,J)
              else
                 S2C(I,J,K,L,1) = 0.0
              end if
              JL = (J-1)+(L-1)
              IK = I-K
              if (JL  <=  IK) then
                 if ((1-L)*JL  <  0) then
                    if (MOD(MIN(JL,(L-1)),2)  /=  0) then
                       BESSEL = -ONE
                    else
                       BESSEL = ONE
                    end if
                 else
                    BESSEL = ONE
                 end if
                 S2C(I,J,K,L,2) = &
                      BESSEL*A(K,L)*A(IK+1,JL+1)*S1C(K,L)/A(I,J)
              else
                 S2C(I,J,K,L,2) = 0.0
              end if
           end do
        end do
     end do
  end do
  !
  ! Spherical harmonic constant array 3
  do I = 1,Terms
     do J = 1,I
        do K = 1,Terms
           do L = 1,K
              if ((J  ==  1) .or. (L .eq. 1)) then
                 BESSEL = ONE
              else
                 if (MOD(MIN(J,L),2)  ==  0) then
                    BESSEL = -ONE
                 else
                    BESSEL = ONE
                 end if
              end if
              if (MOD(k,2)  ==  0) then
                 BESSEL = -BESSEL
              end if
              IK = I+K-1
              LJ = ABS(L-J)+1
              S3C(I,J,K,L,1) = &
                   BESSEL*A(K,L)*A(I,J)*S1C(IK,LJ)/A(IK,LJ)
              BESSEL = ONE
              if (MOD(K,2)  ==  0) then
                 BESSEL = -BESSEL
              end if
              IK = I+K-1
              LJ = ABS((1-L)-(J-1))+1
              S3C(I,J,K,L,2) = &
                   BESSEL*A(K,L)*A(I,J)*S1C(IK,LJ)/A(IK,LJ)
           end do
        end do
     end do
  end do
  !
  ! Spherical harmonic constant array 4
  do I = 1,Terms
     do J = 1,I
        do K = 1,(Terms+1-I)
           do L = 1,K
              NP = K+I-1
              MP = L+J-1
              if ((NP  <=  Terms) .and. (MP .le. Terms)) then
                 BESSEL = A(K,L)*A(I,J)*S1C(K,L)/A(NP,MP)
                 IF (MOD(I,2) == 0) BESSEL = -BESSEL
                 IF (MOD(NP,2) == 0) BESSEL = -BESSEL
                 IF (MP == 1) THEN
                    IF (MOD(J,2) == 0) BESSEL = -BESSEL
                 END IF
                 S4C(I,J,K,L,1) = BESSEL
              END IF
              NP = K+I-1
              MP = J-L
              if ((NP <= Terms).AND.(ABS(MP)+1 .le. Terms)) then
                 BESSEL = A(K,L)*A(I,J)*S1C(K,L)/A(NP,ABS(MP)+1)
                 if (MOD(I,2) == 0) BESSEL = -BESSEL
                 if (MOD(NP,2) == 0) BESSEL = -BESSEL
                 if (ABS(MP) == 0) then
                    if (MOD(J,2) == 0) BESSEL = -BESSEL
                 end if
                 if ((MP < 0).AND.(J  >  1)) then
                    if (MOD(J,2) == 0) BESSEL = -BESSEL
                 end if
                 if ((MP  >  0).AND.((-L) < -1)) then
                    if (MOD(L,2) == 0) BESSEL = -BESSEL
                 end if
                 S4C(I,J,K,L,2) = BESSEL
              end if
           end do
        end do
     end do
  end do
  !
  RETURN
END SUBROUTINE FMACons

SUBROUTINE BoxSIZ(NAtom,X,Y,Z,BoxDim,BoxX,BoxY,BoxZ)
  !-----------------------------------------------------------------------
  !     Calculate the size of the system - will be cubic
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  use stream
  implicit none
  !
  ! Passed
  !
  integer NAtom
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) BoxX(*),BoxY(*),BoxZ(*)
  real(chm_real) BoxDIM
  !
  ! Local
  !
  integer I
  real(chm_real) MAXX,MAXY,MAXZ,MINX,MINY,MINZ
  !
  !---------------------------- begin ----------------------
  !
  MAXX = -ANUM
  MAXY = -ANUM
  MAXZ = -ANUM
  MINX = ANUM
  MINY = ANUM
  MINZ = ANUM
  !
  do I = 1,NAtom
     if (X(I)  >  MAXX) MAXX = X(I)
     if (Y(I)  >  MAXY) MAXY = Y(I)
     if (Z(I)  >  MAXZ) MAXZ = Z(I)
     if (X(I) < MINX) MINX = X(I)
     if (Y(I) < MINY) MINY = Y(I)
     if (Z(I) < MINZ) MINZ = Z(I)
  end do
  !
  BoxX(1) = (MAXX+MINX)/TWO
  BoxY(1) = (MAXY+MINY)/TWO
  BoxZ(1) = (MAXZ+MINZ)/TWO
  !
  MAXX = MAXX-MINX
  MAXY = MAXY-MINY
  MAXZ = MAXZ-MINZ
  !
  BoxDIM = MAX(MAXX,MAXY,MAXZ)+HALF
  !
  RETURN
END SUBROUTINE BoxSIZ

SUBROUTINE DVIDE(NAtom,X,Y,Z,CG,BoxDIM, &
     Level,XL,YL,ZL,CGL,BoxX,BoxY,BoxZ,ATID, &
     Child, ChildNO, PARENT, NPoint, Temp, ACTAT)
  !-----------------------------------------------------------------------
  !     Partition the atoms into Boxes, removing Boxes that contain no
  !     atoms.
  !     determine the interaction list for each Box.
  !     ChildNO - number of sub entries (for lowest Level of the grid
  !                this is the number of atoms in that cell element)
  !     NPoint -
  !     Child -
  !     BoxLOOK -
  !
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  use stream
  use mfmacons
  implicit none
  !
  ! Passed
  !
  integer NAtom,Level,ACTAT
  integer ATID(*),Child(*),ChildNO(*), Parent(*)
  integer NPoint(*),  Temp(*)
  real(chm_real) X(*),Y(*),Z(*),CG(*)
  real(chm_real) XL(*),YL(*),ZL(*),CGL(*)
  real(chm_real) BoxX(*),BoxY(*),BoxZ(*)
  real(chm_real) BoxDIM
  !
  ! Local
  !
  integer I,J,K,L,M,CNUM,NTEMP,NNUM,LevelDIM
  integer StartX,StartY,StartZ,FinishX,FinishY,FinishZ
  integer Box,TEMPS,TREE,PAR,NP,NEAR,NEARKID,TNNUM
  integer LOCAL(8),WIDTH,POINTER,PUTP,LEV,SIDE
  integer XPOS,YPOS,ZPOS,ChildTEMP,OFFSET
  real(chm_real) LENGTH,ORIGX,ORIGY,ORIGZ
  !
  !---------------------------- begin ----------------------
  !
  !     First allocate the box hierarchy - box centre_{xyz} arrays
  PARENT(1) = 1
  CNUM = 1
  LENGTH = BoxDIM
  do I = 1,(Level-1)
     LENGTH = LENGTH/TWO
     do M = Start(I),Finish(I)
        Child(M) = (CNUM+1)
        ChildNO(M) = 8
        ORIGX = BoxX(M)-LENGTH
        ORIGY = BoxY(M)-LENGTH
        ORIGZ = BoxZ(M)-LENGTH
        do J = 1,2
           do K = 1,2
              !
              !     THE FOLLOWING IS NEEDED TO STOP THE SGI SERIAL OPTIMISER
              !     DESTROYING THIS LOOP
              !*$*UNROLL(1)
              do L = 1,2
                 CNUM = CNUM+1
                 BoxX(CNUM) = (L-1)*LENGTH+(LENGTH/TWO)+ORIGX
                 BoxY(CNUM) = (K-1)*LENGTH+(LENGTH/TWO)+ORIGY
                 BoxZ(CNUM) = (J-1)*LENGTH+(LENGTH/TWO)+ORIGZ
                 PARENT(CNUM) = M
              end do
           end do
        end do
     end do
  end do
  !
  do I = 1,NAtom
     TEMP(I) = 1
  end do
  !
  !     Determine which box an atom belongs to from its position
  LENGTH = BoxDIM
  do I = 1,(Level-1)
     LENGTH = LENGTH/TWO
     do J = 1,NAtom
        XPOS = INT((X(J)-BoxX(TEMP(J))+LENGTH)/LENGTH)
        YPOS = INT((Y(J)-BoxY(TEMP(J))+LENGTH)/LENGTH)
        ZPOS = INT((Z(J)-BoxZ(TEMP(J))+LENGTH)/LENGTH)
        TEMP(J) = Child(TEMP(J))+(XPOS)*1+(YPOS)*2+(ZPOS)*4
     end do
  end do
  !
  !     THROW OUT ATOMS WITH ZERO CHARGE
  do I = 1,(Finish(Level)+1)
     NPoint(I) = 0
  end do
  !
  NPoint(Start(Level)) = 1
  do I = 1,NAtom
     if (ABS(CG(I))  >  RSMALL) then
        NPoint(TEMP(I)+1) = NPoint(TEMP(I)+1)+1
     end if
  end do
  !
  do I = Start(Level),Finish(Level)
     ChildNO(I) = NPoint(I+1)
     NPoint(I+1) = NPoint(I+1)+NPoint(I)
     Child(I) = NPoint(I)
  end do
  !
  ACTAT = NPoint(Finish(Level)+1)-1
  !
  do I = 1,NAtom
     if (ABS(CG(I))  >  RSMALL) then
        ATID(NPoint(TEMP(I))) = I
        NPoint(TEMP(I)) = NPoint(TEMP(I))+1
     end if
  end do
  !
  !     FILL THE TEMPORARY ATOM ARRAYS - ONLY USE CHARGED ATOMS
  do I = 1,ACTAT
     XL(I) = X(ATID(I))
     YL(I) = Y(ATID(I))
     ZL(I) = Z(ATID(I))
     CGL(I) = CG(ATID(I))
  end do
  !
  !     Remove empty boxes from the hierarchy
  do I = 1,(Level-1)
     L = Level-I
     do J = Start(L),Finish(L)
        ChildTEMP = 0
        do K = 1,ChildNO(J)
           if (ChildNO(Child(J)+K-1)  >  0) then
              ChildTEMP = ChildTEMP+1
           end if
        end do
        ChildNO(J) = ChildTEMP
     end do
  end do
  !
  PUTP = 1
  POINTER = ChildNO(1)+2
  do LEV = 2,(Level-1)
     TEMPS = PUTP+1
     do I = Start(LEV),Finish(LEV)
        if (ChildNO(I)  >  0) then
           PUTP = PUTP+1
           Child(PUTP) = POINTER
           POINTER = POINTER+ChildNO(I)
           BoxX(PUTP) = BoxX(I)
           BoxY(PUTP) = BoxY(I)
           BoxZ(PUTP) = BoxZ(I)
           ChildNO(PUTP) = ChildNO(I)
        end if
     end do
     Start(LEV) = TEMPS
     Finish(LEV) = PUTP
  end do
  TEMPS = PUTP+1
  do I = Start(Level),Finish(Level)
     if (ChildNO(I)  >  0) then
        PUTP = PUTP+1
        Child(PUTP) = Child(I)
        BoxX(PUTP) = BoxX(I)
        BoxY(PUTP) = BoxY(I)
        BoxZ(PUTP) = BoxZ(I)
        ChildNO(PUTP) = ChildNO(I)
     end if
  end do
  Start(Level) = TEMPS
  Finish(Level) = PUTP
  !
  !     Cull NOW
  !
  !
  !     Make sure boxes have the correct parents
  do LEV = 1,(Level-1)
     do I = Start(LEV),Finish(LEV)
        do J = 1,ChildNO(I)
           PARENT(Child(I)+J-1) = I
        end do
     end do
  end do
  RETURN
END SUBROUTINE DVIDE

SUBROUTINE MEXPAN(Terms,XL,YL,ZL,CGL,N,ORIGX,ORIGY,ORIGZ, &
     PhiX,PhiY,S1C,LEG)
  !-----------------------------------------------------------------------
  !     Calculate the expansion at the centre of a Box due to
  !     the charges in the Box
  !     USES THE OPTIMISATIONS PRESENT IN JOHN BOARD'S CODE
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  use mfmacons
  implicit none
  !
  ! Passed
  !
  integer Terms,N
  real(chm_real) ORIGX,ORIGY,ORIGZ
  real(chm_real) XL(N),YL(N),ZL(N),CGL(N)
  real(chm_real) PhiX(Terms,Terms),PhiY(Terms,Terms)
  real(chm_real) S1C(2*Terms,2*Terms),LEG(Terms,Terms)
  !
  ! Local
  !
  integer I,J,K
  real(chm_real) R,ZR,A,B,RTERM,TEMP,CGIL
  real(chm_real) TEMPX(MXTERM),TEMPY(MXTERM)
  !
  !---------------------------- begin ----------------------
  !
  TEMPX(1) = ONE
  TEMPY(1) = ZERO
  do I = 1,N
     if (ABS(CGL(I))  >  RSMALL) then
        CGIL = CGL(I)
        call ORTOSP(ORIGX,ORIGY,ORIGZ,XL(I),YL(I),ZL(I), &
             R,A,B,ZR)
        call LEGENDRE(LEG,ZR,Terms)
        do J = 2,Terms
           TEMPX(J) = COS((J-ONE)*B)
           TEMPY(J) = SIN((ONE-J)*B)
        end do
        PhiX(1,1) = PhiX(1,1)+CGIL
        RTERM = R
        do J = 2,Terms
           PhiX(J,1) = PhiX(J,1)+(CGIL*RTERM*LEG(J,1))
           do K = 2,J
              TEMP = CGIL*RTERM*S1C(J,K)*LEG(J,K)
              PhiX(J,K) = PhiX(J,K)+TEMP*TEMPX(K)
              PhiY(J,K) = PhiY(J,K)+TEMP*TEMPY(K)
           end do
           RTERM = RTERM*R
        end do
     endif
  end do
  !
  RETURN
END SUBROUTINE MEXPAN

SUBROUTINE LEGENDRE(LEG,VALUE,Terms)
  !-----------------------------------------------------------------------
  !     Calculate the legendre polynomial for value in array
  !     leg(Terms,Terms)
  !     USES THE OPTIMISATIONS PRESENT IN JOHN BOARD'S CODE
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  implicit none
  !
  ! Passed
  !
  integer Terms
  real(chm_real) VALUE
  real(chm_real) LEG(Terms,Terms)
  !
  ! Local
  !
  integer I,J
  real(chm_real) NT,OF,NO,ROOT,ST
  !
  !---------------------------- begin ----------------------
  !
  NT = ONE
  OF = ONE
  NO = ONE
  ROOT = SQRT(ONE-(VALUE*VALUE))
  ST = ONE
  do I = 1,(Terms-1)
     LEG(I,I) = NT*OF*ST
     NT = -NT
     OF = OF*NO
     NO = NO + TWO
     ST = ST*ROOT
     LEG(I+1,I) = VALUE*(TWO*(I-1)+1)*LEG(I,I)
     do J = (I+2),Terms
        LEG(J,I) = (VALUE*(TWO*(J-1)-1)*LEG(J-1,I) &
             -((J-1)+(I-1)-1)*LEG(J-2,I))/(J-I)
     end do
  end do
  LEG(Terms,Terms) = NT*OF*ST
  !
  RETURN
END SUBROUTINE LEGENDRE

SUBROUTINE BTRANS(R,A,B,ZR,PhiXP,PhiYP,PhiXC,PhiYC,Terms, &
     S2C,LEG)
  !-----------------------------------------------------------------------
  !     Translate the expansion at the centre of a Box to the centre
  !     of the parent Box
  !     USES THE OPTIMISATIONS PRESENT IN JOHN BOARD'S CODE
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  use mfmacons
  implicit none
  !
  ! Passed
  !
  integer Terms
  real(chm_real) R,A,B,ZR
  real(chm_real) PhiXP(Terms,Terms),PhiYP(Terms,Terms)
  real(chm_real) PhiXC(Terms,Terms),PhiYC(Terms,Terms)
  real(chm_real) LEG(Terms,Terms)
  real(chm_real) S2C(Terms,Terms,Terms,Terms,2)
  !
  ! Local
  !
  integer I,J,K,M,N,KM,JN
  real(chm_real) RTEMP,TEMP,OX,OY,S2X,S2Y,SUMX,SUMY,SIGNFAC
  real(chm_real) TEMPX(MXTERM),TEMPY(MXTERM)
  real(chm_real) SCALEX(MXTERM,MXTERM),SCALEY(MXTERM,MXTERM)
  !
  !---------------------------- begin ----------------------
  !
  call LEGENDRE(LEG,ZR,Terms)
  TEMPX(1) = ONE
  TEMPY(1) = ZERO
  do I = 2,Terms
     TEMPX(I) = COS((I-ONE)*B)
     TEMPY(I) = SIN((ONE-I)*B)
  end do
  RTEMP = ONE
  do N = 1,Terms
     do M = 1,N
        TEMP = RTEMP*LEG(N,M)
        SCALEX(N,M) = TEMP*TEMPX(M)
        SCALEY(N,M) = TEMP*TEMPY(M)
     end do
     RTEMP = RTEMP*R
  end do
  do J = 1,Terms
     do K = 1,J
        SUMX = ZERO
        SUMY = ZERO
        do N = 1,J
           JN = J-N+1
           TEMP = S2C(J,K,N,1,1)
           OX = PhiXC(JN,K)
           OY = PhiYC(JN,K)
           S2X = SCALEX(N,1)
           SUMX = SUMX+(TEMP*OX*S2X)
           SUMY = SUMY+(TEMP*OY*S2X)
           do M = 2,N
              S2X = SCALEX(N,M)
              S2Y = SCALEY(N,M)
              KM = K-M
              SIGNFAC = SIGN(1,KM)
              KM = ABS(KM)+1
              if (KM <= JN) then
                 TEMP = S2C(J,K,N,M,1)
                 OX = PhiXC(JN,KM)
                 OY = SIGNFAC*PhiYC(JN,KM)
                 SUMX = SUMX+TEMP*(OX*S2X-OY*S2Y)
                 SUMY = SUMY+TEMP*(OX*S2Y+OY*S2X)
              end if
              KM = K+M-1
              if (KM <= JN) then
                 TEMP = S2C(J,K,N,M,2)
                 OX = PhiXC(JN,KM)
                 OY = PhiYC(JN,KM)
                 S2Y = -S2Y
                 SUMX = SUMX+TEMP*(OX*S2X-OY*S2Y)
                 SUMY = SUMY+TEMP*(OX*S2Y+OY*S2X)
              end if
           end do
        end do
        PhiXP(J,K) = PhiXP(J,K)+SUMX
        PhiYP(J,K) = PhiYP(J,K)+SUMY
     end do
  end do
  !
  RETURN
END SUBROUTINE BTRANS

SUBROUTINE ORTOSP(OX,OY,OZ,PX,PY,PZ,R,A,B,ZR)
  !-----------------------------------------------------------------------
  !     Convert the vector between two points o_{xyz} and p_{xyz} to
  !     spherical polar coordinates
  !     USES THE OPTIMISATIONS PRESENT IN JOHN BOARD'S CODE
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  implicit none
  !
  ! Passed
  !
  real(chm_real) OX,OY,OZ,PX,PY,PZ
  real(chm_real) A,B,R,ZR
  !
  ! Local
  !
  real(chm_real) DeltAX,DeltAY,DeltAZ
  !
  !---------------------------- begin ----------------------
  !
  DeltAX = PX-OX
  DeltAY = PY-OY
  DeltAZ = PZ-OZ
  R = SQRT(DeltAX*DeltAX+DeltAY*DeltAY+DeltAZ*DeltAZ)
  ZR = (DeltAZ/R)
  A = ACOS(ZR)
  !
  if ((DeltAX == ZERO).AND.(DeltAY.EQ.ZERO)) then
     B = ZERO
  else
     B = ATAN2(DeltAY,DeltAX)
  end if
  !
  RETURN
END SUBROUTINE ORTOSP

SUBROUTINE LOCEXP(Terms,R,A,B,ZR,PhiX,PhiY,PsiX,PsiY,S3C,LEG)
  !-----------------------------------------------------------------------
  !     Calculate the local expansion at a Box centre due to distant Boxes
  !     USES THE OPTIMISATIONS PRESENT IN JOHN BOARD'S CODE
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  use mfmacons
  implicit none
  !
  ! Passed
  !
  integer Terms
  real(chm_real) R,A,B,ZR
  real(chm_real) PhiX(Terms,Terms),PhiY(Terms,Terms)
  real(chm_real) PsiX(Terms,Terms),PsiY(Terms,Terms)
  real(chm_real) LEG(Terms,Terms)
  real(chm_real) S3C(Terms,Terms,Terms,Terms,2)
  !
  ! Local
  !
  integer I,J,K,M,N,MK,KM
  real(chm_real) RTEMP,TEMP,OX,OY,TX,TY,S3X,S3Y,S3XC,S3YC
  real(chm_real) SIGNFAC
  real(chm_real) TEMPX(MXTERM),TEMPY(MXTERM)
  !
  !---------------------------- begin ----------------------
  !
  TEMPX(1) = ONE
  TEMPY(1) = ZERO
  do I = 2,Terms
     TEMPX(I) = COS((I-ONE)*B)
     TEMPY(I) = SIN((I-ONE)*B)
  end do
  call LEGENDRE(LEG,ZR,Terms)
  RTEMP = ONE/R
  TEMP = RTEMP
  do N = 1,Terms
     do M = 1,N
        LEG(N,M) = LEG(N,M)*RTEMP
     end do
     RTEMP = RTEMP*TEMP
  end do
  do I = 1,Terms
     OX = PhiX(I,1)
     do J = 1,(Terms+1-I)
        PsiX(J,1) = PsiX(J,1)+S3C(J,1,I,1,1)*LEG((I+J-1),1)*OX
     end do
     do K = 2,(Terms+1-I)
        S3X = OX*TEMPX(K)
        S3Y = -OX*TEMPY(K)
        do J = K,(Terms+1-I)
           TEMP = S3C(J,K,I,1,1)*LEG((I+J-1),K)
           PsiX(J,K) = PsiX(J,K)+TEMP*S3X
           PsiY(J,K) = PsiY(J,K)+TEMP*S3Y
        end do
     end do
     do M = 2,I
        OX = PhiX(I,M)
        OY = PhiY(I,M)
        TX = TEMPX(M)
        TY = TEMPY(M)
        S3X = OX*TX-OY*TY
        S3Y = OX*TY+OY*TX
        do J = 1,(Terms+1-I)
           TEMP = S3C(J,1,I,M,1)*LEG((J+I-1),M)
           RTEMP = S3C(J,1,I,M,2)*LEG((J+I-1),M)
           PsiX(J,1) = PsiX(J,1)+(TEMP+RTEMP)*S3X
           PsiY(J,1) = PsiY(J,1)+(TEMP-RTEMP)*S3Y
        end do
        do K = 2,(Terms+1-I)
           MK = M-K
           SIGNFAC = SIGN(1,MK)
           MK = ABS(MK)+1
           KM = M+K-1
           TX = TEMPX(MK)
           TY = SIGNFAC*TEMPY(MK)
           S3X = OX*TX-OY*TY
           S3Y = OX*TY+OY*TX
           TX = TEMPX(KM)
           TY = -TEMPY(KM)
           S3XC = OX*TX+OY*TY
           S3YC = OX*TY-OY*TX
           do J = K,(Terms+1-I)
              TEMP = S3C(J,K,I,M,1)*LEG((J+I-1),MK)
              RTEMP = S3C(J,K,I,M,2)*LEG((J+I-1),KM)
              PsiX(J,K) = PsiX(J,K)+TEMP*S3X+RTEMP*S3XC
              PsiY(J,K) = PsiY(J,K)+TEMP*S3Y+RTEMP*S3YC
           end do
        end do
     end do
  end do
  !
  RETURN
END SUBROUTINE LOCEXP

SUBROUTINE LETRAN(Terms,R,A,B,ZR,PsiXD,PsiYD,PsiXS,PsiYS, &
     S4C,LEG)
  !-----------------------------------------------------------------------
  !     Translate a local expansion to a Child Box centre
  !     USES THE OPTIMISATIONS PRESENT IN JOHN BOARD'S CODE
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  use mfmacons
  implicit none
  !
  ! Passed
  !
  integer Terms
  real(chm_real) R,A,B,ZR
  real(chm_real) PsiXD(Terms,Terms),PsiYD(Terms,Terms)
  real(chm_real) PsiXS(Terms,Terms),PsiYS(Terms,Terms)
  real(chm_real) LEG(Terms,Terms)
  real(chm_real) S4C(Terms,Terms,Terms,Terms,2)
  !
  ! Local
  !
  integer I,J,K,M,N,MK,KM,NP
  real(chm_real) RTEMP,TEMP,OX,OY,S3X,S3Y
  real(chm_real) SIGNFAC,SUMX,SUMY
  real(chm_real) TEMPX(MXTERM),TEMPY(MXTERM)
  real(chm_real) SCALEX(MXTERM,MXTERM),SCALEY(MXTERM,MXTERM)
  !
  !---------------------------- begin ----------------------
  !
  TEMPX(1) = ONE
  TEMPY(1) = ZERO
  do I = 2,Terms
     TEMPX(I) = COS((I-ONE)*B)
     TEMPY(I) = SIN((I-ONE)*B)
  end do
  call LEGENDRE(LEG,ZR,Terms)
  RTEMP = ONE
  do N = 1,Terms
     do M = 1,N
        TEMP = RTEMP*LEG(N,M)
        SCALEX(N,M) = TEMP*TEMPX(M)
        SCALEY(N,M) = TEMP*TEMPY(M)
     end do
     RTEMP = RTEMP*R
  end do
  do J = 1,Terms
     do K = 1,J
        SUMX = ZERO
        SUMY = ZERO
        do N = 1,(Terms+1-J)
           NP = (N+J-1)
           TEMP = S4C(J,K,N,1,1)*SCALEX(N,1)
           OX = PsiXS(NP,K)
           OY = PsiYS(NP,K)
           SUMX = SUMX+TEMP*OX
           SUMY = SUMY+TEMP*OY
           do M = 2,N
              MK = M+K-1
              TEMP = S4C(J,K,N,M,1)
              OX = PsiXS(NP,MK)
              OY = PsiYS(NP,MK)
              S3X = SCALEX(N,M)
              S3Y = SCALEY(N,M)
              SUMX = SUMX+TEMP*(OX*S3X-OY*S3Y)
              SUMY = SUMY+TEMP*(OX*S3Y+OY*S3X)
              KM = K-M
              SIGNFAC = SIGN(1,KM)
              KM = ABS(KM)+1
              TEMP = S4C(J,K,N,M,2)
              OX = PsiXS(NP,KM)
              OY = SIGNFAC*PsiYS(NP,KM)
              S3Y = -S3Y
              SUMX = SUMX+TEMP*(OX*S3X-OY*S3Y)
              SUMY = SUMY+TEMP*(OX*S3Y+OY*S3X)
           end do
        end do
        PsiXD(J,K) = PsiXD(J,K)+SUMX
        PsiYD(J,K) = PsiYD(J,K)+SUMY
     end do
  end do
  !
  RETURN
END SUBROUTINE LETRAN

SUBROUTINE FARPOT(Terms,N,XL,YL,ZL,CGL,ORIGX,ORIGY,ORIGZ, &
     PsiX,PsiY,DXL,DYL,DZL,ELEC,S1C,LEG)
  !-----------------------------------------------------------------------
  !     Calculate the potential and gradients for atoms in a Box due to
  !     atoms in distant Boxes
  !     USES THE OPTIMISATIONS PRESENT IN JOHN BOARD'S CODE
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  use dimens_fcm
  use consta
  use inbnd
  use mfmacons
  implicit none
  !
  ! Passed
  !
  integer Terms,N
  real(chm_real) XL(N),YL(N),ZL(N),CGL(N)
  real(chm_real) ORIGX,ORIGY,ORIGZ,ELEC
  real(chm_real) DXL(N),DYL(N),DZL(N)
  real(chm_real) PsiX(Terms,Terms),PsiY(Terms,Terms)
  real(chm_real) S1C(2*Terms,2*Terms),LEG(Terms,Terms)
  !
  ! Local
  !
  integer AT,I,J
  real(chm_real) CGF,CGI,R,A,B,ZR,ZR2,LR,LA,LB,LPOT,RTERM
  real(chm_real) NI,SI,SUM1,SUM2,SUM3,COSTERM,SINTERM
  real(chm_real) TEMP1,TEMP2,TEMP3,COSTHETA,SINTHETA
  real(chm_real) COSPhi,SINPhi,DeltX,DeltY,DeltZ,ELECL
  real(chm_real) TEMPX(MXTERM),TEMPY(MXTERM)
  real(chm_real) SINA
  !
  !---------------------------- begin ----------------------
  !
  CGF = CCELEC/EPS
  ELECL = ZERO
  TEMPX(1) = ONE
  TEMPY(1) = ZERO
  do AT = 1,N
     if (ABS(CGL(AT))  >  RSMALL) then
        CGI = CGL(AT)*CGF
        call ORTOSP(ORIGX,ORIGY,ORIGZ,XL(AT),YL(AT),ZL(AT), &
             R,A,B,ZR)
        call LEGENDRE(LEG,ZR,Terms)
        do I = 2,Terms
           TEMPX(I) = COS((I-ONE)*B)
           TEMPY(I) = SIN((I-ONE)*B)
        end do
        SINA = SIN(A)
        SINTHETA = SINA
        COSTHETA = ZR
        COSPhi = TEMPX(2)
        SINPhi = TEMPY(2)
        ZR2 = ZR*ZR
        NI = ZR/(ONE-ZR2)
        SI = ONE/SQRT(ONE-ZR2)
        LPOT = PsiX(1,1)
        LR = ZERO
        LA = ZERO
        LB = ZERO
        RTERM = R
        do I = 2,Terms
           SUM1 = ZERO
           SUM2 = ZERO
           SUM3 = ZERO
           do J = 2,I
              TEMP1 = S1C(I,J)*LEG(I,J)
              COSTERM = TEMPX(J)
              SINTERM = TEMPY(J)
              TEMP3 = (COSTERM*PsiX(I,J)-SINTERM*PsiY(I,J))
              SUM1 = SUM1+TEMP1*TEMP3
              if (J < I) then
                 TEMP2 = S1C(I,J)* &
                      ((1-J)*NI*LEG(I,J)-LEG(I,(J+1))*SI)
                 SUM2 = SUM2+TEMP2*TEMP3
              else
                 TEMP2 = S1C(I,J)*((1-J)*NI*LEG(I,J))
                 SUM2 = SUM2+TEMP2*TEMP3
              end if
              SUM3 = SUM3+ &
                   (J-1)*TEMP1*(SINTERM*PsiX(I,J)+COSTERM*PsiY(I,J))
           end do
           LR = LR+(I-1)*RTERM*(TWO*SUM1+LEG(I,1)*PsiX(I,1))
           LPOT = LPOT+RTERM*(TWO*SUM1+LEG(I,1)*PsiX(I,1))
           LA = LA+RTERM*(TWO*SUM2-(LEG(I,2)*SI)*PsiX(I,1))
           LB = LB+RTERM*SUM3
           RTERM = RTERM*R
        end do
        LR = LR/R
        LA = -LA*SINA/R
        LB = -LB*TWO/(R*SINA)
        ELECL = ELECL+CGI*LPOT
        DeltX = CGI * (LR*SINTHETA*COSPhi + &
             LA*COSTHETA*COSPhi-LB*SINPhi)
        DeltY = CGI * (LR*SINTHETA*SINPhi + &
             LA*COSTHETA*SINPhi+LB*COSPhi)
        DeltZ = CGI*(LR*COSTHETA-LA*SINTHETA)
        DXL(AT) = DXL(AT)+DeltX
        DYL(AT) = DYL(AT)+DeltY
        DZL(AT) = DZL(AT)+DeltZ
     endif
  end do
  ELEC = ELEC+ELECL*0.5
  !
  RETURN
END SUBROUTINE FARPOT

SUBROUTINE DIRECT(NAtom,XL,YL,ZL,CGL,DXL,DYL,DZL,Atid,Elec, Enb)
  !-----------------------------------------------------------------------
  !     Calculate the potential and gradients due to other atoms in a Box
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  use dimens_fcm
  use consta
  use inbnd
  use fast
  implicit none
  !
  ! Passed
  !
  integer NAtom
  integer ATID(NAtom)
  real(chm_real) ELEC, Enb
  real(chm_real) XL(NAtom),YL(NAtom),ZL(NAtom),CGL(NAtom)
  real(chm_real) DXL(NAtom),DYL(NAtom),DZL(NAtom)
  !
  ! Local
  !
  integer I, J, IAtId, JAtId
  real(chm_real) ElecL, ElecK, DK, EnbL, EnbK
  real(chm_real) S, R2, R6, CGIL, CGF, CA, CC
  real(chm_real) DeltX, DeltY, DeltZ
  !
  !---------------------------- begin ----------------------
  !
  CGF = CCELEC/EPS
  !
  do I = 1,NAtom
     ElecL = ZERO
     IAtId = AtId(I)
     CGIL = CGL(I)*CGF
     do J = (I+1), NAtom
        JAtId = AtId(J)
        DeltX = XL(I) - XL(J)
        DeltY = YL(I) - YL(J)
        DeltZ = ZL(I) - ZL(J)
        DK = ZERO
        S = DeltX**2+DeltY**2+DeltZ**2
        S = MAX (RSMALL, S)
        R2 = ONE/S
        !
        ElecK = CGIL*CGL(J)*SQRT(R2)
        DK = DK - R2*ELECK
        ElecL = ElecL + ElecK
        !
        DeltX = DeltX*(DK)
        DeltY = DeltY*(DK)
        DeltZ = DeltZ*(DK)
        DXL(I) = DXL(I)+DeltX
        DYL(I) = DYL(I)+DeltY
        DZL(I) = DZL(I)+DeltZ
        DXL(J) = DXL(J)-DeltX
        DYL(J) = DYL(J)-DeltY
        DZL(J) = DZL(J)-DeltZ

     end do
     Elec = Elec + ElecL
  end do
  !
  RETURN
END SUBROUTINE DIRECT

SUBROUTINE DIRECT2(NAtom,XL,YL,ZL,CGL,DXL,DYL,DZL, &
     NAtom2,XL2,YL2,ZL2,CGL2,DXL2,DYL2,DZL2, AtId, &
     ELEC, Enb)
  !-----------------------------------------------------------------------
  !     Calculate the potential and gradients due to atoms
  !     in Neighbouring Boxes
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use number
  use dimens_fcm
  use consta
  use inbnd
  use fast
  implicit none
  !
  ! Passed
  !
  integer NAtom,NAtom2
  integer AtId(NAtom2)
  real(chm_real) ELEC, Enb
  real(chm_real) XL(NAtom),YL(NAtom),ZL(NAtom),CGL(NAtom)
  real(chm_real) DXL(NAtom),DYL(NAtom),DZL(NAtom)
  real(chm_real) XL2(NAtom2),YL2(NAtom2),ZL2(NAtom2), &
       CGL2(NAtom2)
  real(chm_real) DXL2(NAtom2),DYL2(NAtom2),DZL2(NAtom2)
  !
  ! Local
  !
  integer I, J, IAtId, JAtId
  real(chm_real) ElecL, ElecK, EnbK, EnbL, DK
  real(chm_real) S, R2, R6, CGIL, CGF, CA, CC
  real(chm_real) DeltX, DeltY, DeltZ
  !
  !---------------------------- begin ----------------------
  !
  CGF = CCELEC/EPS
  !
  do I = 1,NAtom
     ElecL = ZERO
     CGIL = CGL(I)*CGF
     do J = 1,NAtom2
        JAtId = AtId(J)
        DeltX = XL(I)-XL2(J)
        DeltY = YL(I)-YL2(J)
        DeltZ = ZL(I)-ZL2(J)
        DK = ZERO
        S = DeltX**2+DeltY**2+DeltZ**2
        S = MAX (RSMALL, S)
        R2 = ONE/S
        !
        ElecK = CGIL*CGL2(J)*SQRT(R2)
        DK = DK - R2*ELECK
        ElecL = ElecL + ElecK
        !
        DeltX = DeltX*(DK)
        DeltY = DeltY*(DK)
        DeltZ = DeltZ*(DK)
        DXL(I) = DXL(I)+DeltX
        DYL(I) = DYL(I)+DeltY
        DZL(I) = DZL(I)+DeltZ
        DXL2(J) = DXL2(J)-DeltX
        DYL2(J) = DYL2(J)-DeltY
        DZL2(J) = DZL2(J)-DeltZ
     end do
     Elec = Elec + ElecL
  end do
  !
  RETURN
END SUBROUTINE DIRECT2

SUBROUTINE ARRAYINIT (Level)
  !-----------------------------------------------------------------------
  !     Initialise the box hierarchy pointer arrays
  !     Robert Nagle/Paul Adams
  !
  ! Includes
  !
  use chm_kinds
  use mfmacons
  implicit none
  !
  ! Passed
  !
  integer Level
  !
  ! Local
  !
  integer I
  !
  !---------------------------- begin ----------------------
  !
  Start(1) = 1
  Finish(1) = 1
  do I = 2, Level
     Start(I) = Start(I-1) + 8**(I-2)
     Finish(I) = Finish(I-1) + 8**(I-1)
  end do
  !
  return
end SUBROUTINE ARRAYINIT

#else /* (fma_main)*/
SUBROUTINE FMA(Elec, Enb, Level, Terms)
  use chm_kinds
  implicit none
  real(chm_real) Elec, Enb
  integer Level,Terms
  !
  CALL WRNDIE(-1,'<FMA>','FMA code is NOT compiled.')
  return
end SUBROUTINE FMA

#endif /* (fma_main)*/


subroutine fma_dummy
  RETURN
END SUBROUTINE FMA_DUMMY

