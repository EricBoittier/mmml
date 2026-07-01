SUBROUTINE HELP(OPTION)
  !
  !     Print out command and other useful information.
  !
  use chm_kinds
  use stream
  implicit none
  !
  character(len=*) OPTION
  !
  IF(PRNLEV <= 2) RETURN
  ! . Branch on the option.
  !=======================================================================
  ! . The commands in the main routine.
  IF (OPTION  ==  'CHARMM') THEN
     WRITE(OUTU,'(A)') ' HELP> Commands for the CHARMM program :'
     WRITE(OUTU,'(16(A,/))') &
          ' HELP> . Blank line', &
          ' HELP> . Miscellaneous commands', &
          ' HELP> . ATLImit   BARIer   BLOCk   BUILd', &
          ' HELP> . CONStraints   COORdinates   CORRelation', &
          ' HELP> . CRYStal   DEADline   DELEte_atoms', &
          ' HELP> . DRAW   DYNAmics   ENERgy   GENErate', &
          ' HELP> . GETEnergy   GRAPhics   HBONds   HBUIld', &
          ' HELP> . HBTRim   HELP   IC   IMAGes   IMPAtch', &
          ' HELP> . INQUire   INTEraction_energy   JOIN', &
          ' HELP> . MERGe   MINImize   MONItor   NBONds', &
          ' HELP> . NOE   PATCh   PATH   PERTurbation   PRINt', &
          ' HELP> . READ   RENAme   RMSDifference   SBOUndary', &
          ' HELP> . SCALar   SKIP   SOLAnalysis   STAR', &
          ' HELP> . SYSTem   TEST TRAJectory   UPDAte', &
          ' HELP> . VIBRation WRITe', &
          ' HELP> . STOP'

     !=======================================================================
     ! . The crystal commands.
  ELSE IF (OPTION  ==  'CRYSTL') THEN
     WRITE(OUTU,'(A)') ' HELP> Commands for the CRYSTL module :'
     WRITE(OUTU,'(17(A,/))') &
          ' HELP> CRYStal', &
          ' HELP> . Blank line', &
          ' HELP> . BUILd CUTOff <f> NOINverse PRINt', &
          ' HELP> . DEFIne xtltyp a b c alpha beta gamma NOPEration <i>', &
          ' HELP>   (xtltyp is TRIC, MONO, ORTH, TETR, HEXA or CUBI)', &
          ' HELP> . NOPErations lines defining symmetry operations : ', &
          ' HELP> . ( 3x(+/-X,Y,Z +/- <i/i>) )', &
          ' HELP> . FREE', &
          ' HELP> . HELP', &
          ' HELP> . IMAG', &
          ' HELP> . KEEP', &
          ' HELP> . PRINt TRANsformations', &
          ' HELP> . READ  CARD UNIT <i>', &
          ' HELP> . UNKEep', &
          ' HELP> . UPDAte NOINverse PRINt', &
          ' HELP> . WRITe CARD UNIT <i>', &
          ' HELP> END'
     !=======================================================================
  ELSE
     WRITE(OUTU,'(A)') &
          ' HELP> Information is not available for this option.'
     !=======================================================================
  ENDIF
  !
  RETURN
END SUBROUTINE HELP

