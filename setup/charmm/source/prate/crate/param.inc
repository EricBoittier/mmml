CHARMM Element source/prate/crate/param.inc $Revision: 1.2 $
C#######################################################################
C
C   param2.inc (the old poly6775.inc) 
C
C   This file can be used as the param.inc INCLUDE file required by
C   the POLYRATE source code.
C
C   PARAMETER STATEMENTS FOR POLYRATE 
C
C   THESE ARE THE ONLY PARAMETERS WHICH SHOULD BE MODIFIED BY THE USER.
C
C   NOTE:   IF THESE PARAMETERS ARE MODIFIED THE SOURCE CODE FOR 
C           POLYRATE MUST BE RECOMPILED 
C
C               NSDM      =    THE MAXIMUM NUMBER OF SAVE POINTS 
C                              ALONG THE MINIMUM ENERGY PATH
C
C               NSDML     =    THE MAXIMUM NUMBER OF SAVE POINTS 
C                              FOR THE TUNNELING CALCULATIONS 
C                              (NSDML .LE. NSDM)
C
C               NATOMS    =    MAXIMUM NUMBER OF ATOMS
C
C               NPOTPT    =    MAXIMUM NUMBER OF PIECES OF INFORMATION
C                              WHICH MAY BE PASSED FROM THE POTENTIAL
C                              ENERGY SURFACE (THIS INFORMATION IS NOT
C                              USED IN THE CALCULATION BUT MAY BE 
C                              PRINTED TO UNIT FU6 USING THE LGS2(5)
C                              FLAG IN THE UNIT FU5 INPUT DECK)
C
C               MAXPS     =    MAXIMUM NUMBER OF PRODUCT STATES ALLOWED
C                              FOR THE LCG3 TUNNELING CALCULATION
C
C               MAXWKB    =    MAXIMUM NUMBER OF REACTANT STATES ALLOWED
C                              QUANTIZED REACTANT STATE TUNNELING CALCUL
C
C               NMSPEC    =    MAXIMUM NUMBER OF EXTRA SAVE POINTS
C
C               INMM      =    MAXIMUM VALUE FOR RATIO BETWEEN GRADIENT
C                              SAVE POINTS AND HESSIAN SAVE POINTS NUMBE
C                              (USED ONLY FOR FU31 RUNS)
C
C               FU#       =    FORTRAN UNIT NUMBER FOR SPECIFYING THE
C                              POLYRATE INPUT AND OUTPUT FILES 
C                              (SEE THE POLYRATE MANUAL FOR A FULL 
C                               DESCRIPTION OF THE FILES ASSOCIATED 
C                               WITH EACH #)
C
C
C
C
      PARAMETER (NSDIM= 501)                                            
C
      PARAMETER (NSDM = 501) 
C
      PARAMETER (NSDML = 501)
C
      PARAMETER (NATOMS = 30) 
C
      PARAMETER (MAXINT=3*NATOMS+6, MAXCAR=3*NATOMS)                    
C
      PARAMETER (NPOTPT = 1)
C
      PARAMETER (MAXPS = 5)
C
      PARAMETER (MAXWKB = 50)
C
      PARAMETER (NMSPEC = 20)
C
      PARAMETER (INMM = 10)                                             
C
C  Note: the integer specification must be made before the parameter 
C        declaration.
C
      INTEGER FU1,  FU2,  FU3,  FU5,  FU6,  FU8
      INTEGER FU14, FU15, FU18, FU19
      INTEGER FU20, FU21, FU22, FU25, FU26, FU27, FU28
      INTEGER FU30, FU31
      INTEGER FU71, FU72, FU73, FU74, FU75, FU77, FU78                  
      INTEGER FU40, FU41, FU42, FU43, FU44, FU45, FU46, FU47
      INTEGER FU50, FU51
      INTEGER FU60, FU61, FU65
C
      PARAMETER (FU1 = 1, FU2 = 2, FU3 = 3, FU5 = 5, FU6 = 6, FU8 = 8)
C 
      PARAMETER (FU14 = 14, FU15 = 15, FU18 = 18, FU19 = 19)
C
      PARAMETER (FU20 = 20, FU21 = 21, FU22 = 22)
C
      PARAMETER (FU25 = 25, FU26 = 26, FU27 = 27, FU28 = 28)
C
      PARAMETER (FU30 = 30, FU31 = 31)
C
      PARAMETER (FU71 = 71, FU72 = 72, FU73 = 73, FU74 = 74, FU75 = 75, 
     *           FU77 = 77, FU78 = 78)                                  
C
      PARAMETER (FU40 = 40, FU41 = 41, FU42 = 42, FU43 = 43)
      PARAMETER (FU44 = 44, FU45 = 45, FU46 = 46, FU47 = 47)
      PARAMETER (FU50 = 50, FU51 = 51, FU60 = 60, FU61 = 61, FU65 = 65)
C
C#######################################################################
C
C

