* Cluster file for PARAM22 model 
* no two atoms in this scheme have a relationship
* that is more distant than 1-4.
*

RESI ALA              
CLUSTER  ! #1 
ATOM N                  !     |
ATOM HN                 !     |
CLUSTER   ! #2          !  HN N
ATOM CA                 !     |     HB1
ATOM HA                 !     |    /
ATOM CB                 !  HA-CA--CB-HB2
ATOM HB1                !     |    \
ATOM HB2                !     |     HB3
ATOM HB3                !   O=C
CLUSTER   ! END           !     |
ATOM C               
ATOM O               

RESI ARG              
CLUSTER  !#1 
ATOM N                  
ATOM HN                 !     |                      HH11
CLUSTER !#2             !  HN N                       |
ATOM CA                 !     |   HB1 HG1 HD1 HE     NH1 HH12
ATOM HA                 !     |   |   |   |   |    //(+)  
ATOM CB                 !  HA-CA--CB--CG--CD--NE--CZ
ATOM HB1                !     |   |   |   |         \
ATOM HB2                !     |   HB2 HG2 HD2        NH2 HH22
CLUSTER  !#3              !   O=C                       |
ATOM CG                    !                           HH21
ATOM HG1                                              
ATOM HG2              
ATOM CD               
ATOM HD1              
ATOM HD2              
CLUSTER !#4
ATOM NE               
ATOM HE               
ATOM CZ              
CLUSTER !#5
ATOM NH1             
ATOM HH11             
ATOM HH12             
CLUSTER  !#6
ATOM NH2             
ATOM HH21             
ATOM HH22             
CLUSTER  !END
ATOM C                
ATOM O                

RESI ASN              
CLUSTER  ! #1 
ATOM N                          
ATOM HN                 !     |        
CLUSTER !#2             !  HN N 
ATOM CA                 !     |   HB1 OD1    HD21 (cis to OD1)
ATOM HA                 !     |   |   ||    /
ATOM CB                 !  HA-CA--CB--CG--ND2
ATOM HB1                !     |   |         \
ATOM HB2                !     |   HB2        HD22 (trans to OD1)
CLUSTER ! #3            !   O=C
ATOM CG              
ATOM OD1              
ATOM ND2             
ATOM HD21             
ATOM HD22             
CLUSTER  ! END
ATOM C                
ATOM O               

RESI ASP              
CLUSTER  ! #1 
ATOM N               
ATOM HN                 !     |
CLUSTER ! #2            !  HN N
ATOM CA                 !     |   HB1   OD1
ATOM HA                 !     |   |    //
ATOM CB                 !  HA-CA--CB--CG
ATOM HB1                !     |   |    \
ATOM HB2                !     |   HB2   OD2( )
CLUSTER  ! #3           !   O=C
ATOM CG                 !     |           
ATOM OD1             
ATOM OD2             
CLUSTER  ! END
ATOM C                
ATOM O              

RESI CYS              
CLUSTER  !#1  
ATOM N                          
ATOM HN                 !     |         
CLUSTER !#2             !  HN N
ATOM CA                 !     |   HB1
ATOM HA                 !     |   |  
ATOM CB                 !  HA-CA--CB--SG
ATOM HB1                !     |   |     \
ATOM HB2                !     |   HB2    HG1
CLUSTER !#3             !   O=C
ATOM SG                 !     |           
ATOM HG1              
CLUSTER  ! END
ATOM C                
ATOM O               

RESI GLN              
CLUSTER  !#1  
ATOM N                             
ATOM HN                 !     |            
CLUSTER  !#2            !  HN N   
ATOM CA                 !     |   HB1 HG1 OE1   HE21 (cis to OE1)
ATOM HA                 !     |   |   |   ||    /
ATOM CB                 !  HA-CA--CB--CG--CD--NE2
ATOM HB1                !     |   |   |         \
ATOM HB2                !     |   HB2 HG2       HE22 (trans to OE1)
CLUSTER  !#3            !   O=C
ATOM CG                 
ATOM HG1             
ATOM HG2             
ATOM CD              
ATOM OE1             
CLUSTER !#4
ATOM NE2            
ATOM HE21             
ATOM HE22             
CLUSTER  ! END
ATOM C                
ATOM O               

RESI GLU              
CLUSTER  ! #1 
ATOM N                           
ATOM HN                 !     |           
CLUSTER !#2             !  HN N
ATOM CA                 !     |   HB1 HG1   OE1
ATOM HA                 !     |   |   |    //
ATOM CB                 !  HA-CA--CB--CG--CD
ATOM HB1                !     |   |   |    \
ATOM HB2                !     |   HB2 HG2   OE2( )
CLUSTER  !#3            !   O=C
ATOM CG                 
ATOM HG1             
ATOM HG2              
ATOM CD              
ATOM OE1             
ATOM OE2             
CLUSTER  ! END
ATOM C                
ATOM O               

RESI GLY              
CLUSTER  ! #1 
ATOM N                 
ATOM HN                 !     |  
CLUSTER !#2             !     N H
ATOM CA                 !     |  
ATOM HA1                !     |  
ATOM HA2                ! HA1 CA-HA2
CLUSTER  ! END          !     |
ATOM C                  !     |  
ATOM O                  !     C=O
                        !     |

RESI HSD                ! neutral HIS, proton on ND1
CLUSTER ! #1 
ATOM N                  !     |          HD1    HE1
ATOM HN                 !  HN N           |     /
CLUSTER !#2             !     |   HB1    ND1 -CE1
ATOM CA                 !     |   |     /      ||
ATOM HA                 !  HA CA--CB--CG       ||
ATOM CB                 !     |   |     \\     ||
ATOM HB1                !     |   HB2    CD2 -NE2
ATOM HB2                !   O=C           |
CLUSTER !#3             !     |          HD2
ATOM ND1              
ATOM HD1                
ATOM CG
CLUSTER !#4
ATOM CE1          
ATOM HE1           
ATOM NE2           
CLUSTER ! #5
ATOM CD2           
ATOM HD2           
CLUSTER ! END
ATOM C                
ATOM O                

RESI ILE              
CLUSTER  ! #1 
ATOM N                 
ATOM HN                 !     |    HG21 HG22 
CLUSTER !#2             !  HN N      | / 
ATOM CA                 !     |     CG2 -HG23
ATOM HA                 !     |    /
ATOM CB                 !  HA-CA--CB-HB    HD1
ATOM HB                 !     |    \       /
CLUSTER  ! #3           !     |     CG1 -CD--HD2
ATOM CG2                !   O=C    / \     \
ATOM HG21               !     | HG11 HG12  HD3
ATOM HG22            
ATOM HG23            
CLUSTER !#4
ATOM CG1              
ATOM HG11             
ATOM HG12            
ATOM CD             
ATOM HD1             
ATOM HD2             
ATOM HD3             
CLUSTER  ! END
ATOM C                
ATOM O                

RESI LEU              
CLUSTER  ! #1 
ATOM N                
ATOM HN                 !     |        HD11 HD12
CLUSTER   !#2           !  HN N          | /
ATOM CA                 !     |   HB1   CD1 -HD13
ATOM HA                 !     |   |    /
ATOM CB                 !  HA-CA--CB--CG-HG
ATOM HB1                !     |   |    \
ATOM HB2                !     |   HB2   CD2 -HD23
CLUSTER  !#3            !   O=C          | \
ATOM CG                 !     |        HD21 HD22
ATOM HG              
ATOM CD1            
ATOM HD11            
ATOM HD12            
ATOM HD13            
CLUSTER  ! #4 
ATOM CD2            
ATOM HD21            
ATOM HD22            
ATOM HD23            
CLUSTER  ! END
ATOM C               
ATOM O               

RESI LYS              
CLUSTER  ! #1 
ATOM N                  !     |                   
ATOM HN                 !     |                     
CLUSTER  !#2            !  HN N 
ATOM CA                 !     |   HB1 HG1 HD1 HE1    HZ1
ATOM HA                 !     |   |   |   |   |     /   
ATOM CB                 !  HA-CA--CB--CG--CD--CE--NZ--HZ2
ATOM HB1                !     |   |   |   |   |     \
ATOM HB2                !     |   HB2 HG2 HD2 HE2    HZ3
CLUSTER  !#3            !   O=C
ATOM CG                 !     |
ATOM HG1             
ATOM HG2             
ATOM CD             
ATOM HD1             
ATOM HD2            
CLUSTER !#4
ATOM CE             
ATOM HE1             
ATOM HE2              
ATOM NZ             
ATOM HZ1             
ATOM HZ2             
ATOM HZ3             
CLUSTER  ! END
ATOM C                
ATOM O                

RESI MET              
CLUSTER  ! #1 
ATOM N                                 
ATOM HN                 !     |                     
CLUSTER  !#2            !  HN N   
ATOM CA                 !     |   HB1 HG1     HE1 
ATOM HA                 !     |   |   |       |   
ATOM CB                 !  HA-CA--CB--CG--SD--CE--HE3
ATOM HB1                !     |   |   |       |
ATOM HB2                !     |   HB2 HG2     HE2
CLUSTER !#3             !   O=C
ATOM CG                 !     | 
ATOM HG1               
ATOM HG2             
ATOM SD               
CLUSTER !#4
ATOM CE             
ATOM HE1            
ATOM HE2             
ATOM HE3              
CLUSTER  ! END
ATOM C                
ATOM O                

RESI PHE              
CLUSTER  ! #1 
ATOM N               
ATOM HN                 !     |        HD1  HE1
CLUSTER !#2             !  HN N         |    |
ATOM CA                 !     |   HB1  CD1 -CE1
ATOM HA                 !     |   |    //     \\
ATOM CB                 !  HA-CA--CB--CG      CZ--HZ
ATOM HB1                !     |   |    \  __  /
ATOM HB2                !     |   HB2  CD2 -CE2
CLUSTER !#3             !   O=C         |    |
ATOM CG                 !     |        HD2  HE2
ATOM CD1              
ATOM HD1              
CLUSTER !#4
ATOM CE1              
ATOM HE1              
ATOM CZ               
ATOM HZ               
CLUSTER !#5
ATOM CD2              
ATOM HD2              
ATOM CE2              
ATOM HE2              
CLUSTER  ! END
ATOM C                
ATOM O              

RESI PRO              
CLUSTER  ! #1         !       HD1 HD2
ATOM N                !     |   \ /              
CLUSTER !#2           !     N --CD   HG1
ATOM CD               !     |     \  / 
ATOM HD1              !     |      CG
ATOM HD2              !     |     /  \
CLUSTER !#3           !  HA CA--CB   HG2
ATOM CA               !     |   / \
ATOM HA               !   O=C
ATOM CB               !     |
ATOM HB1               
ATOM HB2               
CLUSTER !#4
ATOM CG                
ATOM HG1               
ATOM HG2             
CLUSTER !END
ATOM C                
ATOM O               

RESI SER              
CLUSTER  ! #1 
ATOM N                   
ATOM HN                 !     |     
CLUSTER !#2             !  HN N 
ATOM CA                 !     |   HB1
ATOM HA                 !     |   |  
ATOM CB                 !  HA CA--CB--OG
ATOM HB1                !     |   |     \
ATOM HB2                !     |   HB2    HG1
CLUSTER !#3             !   O=C
ATOM OG                
ATOM HG1              
CLUSTER  ! END
ATOM C                
ATOM O                

RESI THR              
CLUSTER ! #1  
ATOM N                
ATOM HN                 !     |
CLUSTER !#2             !  HN N
ATOM CA                 !     |     OG1 -HG1
ATOM HA                 !     |    /
ATOM CB                 !  HA CA--CB-HB
ATOM HB                 !     |    \
CLUSTER  !#3            !     |     CG2 -HG21
ATOM OG1                !   O=C    / \
ATOM HG1                !     | HG21 HG22
CLUSTER !#4
ATOM CG2           
ATOM HG21            
ATOM HG22            
ATOM HG23            
CLUSTER  ! END
ATOM C                
ATOM O               

RESI TRP              
CLUSTER ! #1  
ATOM N                 !     |                  HE3
ATOM HN                !  HN N                   |
CLUSTER !#2            !     |   HB1            CE3
ATOM CA                !     |   |             /  \\
ATOM HA                !  HA CA--CB---CG-----CD2   CZ3-HZ3
ATOM CB                !     |   |    ||     ||     |
ATOM HB1               !     |   HB2  CD1    CE2   CH2 HH2
ATOM HB2               !   O=C       /   \   / \  //
CLUSTER ! #3           !     |     HD1    NE1   CZ2
ATOM CG                !                   |     |
ATOM CD1               !                  HE1   HZ2
ATOM HD1              
CLUSTER !#4
ATOM NE1        
ATOM HE1          
ATOM CE2            
ATOM CD2            
CLUSTER ! #5
ATOM CE3              
ATOM HE3              
ATOM CZ3              
ATOM HZ3             
CLUSTER ! #6  
ATOM CZ2              
ATOM HZ2              
ATOM CH2             
ATOM HH2              
CLUSTER  ! END
ATOM C                
ATOM O                

RESI TYR              
CLUSTER ! #1  
ATOM N                  !     |        HD1  HE1    
ATOM HN                 !  HN N         |    |   
CLUSTER !#2             !     |   HB1  CD1 -CE1
ATOM CA                 !     |   |   //      \\
ATOM HA                 !  HA CA--CB--CG      CZ--OH
ATOM CB                 !     |   |    \  __  /     \
ATOM HB1                !     |   HB2  CD2 -CE2     HH
ATOM HB2                !   O=C         |    |  
CLUSTER !#3             !     |        HD2  HE2
ATOM CG                 
ATOM CD1               
ATOM HD1               
CLUSTER !#4
ATOM CE1               
ATOM HE1               
ATOM CZ               
ATOM OH               
ATOM HH               
CLUSTER  ! #5
ATOM CD2               
ATOM HD2               
ATOM CE2               
ATOM HE2               
CLUSTER  ! END
ATOM C                
ATOM O               

RESI VAL              
CLUSTER  ! #1 
ATOM N                  !     |    HG11 HG12
ATOM HN                 !  HN N      | / 
CLUSTER  ! #2           !     |     CG1 -HG13
ATOM CA                 !     |    /
ATOM HA                 !  HA CA--CB-HB
ATOM CB                 !     |    \     
ATOM HB                 !     |     CG2 -HG21
CLUSTER  ! #3           !   O=C    / \
ATOM CG1                !     | HG21 HG22
ATOM HG11             
ATOM HG12            
ATOM HG13            
CLUSTER  ! #4
ATOM CG2            
ATOM HG21            
ATOM HG22            
ATOM HG23            
CLUSTER  ! END
ATOM C                
ATOM O               

RESI TIP3               ! tip3p water model, generate using noangle nodihedral
CLUSTER   
ATOM OH2             
ATOM H1               
ATOM H2               

RESI TP3M               ! "mmff" water model, as an analog of tip3p
CLUSTER   
ATOM OH2               ! these charges are replaced by the mmff setup   
ATOM H1                ! these charges are replaced by the mmff setup
ATOM H2                ! these charges are replaced by the mmff setup

RESI CAL               ! Calcium ion
CLUSTER   
ATOM CAL            

RESI ZN2               ! Zinc ion
CLUSTER   
ATOM ZN              

! if a residue is specified as a bridge residue
! its first cluster is combined with the previous
! residue's last cluster and its last cluster
! is combined with the next residue's first cluster

BRIDge ALA ARG ASN ASP CYS GLN GLU GLY HSD ILE 
BRIDge LEU LYS MET PHE PRO SER THR TRP TYR VAL

! Equivalence used for handling termini. The cluster
! of first atom is set to that of second atom 
! wherever it appears in structure

EQUI OT1 O
EQUI OT2 O
EQUI HT1 N
EQUI HT2 N
EQUI HT3 N
EQUI H5T P
EQUI O5T P
EQUI H3T O3'
END


