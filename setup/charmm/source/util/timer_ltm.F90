module timerm
  use chm_kinds

  implicit none

  !     TIMER - 0=NO TIMING, 1=SOME, 2=INDIVIDAUL ENERGY TERM TIMING
  !moved to chm_kinds-mfc     BOMLEV - -5 TO 5  -5=DONT BOMB, -1=INTERACTIVE, 0=NORMAL
  !moved to chm_kinds-mfc     BOMMIN - Lowest BOMLEV value for the run.
  !     CPULIM - 0.0 NO LIMIT, >0 STOP GRACEFULLY WHEN CPULIM IS REACHED
  !     CPUINI - SECONDS SINCE MIDNIGHT WHEN DEADline COMMAND was given
  !     DEADHR -  CLOCK time deadline hour
  !     DEADMN -  CLOCK time deadline minute
  !     PASMID -  .FALSE. if midnight should pass before stopping
  !     PRHEAP - Flag to print heap data between commands (TEST HEAP).
  !     PRSTCK - Flag to print stack data calls (TEST STACk).
  !     ATLIM  - .TRUE. on time limit
  !     LIMTYP - 'NONE', 'CPU ', 'CLK ', 'BOTH' or 'QUAN' - type of limit reached
  !     ALTLEN - Length of ATLIM command
  !     MXALSZ - Maximum length of the ATLIM command
  !     ALTCOM - The saved ATLIM command
  !

  INTEGER,parameter ::  MXALSZ=80

  real(chm_real)  CPULIM,CPUINI
  INTEGER TIMER,DEADHR,DEADMN,PASMID,ALTLEN
  LOGICAL ATLIM,PRHEAP,PRSTCK
  CHARACTER(len=4) :: LIMTYP
  CHARACTER(len=MXALSZ) :: ALTCOM

  !

contains

  subroutine timer_iniall
    timer=0
    bommin=100
    cpulim=0.0
    deadhr=-1
    return
  end subroutine timer_iniall
end module timerm

