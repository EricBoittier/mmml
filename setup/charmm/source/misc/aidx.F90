module aidxmod
  use chm_kinds
  use dimens_fcm
  implicit none

contains

#if KEY_NOMISC==0 /*nomisc*/
  SUBROUTINE AIDX
    !
    ! Find internal index based on name of chemical atom type
    ! Stefan Boresch, Univ. Vienna, July 2017
    !
    use chm_kinds
    use dimens_fcm
    use exfunc
    use number
    !
    use comand
    use stream
    use string
    use param,only:natc,atc
    use param_store, only: set_param
    
    implicit none

    character(len=8) wrd
    integer iatmtype
    
    CALL TRIMA(COMLYN,COMLEN)
    IF(COMLEN == 0) THEN
       ! Process general information command
       IF(PRNLEV >= 2) WRITE(OUTU,'(A)')&
            ' AIDX: Nothing requested, doing nothing ...'
       RETURN
    ENDIF

    WRD=NEXTA8(COMLYN,COMLEN)
    iatmtype = srchws(atc,natc,wrd)
       IF(PRNLEV >= 2) WRITE(OUTU,'(A,A,A,I4)')&
            ' AIDX: Search for type  ',WRD, ' returned ',iatmtype
    if (iatmtype > 0) then
       call set_param('AIDX',iatmtype)
    else
       ! do we want to die here ?
       call wrndie(0,'<AIDX>','Atom type not found')
       return
    endif

    ! clean up after ourselves
    CALL TRIMA(COMLYN,COMLEN)
    IF(COMLEN > 0) THEN
       ! Process general information command
       IF(PRNLEV >= 2) WRITE(OUTU,'(A)')' AIDX: Ignoring further args.'
       call xtrane(comlyn,comlen,'aidx')
       RETURN
    ENDIF
    

  END SUBROUTINE AIDX
  
#endif /*  (nomisc)*/

  SUBROUTINE NULL_QK
    RETURN
  END SUBROUTINE NULL_QK

end module aidxmod

