MODULE stringutil
  !
  !     string manipulation utilities
  !
CONTAINS
  !______________________________________________________________________
  !______________________________________________________________________
  FUNCTION cmpstr_ins(stringa,stringb)
    !
    !  Comparison of two strings without regard to case.
    !  This routine ignores trailing blanks.
    !
    implicit none
    !
    character(len=*) :: stringa,stringb
    logical :: cmpstr_ins 
    character :: char1,char2
    integer :: pp,lena,lenb,mlen,mlenp1
    logical :: qlocmatch
    !**************************************************************
    lena=len(stringa)
    lenb=len(stringb)
    mlen=min(lena,lenb)
    mlenp1=mlen+1 
    cmpstr_ins=.false.
    !
    qlocmatch=.true.
    pp=1
    do while((pp <= mlen).and.(qlocmatch))
       char1=stringa(pp:pp)
       char2=stringb(pp:pp)
       call lowcsng(char1) !make character lower case
       call lowcsng(char2) !make character lower case
       if(char1 /= char2) qlocmatch = .false.
       pp = pp + 1
    end do
    if(.not.qlocmatch) goto 999
    !
    ! If character  strings are of different lengths
    ! the longer string must have blanks as the extra characters
    ! in order to match
    pp=mlenp1
    qlocmatch=.true.
    do while((pp <= lena).and.(qlocmatch))
       if (stringa(pp:pp) /= ' ' ) qlocmatch=.false.
       pp=pp+1 
    end do
    if(.not.qlocmatch) goto 999 
    !        
    pp=mlenp1
    qlocmatch=.true.
    do while((pp <= lenb).and.(qlocmatch)) 
       if(stringb(pp:pp) /= ' ' ) qlocmatch=.false.
       pp=pp+1
    end do
    if(.not.qlocmatch) goto 999
    !
    cmpstr_ins=.true.
999 CONTINUE       
    return
  end FUNCTION cmpstr_ins
  !______________________________________________________________________
  !______________________________________________________________________
  SUBROUTINE lowcsng(let)
    !
    !     Changes the case of a single letter to lower case
    !
    implicit none
    !
    character(len=1),intent(inout) :: let
    integer :: ival
    !***********************************************************
    ival=ichar(let)
    if(65 <= ival.and.ival.le.90) then
       let=char(ival+32)
    endif
    !
  END SUBROUTINE lowcsng
  !
END MODULE stringutil

