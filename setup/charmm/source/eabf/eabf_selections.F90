!-----------------------------------------------------------
 subroutine MAKE_SELECTIONS(COMLYN,COMLEN)
!----------------------------------------------------------
 use eabfsrc
 use dimens_fcm
 use exfunc
 use number
 use stream
 use psf
 use coord
 use coordc
 use deriv
 use reawri
 use consta
 use contrl
 use chm_kinds
 use memory
 use string
 use image

 implicit none
    CHARACTER(len=*), intent(in) :: comlyn
    INTEGER, intent(in) :: comlen
    !INTEGER :: LDYNA,i

 ALLOCATE (ListA(1:natom))
 ALLOCATE (ListB(1:natom))
 ALLOCATE (ListC(1:natom))
 ALLOCATE (ListD(1:natom))
 ALLOCATE (ListE(1:natom))
 ALLOCATE (ListF(1:natom))
 ALLOCATE (ListG(1:natom))
 ALLOCATE (ListH(1:natom))
 ALLOCATE (ListI(1:natom))
 ALLOCATE (ListJ(1:natom))
 ALLOCATE (ListK(1:natom))
 ALLOCATE (ListL(1:natom))
 ALLOCATE (ListM(1:natom))
 ALLOCATE (ListN(1:natom))
 ALLOCATE (ListO(1:natom))

 NatomA=0
 NatomB=0
 NatomC=0
 NatomD=0
 NatomE=0
 NatomF=0
 NatomG=0
 NatomH=0
 NatomI=0
 NatomJ=0
 NatomK=0
 NatomLL=0
 NatomM=0
 NatomN=0
 NatomO=0

 ListA(:)=0
 ListB(:)=0
 ListC(:)=0
 ListD(:)=0
 ListE(:)=0
 ListF(:)=0
 ListG(:)=0
 ListH(:)=0
 ListI(:)=0
 ListJ(:)=0
 ListK(:)=0
 ListL(:)=0
 ListM(:)=0
 ListN(:)=0
 ListO(:)=0

  call getatomselected2(COMLYN,COMLEN,NatomA,ListA)
  call getatomselected2(COMLYN,COMLEN,NatomB,ListB)
  call getatomselected2(COMLYN,COMLEN,NatomC,ListC)
  call getatomselected2(COMLYN,COMLEN,NatomD,ListD)
  call getatomselected2(COMLYN,COMLEN,NatomE,ListE)
  call getatomselected2(COMLYN,COMLEN,NatomF,ListF)
  call getatomselected2(COMLYN,COMLEN,NatomG,ListG)
  call getatomselected2(COMLYN,COMLEN,NatomH,ListH)
  call getatomselected2(COMLYN,COMLEN,NatomI,ListI)
  call getatomselected2(COMLYN,COMLEN,NatomJ,ListJ)
  call getatomselected2(COMLYN,COMLEN,NatomK,ListK)
  call getatomselected2(COMLYN,COMLEN,NatomLL,ListL)
  call getatomselected2(COMLYN,COMLEN,NatomM,ListM)
  call getatomselected2(COMLYN,COMLEN,NatomN,ListN)
  call getatomselected2(COMLYN,COMLEN,NatomO,ListO)

 end subroutine

!-------------------------------------------------------------------------------------------
  subroutine getatomselected2(COMLYN,COMLEN,NATMSelected,listheap)
!-------------------------------------------------------------------------------------------
 use eabfsrc
 use chm_kinds
 use memory
 use exfunc
 use psf
 use coord
 use dimens_fcm
 use stream
 use select
 use image
 use number

 implicit none
 character(len=*), intent(in) :: COMLYN
 integer, intent(in) :: COMLEN

 integer,dimension(NATOM) :: itemp
 integer :: NATMSelected
 integer :: i,n
 integer,dimension(natom) :: listheap

        itemp(:)=0
         !selecta goes through and assigns itemp 1 for selected atoms
         call selcta(COMLYN,COMLEN,itemp,X,Y,Z,WMAIN,.TRUE.)
         n=0
         NATMSelected=0
         !Change Below
         do i=1,NATOM
           if(itemp(i) .eq. 1)then
              n=n+1
           endif
         enddo
         if(n .eq. 0) then
           !write(*,*) '@@ no atoms selected'
           return
         endif
        listheap(:)=0

         n=0
         !Change Below
         do i=1,NATOM
          if(itemp(i) .eq. 1) then
              n=n+1
              listheap(n)=i
           endif
         enddo

       NATMSelected=n
       return

 end subroutine

!-----------------------------------------------------------
 subroutine DESELECT_LISTS()
!----------------------------------------------------------
 use eabfsrc
 implicit none

 DEALLOCATE (ListA)
 DEALLOCATE (ListB)
 DEALLOCATE (ListC)
 DEALLOCATE (ListD)
 DEALLOCATE (ListE)
 DEALLOCATE (ListF)
 DEALLOCATE (ListG)
 DEALLOCATE (ListH)
 DEALLOCATE (ListI)
 DEALLOCATE (ListJ)
 DEALLOCATE (ListK)
 DEALLOCATE (ListL)
 DEALLOCATE (ListM)
 DEALLOCATE (ListN)
 DEALLOCATE (ListO)

end subroutine

