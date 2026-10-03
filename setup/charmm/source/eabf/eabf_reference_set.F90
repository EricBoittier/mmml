!---------------------------------------------------------------------
SUBROUTINE INITILIZE_REFERENCE(COMLYN,COMLEN)
!---------------------------------------------------------------------
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
use parallel
use energym
use corsubs
implicit none
CHARACTER(len=*), intent(in) :: comlyn
INTEGER, intent(in) :: comlen
integer :: icall,i,ii,k,l,istat
real(chm_real) :: v,prev_value,CMXC,CMYC,xxx,yyy,zzz

!Can currently hold 4 sets of references of 10000 atoms
ALLOCATE(initial_rmsd_refs_x(0:3,0:10000))
ALLOCATE(initial_rmsd_refs_y(0:3,0:10000))
ALLOCATE(initial_rmsd_refs_z(0:3,0:10000))
initial_rmsd_refs_x(:,:)=0
initial_rmsd_refs_y(:,:)=0
initial_rmsd_refs_z(:,:)=0
ALLOCATE(rmsd_refs_x(0:3,0:10000))
ALLOCATE(rmsd_refs_y(0:3,0:10000))
ALLOCATE(rmsd_refs_z(0:3,0:10000))
rmsd_refs_x(:,:)=0
rmsd_refs_y(:,:)=0
rmsd_refs_z(:,:)=0
!Set current number of references to 0 (meaning save in array space 0)
rmsd_ref_count=0
END SUBROUTINE

!---------------------------------------------------------------------
SUBROUTINE COPY_REFERENCE(COMLYN,COMLEN)
!---------------------------------------------------------------------
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
use parallel
use energym
use corsubs
implicit none
CHARACTER(len=*), intent(in) :: comlyn
INTEGER, intent(in) :: comlen
integer :: icall,i,ii,k,l,istat
real(chm_real) :: v,prev_value,CMXC,CMYC,xxx,yyy,zzz

ALLOCATE (rmsd_list_tmp(1:natom))
rmsd_list_tmp(:)=0
rmsd_num_sel=0
call getatomselected2(COMLYN,COMLEN,rmsd_num_sel,rmsd_list_tmp)
!Store Reference
do i=1,rmsd_num_sel
   ii=rmsd_list_tmp(i)
   initial_rmsd_refs_x(rmsd_ref_count,i)=X(ii)
   initial_rmsd_refs_y(rmsd_ref_count,i)=Y(ii)
   initial_rmsd_refs_z(rmsd_ref_count,i)=Z(ii)
enddo
!Write Comparison Coordinates
open(unit=1333,file='eabf_reference_coords.dat')
do i=1,rmsd_num_sel
   write(1333,'(I8,2x,I8,2x,G21.14,2x,G21.14,2x,G21.14)') rmsd_ref_count,i,initial_rmsd_refs_x(rmsd_ref_count,i),initial_rmsd_refs_y(rmsd_ref_count,i),initial_rmsd_refs_z(rmsd_ref_count,i)
   call flush(1333)
enddo
CLOSE (1333,STATUS='KEEP',IOSTAT=I)
DEALLOCATE (rmsd_list_tmp)
!Increase the reference count for next read (if there is one)
rmsd_ref_count=rmsd_ref_count+1
END SUBROUTINE
