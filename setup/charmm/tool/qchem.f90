program no_qchem

  !
  ! Use this program if there is no Q-Chem installed on the system.
  ! Writes the files that are used by gukini.src interface program

  ! Currently it does nothing except creates an empty file, which could be
  ! created in a testing script itself. But we need an executable anyway!!

  ! It can be made more complete, so it can really calulate HF/STO-3G
  ! forces with external charges !!
  !

  open(unit=1,file='qchem.out',status='unknown')

  close(unit=1)

  open(unit=1,file='q1.out',status='unknown')

  close(unit=1)

  open(unit=1,file='hessian.dat',status='unknown')
  write(1,'(i1)')(0,i=1,1000)
  close(unit=1)

end program no_qchem
