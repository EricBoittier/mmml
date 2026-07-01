#if KEY_SSNMR==1
SUBROUTINE CSSET2

   use dimens_fcm
   use psf
   use comand
   use stream
   use string
   use select
   use coord
   use number
   use chutil,only:atomid
   use ssnmr
   use memory

   implicit none

   INTEGER           I,J,II,N
   CHARACTER*8       SIDI,RIDI,RENI,ACI
   CHARACTER*8       SIDJ,RIDJ,RENJ,ACJ

   LOGICAL           DONE,EOF,OK
   CHARACTER(len=4)  WRD
   integer,allocatable,dimension(:) :: islct

   DONE = .FALSE.
   EOF  = .FALSE.
   OK   = .TRUE.

   call chmalloc('ssnmr.src','ssnmr','islct',natom,intg=islct)

   DO WHILE(.NOT.DONE)
      CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE.,'CCS> ')

      IF(EOF)THEN
         CALL PPSTRM(OK)
         IF(.NOT.OK)  RETURN
      ENDIF

      WRD='    '
      WRD=NEXTA4(COMLYN,COMLEN)
      IF (WRD.EQ.'    ') THEN
         CONTINUE

      ELSE IF (WRD.EQ.'RESE') THEN
         QSSNMR  = .FALSE.
         call ccs_uniniall()

      ELSE IF (WRD.EQ.'EXPS') THEN
         if (.not. cs_initialized) call ccs_iniall()
         NUMCAT=NUMCAT+1
         if (numcat > nmrmx) call nmr_add_storage()

         SIGMA(1,NUMCAT)=GTRMF(COMLYN,COMLEN,'S11',64.D0)
         SIGMA(2,NUMCAT)=GTRMF(COMLYN,COMLEN,'S22',77.D0)
         SIGMA(3,NUMCAT)=GTRMF(COMLYN,COMLEN,'S33',217.D0)
         PHICS(NUMCAT)=GTRMF(COMLYN,COMLEN,'PHI',17.D0)
         ! DEFAULT: NH NU_0=19.86KHZ IN D(NH)= 1.07A
         NUDC(NUMCAT)=GTRMF(COMLYN,COMLEN,'NUDC',19.86D0)
         LCONSNU(NUMCAT)=(INDXA(COMLYN, COMLEN,'DCCO').GT.0)
         LDCABS(NUMCAT)=(INDXA(COMLYN, COMLEN,'DCAB').GT.0)
         stoag(1,numcat)=csnum+1

      ELSE IF (WRD.EQ.'ASSI') THEN
         if (.not. cs_initialized) call ccs_iniall()
         QSSNMR = .TRUE.

         CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)

         CSNUM=CSNUM+1
         if (csnum > csmax) call ccs_add_storage()

         LSOFTA(CSNUM)=(INDXA(COMLYN,COMLEN,'SOFT').GT.0)
         LDIPC(CSNUM)=(INDXA(COMLYN,COMLEN,'DIPC').GT.0)
         FORCS(CSNUM)=GTRMF(COMLYN,COMLEN,'FORC',ZERO)
         EXPCS(CSNUM)=GTRMF(COMLYN,COMLEN,'EXP',100.D0)
         ! for soft asymptote
         KMINCS(CSNUM)=GTRMF(COMLYN,COMLEN,'KMIN',1.D0)
         PLCS(CSNUM)=GTRMF(COMLYN,COMLEN,'RMIN',0.D0)
         KMAXCS(CSNUM)=GTRMF(COMLYN,COMLEN,'KMAX',1.D0)
         PUCS(CSNUM)=GTRMF(COMLYN,COMLEN,'RMAX',1.D0)
         RSW(CSNUM)=GTRMF(COMLYN,COMLEN,'RSWI',1.D0)
         EXPT(CSNUM)=GTRMF(COMLYN,COMLEN,'EXPT',1.D0)
         KNEW(CSNUM)=GTRMF(COMLYN,COMLEN,'FMAX',1.D0)
         RSWL(CSNUM)=GTRMF(COMLYN,COMLEN,'RSWL',1.D0)

         CSIPT(CSNUM)=CSNM2+1
         CSINM(CSNUM)=0
         N=CSNM2
         DO I=1,NATOM
            IF(ISLCT(I).EQ.1) THEN
               CSNM2=CSNM2+1
               if (csnm2 > csmax) call ccs_add_storage()

               CSLIS(CSNM2)=I
               CSINM(CSNUM)=CSINM(CSNUM)+1
            ENDIF
         ENDDO

         IF(CSINM(CSNUM).EQ.0) THEN
            CALL WRNDIE(0, '<CSSET>', &
                 'Zero atom selected for this restraint. Ignored.')

            CSNUM=CSNUM-1
            CSNM2=N
            GOTO 120
         ENDIF

         IF(PRNLEV.GE.2) THEN
            IF(LSOFTA(CSNUM)) WRITE(OUTU,*) 'SOFT ASYMPTOTE potential applied'
            WRITE(OUTU,'(3(A,1X,I5))') &
               '  CS: ADDING RESTRAINT #',CSNUM, &
               ', # atoms 1st set',CSINM(CSNUM)

            DO I=1,CSINM(CSNUM)
               II=CSLIS(CSIPT(CSNUM)+I-1)
               CALL ATOMID(II,SIDI,RIDI,RENI,ACI)
               WRITE(OUTU,511) II,SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng)
511               FORMAT('        FIRST SET ATOM:',I5,3(1X,A))
            ENDDO

            WRITE(OUTU,515) FORCs(CSNUM),EXPcs(CSNUM)
515            FORMAT('   FORC=',F10.3,'  EXP=',F10.3)

            IF(LSOFTA(CSNUM)) &
            WRITE(OUTU,*) KMINCS(CSNUM),PLCS(CSNUM),&
               KMAXCS(CSNUM),PUCS(CSNUM),RSW(CSNUM),EXPT(CSNUM),&
               KNEW(CSNUM)
         ENDIF
120      CONTINUE

      ELSE IF (WRD.EQ.'PRIN') THEN
         LANAL=(INDXA(COMLYN,COMLEN,'ANAL').GT.0)
         RMSDVAL=GTRMF(COMLYN,COMLEN,'DIFF',ZERO)
         RMSDVDC=GTRMF(COMLYN,COMLEN,'DIFD',ZERO)
         IUNIJ=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)

         IF (LANAL) THEN
            IF (PRNLEV.GE.6) THEN
               WRITE(OUTU,*) 'CAT  S_11    S_22    S_33     Phi'
               DO I=1,NUMCAT
                  WRITE(OUTU,'(i2,4(1x,f7.3))') I,SIGMA(1,I),SIGMA(2,I),SIGMA(3,I),PHICS(I)
               ENDDO

               WRITE(OUTU,*) 'CSNUM: ',CSNUM
               DO I=1,CSNUM
                  WRITE(OUTU,*) 'CS : Tn sel.atoms : init_index',I,CSINM(I),CSIPT(I)

                  DO J=1,CSINM(I)
                     WRITE(OUTU,*) 'selected atom number',CSLIS(CSIPT(I)+J-1)
                  ENDDO
               ENDDO
            ENDIF
         ENDIF

      ELSE IF (WRD.EQ.'END ') THEN
         DONE = .TRUE.
         if (cs_initialized) STOAG(2, NUMCAT) = CSNUM

      ELSE
         CALL WRNDIE(0,'<CSSET>','UNKNOWN OPTION')

      ENDIF
   ENDDO

   IF(PRNLEV.GE.2) WRITE(OUTU,230) CSNUM
230   FORMAT(' CS:  CURRENT NUMBER OF CONSTRAINTS=',I4)

   IF (DONE) call chmdealloc('ssnmr.src','ssnmr','islct',natom,intg=islct)

RETURN
END SUBROUTINE CSSET2


subroutine cscns(en,dx,dy,dz,x,y,z, &
                 forcs,expcs,csnum,csipt,csinm,cslis, &
                 sigma,pphics, &
                 kmincs,plcs,kmaxcs,pucs,rsw,expt, &
                 knew,rswl,lsofta,ldipc,nnudc, &
                 lanal,stoag,numcat,lldcabs, &
                 llconsnu,rmsdval,rmsdvdc, &
                 iunij)
   !
   ! Routine computes force field for NCS constraints
   !
   ! C   in [KCAL/(M*ppm**2)],  d in [ppm] units.
   !  IJ
   !
   ! author: Jinhyuk Lee and Wonpil Im
   !
   use number
   use dimens_fcm
   use stream
#if KEY_ENSEMBLE==1
   use ensemble
   use mpi
#endif
#if KEY_PARALLEL==1
   use parallel
#endif
   use consta
   use memory
   use chutil, only: atomid
   use param_store, only: set_param

   implicit none

   real(chm_real) en
   real(chm_real) X(*),Y(*),Z(*)
   real(chm_real) DX(*),DY(*),DZ(*)

   integer csnum,csnm2,csipt(*),csinm(*),cslis(*),numcat
   integer stoag(2,*)
   real(chm_real) sigma(3,*),pphics(*),expcs(*),forcs(*)
   logical llconsnu(*),lldcabs(*),ldipc(*)

   ! for soft asymptote
   logical lsofta(*)
   real(chm_real) kmincs(*),plcs(*),kmaxcs(*),pucs(*),rsw(*),expt(*),knew(*),rswl(*)
   real asymp,asympl,bsymp,bsympl

   ! for analysis
   logical lanal
   real rmsdval,rmsdvdc,sumdc,sumcs
   logical lsrm
   integer numdc,numcs
   integer iunij

#if KEY_ENSEMBLE==1
   real(chm_real) sii(3)
   real(chm_real) nuens(1)
   real(chm_real) ave
   real(chm_real),allocatable,dimension(:) :: siibuf,nubuf
   integer ierror,m,n
#endif

   ! local
   real ms11,ms22,ms33,phics,nudc,phics2,phicsrad
   integer i,j,k,atomn
   logical ldcabs,lconsnu,ldipcc,ldipnc
   integer nnum,hnum,onum

   CHARACTER*8 SIDI,RIDI,RENI,ACI
   CHARACTER*8 SIDN,RIDN,RENN,ACN
   CHARACTER*8 SIDH,RIDH,RENH,ACH
   CHARACTER*8 SIDO,RIDO,RENO,ACO
   character*4 atna
   character*1 atn1

   real*8 nhx,nhy,nhz
   real*8 nox,noy,noz
   real*8 s2x,s2y,s2z
   real*8 s22,s11,s33

   ! S_22
   ! N
   real*8 nhxnx,nhynx,nhznx
   real*8 noxnx,noynx,noznx
   real*8 nhxny,nhyny,nhzny
   real*8 noxny,noyny,nozny
   real*8 nhxnz,nhynz,nhznz
   real*8 noxnz,noynz,noznz
   ! H
   real*8 nhxhx,nhyhx,nhzhx
   real*8 noxhx,noyhx,nozhx
   real*8 nhxhy,nhyhy,nhzhy
   real*8 noxhy,noyhy,nozhy
   real*8 nhxhz,nhyhz,nhzhz
   real*8 noxhz,noyhz,nozhz
   ! O
   real*8 nhxox,nhyox,nhzox
   real*8 noxox,noyox,nozox
   real*8 nhxoy,nhyoy,nhzoy
   real*8 noxoy,noyoy,nozoy
   real*8 nhxoz,nhyoz,nhzoz
   real*8 noxoz,noyoz,nozoz

   real*8 s2znx,s2zhx,s2zox
   real*8 s2ynx,s2yhx,s2yox
   real*8 s2xnx,s2xhx,s2xox

   real*8 s2zny,s2zhy,s2zoy
   real*8 s2yny,s2yhy,s2yoy
   real*8 s2xny,s2xhy,s2xoy

   real*8 s2znz,s2zhz,s2zoz
   real*8 s2ynz,s2yhz,s2yoz
   real*8 s2xnz,s2xhz,s2xoz

   real*8 s22nx,s22hx,s22ox
   real*8 s22ny,s22hy,s22oy
   real*8 s22nz,s22hz,s22oz

   real*8 e2znx,e2zhx,e2zox
   real*8 e2ynx,e2yhx,e2yox
   real*8 e2xnx,e2xhx,e2xox

   real*8 e2zny,e2zhy,e2zoy
   real*8 e2yny,e2yhy,e2yoy
   real*8 e2xny,e2xhy,e2xoy

   real*8 e2znz,e2zhz,e2zoz
   real*8 e2ynz,e2yhz,e2yoz
   real*8 e2xnz,e2xhz,e2xoz

   real*8 e2x,e2y,e2z

   ! S_11
   real*8 nhe2x,nhe2y,nhe2z
   real*8 s1x,s1y,s1z
   real*8 nhe2xnx,nhe2xhx,nhe2xox
   real*8 nhe2ynx,nhe2yhx,nhe2yox
   real*8 nhe2znx,nhe2zhx,nhe2zox

   real*8 nhe2xny,nhe2xhy,nhe2xoy
   real*8 nhe2yny,nhe2yhy,nhe2yoy
   real*8 nhe2zny,nhe2zhy,nhe2zoy

   real*8 nhe2xnz,nhe2xhz,nhe2xoz
   real*8 nhe2ynz,nhe2yhz,nhe2yoz
   real*8 nhe2znz,nhe2zhz,nhe2zoz

   real*8 s1xnx,s1xhx,s1xox
   real*8 s1ynx,s1yhx,s1yox
   real*8 s1znx,s1zhx,s1zox

   real*8 s1xny,s1xhy,s1xoy
   real*8 s1yny,s1yhy,s1yoy
   real*8 s1zny,s1zhy,s1zoy

   real*8 s1xnz,s1xhz,s1xoz
   real*8 s1ynz,s1yhz,s1yoz
   real*8 s1znz,s1zhz,s1zoz

   real*8 s11nx,s11hx,s11ox
   real*8 s11ny,s11hy,s11oy
   real*8 s11nz,s11hz,s11oz

   real*8 e1xnx,e1xhx,e1xox
   real*8 e1ynx,e1yhx,e1yox
   real*8 e1znx,e1zhx,e1zox

   real*8 e1xny,e1xhy,e1xoy
   real*8 e1yny,e1yhy,e1yoy
   real*8 e1zny,e1zhy,e1zoy

   real*8 e1xnz,e1xhz,e1xoz
   real*8 e1ynz,e1yhz,e1yoz
   real*8 e1znz,e1zhz,e1zoz

   ! S_33
   real*8 s3x,s3y,s3z
   real*8 s3xnx,s3xhx,s3xox
   real*8 s3ynx,s3yhx,s3yox
   real*8 s3znx,s3zhx,s3zox
   real*8 s33nx,s33hx,s33ox
   real*8 e3xnx,e3xhx,e3xox
   real*8 e3ynx,e3yhx,e3yox
   real*8 e3znx,e3zhx,e3zox

   real*8 s3xny,s3xhy,s3xoy
   real*8 s3yny,s3yhy,s3yoy
   real*8 s3zny,s3zhy,s3zoy
   real*8 s33ny,s33hy,s33oy
   real*8 e3xny,e3xhy,e3xoy
   real*8 e3yny,e3yhy,e3yoy
   real*8 e3zny,e3zhy,e3zoy

   real*8 s3xnz,s3xhz,s3xoz
   real*8 s3ynz,s3yhz,s3yoz
   real*8 s3znz,s3zhz,s3zoz
   real*8 s33nz,s33hz,s33oz
   real*8 e3xnz,e3xhz,e3xoz
   real*8 e3ynz,e3yhz,e3yoz
   real*8 e3znz,e3zhz,e3zoz

   ! final
   real*8 st
   real*8 stnx,sthx,stox
   real*8 stny,sthy,stoy
   real*8 stnz,sthz,stoz

   real*8 enecs
   real*8 enenx,enehx,eneox
   real*8 eneny,enehy,eneoy
   real*8 enenz,enehz,eneoz

   ! DC (dipolar coupling)
   real*8 nh,dcost,nnudc(*),nu
   real*8 nhnx,nhny,nhnz
   real*8 nhhx,nhhy,nhhz
   real*8 costnx,costny,costnz
   real*8 costhx,costhy,costhz
   real*8 nunx,nuny,nunz
   real*8 nuhx,nuhy,nuhz
   real*8 unx,uny,unz
   real*8 uhx,uhy,uhz
   ! nu_0
   real*8 gamn,gamh,gamc,perms
   real*8 chnudc

   ! changable nu
   real*8 chnunx,chnuny,chnunz
   real*8 chnuhx,chnuhy,chnuhz

   ! initialize
   enecs=0.d0
   lsrm=.true.
   sumdc=0.d0
   sumcs=0.d0
   numcs=0
   numdc=0

   ! permeability of space = mu_0 / 8*pi^3
   perms=4.d0*PI*1D-07/(8.d0*PI**3)
   ! gyromagnetic constants
   gamn=2.7128D07
   gamh=26.753D07
   gamc=6.73D07

   do k=1,numcat
      ms11=sigma(1,k)
      ms22=sigma(2,k)
      ms33=sigma(3,k)
      phics=pphics(k)
      nudc=nnudc(k)
      ldcabs=lldcabs(k)
      lconsnu=llconsnu(k)

      if(prnlev.ge.6) then
         write(outu,*) 'Parameters for CS and DC'
         write(outu,*) 'cat S11 S22 S33 phics nudc numres '
         write(outu,'(i2,5f7.3,i3)') k,ms11,ms22,ms33,phics,nudc,stoag(2,k)-stoag(1,k)+1
      endif

      ! number of restraints
      do i=stoag(1,k),stoag(2,k)
         do j=1,csinm(i)
            atomn=cslis(csipt(i)+j-1)
            call atomid(atomn,sidi,ridi,reni,aci)
            if(prnlev.ge.6) then
               write(outu,*) 'atomn',atomn,sidi(1:idleng),ridi(1:idleng),aci(1:idleng)
            endif
            !atna=aci(1:idleng)
            !
            !! jlee 051506 for trp indole CS (CD1-NE1-HE1)
            !! 051806 add N-C dipolr coupling ldipnc = .true.
            !! 010807 add CA-CB DQS for alanine (ta/11) ldipcc = .true.
            !! define N
            !if((atna.eq.'N   ').or.(atna.eq.'NE1 ').or. &
            !   (atna.eq.'CA  ')) then
            !   nnum=atomn
            !endif
            !
            !! define H
            !! add CD for proline
            !if((atna.eq.'HN  ').or.(atna.eq.'HE1 ').or.&
            !   (atna.eq.'C   ').or.(atna.eq.'CB  ').or.&
            !   (atna.eq.'CD  ').or.(atna.eq.'H   ')) then
            !
            !   ! NH DC
            !   ldipnc=.false.
            !   ldipcc=.false.
            !
            !   ! NC DC
            !   if((atna.eq.'C   ')) ldipnc=.true.
            !   ! CC DC
            !   if((atna.eq.'CB  ')) ldipcc=.true.
            !   hnum=atomn
            !endif
            !
            !! define C
            !if((atna.eq.'C   ').or.(atna.eq.'CD1 ')) then
            !   onum=atomn
            !endif

            ! TODO: use multiple selection instead of sorting out atoms based on their name
            atna=aci(1:idleng)
            atn1=aci(1:1)
            if (ldipc(i)) then
               !----------------
               ! DC

               if ((atn1.eq.'N').or.(atna.eq.'CA  ')) nnum=atomn

               if ((atn1.eq.'H').or.(atna.eq.'C   ').or.&
                   (atna.eq.'CB  ')) hnum=atomn

               ldipnc=.false.
               ldipcc=.false.

               ! NC DC
               if (atna.eq.'C   ') ldipnc=.true.
               ! CC DC
               if (atna.eq.'CB  ') ldipcc=.true.

            else
               !----------------
               ! CS

               if (atn1.eq.'N') nnum=atomn
               if (atn1.eq.'H') hnum=atomn
               if (atn1.eq.'C') onum=atomn

            endif
         enddo

         ! S_22 =  NC X NH
         !      = (-nhy noz + nhz noy)i + (-nhz nox + nhx noz)j + (-nhx noy + nhy nox)k
         !      = s_2x i + s_2y j + s_2z k

         ! N->H
         nhx=x(hnum)-x(nnum)
         nhy=y(hnum)-y(nnum)
         nhz=z(hnum)-z(nnum)

         if(.not.ldipc(i)) then
            ! N->C
            nox=x(onum)-x(nnum)
            noy=y(onum)-y(nnum)
            noz=z(onum)-z(nnum)
         endif

         if(ldipc(i)) then
            !----------------
            ! DC
            ! |NH| = sqrt((nhx)^2+(nhy)^2+(nhz)^2)
            nh=sqrt(nhx**2+nhy**2+nhz**2)

            ! changable nudc = (nu_0) = g1  g2  h / r^3  mu_0 / (8*pi^3)
            if(.not.lconsnu) then
               if(ldipnc.or.ldipcc) then
                  ! in case of NC
                  chnudc=gamn*gamc*hplanck*perms*1D-3/(nh*1D-10)**3
                  ! in case of CC
                  if(ldipcc) chnudc=gamc*gamc*hplanck*perms*1D-3/(nh*1D-10)**3
               else
                  ! in case of NH
                  chnudc=gamn*gamh*hplanck*perms*1D-3/(nh*1D-10)**3
               endif
            endif

            if(prnlev.ge.6) then
               if(.not.lconsnu) then
                  write(outu,*) 'changable nu'
                  write(outu,*) 'cal nudc cons_nu',nh,chnudc,nudc
               else
                  write(outu,*) 'constant nu'
                  write(outu,*) 'constant nudc',nh,nudc
               endif
            endif

            ! DC
            ! cos(T)=NH (o) Uz / |NH||Uz| = nhz / |NH|
            dcost=nhz/nh

         else
            !----------------
            ! CS

            ! s_2x = (nhy noz - nhz noy)
            s2x=-nhy*noz+nhz*noy
            s2y=-nhz*nox+nhx*noz
            s2z=-nhx*noy+nhy*nox

            ! |S_22| = sqrt(s2x^2+s2y^2+s2z^2)
            s22=sqrt(s2x*s2x+s2y*s2y+s2z*s2z)

         endif

         ! COMMON
         ! NH & NC by x (N)
         nhxnx=-1.d0
         nhynx=0.d0
         nhznx=0.d0
         noxnx=-1.d0
         noynx=0.d0
         noznx=0.d0
         ! by y
         nhxny=0.d0
         nhyny=-1.d0
         nhzny=0.d0
         noxny=0.d0
         noyny=-1.d0
         nozny=0.d0
         ! by z
         nhxnz=0.d0
         nhynz=0.d0
         nhznz=-1.d0
         noxnz=0.d0
         noynz=0.d0
         noznz=-1.d0
         ! by x (H)
         nhxhx=1.d0
         nhyhx=0.d0
         nhzhx=0.d0
         noxhx=0.d0
         noyhx=0.d0
         nozhx=0.d0
         ! by y
         nhxhy=0.d0
         nhyhy=1.d0
         nhzhy=0.d0
         noxhy=0.d0
         noyhy=0.d0
         nozhy=0.d0
         ! by z
         nhxhz=0.d0
         nhyhz=0.d0
         nhzhz=1.d0
         noxhz=0.d0
         noyhz=0.d0
         nozhz=0.d0
         ! by x (C)
         nhxox=0.d0
         nhyox=0.d0
         nhzox=0.d0
         noxox=1.d0
         noyox=0.d0
         nozox=0.d0
         ! by y
         nhxoy=0.d0
         nhyoy=0.d0
         nhzoy=0.d0
         noxoy=0.d0
         noyoy=1.d0
         nozoy=0.d0
         ! by z
         nhxoz=0.d0
         nhyoz=0.d0
         nhzoz=0.d0
         noxoz=0.d0
         noyoz=0.d0
         nozoz=1.d0

         if(.not.ldipc(i)) then
            !----------------
            ! CS

            ! (s_2z)' by x (N)
            !   = (nhx)' noy + nhx (noy)' - (nhy)' nox - nhy (nox)'
            s2znx= -nhxnx*noy-nhx*noynx+nhynx*nox+nhy*noxnx
            s2zny= -nhxny*noy-nhx*noyny+nhyny*nox+nhy*noxny
            s2znz= -nhxnz*noy-nhx*noynz+nhynz*nox+nhy*noxnz

            ! (s_2z)' by x (H)
            s2zhx= -nhxhx*noy-nhx*noyhx+nhyhx*nox+nhy*noxhx
            s2zhy= -nhxhy*noy-nhx*noyhy+nhyhy*nox+nhy*noxhy
            s2zhz= -nhxhz*noy-nhx*noyhz+nhyhz*nox+nhy*noxhz

            ! (s_2z)' by x (C)
            s2zox= -nhxox*noy-nhx*noyox+nhyox*nox+nhy*noxox
            s2zoy= -nhxoy*noy-nhx*noyoy+nhyoy*nox+nhy*noxoy
            s2zoz= -nhxoz*noy-nhx*noyoz+nhyoz*nox+nhy*noxoz

            ! (s_2y)' by x (N)
            ! = (nhz)' nox + nhz (nox)' - (nhx)' noz - nhx (noz)'
            s2ynx= -nhznx*nox-nhz*noxnx+nhxnx*noz+nhx*noznx
            s2yny= -nhzny*nox-nhz*noxny+nhxny*noz+nhx*nozny
            s2ynz= -nhznz*nox-nhz*noxnz+nhxnz*noz+nhx*noznz

            ! (s_2y)' by x (H)
            s2yhx= -nhzhx*nox-nhz*noxhx+nhxhx*noz+nhx*nozhx
            s2yhy= -nhzhy*nox-nhz*noxhy+nhxhy*noz+nhx*nozhy
            s2yhz= -nhzhz*nox-nhz*noxhz+nhxhz*noz+nhx*nozhz

            ! (s_2y)' by x (C)
            s2yox= -nhzox*nox-nhz*noxox+nhxox*noz+nhx*nozox
            s2yoy= -nhzoy*nox-nhz*noxoy+nhxoy*noz+nhx*nozoy
            s2yoz= -nhzoz*nox-nhz*noxoz+nhxoz*noz+nhx*nozoz

            ! (s_2x)' by x (N)
            ! = (nhy)' noz + nhy (noz)' - (nhz)' noy - nhz (noy)'
            s2xnx= -nhynx*noz-nhy*noznx+nhznx*noy+nhz*noynx
            s2xny= -nhyny*noz-nhy*nozny+nhzny*noy+nhz*noyny
            s2xnz= -nhynz*noz-nhy*noznz+nhznz*noy+nhz*noynz

            ! (s_2y)' by x (H)
            s2xhx= -nhyhx*noz-nhy*nozhx+nhzhx*noy+nhz*noyhx
            s2xhy= -nhyhy*noz-nhy*nozhy+nhzhy*noy+nhz*noyhy
            s2xhz= -nhyhz*noz-nhy*nozhz+nhzhz*noy+nhz*noyhz

            ! (s_2y)' by x (C)
            s2xox= -nhyox*noz-nhy*nozox+nhzox*noy+nhz*noyox
            s2xoy= -nhyoy*noz-nhy*nozoy+nhzoy*noy+nhz*noyoy
            s2xoz= -nhyoz*noz-nhy*nozoz+nhzoz*noy+nhz*noyoz

            ! (|s_22|)' by x at N
            ! x
            s22nx=1.d0/s22*(s2x*s2xnx+s2y*s2ynx+s2z*s2znx)
            s22ny=1.d0/s22*(s2x*s2xny+s2y*s2yny+s2z*s2zny)
            s22nz=1.d0/s22*(s2x*s2xnz+s2y*s2ynz+s2z*s2znz)

            ! (s_22)' by x at H
            ! x
            s22hx=1.d0/s22*(s2x*s2xhx+s2y*s2yhx+s2z*s2zhx)
            s22hy=1.d0/s22*(s2x*s2xhy+s2y*s2yhy+s2z*s2zhy)
            s22hz=1.d0/s22*(s2x*s2xhz+s2y*s2yhz+s2z*s2zhz)

            ! (s_22)' by x at C
            ! x
            s22ox=1.d0/s22*(s2x*s2xox+s2y*s2yox+s2z*s2zox)
            s22oy=1.d0/s22*(s2x*s2xoy+s2y*s2yoy+s2z*s2zoy)
            s22oz=1.d0/s22*(s2x*s2xoz+s2y*s2yoz+s2z*s2zoz)


            ! NC X NH
            ! e_2x
            e2x=s2x/s22
            e2y=s2y/s22
            e2z=s2z/s22

            ! (e_2z)' by x at N
            e2znx= -1.d0 /(s22*s22)*s22nx*s2z+1.d0 /s22*s2znx
            e2zny= -1.d0 /(s22*s22)*s22ny*s2z+1.d0 /s22*s2zny
            e2znz= -1.d0 /(s22*s22)*s22nz*s2z+1.d0 /s22*s2znz

            ! (e_2z)' by x at H
            e2zhx= -1.d0 /(s22*s22)*s22hx*s2z+1.d0 /s22*s2zhx
            e2zhy= -1.d0 /(s22*s22)*s22hy*s2z+1.d0 /s22*s2zhy
            e2zhz= -1.d0 /(s22*s22)*s22hz*s2z+1.d0 /s22*s2zhz

            ! (e_2z)' by x at C
            e2zox= -1.d0 /(s22*s22)*s22ox*s2z+1.d0 /s22*s2zox
            e2zoy= -1.d0 /(s22*s22)*s22oy*s2z+1.d0 /s22*s2zoy
            e2zoz= -1.d0 /(s22*s22)*s22oz*s2z+1.d0 /s22*s2zoz

            ! (e_2y)' by x at N
            e2ynx= -1.d0 /(s22*s22)*s22nx*s2y+1.d0 /s22*s2ynx
            e2yny= -1.d0 /(s22*s22)*s22ny*s2y+1.d0 /s22*s2yny
            e2ynz= -1.d0 /(s22*s22)*s22nz*s2y+1.d0 /s22*s2ynz

            ! (e_2y)' by x at H
            e2yhx= -1.d0 /(s22*s22)*s22hx*s2y+1.d0 /s22*s2yhx
            e2yhy= -1.d0 /(s22*s22)*s22hy*s2y+1.d0 /s22*s2yhy
            e2yhz= -1.d0 /(s22*s22)*s22hz*s2y+1.d0 /s22*s2yhz

            ! (e_2y)' by x at C
            e2yox= -1.d0 /(s22*s22)*s22ox*s2y+1.d0 /s22*s2yox
            e2yoy= -1.d0 /(s22*s22)*s22oy*s2y+1.d0 /s22*s2yoy
            e2yoz= -1.d0 /(s22*s22)*s22oz*s2y+1.d0 /s22*s2yoz

            ! (e_2x)' by x at N
            e2xnx= -1.d0 /(s22*s22)*s22nx*s2x+1.d0 /s22*s2xnx
            e2xny= -1.d0 /(s22*s22)*s22ny*s2x+1.d0 /s22*s2xny
            e2xnz= -1.d0 /(s22*s22)*s22nz*s2x+1.d0 /s22*s2xnz

            ! (e_2x)' by x at H
            e2xhx= -1.d0 /(s22*s22)*s22hx*s2x+1.d0 /s22*s2xhx
            e2xhy= -1.d0 /(s22*s22)*s22hy*s2x+1.d0 /s22*s2xhy
            e2xhz= -1.d0 /(s22*s22)*s22hz*s2x+1.d0 /s22*s2xhz

            ! (e_2x)' by x at C
            e2xox= -1.d0 /(s22*s22)*s22ox*s2x+1.d0 /s22*s2xox
            e2xoy= -1.d0 /(s22*s22)*s22oy*s2x+1.d0 /s22*s2xoy
            e2xoz= -1.d0 /(s22*s22)*s22oz*s2x+1.d0 /s22*s2xoz


            ! S_11
            ! S_11  from nmr_precode.inp
            ! = (nhx cos phi + nhe2x sin phi)i + (nhy cos phi + nhe2y sin phi)j
            ! + (nhz cos phi + nhe2z sin phi)k
            !
            ! nhe2 = NH X e_22
            ! = (nhy e2z - nhz e2y)i + (nhz e2x - nhx e2z)j + (nhx e2y - nhy e2x)k
            ! = nhe2x i + nhe2y j + nhe2z k

            ! |S_11|= sqrt((nhx*cosphi+nhe2x*sinphi)^2+
            !              (nhy*cosphi+nhe2y*sinphi)^2+
            !              (nhz*cosphi+nhe2z*sinphi)^2)

            nhe2x=nhy*e2z-nhz*e2y
            nhe2y=nhz*e2x-nhx*e2z
            nhe2z=nhx*e2y-nhy*e2x

            s1x=nhx*cos(phics*degrad)+nhe2x*sin(phics*degrad)
            s1y=nhy*cos(phics*degrad)+nhe2y*sin(phics*degrad)
            s1z=nhz*cos(phics*degrad)+nhe2z*sin(phics*degrad)

            s11=sqrt(s1x*s1x+s1y*s1y+s1z*s1z)

            ! (nhe2x)' by x (N)
            ! = (nhy)' e2z + nhy * (e2z)' - (nhz)' e2y - nhz (e2y)'
            nhe2xnx=nhynx*e2z+nhy*e2znx-nhznx*e2y-nhz*e2ynx
            nhe2xny=nhyny*e2z+nhy*e2zny-nhzny*e2y-nhz*e2yny
            nhe2xnz=nhynz*e2z+nhy*e2znz-nhznz*e2y-nhz*e2ynz

            ! by x (H)
            nhe2xhx=nhyhx*e2z+nhy*e2zhx-nhzhx*e2y-nhz*e2yhx
            nhe2xhy=nhyhy*e2z+nhy*e2zhy-nhzhy*e2y-nhz*e2yhy
            nhe2xhz=nhyhz*e2z+nhy*e2zhz-nhzhz*e2y-nhz*e2yhz

            ! by x (C)
            nhe2xox=nhyox*e2z+nhy*e2zox-nhzox*e2y-nhz*e2yox
            nhe2xoy=nhyoy*e2z+nhy*e2zoy-nhzoy*e2y-nhz*e2yoy
            nhe2xoz=nhyoz*e2z+nhy*e2zoz-nhzoz*e2y-nhz*e2yoz

            ! (nhe2y)' by x (N)
            ! = (nhz)' e2x + nhz (e2x)' - (nhx)' e2z - nhx (e2z)'
            nhe2ynx=nhznx*e2x+nhz*e2xnx-nhxnx*e2z-nhx*e2znx
            nhe2yny=nhzny*e2x+nhz*e2xny-nhxny*e2z-nhx*e2zny
            nhe2ynz=nhznz*e2x+nhz*e2xnz-nhxnz*e2z-nhx*e2znz

            ! by x (H)
            nhe2yhx=nhzhx*e2x+nhz*e2xhx-nhxhx*e2z-nhx*e2zhx
            nhe2yhy=nhzhy*e2x+nhz*e2xhy-nhxhy*e2z-nhx*e2zhy
            nhe2yhz=nhzhz*e2x+nhz*e2xhz-nhxhz*e2z-nhx*e2zhz

            ! by x (C)
            nhe2yox=nhzox*e2x+nhz*e2xox-nhxox*e2z-nhx*e2zox
            nhe2yoy=nhzoy*e2x+nhz*e2xoy-nhxoy*e2z-nhx*e2zoy
            nhe2yoz=nhzoz*e2x+nhz*e2xoz-nhxoz*e2z-nhx*e2zoz

            ! (nhe2z)' by x (N)
            ! = (nhx)' e2y + nhx (e2y)' - (nhy)' e2x - nhy (e2x)'
            nhe2znx=nhxnx*e2y+nhx*e2ynx-nhynx*e2x-nhy*e2xnx
            nhe2zny=nhxny*e2y+nhx*e2yny-nhyny*e2x-nhy*e2xny
            nhe2znz=nhxnz*e2y+nhx*e2ynz-nhynz*e2x-nhy*e2xnz

            ! by x (H)
            nhe2zhx=nhxhx*e2y+nhx*e2yhx-nhyhx*e2x-nhy*e2xhx
            nhe2zhy=nhxhy*e2y+nhx*e2yhy-nhyhy*e2x-nhy*e2xhy
            nhe2zhz=nhxhz*e2y+nhx*e2yhz-nhyhz*e2x-nhy*e2xhz

            ! by x (C)
            nhe2zox=nhxox*e2y+nhx*e2yox-nhyox*e2x-nhy*e2xox
            nhe2zoy=nhxoy*e2y+nhx*e2yoy-nhyoy*e2x-nhy*e2xoy
            nhe2zoz=nhxoz*e2y+nhx*e2yoz-nhyoz*e2x-nhy*e2xoz

            ! (s_1x)' by x (N)
            ! = (nhx)' cos phi + 0 + (nhe2x)' sin phi + 0
            phicsrad = phics*degrad
            s1xnx=nhxnx*cos(phicsrad)+nhe2xnx*sin(phicsrad)
            s1xny=nhxny*cos(phicsrad)+nhe2xny*sin(phicsrad)
            s1xnz=nhxnz*cos(phicsrad)+nhe2xnz*sin(phicsrad)

            ! by x (H)
            s1xhx=nhxhx*cos(phicsrad)+nhe2xhx*sin(phicsrad)
            s1xhy=nhxhy*cos(phicsrad)+nhe2xhy*sin(phicsrad)
            s1xhz=nhxhz*cos(phicsrad)+nhe2xhz*sin(phicsrad)

            ! by x (C)
            s1xox=nhxox*cos(phicsrad)+nhe2xox*sin(phicsrad)
            s1xoy=nhxoy*cos(phicsrad)+nhe2xoy*sin(phicsrad)
            s1xoz=nhxoz*cos(phicsrad)+nhe2xoz*sin(phicsrad)

            ! (s_1y)' by x (N)
            ! = (nhy)' cos phi + (nhe2y)' sin phi
            s1ynx=nhynx*cos(phicsrad)+nhe2ynx*sin(phicsrad)
            s1yny=nhyny*cos(phicsrad)+nhe2yny*sin(phicsrad)
            s1ynz=nhynz*cos(phicsrad)+nhe2ynz*sin(phicsrad)

            ! by x (H)
            s1yhx=nhyhx*cos(phicsrad)+nhe2yhx*sin(phicsrad)
            s1yhy=nhyhy*cos(phicsrad)+nhe2yhy*sin(phicsrad)
            s1yhz=nhyhz*cos(phicsrad)+nhe2yhz*sin(phicsrad)

            ! by x (C)
            s1yox=nhyox*cos(phicsrad)+nhe2yox*sin(phicsrad)
            s1yoy=nhyoy*cos(phicsrad)+nhe2yoy*sin(phicsrad)
            s1yoz=nhyoz*cos(phicsrad)+nhe2yoz*sin(phicsrad)

            ! (s_1z)' by x (N)
            ! = (nhz)' cos phi + (nhe2z)' sin phi
            s1znx=nhznx*cos(phicsrad)+nhe2znx*sin(phicsrad)
            s1zny=nhzny*cos(phicsrad)+nhe2zny*sin(phicsrad)
            s1znz=nhznz*cos(phicsrad)+nhe2znz*sin(phicsrad)

            ! by x (H)
            s1zhx=nhzhx*cos(phicsrad)+nhe2zhx*sin(phicsrad)
            s1zhy=nhzhy*cos(phicsrad)+nhe2zhy*sin(phicsrad)
            s1zhz=nhzhz*cos(phicsrad)+nhe2zhz*sin(phicsrad)

            ! by x (C)
            s1zox=nhzox*cos(phicsrad)+nhe2zox*sin(phicsrad)
            s1zoy=nhzoy*cos(phicsrad)+nhe2zoy*sin(phicsrad)
            s1zoz=nhzoz*cos(phicsrad)+nhe2zoz*sin(phicsrad)

            ! (|s_11|)' by x (N)
            s11nx=1.d0/s11*(s1x*s1xnx+s1y*s1ynx+s1z*s1znx)
            s11ny=1.d0/s11*(s1x*s1xny+s1y*s1yny+s1z*s1zny)
            s11nz=1.d0/s11*(s1x*s1xnz+s1y*s1ynz+s1z*s1znz)

            ! by x (H)
            s11hx=1.d0/s11*(s1x*s1xhx+s1y*s1yhx+s1z*s1zhx)
            s11hy=1.d0/s11*(s1x*s1xhy+s1y*s1yhy+s1z*s1zhy)
            s11hz=1.d0/s11*(s1x*s1xhz+s1y*s1yhz+s1z*s1zhz)

            ! by x (C)
            s11ox=1.d0/s11*(s1x*s1xox+s1y*s1yox+s1z*s1zox)
            s11oy=1.d0/s11*(s1x*s1xoy+s1y*s1yoy+s1z*s1zoy)
            s11oz=1.d0/s11*(s1x*s1xoz+s1y*s1yoz+s1z*s1zoz)

            ! (e1x)' by x (N)
            e1xnx=-1.d0/(s11*s11)*s11nx*s1x+1.d0/s11*s1xnx
            e1xny=-1.d0/(s11*s11)*s11ny*s1x+1.d0/s11*s1xny
            e1xnz=-1.d0/(s11*s11)*s11nz*s1x+1.d0/s11*s1xnz

            ! by x (H)
            e1xhx=-1.d0/(s11*s11)*s11hx*s1x+1.d0/s11*s1xhx
            e1xhy=-1.d0/(s11*s11)*s11hy*s1x+1.d0/s11*s1xhy
            e1xhz=-1.d0/(s11*s11)*s11hz*s1x+1.d0/s11*s1xhz

            ! by x (C)
            e1xox=-1.d0/(s11*s11)*s11ox*s1x+1.d0/s11*s1xox
            e1xoy=-1.d0/(s11*s11)*s11oy*s1x+1.d0/s11*s1xoy
            e1xoz=-1.d0/(s11*s11)*s11oz*s1x+1.d0/s11*s1xoz

            ! (e1y)' by x (N)
            e1ynx=-1.d0/(s11*s11)*s11nx*s1y+1.d0/s11*s1ynx
            e1yny=-1.d0/(s11*s11)*s11ny*s1y+1.d0/s11*s1yny
            e1ynz=-1.d0/(s11*s11)*s11nz*s1y+1.d0/s11*s1ynz

            ! by x (H)
            e1yhx=-1.d0/(s11*s11)*s11hx*s1y+1.d0/s11*s1yhx
            e1yhy=-1.d0/(s11*s11)*s11hy*s1y+1.d0/s11*s1yhy
            e1yhz=-1.d0/(s11*s11)*s11hz*s1y+1.d0/s11*s1yhz

            ! by x (C)
            e1yox=-1.d0/(s11*s11)*s11ox*s1y+1.d0/s11*s1yox
            e1yoy=-1.d0/(s11*s11)*s11oy*s1y+1.d0/s11*s1yoy
            e1yoz=-1.d0/(s11*s11)*s11oz*s1y+1.d0/s11*s1yoz

            ! (e1z)' by x (N)
            e1znx=-1.d0/(s11*s11)*s11nx*s1z+1.d0/s11*s1znx
            e1zny=-1.d0/(s11*s11)*s11ny*s1z+1.d0/s11*s1zny
            e1znz=-1.d0/(s11*s11)*s11nz*s1z+1.d0/s11*s1znz

            ! by x (H)
            e1zhx=-1.d0/(s11*s11)*s11hx*s1z+1.d0/s11*s1zhx
            e1zhy=-1.d0/(s11*s11)*s11hy*s1z+1.d0/s11*s1zhy
            e1zhz=-1.d0/(s11*s11)*s11hz*s1z+1.d0/s11*s1zhz

            ! by x (C)
            e1zox=-1.d0/(s11*s11)*s11ox*s1z+1.d0/s11*s1zox
            e1zoy=-1.d0/(s11*s11)*s11oy*s1z+1.d0/s11*s1zoy
            e1zoz=-1.d0/(s11*s11)*s11oz*s1z+1.d0/s11*s1zoz

            ! S_33

            ! |S_33|=sqrt((nhx*cosphi2+nhe2x*sinphi2)^2+
            !             (nhy*cosphi2+nhe2y*sinphi2)^2+
            !             (nhz*cosphi2+nhe2z*sinphi2)^2)

            phics2=phics-90.d0
            phicsrad=phics2*degrad

            s3x=nhx*cos(phicsrad)+nhe2x*sin(phicsrad)
            s3y=nhy*cos(phicsrad)+nhe2y*sin(phicsrad)
            s3z=nhz*cos(phicsrad)+nhe2z*sin(phicsrad)

            s33=sqrt(s3x*s3x+s3y*s3y+s3z*s3z)

            ! (s_3x)' by x (N)
            ! = (nhx)' cos phi2 + (nhe2x)' sin phi2
            s3xnx=nhxnx*cos(phicsrad)+nhe2xnx*sin(phicsrad)
            s3xny=nhxny*cos(phicsrad)+nhe2xny*sin(phicsrad)
            s3xnz=nhxnz*cos(phicsrad)+nhe2xnz*sin(phicsrad)

            ! by x (H)
            s3xhx=nhxhx*cos(phicsrad)+nhe2xhx*sin(phicsrad)
            s3xhy=nhxhy*cos(phicsrad)+nhe2xhy*sin(phicsrad)
            s3xhz=nhxhz*cos(phicsrad)+nhe2xhz*sin(phicsrad)

            ! by x (C)
            s3xox=nhxox*cos(phicsrad)+nhe2xox*sin(phicsrad)
            s3xoy=nhxoy*cos(phicsrad)+nhe2xoy*sin(phicsrad)
            s3xoz=nhxoz*cos(phicsrad)+nhe2xoz*sin(phicsrad)

            ! (s_3y)' by x (N)
            ! = (nhy)' cos phi2 + (nhe2y)' sin phi2
            s3ynx=nhynx*cos(phicsrad)+nhe2ynx*sin(phicsrad)
            s3yny=nhyny*cos(phicsrad)+nhe2yny*sin(phicsrad)
            s3ynz=nhynz*cos(phicsrad)+nhe2ynz*sin(phicsrad)

            ! by x (H)
            s3yhx=nhyhx*cos(phicsrad)+nhe2yhx*sin(phicsrad)
            s3yhy=nhyhy*cos(phicsrad)+nhe2yhy*sin(phicsrad)
            s3yhz=nhyhz*cos(phicsrad)+nhe2yhz*sin(phicsrad)

            ! by x (C)
            s3yox=nhyox*cos(phicsrad)+nhe2yox*sin(phicsrad)
            s3yoy=nhyoy*cos(phicsrad)+nhe2yoy*sin(phicsrad)
            s3yoz=nhyoz*cos(phicsrad)+nhe2yoz*sin(phicsrad)

            ! (s_3z)' by x (N)
            ! = (nhz)' cos phi2 + (nhe2z)' sin phi2
            s3znx=nhznx*cos(phicsrad)+nhe2znx*sin(phicsrad)
            s3zny=nhzny*cos(phicsrad)+nhe2zny*sin(phicsrad)
            s3znz=nhznz*cos(phicsrad)+nhe2znz*sin(phicsrad)

            ! by x (H)
            s3zhx=nhzhx*cos(phicsrad)+nhe2zhx*sin(phicsrad)
            s3zhy=nhzhy*cos(phicsrad)+nhe2zhy*sin(phicsrad)
            s3zhz=nhzhz*cos(phicsrad)+nhe2zhz*sin(phicsrad)

            ! by x (C)
            s3zox=nhzox*cos(phicsrad)+nhe2zox*sin(phicsrad)
            s3zoy=nhzoy*cos(phicsrad)+nhe2zoy*sin(phicsrad)
            s3zoz=nhzoz*cos(phicsrad)+nhe2zoz*sin(phicsrad)

            ! (|s_33|)' by x (N)
            s33nx=1.d0/s33*(s3x*s3xnx+s3y*s3ynx+s3z*s3znx)
            s33ny=1.d0/s33*(s3x*s3xny+s3y*s3yny+s3z*s3zny)
            s33nz=1.d0/s33*(s3x*s3xnz+s3y*s3ynz+s3z*s3znz)

            ! by x (H)
            s33hx=1.d0/s33*(s3x*s3xhx+s3y*s3yhx+s3z*s3zhx)
            s33hy=1.d0/s33*(s3x*s3xhy+s3y*s3yhy+s3z*s3zhy)
            s33hz=1.d0/s33*(s3x*s3xhz+s3y*s3yhz+s3z*s3zhz)

            ! by x (C)
            s33ox=1.d0/s33*(s3x*s3xox+s3y*s3yox+s3z*s3zox)
            s33oy=1.d0/s33*(s3x*s3xoy+s3y*s3yoy+s3z*s3zoy)
            s33oz=1.d0/s33*(s3x*s3xoz+s3y*s3yoz+s3z*s3zoz)

            ! (e3x)' by x (N)
            e3xnx=-1.d0/(s33*s33)*s33nx*s3x+1.d0/s33*s3xnx
            e3xny=-1.d0/(s33*s33)*s33ny*s3x+1.d0/s33*s3xny
            e3xnz=-1.d0/(s33*s33)*s33nz*s3x+1.d0/s33*s3xnz

            ! by x (H)
            e3xhx=-1.d0/(s33*s33)*s33hx*s3x+1.d0/s33*s3xhx
            e3xhy=-1.d0/(s33*s33)*s33hy*s3x+1.d0/s33*s3xhy
            e3xhz=-1.d0/(s33*s33)*s33hz*s3x+1.d0/s33*s3xhz

            ! by x (C)
            e3xox=-1.d0/(s33*s33)*s33ox*s3x+1.d0/s33*s3xox
            e3xoy=-1.d0/(s33*s33)*s33oy*s3x+1.d0/s33*s3xoy
            e3xoz=-1.d0/(s33*s33)*s33oz*s3x+1.d0/s33*s3xoz

            ! (e3y)' by x (N)
            e3ynx=-1.d0/(s33*s33)*s33nx*s3y+1.d0/s33*s3ynx
            e3yny=-1.d0/(s33*s33)*s33ny*s3y+1.d0/s33*s3yny
            e3ynz=-1.d0/(s33*s33)*s33nz*s3y+1.d0/s33*s3ynz

            ! by x (H)
            e3yhx=-1.d0/(s33*s33)*s33hx*s3y+1.d0/s33*s3yhx
            e3yhy=-1.d0/(s33*s33)*s33hy*s3y+1.d0/s33*s3yhy
            e3yhz=-1.d0/(s33*s33)*s33hz*s3y+1.d0/s33*s3yhz

            ! by x (C)
            e3yox=-1.d0/(s33*s33)*s33ox*s3y+1.d0/s33*s3yox
            e3yoy=-1.d0/(s33*s33)*s33oy*s3y+1.d0/s33*s3yoy
            e3yoz=-1.d0/(s33*s33)*s33oz*s3y+1.d0/s33*s3yoz

            ! (e3z)' by x (N)
            e3znx=-1.d0/(s33*s33)*s33nx*s3z+1.d0/s33*s3znx
            e3zny=-1.d0/(s33*s33)*s33ny*s3z+1.d0/s33*s3zny
            e3znz=-1.d0/(s33*s33)*s33nz*s3z+1.d0/s33*s3znz

            ! by x (H)
            e3zhx=-1.d0/(s33*s33)*s33hx*s3z+1.d0/s33*s3zhx
            e3zhy=-1.d0/(s33*s33)*s33hy*s3z+1.d0/s33*s3zhy
            e3zhz=-1.d0/(s33*s33)*s33hz*s3z+1.d0/s33*s3zhz

            ! by x (C)
            e3zox=-1.d0/(s33*s33)*s33ox*s3z+1.d0/s33*s3zox
            e3zoy=-1.d0/(s33*s33)*s33oy*s3z+1.d0/s33*s3zoy
            e3zoz=-1.d0/(s33*s33)*s33oz*s3z+1.d0/s33*s3zoz

         endif

         if(ldipc(i)) then
            !----------------
            ! DC
            ! (|NH|)' = 1/ |NH| ( nhx (nhx)' + nhy (nhy)' + nhz (nhz)' )
            ! by x (N)
            nhnx=1.d0/nh*(nhx*nhxnx+nhy*nhynx+nhz*nhznx)
            nhny=1.d0/nh*(nhx*nhxny+nhy*nhyny+nhz*nhzny)
            nhnz=1.d0/nh*(nhx*nhxnz+nhy*nhynz+nhz*nhznz)

            ! by x (H)
            nhhx=1.d0/nh*(nhx*nhxhx+nhy*nhyhx+nhz*nhzhx)
            nhhy=1.d0/nh*(nhx*nhxhy+nhy*nhyhy+nhz*nhzhy)
            nhhz=1.d0/nh*(nhx*nhxhz+nhy*nhyhz+nhz*nhzhz)

            ! (cost)'= - 1 / |nh|^2 (|nh|)' nhz + 1 / |nh| (nhz)'
            ! by x (N)
            costnx=-1.d0/(nh*nh)*nhnx*nhz+1.d0/nh*nhznx
            costny=-1.d0/(nh*nh)*nhny*nhz+1.d0/nh*nhzny
            costnz=-1.d0/(nh*nh)*nhnz*nhz+1.d0/nh*nhznz

            ! by x (H)
            costhx=-1.d0/(nh*nh)*nhhx*nhz+1.d0/nh*nhzhx
            costhy=-1.d0/(nh*nh)*nhhy*nhz+1.d0/nh*nhzhy
            costhz=-1.d0/(nh*nh)*nhhz*nhz+1.d0/nh*nhzhz

            if(lconsnu) then
               ! constant nu
               ! nu=nu_0 ( 3 cos(T)^2 - 1 )/2
               nu=nudc*(3.d0*dcost*dcost-1.d0)/2.d0

            else
               ! changable nu
               ! nu= chnudc ( 3 cos(T)^2 - 1 )/2
               nu=chnudc*(3.d0*dcost*dcost-1.d0)/2.d0
            endif

#if KEY_ENSEMBLE==1
            nuens(1)=nu
            call chmalloc('ssnmr.src','NUBUF','NUBUF',nensem,crl=nubuf)

            if (mynod == 0) then
               call mpi_barrier(comm_master,ierror)
               call mpi_allgather( &
                  nuens, 1, mpi_double_precision, &
                  nubuf, 1, mpi_double_precision, &
                  comm_master,ierror)

               ave = 0.0
               do m=1,nensem
                  ave = ave + nubuf((m-1)+1)
               enddo
               nuens(1) = ave/real(nensem)

            endif
            !call ensave(nuens,nubuf,1)
            call chmdealloc('ssnmr.src','NUBUF','NUBUF',nensem,crl=nubuf)
            nu=nuens(1)
#endif

            if(lconsnu) then
               ! constant nu
               ! (nu)'= 3 nu_0 cost (cost)'
               ! by x (N)
               nunx=3.d0*nudc*dcost*costnx
               nuny=3.d0*nudc*dcost*costny
               nunz=3.d0*nudc*dcost*costnz

               ! by x (H)
               nuhx=3.d0*nudc*dcost*costhx
               nuhy=3.d0*nudc*dcost*costhy
               nuhz=3.d0*nudc*dcost*costhz
            else

               ! (nu_0)' = g1 g2 h mu_0 / (8 pi^3) (-3) |r|^-4 (r)'
               !         = g1 g2 h mu_0 / ( 8  pi^3) |r|^-3 (-3) / |r| (|r|)'
               !         = chnudc (khz) (-3) / |r| (|r|)'
               ! |r| = |NH| = |H - N|
               ! by x (N)
               chnunx=chnudc*(-3.d0)/nh*nhnx
               chnuny=chnudc*(-3.d0)/nh*nhny
               chnunz=chnudc*(-3.d0)/nh*nhnz
               ! by x (H)
               chnuhx=chnudc*(-3.d0)/nh*nhhx
               chnuhy=chnudc*(-3.d0)/nh*nhhy
               chnuhz=chnudc*(-3.d0)/nh*nhhz

               ! 051506 changable nu
               ! (nu)'= 3 nu_0 cost (cost)' + ( 3 cost^2 - 1 ) / 2 (nu_0)'
               ! by x (N)
               nunx=3.d0*chnudc*dcost*costnx+(3.d0*dcost*dcost-1.d0)/2.d0*chnunx
               nuny=3.d0*chnudc*dcost*costny+(3.d0*dcost*dcost-1.d0)/2.d0*chnuny
               nunz=3.d0*chnudc*dcost*costnz+(3.d0*dcost*dcost-1.d0)/2.d0*chnunz

               ! by x (H)
               nuhx=3.d0*chnudc*dcost*costhx+(3.d0*dcost*dcost-1.d0)/2.d0*chnuhx
               nuhy=3.d0*chnudc*dcost*costhy+(3.d0*dcost*dcost-1.d0)/2.d0*chnuhy
               nuhz=3.d0*chnudc*dcost*costhz+(3.d0*dcost*dcost-1.d0)/2.d0*chnuhz
            endif

         else
            !----------------
            ! CS
            ! st=e_1z^2*ms11+e_2z^2*ms22+e_3z^2*ms33

#if KEY_ENSEMBLE==1
            sii(1)=(s1z/s11)**2
            sii(2)=(s2z/s22)**2
            sii(3)=(s3z/s33)**2

            call chmalloc('ssnmr.src','SIIBUF','SIIBUF',3*nensem,crl=siibuf)
            if (mynod == 0) then
               call mpi_barrier(comm_master,ierror)
               call mpi_allgather( &
                  sii, 3, mpi_double_precision, &
                  siibuf, 3, mpi_double_precision, &
                  comm_master,ierror)

               do n=1,3
                  ave = 0.0
                  do m=1,nensem
                     ave = ave + siibuf((m-1)*3+n)
                  enddo
                  sii(n) = ave/real(nensem)
               enddo

            endif
            !call ensave(sii,siibuf,3)
            call chmdealloc('ssnmr.src','SIIBUF','SIIBUF',3*nensem,crl=siibuf)
            st=sii(1)*ms11+sii(2)*ms22+sii(3)*ms33
#else
            st=((s1z/s11)**2)*ms11+((s2z/s22)**2)*ms22+((s3z/s33)**2)*ms33
#endif

            ! (st)' by x (N)
            ! = 2 e1z (e1z)' ms11 + 2 e2z (e2z)' ms22 + 2 e3z (e3z)' ms33
            stnx=2*S1Z/S11*e1znx*ms11+2*S2Z/S22*e2znx*ms22+2*S3Z/S33*e3znx*ms33
            stny=2*S1Z/S11*e1zny*ms11+2*S2Z/S22*e2zny*ms22+2*S3Z/S33*e3zny*ms33
            stnz=2*S1Z/S11*e1znz*ms11+2*S2Z/S22*e2znz*ms22+2*S3Z/S33*e3znz*ms33

            ! by x (H)
            sthx=2*S1Z/S11*e1zhx*ms11+2*S2Z/S22*e2zhx*ms22+2*S3Z/S33*e3zhx*ms33
            sthy=2*S1Z/S11*e1zhy*ms11+2*S2Z/S22*e2zhy*ms22+2*S3Z/S33*e3zhy*ms33
            sthz=2*S1Z/S11*e1zhz*ms11+2*S2Z/S22*e2zhz*ms22+2*S3Z/S33*e3zhz*ms33
! by x (C)
            stox=2*S1Z/S11*e1zox*ms11+2*S2Z/S22*e2zox*ms22+2*S3Z/S33*e3zox*ms33
            stoy=2*S1Z/S11*e1zoy*ms11+2*S2Z/S22*e2zoy*ms22+2*S3Z/S33*e3zoy*ms33
            stoz=2*S1Z/S11*e1zoz*ms11+2*S2Z/S22*e2zoz*ms22+2*S3Z/S33*e3zoz*ms33

         endif

         ! Ut = kt (st - st(exp))^2
         ! energy
         if(ldipc(i)) then
            !----------------
            ! DC

            if(lsofta(i)) then
               !soft asymptote
               if(nu.le.(plcs(i)-rswl(i))) then
                  bsympl=-(knew(i)-2.d0*kmincs(i)*rswl(i))/expt(i)/((-rswl(i))**(-expt(i)-1.d0))
                  asympl=kmincs(i)*(rswl(i)**2.d0)-bsympl/((-rswl(i))**expt(i))-knew(i)*rswl(i)
                  enecs=enecs+asympl+bsympl/((nu-plcs(i))**expt(i))-knew(i)*(nu-plcs(i))

               else if((nu.gt.(plcs(i)-rswl(i))).and.(nu.le.plcs(i))) then
                  enecs=enecs+kmincs(i)*(nu-plcs(i))*(nu-plcs(i))
               else if((nu.gt.plcs(i)).and.(nu.le.pucs(i))) then
                  enecs=enecs+0.d0
               else if((nu.gt.pucs(i)).and.(nu.le.(pucs(i)+rsw(i)))) then
                  enecs=enecs+kmaxcs(i)*(nu-pucs(i))*(nu-pucs(i))
               else
                  bsymp=(knew(i)-2.d0*kmaxcs(i)*rsw(i))/expt(i)/(rsw(i)**(-expt(i)-1.d0))
                  asymp=kmaxcs(i)*(rsw(i)**2.d0)-bsymp/(rsw(i)**expt(i))-knew(i)*rsw(i)
                  enecs=enecs+asymp+bsymp/((nu-pucs(i))**expt(i))+knew(i)*(nu-pucs(i))
               endif

            else

               ! 051606 dc abs on/off
               ! 050206 experimental nu always positive
               if(ldcabs) then
                  if(nu .ge. 0.d0) then
                      enecs=enecs+forcs(i)*(nu-expcs(i))*(nu-expcs(i))
                  else
                      enecs=enecs+forcs(i)*(-nu-expcs(i))*(-nu-expcs(i))
                  endif

               else
                  enecs=enecs+forcs(i)*(nu-expcs(i))*(nu-expcs(i))
               endif
            endif

         else
            !----------------
            ! CS

            if(lsofta(i)) then
               !soft asymptote
               if(st.le.(plcs(i)-rswl(i))) then
                  bsympl=-(knew(i)-2.d0*kmincs(i)*rswl(i))/expt(i)/((-rswl(i))**(-expt(i)-1.d0))
                  asympl=kmincs(i)*(rswl(i)**2.d0)-bsympl/((-rswl(i))**expt(i))-knew(i)*rswl(i)
                  enecs=enecs+asympl+bsympl/((st-plcs(i))**expt(i))-knew(i)*(st-plcs(i))

               else if((st.gt.(plcs(i)-rswl(i))).and.(st.le.plcs(i))) then
                  enecs=enecs+kmincs(i)*(st-plcs(i))*(st-plcs(i))

               else if((st.gt.plcs(i)).and.(st.le.pucs(i))) then
                  enecs=enecs+0.d0

               else if((st.gt.pucs(i)).and.(st.le.(pucs(i)+rsw(i)))) then
                  enecs=enecs+kmaxcs(i)*(st-pucs(i))*(st-pucs(i))

               else
                  bsymp=(knew(i)-2.d0*kmaxcs(i)*rsw(i))/expt(i)/(rsw(i)**(-expt(i)-1.d0))
                  asymp=kmaxcs(i)*(rsw(i)**2.d0)-bsymp/(rsw(i)**expt(i))-knew(i)*rsw(i)
                  enecs=enecs+asymp+bsymp/((st-pucs(i))**expt(i))+knew(i)*(st-pucs(i))

               endif

            else
               enecs=enecs+forcs(i)*(st-expcs(i))*(st-expcs(i))
            endif

         endif

         ! (Ut)' CS for harmonic pot.
         ! = 2 kt (st - st(exp)) (st)'
         ! force

         if(ldipc(i)) then

            if(lsofta(i)) then
               !soft asymptote force DC
               if(nu.le.(plcs(i)-rswl(i))) then
                  ! N x
                  unx=-bsympl/(nu-plcs(i))**(2.d0*expt(i)) &
                         *expt(i)*(nu-plcs(i))**(expt(i)-1.d0) &
                         *nunx-knew(i)*nunx
                  uny=-bsympl/(nu-plcs(i))**(2.d0*expt(i)) &
                         *expt(i)*(nu-plcs(i))**(expt(i)-1.d0) &
                         *nuny-knew(i)*nuny
                  unz=-bsympl/(nu-plcs(i))**(2.d0*expt(i)) &
                         *expt(i)*(nu-plcs(i))**(expt(i)-1.d0) &
                         *nunz-knew(i)*nunz

                  ! H x
                  uhx=-bsympl/(nu-plcs(i))**(2.d0*expt(i)) &
                         *expt(i)*(nu-plcs(i))**(expt(i)-1.d0) &
                         *nuhx-knew(i)*nuhx
                  uhy=-bsympl/(nu-plcs(i))**(2.d0*expt(i)) &
                         *expt(i)*(nu-plcs(i))**(expt(i)-1.d0) &
                         *nuhy-knew(i)*nuhy
                  uhz=-bsympl/(nu-plcs(i))**(2.d0*expt(i)) &
                         *expt(i)*(nu-plcs(i))**(expt(i)-1.d0) &
                         *nuhz-knew(i)*nuhz

               else if((nu.gt.(plcs(i)-rswl(i))).and.(nu.le.plcs(i))) then
                  ! N x
                  unx=2.d0*kmincs(i)*(nu-plcs(i))*nunx
                  uny=2.d0*kmincs(i)*(nu-plcs(i))*nuny
                  unz=2.d0*kmincs(i)*(nu-plcs(i))*nunz
                  ! H x
                  uhx=2.d0*kmincs(i)*(nu-plcs(i))*nuhx
                  uhy=2.d0*kmincs(i)*(nu-plcs(i))*nuhy
                  uhz=2.d0*kmincs(i)*(nu-plcs(i))*nuhz

               else if((nu.gt.plcs(i)).and.(nu.le.pucs(i))) then
                  ! N x
                  unx=0.d0
                  uny=0.d0
                  unz=0.d0
                  ! H x
                  uhx=0.d0
                  uhy=0.d0
                  uhz=0.d0

               else if((nu.gt.pucs(i)).and.(nu.le.(pucs(i)+rsw(i)))) then
                  ! N x
                  unx=2.d0*kmaxcs(i)*(nu-pucs(i))*nunx
                  uny=2.d0*kmaxcs(i)*(nu-pucs(i))*nuny
                  unz=2.d0*kmaxcs(i)*(nu-pucs(i))*nunz
                  ! H x
                  uhx=2.d0*kmaxcs(i)*(nu-pucs(i))*nuhx
                  uhy=2.d0*kmaxcs(i)*(nu-pucs(i))*nuhy
                  uhz=2.d0*kmaxcs(i)*(nu-pucs(i))*nuhz

               else
                  ! N x
                  unx=-bsymp/(nu-pucs(i))**(2.d0*expt(i)) &
                       *expt(i)*(nu-pucs(i))**(expt(i)-1.d0) &
                       *nunx+knew(i)*nunx
                  uny=-bsymp/(nu-pucs(i))**(2.d0*expt(i)) &
                       *expt(i)*(nu-pucs(i))**(expt(i)-1.d0) &
                       *nuny+knew(i)*nuny
                  unz=-bsymp/(nu-pucs(i))**(2.d0*expt(i)) &
                       *expt(i)*(nu-pucs(i))**(expt(i)-1.d0) &
                       *nunz+knew(i)*nunz
                  ! H x
                  uhx=-bsymp/(nu-pucs(i))**(2.d0*expt(i)) &
                       *expt(i)*(nu-pucs(i))**(expt(i)-1.d0) &
                       *nuhx+knew(i)*nuhx
                  uhy=-bsymp/(nu-pucs(i))**(2.d0*expt(i)) &
                       *expt(i)*(nu-pucs(i))**(expt(i)-1.d0) &
                       *nuhy+knew(i)*nuhy
                  uhz=-bsymp/(nu-pucs(i))**(2.d0*expt(i)) &
                       *expt(i)*(nu-pucs(i))**(expt(i)-1.d0) &
                       *nuhz+knew(i)*nuhz
               endif

            else

               if(ldcabs) then
                  ! 050206 experimental nu always positive
                  ! DC force constant
                  ! (U)'= 2 k (nu-nu_e)(nu)' if nu >= 0.0
                  ! (U)'= - 2 k (-nu-nu_e)(nu)' if nu <= 0.0
                  if(nu .ge. 0.d0) then
                     ! by x (N)
                     unx=2.d0*forcs(i)*(nu-expcs(i))*nunx
                     uny=2.d0*forcs(i)*(nu-expcs(i))*nuny
                     unz=2.d0*forcs(i)*(nu-expcs(i))*nunz
                     ! by x (H)
                     uhx=2.d0*forcs(i)*(nu-expcs(i))*nuhx
                     uhy=2.d0*forcs(i)*(nu-expcs(i))*nuhy
                     uhz=2.d0*forcs(i)*(nu-expcs(i))*nuhz
                  else
                     ! by x (N)
                     unx=-2.d0*forcs(i)*(-nu-expcs(i))*nunx
                     uny=-2.d0*forcs(i)*(-nu-expcs(i))*nuny
                     unz=-2.d0*forcs(i)*(-nu-expcs(i))*nunz
                     ! by x (H)
                     uhx=-2.d0*forcs(i)*(-nu-expcs(i))*nuhx
                     uhy=-2.d0*forcs(i)*(-nu-expcs(i))*nuhy
                     uhz=-2.d0*forcs(i)*(-nu-expcs(i))*nuhz
                  endif
               else
                  ! by x (N)
                  unx=2.d0*forcs(i)*(nu-expcs(i))*nunx
                  uny=2.d0*forcs(i)*(nu-expcs(i))*nuny
                  unz=2.d0*forcs(i)*(nu-expcs(i))*nunz
                  ! by x (H)
                  uhx=2.d0*forcs(i)*(nu-expcs(i))*nuhx
                  uhy=2.d0*forcs(i)*(nu-expcs(i))*nuhy
                  uhz=2.d0*forcs(i)*(nu-expcs(i))*nuhz

               endif

            endif

         else
            !----------------
            ! CS

            if(lsofta(i)) then

               !soft asymptote force
               if(st.le.(plcs(i)-rswl(i))) then

                  ! N x
                  enenx=-bsympl/(st-plcs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-plcs(i))**(expt(i)-1.d0) &
                          *stnx-knew(i)*stnx
                  eneny=-bsympl/(st-plcs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-plcs(i))**(expt(i)-1.d0) &
                          *stny-knew(i)*stny
                  enenz=-bsympl/(st-plcs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-plcs(i))**(expt(i)-1.d0) &
                          *stnz-knew(i)*stnz
                  ! H x
                  enehx=-bsympl/(st-plcs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-plcs(i))**(expt(i)-1.d0) &
                          *sthx-knew(i)*sthx
                  enehy=-bsympl/(st-plcs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-plcs(i))**(expt(i)-1.d0) &
                          *sthy-knew(i)*sthy
                  enehz=-bsympl/(st-plcs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-plcs(i))**(expt(i)-1.d0) &
                          *sthz-knew(i)*sthz
                  ! O x
                  eneox=-bsympl/(st-plcs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-plcs(i))**(expt(i)-1.d0) &
                          *stox-knew(i)*stox
                  eneoy=-bsympl/(st-plcs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-plcs(i))**(expt(i)-1.d0) &
                          *stoy-knew(i)*stoy
                  eneoz=-bsympl/(st-plcs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-plcs(i))**(expt(i)-1.d0) &
                          *stoz-knew(i)*stoz

               else if((st.gt.(plcs(i)-rswl(i))).and.(st.le.plcs(i))) then

                  ! N x
                  enenx=2.d0*kmincs(i)*(st-plcs(i))*stnx
                  eneny=2.d0*kmincs(i)*(st-plcs(i))*stny
                  enenz=2.d0*kmincs(i)*(st-plcs(i))*stnz

                  ! H x
                  enehx=2.d0*kmincs(i)*(st-plcs(i))*sthx
                  enehy=2.d0*kmincs(i)*(st-plcs(i))*sthy
                  enehz=2.d0*kmincs(i)*(st-plcs(i))*sthz

                  ! O x
                  eneox=2.d0*kmincs(i)*(st-plcs(i))*stox
                  eneoy=2.d0*kmincs(i)*(st-plcs(i))*stoy
                  eneoz=2.d0*kmincs(i)*(st-plcs(i))*stoz

               else if((st.gt.plcs(i)).and.(st.le.pucs(i))) then

                  ! N x
                  enenx=0.d0
                  eneny=0.d0
                  enenz=0.d0

                  ! H x
                  enehx=0.d0
                  enehy=0.d0
                  enehz=0.d0

                  ! O x
                  eneox=0.d0
                  eneoy=0.d0
                  eneoz=0.d0

               else if((st.gt.pucs(i)).and.(st.le.(pucs(i)+rsw(i)))) then

                  ! N x
                  enenx=2.d0*kmaxcs(i)*(st-pucs(i))*stnx
                  eneny=2.d0*kmaxcs(i)*(st-pucs(i))*stny
                  enenz=2.d0*kmaxcs(i)*(st-pucs(i))*stnz

                  ! H x
                  enehx=2.d0*kmaxcs(i)*(st-pucs(i))*sthx
                  enehy=2.d0*kmaxcs(i)*(st-pucs(i))*sthy
                  enehz=2.d0*kmaxcs(i)*(st-pucs(i))*sthz

                  ! O x
                  eneox=2.d0*kmaxcs(i)*(st-pucs(i))*stox
                  eneoy=2.d0*kmaxcs(i)*(st-pucs(i))*stoy
                  eneoz=2.d0*kmaxcs(i)*(st-pucs(i))*stoz

               else
                  ! N x
                  enenx=-bsymp/(st-pucs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-pucs(i))**(expt(i)-1.d0) &
                          *stnx+knew(i)*stnx
                  eneny=-bsymp/(st-pucs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-pucs(i))**(expt(i)-1.d0) &
                          *stny+knew(i)*stny
                  enenz=-bsymp/(st-pucs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-pucs(i))**(expt(i)-1.d0) &
                          *stnz+knew(i)*stnz

                  ! H x
                  enehx=-bsymp/(st-pucs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-pucs(i))**(expt(i)-1.d0) &
                          *sthx+knew(i)*sthx
                  enehy=-bsymp/(st-pucs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-pucs(i))**(expt(i)-1.d0) &
                          *sthy+knew(i)*sthy
                  enehz=-bsymp/(st-pucs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-pucs(i))**(expt(i)-1.d0) &
                          *sthz+knew(i)*sthz

                  ! O x
                  eneox=-bsymp/(st-pucs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-pucs(i))**(expt(i)-1.d0) &
                          *stox+knew(i)*stox
                  eneoy=-bsymp/(st-pucs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-pucs(i))**(expt(i)-1.d0) &
                          *stoy+knew(i)*stoy
                  eneoz=-bsymp/(st-pucs(i))**(2.d0*expt(i)) &
                          *expt(i)*(st-pucs(i))**(expt(i)-1.d0) &
                          *stoz+knew(i)*stoz

               endif

            else

               !harmonic force
               ! N x
               enenx=2.d0*forcs(i)*(st-expcs(i))*stnx
               eneny=2.d0*forcs(i)*(st-expcs(i))*stny
               enenz=2.d0*forcs(i)*(st-expcs(i))*stnz

               ! H x
               enehx=2.d0*forcs(i)*(st-expcs(i))*sthx
               enehy=2.d0*forcs(i)*(st-expcs(i))*sthy
               enehz=2.d0*forcs(i)*(st-expcs(i))*sthz

               ! O x
               eneox=2.d0*forcs(i)*(st-expcs(i))*stox
               eneoy=2.d0*forcs(i)*(st-expcs(i))*stoy
               eneoz=2.d0*forcs(i)*(st-expcs(i))*stoz

            endif
         endif

         if(ldipc(i)) then
            ! DC
            ! add force to original.
            ! N x
            dx(nnum)=dx(nnum)+unx
            dy(nnum)=dy(nnum)+uny
            dz(nnum)=dz(nnum)+unz
            ! H x
            dx(hnum)=dx(hnum)+uhx
            dy(hnum)=dy(hnum)+uhy
            dz(hnum)=dz(hnum)+uhz

         else
            ! CS
            ! add force to original.
            ! N x
            dx(nnum)=dx(nnum)+enenx
            dy(nnum)=dy(nnum)+eneny
            dz(nnum)=dz(nnum)+enenz

            ! H x
            dx(hnum)=dx(hnum)+enehx
            dy(hnum)=dy(hnum)+enehy
            dz(hnum)=dz(hnum)+enehz

            ! O x
            dx(onum)=dx(onum)+eneox
            dy(onum)=dy(onum)+eneoy
            dz(onum)=dz(onum)+eneoz

         endif

103  FORMAT(6X,A8,1X,I4,3X,6(1X,A),15X,3F9.3)
104  FORMAT(6X,A8,1X,I4,3X,9(1X,A),3F9.3)

         if(ldipc(i)) then
            if(lanal) then
               if(prnlev.ge.2) then
                  call atomid(nnum,sidn,ridn,renn,acn)
                  call atomid(hnum,sidh,ridh,renh,ach)
                  write(OUTU,103) "SSNMR DC",i,&
                        sidn(1:idleng),ridn(1:idleng),acn(1:idleng),&
                        sidh(1:idleng),ridh(1:idleng),ach(1:idleng),&
                        nu,expcs(i),enecs
               endif

               ! jlee 010807 add rmsd for dcabs
               if(ldcabs) then
                  sumdc=sumdc+(abs(nu)-expcs(i))*(abs(nu)-expcs(i))
               else
                  sumdc=sumdc+(nu-expcs(i))*(nu-expcs(i))
               endif
               numdc=numdc+1

            endif
         else
            if(lanal) then
               if(prnlev.ge.2) then
                  call atomid(nnum,sidn,ridn,renn,acn)
                  call atomid(hnum,sidh,ridh,renn,ach)
                  call atomid(onum,sido,rido,reno,aco)
                  write(OUTU,104) "SSNMR CS",i,&
                        sidn(1:idleng),ridn(1:idleng),acn(1:idleng),&
                        sidh(1:idleng),ridh(1:idleng),ach(1:idleng),&
                        sido(1:idleng),rido(1:idleng),aco(1:idleng),&
                        st,expcs(i),enecs
               endif

               sumcs=sumcs+(st-expcs(i))*(st-expcs(i))
               numcs=numcs+1
            endif
         endif

         if(prnlev.ge.6) then
             write(OUTU,*) 'num st energy', i, st, enecs
             write(OUTU,*) 'N H O x forces', stnx,sthx,stox
             if(ldipc(i)) write(OUTU,*) 'DC', nu, enecs
         endif

         ! initialize a and b
         asymp=0.d0
         bsymp=0.d0
         asympl=0.d0
         bsympl=0.d0

      enddo
   enddo

   en=enecs

   if(lanal) then
      ! rmsd (standard deviation)
      sumdc = sqrt(sumdc / numdc)
      sumcs = sqrt(sumcs / numcs)

      call set_param('RMSC', real(sumcs, chm_real))
      call set_param('RMSD', real(sumdc, chm_real))
      write(OUTU,*) 'rmsd for cs and dc', sumcs,sumdc
      if((sumcs.le.rmsdval).and.(sumdc.le.rmsdvdc)) then
         write(OUTU,*) 'satisfying the rmsd',rmsdval,'ppm',rmsdvdc,'kHz'
         call set_param('SRMV',1)
      else
         call set_param('SRMV',0)
      endif
   endif

return
end subroutine cscns

#endif
