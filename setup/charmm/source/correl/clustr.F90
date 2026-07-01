subroutine clustr(iserie, veccod, maxser, maxtot,  &
     tq, nserie, sname, iclass, stot)
  !-----------------------------------------------------------------------
  !     This routine interprets the CLUSTER command and allocates
  !     memory for the routine art2p.
  !
  !     Mary E. Karpen, October 2, 1991
  !     incorporated into CHARMM 23f3 January 21, 1994
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number

  use comand
  use stream
  use string

  implicit none

  integer iserie, veccod(*)
  integer maxser, maxtot
  real(chm_real)  tq(maxtot,1)
  integer nserie
  character(len=*) sname(*), iclass(*)
  integer stot(*)
  !
  !     local variables
  integer maxcluster, maxiteration
  integer unicst, unimember, uniinit
  integer nfeature, cstep
  real(chm_real)  radius, maxerror
  logical angle
  integer mintot, npattern
  integer tstop, tbegin
  integer lastserie
  integer ncluster
  integer i

  ! begin
  maxcluster=gtrmi(comlyn,comlen,'MAXC',50)
  maxiteration=gtrmi(comlyn,comlen,'MAXI',20)
  nfeature = gtrmi(comlyn,comlen,'NFEA',veccod(iserie))
  cstep=gtrmi(comlyn,comlen,'CSTE',nfeature)
  unicst=gtrmi(comlyn,comlen,'UNIC',-1)
  unimember=gtrmi(comlyn,comlen,'UNIM',-1)
  uniinit=gtrmi(comlyn,comlen,'UNII',-1)
  radius=gtrmf(comlyn,comlen,'RADI',ZERO)
  maxerror=gtrmf(comlyn,comlen,'MAXE',PT001)
  angle = .false.
  if (indxa(comlyn,comlen,'ANGL') .gt. 0) angle = .true.

  lastserie = ((veccod(iserie) - nfeature)/cstep)*cstep + &
       iserie + nfeature - 1
  if (lastserie .gt. nserie) then
     call wrndie(0, '<CLUSTR>',  &
          'More time series specified than available.')
     if (prnlev .ge. 2) write(outu,'(a)') &
          ' Maximum series available used.'
     lastserie = nserie
  endif

  if (lastserie - iserie + 1 .lt. nfeature) then
     call wrndie(0, '<CLUSTR>', &
          'More features specified than time series available.')
     nfeature = lastserie - iserie + 1
     if (prnlev .ge. 2) write(outu,510) nfeature
  endif
510 format(' Number of features changed to', i8)

  mintot = stot(iserie)
  do i = iserie + 1, lastserie
     mintot = min(mintot, stot(i))
  enddo

  tbegin=gtrmi(comlyn,comlen,'BEGI',1)
  tstop=gtrmi(comlyn,comlen,'STOP',mintot)

  if (tstop .gt. mintot) then
     call wrndie(0, '<CLUSTR>', 'Too many timesteps specified.')
     tstop = mintot
     if (prnlev .ge. 2) write(outu,520) tstop
  endif
520 format(' Number of timesteps changed to ', i8)

  npattern = 1 + int((lastserie - iserie - nfeature + 1)/cstep)

  call art2p(iserie, lastserie, tq, maxtot, tbegin, tstop,  &
       nserie, sname,  &
       cstep, nfeature, angle, unicst, unimember, uniinit,  &
       radius, maxiteration, maxerror, maxcluster,  &
       ncluster, npattern)

  return
end subroutine clustr

subroutine art2p(iserie, lastserie, tq, maxtot, tbegin, &
     tstop, nserie, sname, &
     cstep, nfeature, angle, unicst, unimember, uniinit,  &
     radius, maxiteration, maxerror, maxcluster,  &
     ncluster, npattern)
  !-----------------------------------------------------------------------
  !     Program Name: art2p.for
  !     Author:       Mary E. Karpen
  !     Date:         13-Jun-89
  !
  !     Program Description
  !
  !     This program is a version of the program ART2P that has been
  !     modified for use as a subroutine in the molecular mechanics
  !     program, CHARMM.
  !
  !     The program implements the Adaptive Resonance Theory (ART2)
  !     algorithm, which creates a self-organizing net with each output
  !     node representing a cluster center (or prototype).  The number
  !     of pattern features is equal to the number of input nodes.  The
  !     weights of the connections between the input layer (layer i) and
  !     the output layer (layer j) are denoted by b(j,i).  To create the
  !     net (which is synonomous to learning the classification scheme
  !     or cluster centers) the following algorithm is implemented:
  !
  !     1. Initialize the network: 1) assign b(1,i) equal to the first
  !        pattern tq(1,i) for i = 1, nfeature, or 2) read in a set of
  !        initial clusters.
  !
  !     2. For each pattern number k, calculate the Euclidean distance (rms)
  !        between the pattern tq(k,i) and all prototypes b(j,i), where j
  !        is the cluster index.
  !
  !        rms(j,k) = sum [(b(j,i)-tq(k,i))**2] for i = 1, nfeature
  !
  !     3. Find cluster j such that rms(j,k) <= rms(n,k) for all n.  If
  !        rms(j,k) <= Threshold, then update b(j,i):
  !
  !        b(j,i) = ((m-1)*b(j,i) + tq(k,i))/m,
  !
  !        where m is the number of prior updates of b(j,i).  Note that
  !        b(j,i) is the average of feature i for all patterns within
  !        cluster j.
  !
  !     4. If rms > Threshold for all existing cluster centers,
  !        then create a new cluster center by incrementing the number of
  !        output nodes by one (increment ncluster by one), and assign
  !        the weights b(ncluster,i) of this node the value of the
  !        pattern tq(k,i).
  !
  !     5. Repeat 2.-4. until all patterns have been input.
  !
  !     6. Compare the new set of prototypes with the last set of prototypes.
  !        If the difference between them is less than MAXERROR, halt
  !        clustering.
  !
  !     7. If the difference between the sets of prototypes is greater than
  !        MAXERROR, then use the new set of prototypes as the starting
  !        cluster centers, and repeat steps 2.-6.  Note that except for the
  !        first iteration with no initial cluster centers (where b(j,i,1) is
  !        used throughout), the current set of cluster centers (b(j,i,2))
  !        is used in step 2, and a new set of cluster centers (b(j,i,1))
  !        is created in step 3. In step 6, each b(j,i,1) is compared with
  !        b(j,i,2).
  !
  !     This program uses patterns formed from CHARMM time series data.
  !
  !   Input Required
  !
  !     The following information must be included in the CLUSTER command
  !     line:
  !
  !     1. radius:    Maximum radius of cluster.  The rms
  !          cutoff or threshold for assignment to a cluster is
  !          equal the square of Radius.
  !
  !     2. nfeature:  This variable gives the number of features in
  !          the input pattern, that is, the number of time series
  !          to be clustered at a time.  The default is the number
  !          of time series, nserie.
  !
  !     3. cstep:     The spacing between time series in the input vector.
  !          For each timepoint k, the set of patterns clustered is
  !          tq(k,1) -> tq(k,nfeature),
  !          tq(k,1 + cstep) -> tq(k,nfeature + cstep),...,
  !          tq(k,nserie - nfeature + 1) -> tq(k,nserie).
  !          The default is nfeature.
  !
  !     4. uniinit:   The unit number of the file with the initial
  !          cluster centers.  If uniinit = -1 (the default),
  !          no initial cluster centers are specified.
  !
  !     5. unicst:    The unit number of the output cluster file.  If
  !          unicst = -1 (the default), the cluster parameters are
  !          output to the standard output.
  !
  !     6. unimember: The unit number of the output membership file.
  !          This file lists each time point and the cluster(s)
  !          associated with the specified time series at that time
  !          point.  The cluster is assigned to the first time series
  !          in the pattern vector.
  !
  !     7. maxiteration: The maximum number of iterations allowed.  If
  !          the clustering has not converged by this number of
  !          iterations, all clusters are output. (default = 20)
  !
  !     8. maxerror: If the rms difference between the positions of the
  !          cluster centers for the last two iterations is less than
  !          maxerror, the system is considered converged and thus the
  !          clustering is halted.
  !
  !     9. maxcluster:  Maximum number of clusters.
  !
  !     10. angle:  A logical flag which when true specified angle data
  !           is to be clustered. (default = .FALSE.)
  !
  !     11. begin:  Starting frame of time series to be clustered.
  !
  !     12. stop:  Ending frame of time series to be clustered.
  !
  !   Usage:
  !       CORREL ...
  !       ENTE ...
  !       TRAJ ...
  !       CLUSter seriesname MAXCluster <int> [MAXIteration <int>] -
  !           MAXError <real> RADIus <real> NFEAtures <int> -
  !           UNICluster <int> UNIMember <int> [UNIInitial <int>] -
  !           [CSTEP <int>] [BEGIn <int>] [STOP <int>] [ANGLE]
  !       END
  !
  !   Routines Called
  !       readcst
  !       ismin
  !-----------------------------------------------------------------------
  !
  use chm_kinds
  use number
  use stream
  implicit none
  !
  !     passed variables
  integer maxtot
  real(chm_real)  tq(maxtot,nserie)
  logical angle
  integer nserie
  integer cstep
  real(chm_real) radius
  character(len=*) sname(*)
  integer iserie, lastserie
  integer nfeature
  integer tbegin,tstop
  integer unicst, unimember, uniinit
  real(chm_real) maxerror
  integer maxiteration
  integer maxcluster
  integer ncluster
  integer npattern
  ! Automatically allocated arrays
  real(chm_real) b(maxcluster, nfeature, 2)
  integer nmember(maxcluster)
  integer memberindex(npattern, tbegin:tstop)
  integer cummember(maxcluster)
  real(chm_real) std(maxcluster), maxdist(maxcluster)
  real(chm_real) sum(maxcluster)
  !
  !     local variables
  real(chm_real) sqmaxerror
  integer i, j, jj, k
  integer ifeature, ipattern, jserie, icluster
  integer totalpattern
  integer numclusters(2), compare
  integer bestfit
  integer iteration
  integer ismin

  integer,parameter :: maxfeature = 20
  real(chm_real) threshold
  real(chm_real) rms, diff
  real(chm_real) totalerror

  logical error, qprint
  external ismin
  external readcst
  !
  !     begin
  !
  !     INITIALIZE PARAMETERS
5 continue
  do j = 1, maxcluster
     nmember(j) = 0
  enddo

  do i = 1, nfeature
     do j = 1, maxcluster
        b(j,i,1) = ZERO
        b(j,i,2) = ZERO
     enddo
  enddo

  compare = 1
  numclusters(1) = 1
  iteration = 0

  threshold = (radius**2)*nfeature
  sqmaxerror = (maxerror**2)*nfeature

  if (unicst .eq. -1) unicst = outu
  !
  !     INITIALIZE NET
  !
  if (uniinit .ge. 0) then
     !
     !     Initial Weights Specified:
     ifeature = nfeature
     icluster = maxcluster
     call readcst (uniinit, icluster, ifeature, b, error)

     if (error) then
        call wrndie(0, '<ART2P>',  &
             'Error reading initial cluster file.  Clustering aborted.')
        return
     endif

     numclusters(2) = icluster
     numclusters(1) = numclusters(2)
     compare = 2

  else
     !     No Initial Clusters Were Specified:
     do i = 1, nfeature
        b(1,i,1) = tq(1,i + iserie - 1)
     enddo
     compare = 1
     numclusters(2) = 1

  endif
  !
  !     LOCATE INVALID PATTERNS
  !
  totalpattern = 0

  do k = tbegin, tstop
     ipattern = 0
     do jserie = iserie - 1, lastserie - nfeature, cstep
        ipattern = ipattern + 1
        memberindex(ipattern,k) = 1
        do ifeature = 1, nfeature
           if (tq(k,jserie + ifeature) .eq. ANUM)  &
                memberindex(ipattern,k) = 0
        enddo
        totalpattern = totalpattern + memberindex(ipattern,k)
     enddo
  enddo
  !
  !     SORT PATTERNS INTO CLUSTERS
  !
80 continue                  !Each iteration starts here

  do k = tbegin, tstop
     ipattern = 0
     do jserie = iserie - 1, lastserie - nfeature, cstep
        ipattern = ipattern + 1
        !     Skip past invalid conformations
        if (memberindex(ipattern,k) .eq. 0) cycle
        !
        !     Compare pattern to each cluster center:
        !     A record is kept of the previously updated set of cluster
        !     centers(b(j,i,2)) and the set currently being updated as each
        !     pattern is assigned (b(j,i,1)).  During the first iteration, if
        !     a set of initial clusters have not been specified, the patterns
        !     are compared to the cluster centers currently being updated
        !     (compare = 1).  In all subsequent iterations, or if initial
        !     clustercenters have been specified (which are assigned to
        !     b(j,i,2)), the patterns are compared to the previously updated
        !     clusters.
        !
        do j = 1, numclusters(compare)
           sum(j) = ZERO
        enddo
        !
        !     The following loop calculates the Euclidean distance between the
        !     pattern and each cluster center.
        !
        !     Correct for angle periodicity...
        if (angle) then
           do i = 1, nfeature
              do j = 1, numclusters(compare)
                 diff = abs(tq(k,jserie + i) &
                      - b(j,i,compare))
                 sum(j) = sum(j) +  &
                      min(diff, thr6ty - diff)**2
                 !     (thr6ty = 360, defined in number.f90)
              enddo
           enddo
        else
           do i = 1, nfeature
              do j = 1, numclusters(compare)
                 sum(j) = sum(j) +  &
                      (tq(k,jserie + i) - b(j,i,compare))**2
              enddo
           enddo
        endif
        !
        !     Find the cluster center most similar to the pattern.
        !
        bestfit = ismin(numclusters(compare),sum)
        !
        !     Assign the pattern to the cluster with the minimum rms, and update
        !     cluster center.
        !
        if (sum(bestfit) .le. threshold) then
           !     Pattern fits cluster...
           !
           nmember(bestfit) = nmember(bestfit) + 1

           if (angle) then
              do i = 1, nfeature !Update b
                 sum(i) = (tq(k,jserie + i) - b(bestfit,i,1))
                 b(bestfit,i,1) =  &
                      b(bestfit,i,1) + &
                      (mod(sum(i) + 540., thr6ty) - 180.)/ &
                      nmember(bestfit)
                 b(bestfit,i,1) =  &
                      mod(b(bestfit,i,1) + 540., thr6ty) &
                      - 180.
              enddo

           else
              do i = 1, nfeature !Update b
                 sum(i) = (tq(k,jserie + i) - b(bestfit,i,1))
                 b(bestfit,i,1) =  &
                      b(bestfit,i,1) + sum(i)/ &
                      nmember(bestfit)
              enddo
           endif
           memberindex(ipattern,k) = bestfit
           !
           !     The pattern can not be assigned (rms > threshold for all clusters).
           !     Therefore, create a new cluster and assign the pattern as its
           !     cluster center.
           !
        else
           numclusters(1) = numclusters(1) + 1
           if (numclusters(1) .gt. maxcluster) &
                call wrndie(-2, '<CLUSTR>',  &
                'Number of clusters exceeds MAXCluster.')

           do i = 1, nfeature
              b(numclusters(1),i,1) = tq(k,jserie + i)
           enddo
           nmember(numclusters(1)) = 1
           memberindex(ipattern,k) = numclusters(1)

        endif

     enddo
  enddo
  !
  !     CHECK FOR CONVERGENCE OF CLUSTERING
  !
  totalerror = 0
  loop160: do j = 1, min(numclusters(1), numclusters(2))
     !
     !     Calculate Euclidean distance.
     rms = ZERO

     if (angle) then
        do i = 1, nfeature
           rms = rms + min(abs(b(j,i,1) - b(j,i,2)),  &
                thr6ty - abs(b(j,i,1) - b(j,i,2)))**2
        enddo
     else
        do i = 1, nfeature
           rms = rms + (b(j,i,1) - b(j,i,2))**2
        enddo
     endif
     !
     !     Compare the previous set of clusters with the updated set.  When
     !     the difference between all pairs of clusters is less than
     !     MAXERROR, clustering is completed.
     !     Convergence?
     if (rms .gt. sqmaxerror) then
        !     No!
        iteration = iteration + 1
        if (iteration .gt. maxiteration) then
           if (prnlev .ge. 2) write(outu,620) maxiteration
           exit loop160
        endif

        if (prnlev .ge. 2) write(outu,610) iteration, &
             j, numclusters(1)
        !
        !     Store updated cluster centers and clear 'learning' cluster centers
        do jj = 1, numclusters(1)
           nmember(jj) = 0
        enddo

        do i = 1, nfeature
           do jj = 1, numclusters(1)
              b(jj,i,2) = b(jj,i,1)
              b(jj,i,1) = 0
           enddo
        enddo

        numclusters(2) = numclusters(1)
        compare = 2
        goto 80            !Repeat update

     else
        totalerror = totalerror + rms
     endif

  enddo loop160
  !
  !     CONVERGENCE or MAXITERATIONs
  !
  totalerror = sqrt(totalerror/(nfeature*numclusters(1)))
  ncluster = numclusters(1)
  if (prnlev .ge. 2 .and. unicst .ne. outu)  &
       write(outu,640) iteration, ncluster, totalerror
  !
  !-----------------------------------------------------------------------
  !
  !     COMPILE CLUSTER STATISTICS & WRITE OUTPUT FILES
  !
  !     Write membership file header
  qprint = ((unimember == outu .and. prnlev >= 2) .or. iolev > 0)
  if (unimember .ge. 0 .and. qprint)  &
       write(unimember,800) tbegin, tstop, radius, totalpattern, &
       ncluster
  !
  !     The total number of patterns within RADIUS of each
  !     cluster center j is stored in cummember(j), for j = 1, ncluster.
  !     The standard deviation of patterns assigned to a cluster
  !     from the cluster center are stored in std(j).  The largest distance
  !     of a member from the cluster center is stored in maxdist(j).
  !
  do j = 1, ncluster
     cummember(j) = 0
     std(j) = ZERO
     maxdist(j) = ZERO
  enddo

  do k = tbegin, tstop
     do jserie = iserie - 1, lastserie - nfeature, cstep
        do j = 1, ncluster
           sum(j) = ZERO
        enddo

        if (angle) then
           do i = 1, nfeature
              do j = 1, ncluster
                 diff = abs(tq(k,jserie + i) - b(j,i,1))
                 sum(j) = sum(j) + min(diff, thr6ty - diff)**2
              enddo
           enddo
        else
           do i = 1, nfeature
              do j = 1, ncluster
                 sum(j) = sum(j) + (tq(k,jserie + i) - b(j,i,1))**2
              enddo
           enddo
        endif

        bestfit = ismin(ncluster,sum)
        do j = 1, ncluster
           if (sum(j) .le. threshold) &
                cummember(j) = cummember(j) + 1
        enddo

        maxdist(bestfit) = max(maxdist(bestfit),sum(bestfit))
        std(bestfit) = std(bestfit) + sum(bestfit)
        if (unimember .ge. 0 .and. qprint) &
             write(unimember,830) bestfit, k,  &
             jserie + 1, sqrt(sum(bestfit)/nfeature)

     enddo
  enddo

  do j = 1, ncluster
     std(j) = sqrt(std(j)/(nfeature*nmember(j)))
  enddo

  if (unimember.gt.0 .and. unimember.ne.outu) close(unimember)
  !
  !     Write cluster file
  !
  qprint = ((unicst == outu .and. prnlev >= 2) .or. iolev > 0)
  if(qprint) then
     write(unicst,630) iteration, ncluster, totalerror
     if (iteration .gt. maxiteration)  &
          write(unicst,650) maxiteration
     write(unicst,660) radius
     write(unicst,670) nfeature
     write(unicst,680) totalpattern
     write(unicst,690) ncluster
     write(unicst,700) lastserie - iserie + 1, &
          (j, sname(j)(1:idleng), j = iserie, lastserie)
     write(unicst,710)
     do j = 1, ncluster
        write(unicst,720) j, &
             nmember(j), &
             cummember(j), &
             std(j), sqrt(maxdist(j)/nfeature), &
             (b(j,i,1), i = 1, nfeature)
     enddo
     if (unicst.ne.outu) close(unicst)
  endif
  !
  !-----------------------------------------------------------------------
  !
610 format(' CLUSTER> Iteration =',i4,' MAXError exceeded', &
       ' for cluster',i7,'  No. of clusters =',i7)
620 format(' CLUSTER> Greater than',i4, &
       ' iterations, program halted.')
630 format(' Cluster File'// &
       ' Total Iterations =',i5,'  No. of clusters =',i7/ &
       ' RMS difference in cluster centers from last iteration =', &
       e11.3/)
640 format(/' CLUSTER> Total Iterations =',i5, &
       '  No. of clusters =',i8/ &
       ' RMS difference in cluster centers from last iteration =', &
       e11.3/)
650 format(' GREATER THAN',i4,' ITERATIONS, program halted.')
660 format(/' Max. Cluster Radius  =',e11.4)
670 format(' Number of Features   =',i8)
680 format(' Number of Patterns   =',i8)
690 format(' Number of Clusters   =',i8)
700 format(' Number of Series     =',i8/('   Series #',i8,3x,a))
710 format(/11x,'No._of_Members   Member  Maximum'/ &
       ' Cluster   Final   Cumul     Std    ', &
       'Distance     Cluster Center Pattern')
720 format(3i8,2e10.3,(t47,9f9.3))
800 format(' Cluster Membership File'// &
       ' Time Series Frames Clustered:',i8,',',i8/ &
       ' Maximum Cluster Radius      :',e12.4/ &
       ' Number of Patterns Clustered:',i8/ &
       ' Number of Clusters          :',i8// &
       ' Cluster   Frame 1stSeries  Distance')
830 format(3i8,e12.4)

  return
end subroutine art2p

subroutine readcst(unit, ncluster, nfeature, x, error)
  !-----------------------------------------------------------------------
  !     Program Name: READCST.FOR
  !     Author:       M. E. Karpen
  !     Date:         08-Jan-88
  !
  !   Purpose
  !
  !     This subroutine reads in cluster data created by the subroutine
  !     ART2P.
  !-----------------------------------------------------------------------
  use chm_kinds
  use number
  use stream
  implicit none
  !
  !     passed variables
  integer unit
  integer ncluster, nfeature
  real(chm_real) x(ncluster,nfeature,2)
  logical error
  !
  !     local variables
  integer i, j
  integer icluster, ifeature
  integer nmember
  integer totalpattern
  integer numseries
  character(len=8) seriesname
  character(len=130) rec
  character(len=4) start
  data start/' Max'/
  !
  !     begin
  error = .false.
  !
  !     Read header
10 continue
  read(unit,200,end = 1000) rec
  if (rec(1:4) .ne. start) goto 10

  read(unit,210,end = 1000) ifeature
  read(unit,210,end = 1000) totalpattern
  read(unit,210,end = 1000) icluster
  read(unit,210,end = 1000) numseries
  do i = 1, numseries
     read(unit,220,end = 1000) seriesname
  enddo
220 format(22x,a)

  if (prnlev .ge. 2) write(outu,500) icluster
500 format(/' CLUSTER> Number of clusters in initial file =', i8)
  if (icluster .gt. ncluster) then
     call wrndie(0, '<READCST>',  &
          'This exceeds the maximum specified.')
     icluster = ncluster
     if (prnlev .ge. 2) write(outu,510) ncluster
510  format(' Reading only first',i8,' clusters.')
  endif
  ncluster = icluster

  if (prnlev .ge. 2) write(outu,520) ifeature
520 format(' CLUSTER> Number of features per cluster in initial', &
       ' file = ',i8/)
  if (ifeature .gt. nfeature) then
     call wrndie(0, '<READCST>',  &
          'This exceeds the number of time series specified')
     ifeature = nfeature
     if (prnlev .ge. 2) write(outu,530) nfeature
530  format(' Only first',i8,' features will be read')
  elseif (nfeature .gt. ifeature) then
     call wrndie(-1, '<READCST>',  &
          'This is less than the number of time series specified')

     do i = ifeature + 1, nfeature
        do j = 1, ncluster
           x(j,i,2) = ZERO
        enddo
     enddo
  endif

  read(unit,230,end = 1000)
  !
  !     Read data
  do j = 1, ncluster
     read(unit,240,end = 1000) nmember, &
          (x(j,i,2),i = 1, ifeature)
  enddo

  ifeature = nfeature

  close(unit)
  return

200 format(a)
210 format(23x,i8)
230 format(//)
240 format(8x,i8,28x,(t47,9f9.3))

1000 call wrndie(0, '<READCST>', 'Error on read')
  error = .true.
  close(unit)
  return

end subroutine readcst

integer function ismin(n, x)
  !-----------------------------------------------------------------------
  use chm_kinds
  use number
  implicit none

  integer n
  real(chm_real) x(n)
  real(chm_real) xmin
  integer i

  ismin = -1
  xmin = RBIGST
  do i = 1, n
     if (x(i) .lt. xmin) then
        xmin = x(i)
        ismin = i
     endif
  enddo

  return
END function ismin

