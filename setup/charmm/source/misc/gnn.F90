module gnn

  use chm_kinds
  implicit none
#if KEY_GNN==1 /*gnn_fcm*/
  integer,parameter :: maxdes = 4, maxhidden = 10, maxout = 5
  integer,save :: nrun, idpara(maxdes)
  real(chm_real),save :: minex
  real(chm_real),save :: w10(maxhidden, maxdes + 1),  &
       w21(maxout, maxhidden + 1),  &
       dw10(maxhidden, maxdes + 1), dw21(maxout, maxhidden + 1), &
       del1(maxhidden), del2(maxout), &
       net1(maxhidden), net2(maxout), &
       out1(maxhidden), out2(maxout)

  real(chm_real),save,allocatable,dimension(:) :: moderr,para,rpara, &
       targ, rtarg, pred
  integer,save,allocatable,dimension(:) :: model

#endif /* (gnn_fcm)*/


  !---------------------------------------
Contains
  !---------------------------------------


#if KEY_GNN==0 /*gnn*/
  subroutine gnncall(comlyn, comlen)
    integer comlen
    character(len=*) comlyn
    CALL WRNDIE(0,'<GNNCALL>','GNN code not compiled')
    return
  end subroutine gnncall
#else /* (gnn)*/
  subroutine gnncall(comlyn, comlen)
    !       Parsing the gnn command
    !       Jie Hu, 2007-12-28
    use number
    use stream
    use string
    use chm_kinds

    implicit none
    integer comlen
    character(len=*) comlyn
    integer ndata, nprod, npara, ndes, nhidden, ntarg
    integer nsweep, npopu, ngen
    integer punit, seed
    integer nmodel
    real(chm_real) mu, eta, fitness
    logical exhaust, gfa
    ndata = gtrmi(comlyn, comlen, 'NDAT', 1)
    nprod = gtrmi(comlyn, comlen, 'NPRO', 0)
    npara = gtrmi(comlyn, comlen, 'NPAR', 1)
    ndes = gtrmi(comlyn, comlen, 'NDES', 1)
    nhidden = gtrmi(comlyn, comlen, 'NHID', 2)
    ntarg = gtrmi(comlyn, comlen, 'NTAR', 1)
    nsweep = gtrmi(comlyn, comlen, 'NSWE', 100)
    npopu = gtrmi(comlyn, comlen, 'NPOP', 500)
    ngen = gtrmi(comlyn, comlen, 'NGEN', 200)
    punit = gtrmi(comlyn, comlen, 'UNIT', -1)
    seed  = gtrmi(comlyn,comlen,'SEED', 123)
    mu = gtrmf(comlyn, comlen, 'MU', half)
    eta = gtrmf(comlyn, comlen, 'ETA', half)
    fitness = gtrmf(comlyn, comlen, 'FITN', five)
    exhaust = .true.
    gfa = .false.
    if (indxa(comlyn, comlen, 'GFA') .gt. 0) then
       if (ndes .gt. 1) gfa = .true.
       exhaust = .false.
    endif
    if (indxa(comlyn, comlen, 'EP') .gt. 0) exhaust = .false.
    if (indxa(comlyn, comlen, 'EXHA') .gt. 0) exhaust = .true.
    if (ndes .gt. maxdes .or. nhidden .gt. maxhidden .or. &
         ntarg .gt. maxout) then
       write(outu,'(a)') ' GNN> Too complex networks, cannot handle!'
       return
    endif
    if (punit .lt. 1) then
       write(outu,'(a)') ' GNN> Input data file required!'
       return
    endif
    call gnnread(ndata + nprod, npara, ntarg, punit)
    nrun = 0
    if (exhaust) then
       call gnnexha(ndata, nprod, npara, ndes, nhidden, ntarg, &
            nsweep, nmodel, seed, mu, eta)
    else
       call gnnctrl(ndata, nprod, npara, ndes, nhidden, ntarg, &
            nsweep, npopu, ngen, nmodel, seed, mu, eta, &
            fitness, gfa)
    endif
    call gnnstat(ndata, nprod, npara, ndes, ntarg, nmodel)

    if(allocated(para)) deallocate(para)
    if(allocated(targ)) deallocate(targ)
    if(allocated(rpara)) deallocate(rpara)
    if(allocated(rtarg)) deallocate(rtarg)
    if(allocated(pred)) deallocate(pred)


    if (exhaust) then
       if(allocated(model)) deallocate(model)
       if ( allocated(moderr)) deallocate(moderr)
    else
       if(allocated(model)) deallocate(model)
       if ( allocated(moderr)) deallocate(moderr)
    endif
    return
  end subroutine gnncall


  subroutine gnnread(ndata, npara, ntarg, punit)
    !       Reading parameter time-series
    !       Jie Hu, 2007-12-28
    use chm_kinds
    use stream
    implicit none
    integer ndata, npara, ntarg, punit
    integer,parameter :: maxp = 10000
    integer id, ip, dim1
    real(chm_real) v, vmin, vmax, scratch(maxp)

    dim1=ndata*npara
    allocate(para(dim1))
    allocate(rpara(dim1))

    dim1=ndata*ntarg
    allocate(targ(dim1))
    allocate(rtarg(dim1))
    allocate(pred(dim1))
    do id = 1, ndata
       read(punit,*) (scratch(ip), ip = 1, npara + ntarg)
       do ip = 1, npara
          rpara((ip - 1) * ndata + id) = scratch(ip)
       enddo
       do ip = 1, ntarg
          rtarg((ip - 1) * ndata + id) = scratch(npara + ip)
       enddo
    enddo
    do ip = 1, npara
       vmax = rpara( (ip - 1) * ndata + 1)
       vmin = vmax
       do id = 1, ndata
          v = rpara( (ip - 1) * ndata + id)
          if (v .gt. vmax) vmax = v
          if (v .lt. vmin) vmin = v
       enddo
       do id = 1, ndata
          para((ip - 1) * ndata + id) = (rpara((ip - 1) * ndata + id) - vmin) &
               * 0.8 / (vmax - vmin) + 0.1
       enddo
    enddo
    do ip = 1, ntarg
       vmax = rtarg((ip - 1) * ndata + 1)
       vmin = vmax
       do id = 1, ndata
          v = rtarg( (ip - 1) * ndata + id)
          if (v .gt. vmax) vmax = v
          if (v .lt. vmin) vmin = v
       enddo
       do id = 1, ndata
          targ((ip - 1) * ndata + id) = (rtarg((ip - 1) * ndata + id) - vmin) &
               * 0.8 / (vmax - vmin) + 0.1
       enddo
    enddo
    return
  end  subroutine gnnread


  subroutine gnnexha(ndata, nprod, npara, ndes, nhidden, ntarg, &
       nsweep, nmodel, seed, mu, eta)
    !       Exhaustive enumeration
    !       Jie Hu, 2007-12-28
    use chm_kinds
    use stream
    implicit none
    integer ndata, nprod, npara, ndes, nhidden, ntarg, nsweep
    integer nmodel
    integer seed
    real(chm_real) mu, eta
    integer id, n,dim1
    logical enumerated

    nmodel = bico(npara, ndes)

    dim1=nmodel*ndes
    allocate(model(dim1))
    allocate(moderr(nmodel))
    n = 0
    enumerated = .false.
    do id = 1, ndes
       idpara(id) = id
    enddo
    do
       call gnnexharun(ndata, nprod, ndes, nhidden, ntarg, nsweep, n, &
            seed, mu, eta)
       id = 1
       do
          if (id .eq. 1 .and. idpara(id) .ge. (npara - ndes + 1)) then
             enumerated = .true.
          else if (id .eq. ndes) then
             idpara(id) = idpara(id) + 1
          else if (idpara(id + 1) .ge. (npara - ndes + id + 1)) then
             idpara(id) = idpara(id) + 1
             call enumerone(ndes, id)
             id = ndes
          endif
          id = id + 1
          if (id .gt. ndes .and. valid(npara, ndes) .or. enumerated) &
               exit
       enddo
       if (enumerated) exit
    enddo
    call gnnrank(ndes, n)
    if (nmodel .ne. n) &
         write(outu,'(a)') ' GNN> Errors in exhaustive enumeration!'
    return
  end  subroutine gnnexha


  subroutine enumerone(ndes, id)
    !       Enumerating one
    !       Jie Hu, 2007-12-28
    use chm_kinds
    implicit none
    integer ndes, id, ia
    do ia = id + 1, ndes
       idpara(ia) = idpara(ia - 1) + 1
    enddo
    return
  end subroutine enumerone


  logical function valid(npara, ndes)
    !       Testing if the enumerated model is a valid one
    !       Jie Hu, 2007-12-28
    use chm_kinds
    implicit none
    integer npara, ndes, id
    valid = .true.
    do id = 1, ndes - 1
       if (idpara(id) .ge. idpara(id + 1)) valid = .false.
    enddo
    if (idpara(ndes) .gt. npara) valid = .false.
    return
  end function valid


  subroutine gnnctrl(ndata, nprod, npara, ndes, nhidden, ntarg, &
       nsweep, npopu, ngen, nmodel, seed, mu, eta, &
       fitness, gfa)
    !       Genetic algorithms
    !       Jie Hu, 2007-12-28
    use chm_kinds
    use number
    use stream
    implicit none
    integer ndata, nprod, npara, ndes, nhidden, ntarg, nsweep
    integer npopu, ngen
    integer nmodel
    integer seed
    real(chm_real) mu, eta, fitness
    logical gfa
    integer ip, ig, dim1
    real(chm_real) a
    nmodel = bico(npara, ndes)
    if (nmodel .lt. 2 * npopu) then
       npopu = nmodel / 2
       write(outu,'(a)') ' GNN> Wrong number of models in the pool!'
       write(outu,'(a,i5)') ' GNN> Will use npopu = ', npopu
    endif
    nmodel = npopu

    dim1 = 2 * npopu * ndes
    allocate(model(dim1))
    allocate(moderr(nmodel))
    call gnngenpopu(npara, ndes, npopu, seed)
    call gnnpopurun(ndata, nprod, ndes, nhidden, ntarg, nsweep, 1, &
         npopu, seed, mu, eta)
    call gnnrank(ndes, npopu)
    ig = 1
    do
       a = zero
       do ip = 1, npopu
          a = a + one / moderr(ip)
       enddo
       a = a / dble(npopu)
       if (ig .ge. ngen .or. a .gt. fitness) exit
       if (gfa) then
          call gnngfa(npara, ndes, npopu, &
               seed)
          call gnnpopurun(ndata, nprod, ndes, nhidden, ntarg, nsweep, &
               2, npopu, seed, mu, eta)
          call gnnrank(ndes, npopu)
       else
          call gnnep(npara, ndes, npopu, seed)
          call gnnpopurun(ndata, nprod, ndes, nhidden, ntarg, nsweep, &
               npopu + 1, 2 * npopu, seed, mu, &
               eta)
          call gnnrank(ndes, 2 * npopu)
       endif
       ig = ig + 1
    enddo
    return
  end  subroutine gnnctrl


  subroutine gnngenpopu(npara, ndes, npopu, seed)
    !       Generating a starting population
    !       Jie Hu, 2007-12-28
    use chm_kinds
    implicit none
    integer npara, ndes, npopu, seed
    integer id, ip
    ip = 1
    do
       call ranone(npara, ndes, seed)
       if (valid(npara, ndes) .and. .not. same(ndes, ip - 1)) &
            then
          do id = 1, ndes
             model((ip - 1) * ndes + id) = idpara(id)
          enddo
          ip = ip + 1
       endif
       if (ip .gt. npopu) exit
    enddo
    return
  end  subroutine gnngenpopu


  subroutine ranone(npara, ndes, seed)
    !       Generating one randomly
    !       Jie Hu, 2007-12-28
    use clcg_mod,only:random
    use chm_kinds
    implicit none
    integer npara, ndes, seed
    integer id, ia, ib, n
    logical existing
    do id = 1, ndes
       do
          existing = .false.
          n = int(npara * random(seed)) + 1
          do ia = 1, id - 1
             if (n .eq. idpara(ia)) existing = .true.
          enddo
          if (.not. existing) exit
       enddo
       idpara(id) = n
       do ia = 1, id - 1
          if (n .lt. idpara(ia)) then
             do ib = id - 1, ia, -1
                idpara(ib + 1) = idpara(ib)
             enddo
             idpara(ia) = n
             exit
          endif
       enddo
    enddo
    return
  end  subroutine ranone


  logical function same(ndes, n)
    !       Testing if the model exists
    !       Jie Hu, 2007-12-28
    use chm_kinds
    implicit none
    integer ndes, n
    integer id, io
    logical existing
    do io = 1, n
       existing = .true.
       do id = 1, ndes
          if (model((io - 1) * ndes + id) .ne. idpara(id)) &
               existing = .false.
       enddo
       if (existing) then
          same = .true.
          return
       endif
    enddo
    same = .false.
    return
  end function same


  subroutine gnnrank(ndes, n)
    !       Ranking the n individuals
    !       Jie Hu, 2007-12-28
    use chm_kinds
    implicit none
    integer ndes, n
    integer id, io, ip, s
    real(chm_real) r
    do io = 1, n - 1
       do ip = io + 1, n
          if (moderr(ip) < moderr(io)) then
             r = moderr(ip)
             moderr(ip) = moderr(io)
             moderr(io) = r
             do id = 1, ndes
                s = model((ip - 1) * ndes + id)
                model((ip - 1) * ndes + id) = model((io - 1) * ndes + &
                     id)
                model((io - 1) * ndes + id) = s
             enddo
          endif
       enddo
    enddo
    return
  end  subroutine gnnrank


  subroutine gnngfa(npara, ndes, npopu, seed)
    !       Genetic function approximation
    !       Jie Hu, 2007-12-28
    use clcg_mod,only:random
    use chm_kinds
    use number
    use stream
    implicit none
    integer npara, ndes, npopu, seed
    integer id, ip
    integer,allocatable,dimension(:) :: na, nb
    real(chm_real) a

    allocate(na(npopu-1),nb(npopu-1))
    call pickpar(npopu, seed, na, nb)
    do ip = 1, npopu - 1
       do
          do id = 1, ndes
             a = random(seed)
             if (a .lt. half) then
                idpara(id) = model(na(ip) * ndes + id)
             else
                idpara(id) = model(nb(ip) * ndes + id)
             endif
          enddo
          call gnnmutate(npara, ndes, seed)
          if (valid(npara, ndes) .and. .not. same(ndes, npopu + ip &
               - 1)) exit
       enddo
       do id = 1, ndes
          model((npopu + ip - 1) * ndes + id) = idpara(id)
       enddo
    enddo
    do ip = 1, npopu - 1
       do id = 1, ndes
          model(ip * ndes + id) = model((npopu + ip - 1) * ndes + id)
       enddo
    enddo

    deallocate(na,nb)
    return
  end  subroutine gnngfa


  subroutine gnnep(npara, ndes, npopu, seed)
    !       Evolutionary programming
    !       Jie Hu, 2007-12-28
    use chm_kinds
    implicit none
    integer npara, ndes, npopu, seed
    integer id, ip
    do ip = 1, npopu
       do
          do id = 1, ndes
             idpara(id) = model((ip - 1) * ndes + id)
          enddo
          call gnnmutate(npara, ndes, seed)
          if (valid(npara, ndes) .and. .not. same(ndes, npopu + ip &
               - 1)) exit
       enddo
       do id = 1, ndes
          model((npopu + ip - 1) * ndes + id) = idpara(id)
       enddo
    enddo
    return
  end subroutine gnnep


  subroutine pickpar(npopu, seed, na, nb)
    !       Picking a parent for reproduction
    !       Jie Hu, 2007-12-28
    use clcg_mod,only:random
    use chm_kinds
    use memory
    use number
    implicit none
    integer npopu, seed, na(*), nb(*)
    integer io, ip, n, ia, ib
    integer,allocatable,dimension(:) :: np
    real(chm_real),allocatable,dimension(:) :: r
    real(chm_real) ave, a, atot, sa, sb
    logical apicked, bpicked, elite
    call chmalloc('gnn.src','pickpar','np',npopu,intg=np)
    call chmalloc('gnn.src','pickpar','r',npopu,crl=r)
    atot = zero
    do ip = 1, npopu
       r(ip) = one / moderr(ip)
       atot = atot + r(ip)
    enddo
    ave = atot / dble(npopu)
    do ip = 1, npopu
       if (r(ip) .lt. ave) exit
    enddo
    n = ip - 1
    do
       np(1:npopu) = 0
       do io = 1, npopu - 1
          do
             sa = atot * random(seed)
             sb = atot * random(seed)
             apicked = .false.
             bpicked = .false.
             a = zero
             do ip = 1, npopu
                a = a + r(ip)
                if (sa < a .and. .not. apicked) then
                   ia = ip
                   apicked = .true.
                endif
                if (sb < a .and. .not. bpicked) then
                   ib = ip
                   bpicked = .true.
                endif
             enddo
             if (ia .ne. ib) exit
          enddo
          na(io) = ia
          nb(io) = ib
          np(ia) = np(ia)+1
          np(ib) = np(ib)+1
       enddo
       elite = .true.
       do ip = 1, n
          if (np(ip) == 0) elite = .false.
       enddo
       if (elite) exit
    enddo
    call chmdealloc('gnn.src','pickpar','np',npopu,intg=np)
    call chmdealloc('gnn.src','pickpar','r',npopu,crl=r)
    return
  end  subroutine pickpar

  subroutine gnnmutate(npara, ndes, seed)
    !       Mutating a single gene
    !       Jie Hu, 2007-12-28
    use clcg_mod,only:random
    use chm_kinds
    implicit none
    integer npara, ndes, seed
    integer id, ia, ib, n
    logical existing
    id = int(ndes * random(seed)) + 1
    do
       existing = .false.
       n = int(npara * random(seed)) + 1
       do ia = 1, ndes
          if (n .eq. idpara(ia)) existing = .true.
       enddo
       if (.not. existing) exit
    enddo
    do ia = id + 1, ndes
       idpara(ia - 1) = idpara(ia)
    enddo
    idpara(ndes) = n
    do ia = 1, ndes - 1
       if (n .lt. idpara(ia)) then
          do ib = ndes - 1, ia, -1
             idpara(ib + 1) = idpara(ib)
          enddo
          idpara(ia) = n
          exit
       endif
    enddo
    return
  end  subroutine gnnmutate


  subroutine gnnexharun(ndata, nprod, ndes, nhidden, ntarg, nsweep, &
       n, seed, mu, eta)
    !       Running neural networks with exhaustive enumeration
    !       Jie Hu, 2007-12-28
    use chm_kinds
    implicit none
    integer ndata, nprod, ndes, nhidden, ntarg, nsweep
    integer n, seed
    real(chm_real) mu, eta
    integer id
    real(chm_real) error
    call gnnrun(ndata, nprod, ndes, nhidden, ntarg, nsweep, seed, &
         mu, eta, error)
    n = n + 1
    do id = 1, ndes
       model((n - 1) * ndes + id) = idpara(id)
    enddo
    moderr(n) = error
    return
  end subroutine gnnexharun


  subroutine gnnpopurun(ndata, nprod, ndes, nhidden, ntarg, nsweep, &
       n, m, seed, mu, eta)
    !       Running neural networks with a population of n individuals
    !       Jie Hu, 2007-12-28
    use chm_kinds
    implicit none
    integer ndata, nprod, ndes, nhidden, ntarg, nsweep
    integer n, m, seed
    real(chm_real) mu, eta
    integer id, ip
    real(chm_real) error
    do ip = n, m
       do id = 1, ndes
          idpara(id) = model((ip - 1) * ndes + id)
       enddo
       call gnnrun(ndata, nprod, ndes, nhidden, ntarg, nsweep, seed, &
            mu, eta, error)
       moderr(ip) = error
    enddo
    return
  end  subroutine gnnpopurun


  subroutine gnnrun(ndata, nprod, ndes, nhidden, ntarg, nsweep, &
       seed, mu, eta, error)
    !       Running neural networks
    !       Jie Hu, 2007-12-28
    use clcg_mod,only:random
    use chm_kinds
    use memory
    use number
    implicit none
    integer ndata, nprod, ndes, nhidden, ntarg, nsweep
    integer seed
    real(chm_real) mu, eta, error
    integer id, ip, irun, io
    real(chm_real),allocatable,dimension(:) :: scratch
    real(chm_real) r
    nrun = nrun + 1
    error = zero
    call chmalloc('gnn.src','gnnrun','scratch',(ndata+nprod)*ntarg,crl=scratch)
    if (nprod .le. 0) then
       do id = 1, ndata
          call gnninit(ndes, nhidden, ntarg, seed)
          do io = 1, nsweep * ndata
             irun = int(ndata * random(seed)) + 1
             if (irun .ne. id) then
                call gnnfire(ndata, ndes, nhidden, ntarg, irun)
                call gnncorr(ndata, ndes, nhidden, ntarg, irun, mu, &
                     eta)
             endif
          enddo
          call gnnfire(ndata, ndes, nhidden, ntarg, id)
          do ip = 1, ntarg
             r = targ((ip - 1) * ndata + id) - out2(ip)
             error = error + r * r
             scratch((ip - 1) * ndata + id) = out2(ip)
          enddo
       enddo
       error = dsqrt(error / dble(ndata))
    else
       call gnninit(ndes, nhidden, ntarg, seed)
       do io = 1, nsweep * ndata
          irun = int(ndata * random(seed)) + 1
          call gnnfire(ndata + nprod, ndes, nhidden, ntarg, irun)
          call gnncorr(ndata + nprod, ndes, nhidden, ntarg, irun, &
               mu, eta)
       enddo
       do id = ndata + 1, ndata + nprod
          call gnnfire(ndata + nprod, ndes, nhidden, ntarg, id)
          do ip = 1, ntarg
             r = targ((ip - 1) * (ndata + nprod) + id) - out2(ip)
             error = error + r * r
             scratch((ip-1)*(ndata+nprod) + id) = out2(ip)
          enddo
       enddo
       error = dsqrt(error / dble(nprod))
    endif
    if (nrun .eq. 1 .or. error .lt. minex) then
       minex = error
       do irun = 1, (ndata + nprod) * ntarg
          pred(irun) = scratch(irun)
       enddo
    endif
    call chmdealloc('gnn.src','gnnrun','scratch',(ndata+nprod)*ntarg,crl=scratch)
    return
  end  subroutine gnnrun


  subroutine gnninit(ndes, nhidden, ntarg, seed)
    !       Initialization
    !       Jie Hu, 2007-12-28
    use clcg_mod,only:random
    use chm_kinds
    use number
    implicit none
    integer ndes, nhidden, ntarg
    integer seed
    integer id, io, ip
    do io = 1, nhidden
       do id = 1, ndes + 1
          w10(io, id) = (one / dble(nhidden * (ndes + 1))) * &
               (two * random(seed) - one)
          dw10(io, id) = zero
       enddo
       del1(io) = zero
    enddo
    do ip = 1, ntarg
       do io = 1, nhidden + 1
          w21(ip, io) = (one / dble(ntarg * (nhidden + 1))) * &
               (two * random(seed) - one)
          dw21(ip, io) = zero
       enddo
       del2(ip) = zero
    enddo
    return
  end subroutine gnninit


  subroutine gnnfire(ndata, ndes, nhidden, ntarg, irun)
    !       Propagation through the network
    !       Jie Hu, 2007-12-28
    use chm_kinds
    use number
    implicit none
    integer ndata, ndes, nhidden, ntarg
    integer irun
    integer id, io, ip
    do io = 1, nhidden
       net1(io) = zero
       do id = 1, ndes
          net1(io) = net1(io) + w10(io, id) * &
               para((idpara(id) - 1) * ndata + irun)
       enddo
       net1(io) = net1(io) + w10(io, ndes + 1)
       out1(io) = sigmod(net1(io))
    enddo
    do ip = 1, ntarg
       net2(ip) = zero
       do io = 1, nhidden
          net2(ip) = net2(ip) + w21(ip, io) * out1(io)
       enddo
       net2(ip) = net2(ip) + w21(ip, nhidden + 1)
       out2(ip) = sigmod(net2(ip))
    enddo
    return
  end  subroutine gnnfire


  function sigmod(net) result(sigmod1)
    !       Sigmoidal transfer function
    !       Jie Hu, 2007-12-28
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: sigmod1
    real(chm_real) net
    sigmod1 = one / (one + dexp(-net))
    return
  end  function sigmod


  subroutine gnncorr(ndata, ndes, nhidden, ntarg, irun, mu, eta)
    !       Correcting the weights
    !       Jie Hu, 2007-12-28
    use chm_kinds
    use number
    implicit none
    integer ndata, ndes, nhidden, ntarg
    integer irun
    real(chm_real) mu, eta
    integer id, io, ip
    do ip = 1, ntarg
       del2(ip) = (targ((ip - 1) * ndata + irun) - out2(ip)) * &
            out2(ip) * (one - out2(ip))
       do io = 1, nhidden
          dw21(ip, io) = eta * del2(ip) * out1(io) + &
               mu * dw21(ip, io)
       enddo
       dw21(ip, nhidden + 1) = eta * del2(ip) + &
            mu * dw21(ip, nhidden + 1)
       do io = 1, nhidden + 1
          w21(ip, io) = w21(ip, io) + dw21(ip, io)
       enddo
    enddo
    do io = 1, nhidden
       del1(io) = zero
       do ip = 1, ntarg
          del1(io) = del1(io) + del2(ip) * w21(ip, io)
       enddo
       del1(io) = del1(io) * out1(io) * (one - out1(io))
       do id = 1, ndes
          dw10(io, id) = eta * del1(io) * &
               para((idpara(id) - 1) * ndata + irun) + &
               mu * dw10(io, id)
       enddo
       dw10(io, ndes + 1) = eta * del1(io) + &
            mu * dw10(io, ndes + 1)
       do id = 1, ndes + 1
          w10(io, id) = w10(io, id) + dw10(io, id)
       enddo
    enddo
    return
  end  subroutine gnncorr


  subroutine gnnstat(ndata, nprod, npara, ndes, ntarg, nmodel)
    !       Statistics
    !       Jie Hu, 2007-12-28
    use chm_kinds
    use number
    use stream
    implicit none
    integer ndata, nprod, npara, ndes, ntarg, nmodel
    integer id, io, ip, na, nb
    if (dabs(minex - moderr(1)) .gt. rsmall) then
       write(outu,'(a)') ' GNN> Oh my god, bug somewhere...'
    endif
    write(outu,'(a,i5)') ' GNN> Number of Example = ', ndata
    if (nprod .gt. 0) &
         write(outu, '(a,i5)') ' GNN> Number of Tested Example = ', &
         nprod
    write(outu,'(a,i5)') ' GNN> Number of Input = ', npara
    write(outu,'(a,i5)') ' GNN> Descriptor Used = ', ndes
    write(outu,'(a,i5)') ' GNN> Number of Output = ', ntarg
    write(outu,'(a,i5)') ' GNN> Models Generated = ', nmodel
    write(outu,'(a,i5)') ' GNN> Neural Networks Run = ', nrun
    do io = 1, nmodel
       write(outu,'(a,i5,a,f9.4,a,4i5)') ' Model: ', &
            io, ' Error ', moderr(io), ' Variable ', (model((io - 1) * &
            ndes + id), id = 1, ndes)
    enddo
    write(outu,'(a,5(a,a))') '     ', ('    P    ', &
         '    E    ', ip = 1, ntarg)
    if (nprod .gt. 0) then
       na = ndata + 1
       nb = ndata + nprod
    else
       na = 1
       nb = ndata
    endif
    do io = na, nb
       write(outu,'(i5,5(2(f9.4)))') io, &
            (pred((ip - 1) * (ndata + nprod) + io), targ((ip - 1) * &
            (ndata + nprod) + io), ip = 1, ntarg)
    enddo
    write(outu,'(a,f9.4)') ' RMS = ', minex
    return
  end subroutine gnnstat


  integer function bico(n, k)
    !       Binomical coefficient
    !       Jie Hu, 2007-12-28
    use chm_kinds
    implicit none
    integer k, n

    bico = nint(dexp(factor(n) - factor(k) - factor(n - k)))
    return
  end function bico


  function factor(n) result(factor1)
    !       Factorial
    !       Jie Hu, 2007-12-28
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: factor1
    integer n, i
    factor1 = zero
    do i = 2, n
       factor1 = factor1 + dlog(dble(i))
    enddo
    return
  end  function factor
#endif /* (gnn)*/
end module gnn

