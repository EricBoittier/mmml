module kernn

    ! Some documentation


    ! Variables:
    ! qkernn, qfkernn :: Logical variables for triakern, used as flags to activate KerNN
    ! na :: number of atoms
    ! nr :: number of interatomic distance
    ! n_input :: number of nodes in the input layer (can be equal to na, or be larger as in a permut. invariant case)
    ! n_hidden :: number of hidden layers
    ! n_width :: number of nodes in the hidden layers.
    ! w_hidden :: weights of the hidden layers
    ! w_in, :: weights of the input layer
    ! b_out :: bias of the output layer
    ! meanE, stdE :: mean and standard deviation of the training energies
    ! b_hidden :: biases of all layers but the output layer
    ! w_out :: weights of the output layer
    ! meank, stdk :: mean and standard deviation of the 1D Kernels
    ! kernel :: 1D kernel vector that is used as descriptor
    ! derkernel :: derivative of the 1D kernels 

    ! Note: currently the system size and the NN architecture is hard-coded. Will be generalized in the next version


    use chm_kinds

    implicit none

    logical, save :: qkernn, qfkernn
    integer, parameter :: na=4, nr=na*(na-1)/2, n_input=6, n_hidden=2, nn_width=20
    real(chm_real), dimension(nn_width, nn_width, n_hidden) :: w_hidden 
    real(chm_real) :: b_out, meanE, stdE
    real(chm_real), dimension((n_hidden+1), nn_width) :: b_hidden
    real(chm_real), dimension(n_input, nn_width) :: w_in
    real(chm_real), dimension(nn_width):: w_out
    real(chm_real), dimension(n_input):: meank, stdk, minr, kernel, derkernel

    contains

    function softplus(i, nn_width) result(j)
        ! Softplus functtion
        ! Activation function that is used in KerNN

        ! i :: input vector to the activation function (is applied entry wise)
        ! nn_width :: dimension of the in put vector
        ! j : output of the function
        implicit none
        integer, intent(in) :: nn_width ! just to allocate the correct dimension
        real(chm_real), dimension(nn_width), intent(in)  :: i ! input
        real(chm_real), dimension(nn_width) :: j ! output      
        j = log(exp(i) + 1.0 )
    end function
    
    function softplus_prime(i, nn_width) result(j)
        ! Derivative of the softplus functtion
        ! Required for the back-prop to determine the forces

        ! i :: input vector to the activation function (is applied entry wise)
        ! nn_width :: dimension of the in put vector
        ! j : output of the function
        implicit none
        integer, intent(in) :: nn_width ! just to allocate the correct dimension
        real(chm_real), dimension(nn_width), intent(in)  :: i ! input
        real(chm_real), dimension(nn_width) :: j ! output      
        j = exp(i) / (exp(i) + 1)
    end function
    
    function drker33(x,xi) 
        ! One-dimensional reciprocal power reproducing kernels
        ! Used as a descriptor of the query molecule

        ! x :: interatomic distances of the query molecule
        ! xi :: interatomic distances of a reference molecule (here: minimum energy configuration)
        implicit none
        real(chm_real), intent(in) :: x, xi
        real(chm_real) :: drker33, xl, xs

        xl = x
        xs = xi
        if (x .le. xi) then
            xl = xi
            xs = x
        end if

        drker33=3d0/(20d0*xl**4) - 6d0/35d0 * xs/xl**5 + 3d0/56d0 * xs**2/xl**6

    end function drker33

    function dkdrker33(x,xi)
        ! Derivatives of the one-dimensional reciprocal power reproducing kernels
        ! Required for the back-prop to determine the forces

        ! x :: interatomic distances of the query molecule
        ! xi :: interatomic distances of a reference molecule (here: minimum energy configuration)
        implicit none
        real(chm_real), intent(in) :: x, xi
        real(chm_real) :: dkdrker33

        if (x .le. xi) then
            dkdrker33 = 3.0d0/28.0d0 * x/xi**6 - 6.0d0/(35.0d0*xi**5)
        else
            dkdrker33 = -3.0d0/(5.0d0*x**5)  + 6.0d0/7.0d0 *xi/x**6 - 9.0d0/28.0d0*xi**2/x**7
        end if

    end function dkdrker33
    
   
   
    subroutine kernn_set(comlyn,comlen)
        ! This subroutine reads the input command
        ! Currently it only sets the appropriate flags
        !(qkernn and qfkernn) if the ACTI or CLEA
        ! keywords are used and reads all the weights
        ! and biases of the neural network
        ! 

        ! Can be used to parse more options in future versions.
        use string

        implicit none


        character(len=6)::symbol
        logical :: storedsk = .false.
        integer :: i, ii, jj

        character(len=4) :: wrd, word
        integer :: inuni, idx, ic
        character(len=*), intent(in) :: comlyn
        integer, intent(in) :: comlen
        ! Variables
        ! wrd, word :: Character, defines if we are modifying or adding bonds.
        ! ic :: Integer number corresponding to capital or small letter

        if(.not.qkernn) qkernn = .true.


        wrd = nexta4(comlyn, comlen)

        ! Convert the character of word in small letters
        word = wrd
        do idx = 1, 4
            ic = ichar(word(idx:idx))
            if (ic >= 65 .and. ic < 91) word(idx:idx) = char(ic + 32)
        end do
        wrd = word

        select case(wrd)
            case('acti') ! Modifying existing bond/angle
                if(.not.qfkernn) qfkernn = .true.
            case ('clea')
                qkernn = .false.
                qfkernn = .false.
            case default  ! in case of unrecognized variable
            call wrndie(-3,'<triakern>','Unrecognized option.')
        end select
      




        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !Read all neural network parameters
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (.not. storedsk ) then
       
            open(11, file="wandb/w_in") !corresponds to the weight of first layer
            read(11,*) w_in
            !write(*,*) w_in
        
            open(12, file="wandb/b_hidden") !corresponds to the biases of all but the output layer
            do jj = 1, (n_hidden+1)
                do ii = 1,nn_width
                    read(12,*) b_hidden(jj, ii)
                end do
            end do
        
            open(13, file="wandb/w_hidden")
            !read weights (total of nhidden blocks)
            read(13,*) w_hidden
            !write(*,*) shape(w_hidden)

            open(14, file="wandb/w_out")
            !read weights of the output layer
            read(14,*) w_out
            !write(*,*) w_out

        
            open(15, file="wandb/b_out") !corresponds to the bias of the last dense layer with no activation
            read(15,*) b_out
            !write(*,*) b_out
        
            open(16, file="wandb/meanE") 
            read(16,*) meanE
            !write(*,*) meanE
        
            open(17, file="wandb/stdE")
            read(17,*) stdE
            !write(*,*) stdE    

            open(18, file="wandb/meank") 
            read(18,*) meank
            !write(*,*) meank
        

            open(19, file="wandb/stdk")
            read(19,*) stdk
            !write(*,*) stdk   


            open(20, file="wandb/minr")
            read(20,*) minr
            !write(*,*) minr   
        
        
            close(11)
            close(12)
            close(13)
            close(14)
            close(15)
            close(16)
            close(17)
            close(18)
            close(19)

            storedsk = .true.

        end if


        return

    end subroutine kernn_set
   
   
   
    subroutine kernn_ener(ebond,x,y,z,dx,dy,dz)
        ! kernn_ener is the core of KerNN. It calculates energies/forces and
        ! modifies the CHARMM contributions.
        
    
        ! Variables
        ! ebond     :: CHARMM bond energy
        ! x,y,z     :: Atomic coordinates
        ! dx,dy,dz  :: Atomic potential Derivatives
        ! ekernn    :: Energy contribution of the Kernels
        ! h1_in, h1_out :: Results from the first layer, before and after activation function
        ! h2_in, h2_out :: Results from the second layer, before and after activation function
        ! h3_in, h3_out :: Results from the third layer, before and after activation function
        ! pos0 :: Positions of the atoms, used as helper variable
        ! forces :: Forces of the atoms, used again as helper variable
        ! r0 :: Interatomic distances of the query molecule
        ! dEdk :: Derivative of the energy with respect to the 1D kernels
        ! dEdr :: Derivative of the energy with respect to the interatomic distances


        use code
        use number
        use parallel
        use psf
        use string
      
        implicit none

        real(chm_real) :: ebond
        real(chm_real) :: x(*), y(*), z(*), dx(*), dy(*), dz(*)

        real(chm_real) :: ekernn

        integer :: i, ii, jj

        real(chm_real), dimension(nn_width):: h1_in, h1_out, h2_in, h2_out, h3_in, h3_out
        real(chm_real), dimension(na,3):: forces, pos0
        real(chm_real), dimension(nr) :: r0
	real(chm_real), dimension(nr) :: dEdk, dEdr
	    


        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !Calculate the quantities that are required for prediction
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        !calculate interatomic distances
        r0(1) = sqrt((x(1)-x(2))**2 + (y(1)-y(2))**2 + (z(1)-z(2))**2)
        r0(2) = sqrt((x(1)-x(3))**2 + (y(1)-y(3))**2 + (z(1)-z(3))**2)
        r0(3) = sqrt((x(1)-x(4))**2 + (y(1)-y(4))**2 + (z(1)-z(4))**2)
        r0(4) = sqrt((x(2)-x(3))**2 + (y(2)-y(3))**2 + (z(2)-z(3))**2)
        r0(5) = sqrt((x(2)-x(4))**2 + (y(2)-y(4))**2 + (z(2)-z(4))**2)
        r0(6) = sqrt((x(3)-x(4))**2 + (y(3)-y(4))**2 + (z(3)-z(4))**2)

        !calculate 1d kernels for each of the interatomic distances
        do ii = 1, nr
            kernel(ii)=drker33(r0(ii),minr(ii))
        end do 

        !calculate derivative of kernels for each of the interatomic distances
        do ii = 1, nr
            derkernel(ii)=dkdrker33(r0(ii),minr(ii))
        end do 

        !normalize k
        kernel = (kernel - meank) / stdk

        !normalize derk
        derkernel = derkernel / stdk
        

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !Do the forward-pass
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	    !Input layer
	    h1_in = matmul(kernel, w_in) + b_hidden(1,:)
	    h1_out = softplus(h1_in, nn_width)

	    !Hidden layers
	    h2_in = matmul(h1_out,w_hidden(:,:,1))  + b_hidden(2,:)
	    h2_out = softplus(h2_in, nn_width)
	    
	    h3_in = matmul(h2_out,w_hidden(:,:,2))  + b_hidden(3,:)
	    h3_out = softplus(h3_in, nn_width)
	    
	    !Output layer
	    ekernn = (dot_product(h3_out, w_out) + b_out)* stdE + meanE


        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !Do the back-pass
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    !get gradients with respect to input r
	    dEdk =  -matmul(w_in, matmul(w_hidden(:, :, 1), matmul(w_hidden(:, :, 2), (w_out * softplus_prime(h3_in, nn_width))) &
	    * softplus_prime(h2_in, nn_width)) * softplus_prime(h1_in, nn_width)) * stdE
	    

	    !transform to from dE/dk to dE/dr 
        do ii = 1, nr
            dEdr(ii) = derkernel(ii)*dEdk(ii)
        end do 



        !just to make calculation easier I create a position array
        do ii = 1, na
            pos0(ii,1) = x(ii)
            pos0(ii,2) = y(ii)
            pos0(ii,3) = z(ii)
        end do 


        !calculate Cartesian forces
        !ATTENTION forces(1,:) correspond to forces acting on atom 1  
            
        forces(1,:) = (pos0(1,:) - pos0(2,:))/r0(1)*dEdr(1)+ (pos0(1,:) - pos0(3,:))/r0(2)*dEdr(2) + (pos0(1,:) &
                - pos0(4,:))/r0(3)*dEdr(3)
        forces(2,:) = (pos0(2,:) - pos0(3,:))/r0(4)*dEdr(4)+ (pos0(2,:) - pos0(4,:))/r0(5)*dEdr(5) + (pos0(2,:) &
                - pos0(1,:))/r0(1)*dEdr(1)
        forces(3,:) = (pos0(3,:) - pos0(4,:))/r0(6)*dEdr(6)+ (pos0(3,:) - pos0(1,:))/r0(2)*dEdr(2) + (pos0(3,:) &
                - pos0(2,:))/r0(4)*dEdr(4)
        forces(4,:) = (pos0(4,:) - pos0(1,:))/r0(3)*dEdr(3)+ (pos0(4,:) - pos0(2,:))/r0(5)*dEdr(5) + (pos0(4,:) &
                - pos0(3,:))/r0(6)*dEdr(6)


        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !Update the CHARMM contributions
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        !unit conversion - standard for KerNN potentials is ev, ev/angstrom, angstrom
        ekernn = ekernn * 23.0605419d0
        forces(:,:) = -1 * forces(:,:)*23.0605419d0

        !Add NN energy to charmm energy
        ebond = ebond + ekernn
    
        !Add forces to charmm forces
        do ii = 1, na
            dx(ii) = dx(ii) + forces(ii, 1)
            dy(ii) = dy(ii) + forces(ii, 2)
            dz(ii) = dz(ii) + forces(ii, 3)
        end do 
        
        
        
   end subroutine kernn_ener

end module kernn
