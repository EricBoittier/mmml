
MODULE OpenMMMSES_Types
    implicit none

    ! Global Constants


    ! Type Declarations

    type OpenMMMSES_MSESForce
        integer*8 :: handle = 0
    end type
    integer*4, parameter :: OpenMMMSES_MSESForce_NoCutoff = 0
    integer*4, parameter :: OpenMMMSES_MSESForce_CutoffNonPeriodic = 1
    integer*4, parameter :: OpenMMMSES_MSESForce_CutoffPeriodic = 2


END MODULE OpenMMMSES_Types

MODULE OpenMMMSES
    use OpenMM_Types
    use OpenMM
    use OpenMMMSES_Types
    implicit none
    interface
        ! OpenMMMSES::MSESForce
        subroutine OpenMMMSES_MSESForce_create(result)
            use OpenMM_Types
            use OpenMM
            use OpenMMMSES_Types
            implicit none
            type (OpenMMMSES_MSESForce) result
        end subroutine
        function OpenMMMSES_MSESForce_getNumDistPair(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMMSES_Types
            implicit none
            type (OpenMMMSES_MSESForce) target
            integer*4 OpenMMMSES_MSESForce_getNumDistPair
        end function
        function OpenMMMSES_MSESForce_addDistPair(target,at1,at2,at3,at4,kc,fmax,dcut,sexp)
            use OpenMM_Types
            use OpenMM
            use OpenMMMSES_Types
            implicit none
            type (OpenMMMSES_MSESForce) target
            integer at1,at2,at3,at4
            real*8 kc,fmax,dcut,sexp
            integer*4 OpenMMMSES_MSESForce_addDistPair
        end function
        subroutine OpenMMMSES_MSESForce_setDistPairParameters(target,idx,at1,at2,at3,at4,kc,fmax,dcut,sexp)
            use OpenMM_Types
            use OpenMM
            use OpenMMMSES_Types
            implicit none
            type (OpenMMMSES_MSESForce) target
            integer idx,at1,at2,at3,at4
            real*8 kc,fmax,dcut,sexp
        end subroutine
        subroutine OpenMMMSES_MSESForce_updateParametersInContext(target, context)
            use OpenMM_Types
            use OpenMM
            use OpenMMMSES_Types
            implicit none
            type (OpenMMMSES_MSESForce) target
            type (OpenMM_Context) context
        end subroutine
        function OpenMMMSES_MSESForce_usesPeriodicBoundaryConditions(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMMSES_Types
            implicit none
            type (OpenMMMSES_MSESForce) target
            integer*4 OpenMMMSES_MSESForce_usesPeriodicBoundaryConditions
        end function
    end interface
END MODULE OpenMMMSES
