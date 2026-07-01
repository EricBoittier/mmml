
MODULE OpenMM_Charmm_Types
    implicit none
    ! Type Declarations
 
    type OpenMM_MonteCarloBarostat2
        integer*8 :: handle = 0
    end type
 
END MODULE OpenMM_Charmm_Types

MODULE OpenMM_Charmm
    use OpenMM_Charmm_Types
    implicit none

    interface
 
 

        ! OpenMM::MonteCarloBarostat2
        subroutine OpenMM_MonteCarloBarostat2_create(result, defaultPressure, temperature, pressure3D, surfaceTension, frequency)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) result
            real*8 defaultPressure
            real*8 temperature
            real*8 pressure3D(3)
            real*8 surfaceTension
            integer*4 frequency
        end subroutine
        subroutine OpenMM_MonteCarloBarostat2_destroy(destroy)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) destroy
        end subroutine
        subroutine OpenMM_MonteCarloBarostat2_Pressure(result)
            use OpenMM_Charmm_Types; implicit none
            character(*) result
        end subroutine
        function OpenMM_MonteCarloBarostat2_getDefaultPressure(target)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            real*8  OpenMM_MonteCarloBarostat2_getDefaultPressure
        end function
        function OpenMM_MonteCarloBarostat2_getFrequency(target)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            integer*4  OpenMM_MonteCarloBarostat2_getFrequency
        end function
        subroutine OpenMM_MonteCarloBarostat2_setFrequency(target, &
                          freq)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            integer*4 freq
        end subroutine
        function OpenMM_MonteCarloBarostat2_getTemperature(target)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            real*8  OpenMM_MonteCarloBarostat2_getTemperature
        end function
        subroutine OpenMM_MonteCarloBarostat2_setTemperature(target, &
                          temp)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            real*8 temp
        end subroutine
        function OpenMM_MonteCarloBarostat2_getRandomNumberSeed(target)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            integer*4  OpenMM_MonteCarloBarostat2_getRandomNumberSeed
        end function
        subroutine OpenMM_MonteCarloBarostat2_setPressureIn3Dimensions(target, &
                          pressures)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            real*8 pressures(3)
        end subroutine
        subroutine OpenMM_MonteCarloBarostat2_getPressureIn3Dimensions(target, &
                          result)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            real*8 result(3)
        end subroutine
        subroutine OpenMM_MonteCarloBarostat2_setSurfaceTension(target, &
                          tension)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            real*8 tension
        end subroutine
        function OpenMM_MonteCarloBarostat2_getSurfaceTension(target)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            real*8  OpenMM_MonteCarloBarostat2_getSurfaceTension
        end function
        subroutine OpenMM_MonteCarloBarostat2_setRandomNumberSeed(target, &
                          seed)
            use OpenMM_Charmm_Types; implicit none
            type (OpenMM_MonteCarloBarostat2) target
            integer*4 seed
        end subroutine
    end interface
END MODULE OpenMM_Charmm

