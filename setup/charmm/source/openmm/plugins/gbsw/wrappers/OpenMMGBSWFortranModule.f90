
MODULE OpenMMGBSW_Types
    implicit none

    ! Global Constants


    ! Type Declarations

    type OpenMMGBSW_GBSWForce
        integer*8 :: handle = 0
    end type
    integer*4, parameter :: OpenMMGBSW_GBSWForce_NoCutoff = 0
    integer*4, parameter :: OpenMMGBSW_GBSWForce_CutoffNonPeriodic = 1
    integer*4, parameter :: OpenMMGBSW_GBSWForce_CutoffPeriodic = 2


END MODULE OpenMMGBSW_Types

MODULE OpenMMGBSW
    use OpenMM_Types
    use OpenMM
    use OpenMMGBSW_Types
    implicit none
    interface

        

        ! OpenMMGBSW::GBSWForce
        subroutine OpenMMGBSW_GBSWForce_create(result)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) result
        end subroutine
        subroutine OpenMMGBSW_GBSWForce_destroy(destroy)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) destroy
        end subroutine

        subroutine OpenMMGBSW_GBSWForce_setDoingDynamics(target, context, flag)
            use OpenMM_Types
            use OpenMMGBSW_Types

            implicit none

            type (OpenMMGBSW_GBSWForce) :: target
            type (OpenMM_Context) :: context
            integer*4 :: flag
        end subroutine

        subroutine OpenMMGBSW_GBSWForce_addCPHMDForce(target, pH, &
T_theta, &
mTheta, &
ts_theta, &
beta, &
outFreq, &
fileName)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 pH
            real*8 T_theta
            real*8 mTheta
            real*8 ts_theta
            real*8 beta
            integer*4 outFreq
            character(*) fileName
        end subroutine
        function OpenMMGBSW_GBSWForce_usingCPHMD(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 OpenMMGBSW_GBSWForce_usingCPHMD
        end function
        function OpenMMGBSW_GBSWForce_getNumTitratingGroups(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 OpenMMGBSW_GBSWForce_getNumTitratingGroups
        end function
        function OpenMMGBSW_GBSWForce_addTitratingGroupParameters(target, &
resPKA1, resPKA2, &
barrier1, barrier2, &
a0, a1, a2, a3, a4, a5, a6, a7, a8)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 resPKA1
            real*8 resPKA2
            real*8 barrier1
            real*8 barrier2
            real*8 a0
            real*8 a1
            real*8 a2
            real*8 a3
            real*8 a4
            real*8 a5
            real*8 a6
            real*8 a7
            real*8 a8
            integer*4 OpenMMGBSW_GBSWForce_addTitratingGroupParameters
        end function
        subroutine OpenMMGBSW_GBSWForce_getTitratingGroupParameters(target, index, &
resPKA1, &
resPKA2, &
barrier1, &
barrier2, &
a0, &
a1, &
a2, &
a3, &
a4, &
a5, &
a6, &
a7, a8)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 index
            real*8 resPKA1
            real*8 resPKA2
            real*8 barrier1
            real*8 barrier2
            real*8 a0
            real*8 a1
            real*8 a2
            real*8 a3
            real*8 a4
            real*8 a5
            real*8 a6
            real*8 a7
            real*8 a8
        end subroutine
        subroutine OpenMMGBSW_GBSWForce_setTitratingGroupParameters(target, index, &
resPKA1, &
resPKA2, &
barrier1, &
barrier2, &
a0, &
a1, &
a2, &
a3, &
a4, &
a5, &
a6, &
a7, a8)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 index
            real*8 resPKA1
            real*8 resPKA2
            real*8 barrier1
            real*8 barrier2
            real*8 a0
            real*8 a1
            real*8 a2
            real*8 a3
            real*8 a4
            real*8 a5
            real*8 a6
            real*8 a7
            real*8 a8
        end subroutine
        function OpenMMGBSW_GBSWForce_addNonbondedException(target, atom1, &
atom2)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 atom1
            integer*4 atom2
            integer*4 OpenMMGBSW_GBSWForce_addNonbondedException
        end function
        subroutine OpenMMGBSW_GBSWForce_getNonbondedException(target, index, &
atom1, &
atom2)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 index
            integer*4 atom1
            integer*4 atom2
        end subroutine
        function OpenMMGBSW_GBSWForce_getNumNonbondedExceptions(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 OpenMMGBSW_GBSWForce_getNumNonbondedExceptions
        end function
        function OpenMMGBSW_GBSWForce_getNumParticles(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 OpenMMGBSW_GBSWForce_getNumParticles
        end function
        function OpenMMGBSW_GBSWForce_addParticle(target, charge, &
radius)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 charge
            real*8 radius
            integer*4 OpenMMGBSW_GBSWForce_addParticle
        end function
        subroutine OpenMMGBSW_GBSWForce_getParticleParameters(target, index, &
charge, &
radius)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 index
            real*8 charge
            real*8 radius
        end subroutine
        subroutine OpenMMGBSW_GBSWForce_setParticleParameters(target, index, &
charge, &
radius)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 index
            real*8 charge
            real*8 radius
        end subroutine
        function OpenMMGBSW_GBSWForce_addCphmdParameters(target, titrateResID, &
refChargeState1, &
refChargeState2, &
chargeState1, &
chargeState2)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 titrateResID
            real*8 refChargeState1
            real*8 refChargeState2
            real*8 chargeState1
            real*8 chargeState2
            integer*4 OpenMMGBSW_GBSWForce_addCphmdParameters
        end function
        subroutine OpenMMGBSW_GBSWForce_getCphmdParameters(target, index, &
titrateResID, &
refChargeState1, &
refChargeState2, &
chargeState1, &
chargeState2)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 index
            integer*4 titrateResID
            real*8 refChargeState1
            real*8 refChargeState2
            real*8 chargeState1
            real*8 chargeState2
        end subroutine
        subroutine OpenMMGBSW_GBSWForce_setCphmdParameters(target, index, &
titrateResID, &
refChargeState1, &
refChargeState2, &
chargeState1, &
chargeState2)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 index
            integer*4 titrateResID
            real*8 refChargeState1
            real*8 refChargeState2
            real*8 chargeState1
            real*8 chargeState2
        end subroutine
        subroutine OpenMMGBSW_GBSWForce_getLambdaOutputFile(target, result)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            character(*) result
        end subroutine
        subroutine OpenMMGBSW_GBSWForce_setLambdaOutputFile(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            character(*) tmp
        end subroutine
        function OpenMMGBSW_GBSWForce_getSystemPH(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getSystemPH
        end function
        subroutine OpenMMGBSW_GBSWForce_setSystemPH(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBSW_GBSWForce_getThetaTemp(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getThetaTemp
        end function
        subroutine OpenMMGBSW_GBSWForce_setThetaTemp(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBSW_GBSWForce_getThetaMass(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getThetaMass
        end function
        subroutine OpenMMGBSW_GBSWForce_setThetaMass(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBSW_GBSWForce_getLambdaOutputFrequency(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 OpenMMGBSW_GBSWForce_getLambdaOutputFrequency
        end function
        subroutine OpenMMGBSW_GBSWForce_setLambdaOutputFrequency(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 tmp
        end subroutine
        function OpenMMGBSW_GBSWForce_getPHbeta(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getPHbeta
        end function
        subroutine OpenMMGBSW_GBSWForce_setPHbeta(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBSW_GBSWForce_getSolventDielectric(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getSolventDielectric
        end function
        subroutine OpenMMGBSW_GBSWForce_setSolventDielectric(target, dielectric)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 dielectric
        end subroutine
        function OpenMMGBSW_GBSWForce_getSoluteDielectric(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getSoluteDielectric
        end function
        subroutine OpenMMGBSW_GBSWForce_setSoluteDielectric(target, dielectric)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 dielectric
        end subroutine
        function OpenMMGBSW_GBSWForce_getSurfaceAreaEnergy(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getSurfaceAreaEnergy
        end function
        subroutine OpenMMGBSW_GBSWForce_setSurfaceAreaEnergy(target, energy)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 energy
        end subroutine
        function OpenMMGBSW_GBSWForce_getAA0(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getAA0
        end function
        subroutine OpenMMGBSW_GBSWForce_setAA0(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 value
        end subroutine
        function OpenMMGBSW_GBSWForce_getAA1(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getAA1
        end function
        subroutine OpenMMGBSW_GBSWForce_setAA1(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 value
        end subroutine
        function OpenMMGBSW_GBSWForce_getNumGauLegRad(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 OpenMMGBSW_GBSWForce_getNumGauLegRad
        end function
        subroutine OpenMMGBSW_GBSWForce_setNumGauLegRad(target, number)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 number
        end subroutine
        function OpenMMGBSW_GBSWForce_getNumLebAng(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 OpenMMGBSW_GBSWForce_getNumLebAng
        end function
        subroutine OpenMMGBSW_GBSWForce_setNumLebAng(target, number)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 number
        end subroutine
        function OpenMMGBSW_GBSWForce_getDebyeHuckelLength(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getDebyeHuckelLength
        end function
        subroutine OpenMMGBSW_GBSWForce_setDebyeHuckelLength(target, length)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 length
        end subroutine
        function OpenMMGBSW_GBSWForce_getSwitchingLength(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getSwitchingLength
        end function
        subroutine OpenMMGBSW_GBSWForce_setSwitchingLength(target, length)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 length
        end subroutine
        function OpenMMGBSW_GBSWForce_getMembraneThickness(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getMembraneThickness
        end function
        subroutine OpenMMGBSW_GBSWForce_setMembraneThickness(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBSW_GBSWForce_getMembraneSwLen(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getMembraneSwLen
        end function
        subroutine OpenMMGBSW_GBSWForce_setMembraneSwLen(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBSW_GBSWForce_getNonbondedMethod(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 OpenMMGBSW_GBSWForce_getNonbondedMethod
        end function
        subroutine OpenMMGBSW_GBSWForce_setNonbondedMethod(target, method)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 method
        end subroutine
        function OpenMMGBSW_GBSWForce_getCutoffDistance(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getCutoffDistance
        end function
        subroutine OpenMMGBSW_GBSWForce_setCutoffDistance(target, distance)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 distance
        end subroutine
        function OpenMMGBSW_GBSWForce_getReactionFieldDielectric(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 OpenMMGBSW_GBSWForce_getReactionFieldDielectric
        end function
        subroutine OpenMMGBSW_GBSWForce_setReactionFieldDielectric(target, distance)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            real*8 distance
        end subroutine
        subroutine OpenMMGBSW_GBSWForce_updateParametersInContext(target, context)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            type (OpenMM_Context) context
        end subroutine
        function OpenMMGBSW_GBSWForce_usesPeriodicBoundaryConditions(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBSW_Types
            implicit none
            type (OpenMMGBSW_GBSWForce) target
            integer*4 OpenMMGBSW_GBSWForce_usesPeriodicBoundaryConditions
        end function
        subroutine OpenMMGBSW_GBSWForce_setLambdaState(target, context, lambda_state)
          use OpenMM_Types
          use OpenMM
          use OpenMMGBSW_Types
          implicit none
          type (OpenMMGBSW_GBSWForce) target
          type (OpenMM_Context) context
          type (OpenMM_DoubleArray) lambda_state
        end subroutine
        subroutine OpenMMGBSW_GBSWForce_getLambdaState(target, context, lambda_state)
          use OpenMM_Types
          use OpenMM
          use OpenMMGBSW_Types
          implicit none
          type (OpenMMGBSW_GBSWForce) target
          type (OpenMM_Context) context
          type (OpenMM_DoubleArray) lambda_state
        end subroutine
    end interface
END MODULE OpenMMGBSW
