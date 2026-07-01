
MODULE OpenMMGBMV_Types
    implicit none

    ! Global Constants


    ! Type Declarations

    type OpenMMGBMV_GBMVForce
        integer*8 :: handle = 0
    end type
    integer*4, parameter :: OpenMMGBMV_GBMVForce_NoCutoff = 0
    integer*4, parameter :: OpenMMGBMV_GBMVForce_CutoffNonPeriodic = 1
    integer*4, parameter :: OpenMMGBMV_GBMVForce_CutoffPeriodic = 2


END MODULE OpenMMGBMV_Types

MODULE OpenMMGBMV
    use OpenMM_Types
    use OpenMM
    use OpenMMGBMV_Types
    implicit none
    interface

        

        ! OpenMMGBMV::GBMVForce
        subroutine OpenMMGBMV_GBMVForce_create(result)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) result
        end subroutine
        subroutine OpenMMGBMV_GBMVForce_destroy(destroy)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) destroy
        end subroutine
        subroutine OpenMMGBMV_GBMVForce_addCPHMDForce(target, pH, &
T_theta, &
mTheta, &
ts_theta, &
beta, &
outFreq, &
fileName)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 pH
            real*8 T_theta
            real*8 mTheta
            real*8 ts_theta
            real*8 beta
            integer*4 outFreq
            character(*) fileName
        end subroutine
        function OpenMMGBMV_GBMVForce_usingCPHMD(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 OpenMMGBMV_GBMVForce_usingCPHMD
        end function
        function OpenMMGBMV_GBMVForce_getNumTitratingGroups(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 OpenMMGBMV_GBMVForce_getNumTitratingGroups
        end function
        function OpenMMGBMV_GBMVForce_addTitratingGroupParameters(target, &
resPKA1, resPKA2, &
barrier1, barrier2, &
a0, a1, a2, a3, a4, a5, a6, a7, a8)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
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
            integer*4 OpenMMGBMV_GBMVForce_addTitratingGroupParameters
        end function
        subroutine OpenMMGBMV_GBMVForce_getTitratingGroupParameters(target, index, &
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
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
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
        subroutine OpenMMGBMV_GBMVForce_setTitratingGroupParameters(target, index, &
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
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
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
        function OpenMMGBMV_GBMVForce_addNonbondedException(target, atom1, &
atom2)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 atom1
            integer*4 atom2
            integer*4 OpenMMGBMV_GBMVForce_addNonbondedException
        end function
        subroutine OpenMMGBMV_GBMVForce_getNonbondedException(target, index, &
atom1, &
atom2)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 index
            integer*4 atom1
            integer*4 atom2
        end subroutine
        function OpenMMGBMV_GBMVForce_getNumNonbondedExceptions(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 OpenMMGBMV_GBMVForce_getNumNonbondedExceptions
        end function
        function OpenMMGBMV_GBMVForce_getNumParticles(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 OpenMMGBMV_GBMVForce_getNumParticles
        end function
        function OpenMMGBMV_GBMVForce_addParticle(target, charge, &
radius)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 charge
            real*8 radius
            integer*4 OpenMMGBMV_GBMVForce_addParticle
        end function
        subroutine OpenMMGBMV_GBMVForce_getParticleParameters(target, index, &
charge, &
radius)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 index
            real*8 charge
            real*8 radius
        end subroutine
        subroutine OpenMMGBMV_GBMVForce_setParticleParameters(target, index, &
charge, &
radius)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 index
            real*8 charge
            real*8 radius
        end subroutine
        function OpenMMGBMV_GBMVForce_addCphmdParameters(target, titrateResID, &
refChargeState1, &
refChargeState2, &
chargeState1, &
chargeState2)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 titrateResID
            real*8 refChargeState1
            real*8 refChargeState2
            real*8 chargeState1
            real*8 chargeState2
            integer*4 OpenMMGBMV_GBMVForce_addCphmdParameters
        end function
        subroutine OpenMMGBMV_GBMVForce_getCphmdParameters(target, index, &
titrateResID, &
refChargeState1, &
refChargeState2, &
chargeState1, &
chargeState2)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 index
            integer*4 titrateResID
            real*8 refChargeState1
            real*8 refChargeState2
            real*8 chargeState1
            real*8 chargeState2
        end subroutine
        subroutine OpenMMGBMV_GBMVForce_setCphmdParameters(target, index, &
titrateResID, &
refChargeState1, &
refChargeState2, &
chargeState1, &
chargeState2)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 index
            integer*4 titrateResID
            real*8 refChargeState1
            real*8 refChargeState2
            real*8 chargeState1
            real*8 chargeState2
        end subroutine
        subroutine OpenMMGBMV_GBMVForce_getLambdaOutputFile(target, result)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            character(*) result
        end subroutine
        subroutine OpenMMGBMV_GBMVForce_setLambdaOutputFile(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            character(*) tmp
        end subroutine
        function OpenMMGBMV_GBMVForce_getSystemPH(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getSystemPH
        end function
        subroutine OpenMMGBMV_GBMVForce_setSystemPH(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBMV_GBMVForce_getThetaTemp(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getThetaTemp
        end function
        subroutine OpenMMGBMV_GBMVForce_setThetaTemp(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBMV_GBMVForce_getThetaMass(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getThetaMass
        end function
        subroutine OpenMMGBMV_GBMVForce_setThetaMass(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBMV_GBMVForce_getLambdaOutputFrequency(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 OpenMMGBMV_GBMVForce_getLambdaOutputFrequency
        end function
        subroutine OpenMMGBMV_GBMVForce_setLambdaOutputFrequency(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 tmp
        end subroutine
        function OpenMMGBMV_GBMVForce_getPHbeta(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getPHbeta
        end function
        subroutine OpenMMGBMV_GBMVForce_setPHbeta(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBMV_GBMVForce_getSolventDielectric(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getSolventDielectric
        end function
        subroutine OpenMMGBMV_GBMVForce_setSolventDielectric(target, dielectric)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 dielectric
        end subroutine
        function OpenMMGBMV_GBMVForce_getSoluteDielectric(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getSoluteDielectric
        end function
        subroutine OpenMMGBMV_GBMVForce_setSoluteDielectric(target, dielectric)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 dielectric
        end subroutine
        function OpenMMGBMV_GBMVForce_getSurfaceAreaEnergy(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getSurfaceAreaEnergy
        end function
        subroutine OpenMMGBMV_GBMVForce_setSurfaceAreaEnergy(target, energy)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 energy
        end subroutine
        function OpenMMGBMV_GBMVForce_getAA0(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getAA0
        end function
        subroutine OpenMMGBMV_GBMVForce_setAA0(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getAA1(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getAA1
        end function
        subroutine OpenMMGBMV_GBMVForce_setAA1(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getNumGauLegRad(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 OpenMMGBMV_GBMVForce_getNumGauLegRad
        end function
        subroutine OpenMMGBMV_GBMVForce_setNumGauLegRad(target, number)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 number
        end subroutine
        function OpenMMGBMV_GBMVForce_getNumLebAng(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 OpenMMGBMV_GBMVForce_getNumLebAng
        end function
        subroutine OpenMMGBMV_GBMVForce_setNumLebAng(target, number)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 number
        end subroutine
        function OpenMMGBMV_GBMVForce_getDebyeHuckelLength(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getDebyeHuckelLength
        end function
        subroutine OpenMMGBMV_GBMVForce_setDebyeHuckelLength(target, length)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 length
        end subroutine
        function OpenMMGBMV_GBMVForce_getSwitchingLength(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getSwitchingLength
        end function
        subroutine OpenMMGBMV_GBMVForce_setSwitchingLength(target, length)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 length
        end subroutine
! GBMV2: Parameters of GBMV2 model
        function OpenMMGBMV_GBMVForce_getBETA_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getBETA_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setBETA_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getLAMBDA_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getLAMBDA_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setLAMBDA_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getALPHA_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getALPHA_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setALPHA_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getSLOPE_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getSLOPE_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setSLOPE_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getSHIFT_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getSHIFT_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setSHIFT_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getHSX1_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getHSX1_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setHSX1_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getHSX2_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getHSX2_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setHSX2_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getONX_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getONX_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setONX_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getOFFX_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getOFFX_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setOFFX_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getP1_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getP1_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setP1_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getP2_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getP2_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setP2_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getP3_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getP3_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setP3_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getP6_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getP6_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setP6_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 value
        end subroutine
        function OpenMMGBMV_GBMVForce_getCUTNUM_GBMV2(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer OpenMMGBMV_GBMVForce_getCUTNUM_GBMV2
        end function
        subroutine OpenMMGBMV_GBMVForce_setCUTNUM_GBMV2(target, value)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer value
        end subroutine
! END GBMV2
        function OpenMMGBMV_GBMVForce_getMembraneThickness(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getMembraneThickness
        end function
        subroutine OpenMMGBMV_GBMVForce_setMembraneThickness(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBMV_GBMVForce_getMembraneSwLen(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getMembraneSwLen
        end function
        subroutine OpenMMGBMV_GBMVForce_setMembraneSwLen(target, tmp)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 tmp
        end subroutine
        function OpenMMGBMV_GBMVForce_getNonbondedMethod(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 OpenMMGBMV_GBMVForce_getNonbondedMethod
        end function
        subroutine OpenMMGBMV_GBMVForce_setNonbondedMethod(target, method)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 method
        end subroutine
        function OpenMMGBMV_GBMVForce_getCutoffDistance(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getCutoffDistance
        end function
        subroutine OpenMMGBMV_GBMVForce_setCutoffDistance(target, distance)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 distance
        end subroutine
        function OpenMMGBMV_GBMVForce_getCutonDistance(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getCutonDistance
        end function
        subroutine OpenMMGBMV_GBMVForce_setCutonDistance(target, distance)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 distance
        end subroutine
        function OpenMMGBMV_GBMVForce_getReactionFieldDielectric(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 OpenMMGBMV_GBMVForce_getReactionFieldDielectric
        end function
        subroutine OpenMMGBMV_GBMVForce_setReactionFieldDielectric(target, distance)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            real*8 distance
        end subroutine
        subroutine OpenMMGBMV_GBMVForce_updateParametersInContext(target, context)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            type (OpenMM_Context) context
        end subroutine
        function OpenMMGBMV_GBMVForce_usesPeriodicBoundaryConditions(target)
            use OpenMM_Types
            use OpenMM
            use OpenMMGBMV_Types
            implicit none
            type (OpenMMGBMV_GBMVForce) target
            integer*4 OpenMMGBMV_GBMVForce_usesPeriodicBoundaryConditions
        end function
        subroutine OpenMMGBMV_GBMVForce_setLambdaState(target, context, lambda_state)
          use OpenMM_Types
          use OpenMM
          use OpenMMGBMV_Types
          implicit none
          type (OpenMMGBMV_GBMVForce) target
          type (OpenMM_Context) context
          type (OpenMM_DoubleArray) lambda_state
        end subroutine
        subroutine OpenMMGBMV_GBMVForce_getLambdaState(target, context, lambda_state)
          use OpenMM_Types
          use OpenMM
          use OpenMMGBMV_Types
          implicit none
          type (OpenMMGBMV_GBMVForce) target
          type (OpenMM_Context) context
          type (OpenMM_DoubleArray) lambda_state
        end subroutine
    end interface
END MODULE OpenMMGBMV
