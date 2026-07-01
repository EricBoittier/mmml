module sanity_checks

!-----------------------------------------------------------------
!------------------------- MNDO97 -------------------------------
#if KEY_MNDO97==1 /*mndo97_fcm*/

#if KEY_QUANTUM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 /*mndo_check*/
#error  'QUANTUM, GAMESS(UK) and MNDO97 code are mutually exclusive.'
#endif /* (mndo_check)*/

#if KEY_SCCDFTB==1 || KEY_CADPAC==1 /*mndo_check*/
#error  'SCCDFTB, CADPAC, and MNDO97 code are mutually exclusive.'
#endif /* (mndo_check)*/

#if KEY_SQUANTM==1
#error  'SQUANTM and MNDO97 code are mutally exclusive.'
#endif 

#endif /* (mndo97_fcm)*/

!-----------------------------------------------------------------
!------------------------- CADPAC -------------------------------
#if KEY_CADPAC==1 /*cadpac_fcm*/

#if KEY_QUANTUM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 /*qm_check*/
#error  'QUANTUM, CADPAC, and GAMESS(UK) code are mutually exclusive.'
#endif /* (qm_check)*/

#if KEY_SCCDFTB==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
#error  'SCCDFTB, MNDO97, SQUANTM, and CADPAC code are mutually exclusive.'
#endif 

#endif /* (cadpac_fcm)*/

!-----------------------------------------------------------------
!------------------------- QUANTUM -------------------------------
#if KEY_QUANTUM==1 /*quantum_check*/

#if KEY_CADPAC==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 /*gms_check*/
  Kill the compile, see preprocessor message
#error  'QUANTUM, CADPAC, GAMESS and GAMESS-UK code are mutually exclusive.'
#endif /* (gms_check)*/

#if KEY_SCCDFTB==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
  Kill the compile, see preprocessor message
#error  'QUANTUM, MNDO97, SQUANTM, and SCCDFTB code are mutually exclusive.'
#endif 

#endif /* (quantum_check)*/

!-----------------------------------------------------------------
!------------------------- GAMESS -------------------------------
#if KEY_GAMESS==1 /*gamess_check*/
#if KEY_QUANTUM==1 || KEY_CADPAC==1 || KEY_GAMESSUK==1 /*qms_check*/
  Kill the compile, see preprocessor message
#error  'QUANTUM, CADPAC, GAMESS and GAMESS-UK code are mutually exclusive.'
#endif /* (qms_check)*/
#if KEY_SCCDFTB==1 || KEY_MNDO97==1 /*qms_check*/ /* kn_080713 remove SQUANTM*/
  Kill the compile, see preprocessor message
#error  'GAMESS, SCCDFTB, and MNDO97 code are mutually exclusive.'
#endif /* (qms_check)*/
#endif /* (gamess_check)*/
     !
#if KEY_GAMESSUK==1 /*guk_check*/
#if KEY_QUANTUM==1 || KEY_CADPAC==1 || KEY_GAMESS==1 /*qgms_check*/
  Kill the compile, see preprocessor message
#error  'QUANTUM, CADPAC, GAMESS and GAMESS-UK code are mutually exclusive.'
#endif /* (qgms_check)*/
#if KEY_SCCDFTB==1 || KEY_MNDO97==1 /*qgms_check*/ /* kn_080713 remove SQUANTM*/
  Kill the compile, see preprocessor message
#error  'SCCDFTB, MNDO97, and GAMESS(UK) code are mutually exclusive.'
#endif /* (qgms_check)*/
#endif /* (guk_check)*/
     !
#if KEY_SCCDFTB==1 /*sccdft_fcm*/
#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
  Kill the compile, see preprocessor message
#error  'QUANTUM, MNDO97, SQUANTM, and SCCDFTB code are mutually exclusive.'
#endif /* */
#endif /* (sccdft_fcm)*/
     !
#if KEY_SQUANTM==1 /*squantm_fcm*/
#if KEY_QUANTUM==1 || KEY_MNDO97==1
  Kill the compile, see preprocessor message
#error  'QUANTUM, MNDO97, and SQUANTM code are mutually exclusive.'
#endif 
#if KEY_CADPAC==1 || KEY_SCCDFTB==1 /* kn_080713 remove GAMESS and GAMESSUK*/
  Kill the compile, see preprocessor message
#error  'CADPAC, SCCDFTB and SQUANTM code are mutually exclusive.'
#endif 
#endif /* (squantm_fcm)*/

!-----------------------------------------------------------------
!------------------------- COLFFT -------------------------------
#if KEY_COLFFT==1 /*col*/
#if KEY_IFN==1 || KEY_PARALLEL==1
 /* (pll) */
#endif
!  Kill the compile, COLFFT needs PARALLEL, column-fft is for parallel only
!##ERROR ' COLFFT specified without PARALLEL, COLFFT is for parallel only'
#if KEY_ENDIF==1
 /* (pll) */
#endif
#endif /* (col)*/

!-----------------------------------------------------------------
!--------------------------- FFTW --------------------------------
#if KEY_FFTW==1 /*fftw*/
#if KEY_MKL==1
  Kill the compile, cannot have FFTW and MKL
#error  'FFTW and MKL are mutually exclusive.'
#endif 
#if KEY_PATHSCALE==1
  Kill the compile, cannot link to FFTW with Pathscale
#error  'FFTW and PATHSCALE are mutually exclusive'
#endif 
#endif /* (fftw)*/

!-----------------------------------------------------------------
!---------------------------- MKL --------------------------------
#if KEY_MKL==1 /*mkl*/
#if KEY_FFTW==1
  Kill the compile, cannot have FFTW and MKL
#error  'FFTW and MKL are mutually exclusive.'
#endif 
#endif /* (mkl)*/

!-----------------------------------------------------------------
!-------------------------- DOMDEC -------------------------------
#if KEY_DOMDEC==1 /*domdec*/
#if KEY_PARALLEL==0 /*pll*/
  Kill the compile, DOMDEC needs PARALLEL
#error  ' DOMDEC specified without PARALLEL, DOMDEC is for parallel only'
#endif /* (pll)*/
#if KEY_PARAFULL==0 /*pf*/
  Kill the compile, DOMDEC needs PARAFULL
#error  ' DOMDEC specified without PARAFULL'
#endif /* (pf)*/
#if KEY_SPACDEC==1 || KEY_PARASCAL==1 || KEY_CMPI==1 /*others*/
  Kill the compile, see preprocessor message
#error  ' DOMDEC does not work with SPACDEC, PARASCAL, or CMPI'
#endif /* (others)*/
#endif /* (domdec)*/

#if KEY_DOMDEC_GPU==1 /*domdec_gpu*/
#if KEY_DOMDEC==0 /*domdec*/
  Kill the compile, DOMDEC_GPU requires DOMDEC
#error  'DOMDEC_GPU requires DOMDEC'
#endif /* (domdec)*/
#endif /* (domdec_gpu)*/

!-----------------------------------------------------------------
!------------------------- IMCUBES -------------------------------

!-----------------------------------------------------------------
!------------------------- TAMD -------------------------------
#if KEY_TAMD==1 /*tamdcheck*/
#if KEY_PARALLEL==1 /*pll*/
  This line to break compiler when both TAMD and PARALLEL are in pref.dat
#error  ' TAMD cannot run in parallel, remove TAMD from pref.dat for parallel'
#endif /*       (pll)*/
#endif /* (tamdcheck)*/

!-----------------------------------------------------------------
!------------------------- PARALLEL -------------------------------
#if KEY_PARALLEL==1 /*parallel*/
#if KEY_PARAFULL==1 /*pf*/

#if KEY_PARASCAL==1
#error  'PARALLEL cannot have both PARAFULL and PARASCAL'
#elif KEY_SPACDEC==1
#error  'PARALLEL cannot have both PARAFULL and SPACDEC'
! else only PARAFULL
#endif 

#elif KEY_PARASCAL==1 /*pf*/ /* not PARAFULL*/

#if KEY_SPACDEC==1
#error  'PARALLEL cannot have both PARASCAL and SPACDEC'
! else only PARASCAL
#endif 

#else /* (pf)  neither PARAFULL nor PARASCAL*/

#if KEY_SPACDEC==0
#error  'PARALLEL requires either PARAFULL, SPACDEC, or PARASCAL'
! else only SPACDEC
#endif 

#endif /* (pf)*/
#endif /* (parallel)*/


!-----------------------------------------------------------------
!------------------------- CSA, DISTENE --------------------------
#if KEY_CSA==1 || KEY_DISTENE==1
!  Break compile     CSA, DISTENE work only in parallel
#if KEY_PARALLEL==0
#error  'CSA/DISTENE needs to be compiled with the PARALLEL in pref.dat'
#endif 
#endif 

!-----------------------------------------------------------------
!------------------------- BLOCK -------------------------------
#if KEY_BLOCK==0
#if KEY_DOCK==1
#error  'The DOCK compile keyword requires the BLOCK keyword'
#endif 
#endif 

!-----------------------------------------------------------------
!------------------------- REPDSTR -------------------------------
#if KEY_REPDSTR==1 && KEY_PARALLEL==0
#error  'Either add PARALLEL or delete REPDSTR from pref.dat'
#endif 

!-----------------------------------------------------------------
!------------------------- PERT -------------------------------
#if KEY_PERT==0 /*nopert*/
#if KEY_CHEMPERT==1 || KEY_WCA==1
See preprocessor message
#error  'Keywords CHEMPERT and WCA require PERT'
#endif 
#endif /* (nopert)*/

!-----------------------------------------------------------------
!------------------------- ACTBOND -------------------------------
#if KEY_ACTBOND==1 /*actbond0*/
#if KEY_GENETIC==1
#error  'Keywords GENETIC and ACTBOND are incompatible'
#endif 
#else /* (actbond0)*/
#if KEY_ZEROM==1
#error  'Keyword ZEROM requires ACTBOND'
#endif 
#endif /* (actbond0)*/

!-----------------------------------------------------------------
!------------------------- MSCALE --------------------------------
#if KEY_MSCALE==1 /*mscaletst*/
#if KEY_PARALLEL==0 /*partest*/
#error  'keyword MSCALE REQUIRES PARALLEL'
#endif /* (partest)*/
#endif /* (mscaletst)*/

!-----------------------------------------------------------------
!------------------------- HFB/TRAVEL -------------------------------
#if KEY_HFB==1
#if KEY_TRAVEL==0
#error  'keyword HFB requires TRAVEL keyword'
#endif 
#endif 

!-----------------------------------------------------------------
!------------------------- PRIMO/EPMF-----------------------------
#if KEY_PRIMO==1
#if KEY_EPMF==0
#error  'keyword PRIMO requires EPMF keyword'
#endif 
#endif 

!-----------------------------------------------------------------
!------------------------- EPMF/CMAP -----------------------------
#if KEY_EPMF==1
#if KEY_CMAP==0
#error  'keyword EPMF requires CMAP keyword'
#endif 
#endif 

!------------------------- ENSEMBLE ------------------------------
#if KEY_ENSEMBLE==1 /*ens*/

#if KEY_PARALLEL==0
#error  'keyword ENSEMBLE requires PARALLEL keyword'
#endif 

#if KEY_CMPI==1
#error  'keyword ENSEMBLE cannot have CMPI keyword'
#endif 

#endif /* (ens)*/

!------------------------- STRINGM ------------------------------
#if KEY_STRINGM==1 && ((KEY_PARALLEL==0) || (KEY_MPI==0))
#error 'the string method (STRINGM) requires PARALLEL and MPI'
#endif

!-----------------------------------------------------------------
!------------------------- feature -------------------------------

  !===========================================================================
  !            End Sanity Checks
  !===========================================================================
end module sanity_checks

