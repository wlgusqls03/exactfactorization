! Adapter additions to G-CTMQC (not upstream code), 2026-11-04.
! Distributed with the modified build under LGPL-3.0-or-later.
! I/O and monitoring only. No alteration of the upstream propagation equations.
SUBROUTINE adapter_initial_coefficients
  INTEGER :: unit, itraj, ios
  REAL(KIND=dp) :: re(nstates), im(nstates)
  COMPLEX(KIND=qp) :: fixed(nstates)
  OPEN(NEWUNIT=unit,FILE='adapter_initial.dat',STATUS='old',IOSTAT=ios)
  IF (ios /= 0) ERROR STOP 'Missing adapter_initial.dat'
  DO itraj=1,ntraj
    READ(unit,*,IOSTAT=ios) re, im
    IF (ios /= 0) ERROR STOP 'Invalid adapter initial coefficients'
    fixed=cmplx(re,im,kind=qp)
    IF (abs(sum(abs(fixed)**2)-1.0_dp)>1.0e-10_dp) ERROR STOP 'Initial norm != 1'
    BOcoeff(itraj,:)=matmul(transpose(Vec0(:,:,itraj)),fixed)
  END DO
  CLOSE(unit)
END SUBROUTINE adapter_initial_coefficients

SUBROUTINE adapter_snapshot(step)
  USE iso_fortran_env, ONLY: int64
  USE, INTRINSIC :: ieee_arithmetic, ONLY: ieee_is_finite
  INTEGER, INTENT(IN) :: step
  INTEGER :: unit, itraj
  CHARACTER(LEN=16) :: idx
  REAL(KIND=dp) :: mf(ntraj,n_dof), cf(ntraj,n_dof), total(n_dof)
  REAL(KIND=dp) :: no_coupling(nstates,nstates)
  COMPLEX(KIND=dp) :: fixed(ntraj,nstates)
  no_coupling=0.0_dp
  DO itraj=1,ntraj
    fixed(itraj,:)=cmplx(matmul(Vec0(:,:,itraj),BOcoeff(itraj,:)),kind=dp)
    CALL non_adiabatic_force(BOcoeff(itraj,:),mf(itraj,:), &
         my_force(itraj,:,:),no_coupling,itraj)
    CALL non_adiabatic_force(BOcoeff(itraj,:),total, &
         my_force(itraj,:,:),k_li(itraj,:,:),itraj)
    cf(itraj,:)=total-mf(itraj,:)
  END DO
  IF (.NOT.all(ieee_is_finite(Rcl)) .OR. .NOT.all(ieee_is_finite(Vcl)) .OR. &
      .NOT.all(ieee_is_finite(real(fixed))) .OR. .NOT.all(ieee_is_finite(aimag(fixed)))) &
      ERROR STOP 'Nonfinite author state'
  WRITE(idx,'(i10.10)') step
  OPEN(NEWUNIT=unit,FILE='adapter_frames/frame_'//trim(idx)//'.bin', &
       ACCESS='stream',FORM='unformatted',STATUS='new')
  WRITE(unit) int(ntraj,int64),int(n_dof,int64),int(nstates,int64),int(step,int64)
  WRITE(unit) real(step*dt,dp),adapter_norm_max
  WRITE(unit) Rcl, Vcl, fixed, BOenergy, mf, cf, my_force, k_li
  CLOSE(unit)
  adapter_norm_max=0.0_dp
END SUBROUTINE adapter_snapshot
