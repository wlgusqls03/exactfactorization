! QML ABI probe; no CTMQC propagation or source modification.
program model_probe
  use iso_fortran_env, only: real64
  implicit none
  integer :: n, d, ios, i, unit
  character(len=100) :: model
  real(real64), allocatable :: q(:), v(:,:), grad(:,:,:), nac(:,:,:), vec(:,:)
  namelist /adapter_probe/ model,n,d
  read(*,adapter_probe)
  open(newunit=unit,file='probe_qml.log',status='replace')
  call set_Qmodel_out_unit(unit)
  call sub_Init_Qmodel(d,n,model,.true.,1)
  call set_Qmodel_Phase_Checking(.true.)
  call set_Qmodel_Phase_following(.true.)
  allocate(q(d),v(n,n),grad(n,n,d),nac(n,n,d),vec(n,n))
  open(unit=31,file='probe_points.dat',status='old')
  open(unit=32,file='probe_values.dat',status='new')
  do
    read(31,*,iostat=ios) q
    if (ios /= 0) exit
    vec=0.0_real64
    call sub_Qmodel_VG_NAC_Vec0(v,grad,nac,vec,q)
    write(32,'(*(es25.16e3,1x))') v,grad,nac,vec
  end do
  close(31)
  close(32)
end program
