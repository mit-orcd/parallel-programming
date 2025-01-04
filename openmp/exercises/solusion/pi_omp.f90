program pi_omp
  
  implicit none
  integer, parameter :: dp=selected_real_kind(14)
  integer, parameter :: ip=selected_int_kind(15)
  integer(ip) :: i
  integer(ip), parameter :: n=10000000
  real(dp) :: x,pi,sum,step,start_time,end_time
  integer, dimension(8) :: value

  sum = 0d0
  step = 1.d0/float(n)

  call date_and_time(VALUES=value)
  start_time = float(value(6)*60) + float(value(7)) + float(value(8))/1000d0
  !$omp parallel do default(shared) private(i,x) reduction(+:sum)
  do i = 0, n
     x = (i + 0.5d0) * step
     sum = sum + 4.d0 / (1.d0 + x ** 2)
  end do
  !$omp end parallel do
  pi = step * sum
  call date_and_time(VALUES=value)
  end_time = float(value(6)*60) + float(value(7)) + float(value(8))/1000d0
  if ( start_time > end_time ) end_time = end_time + 3600d0

  print '(a,f17.15)', "pi = ", pi
  print '(a,f9.3,a)', "time to compute =",end_time - start_time, " seconds"

end program pi_omp
