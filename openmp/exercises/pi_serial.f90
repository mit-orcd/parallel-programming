program pi_serial

  implicit none
  integer, parameter :: dp=selected_real_kind(14)
  integer :: i
  integer, parameter :: n=100000000
  real(dp) :: x,pi,sum,step,start_time,end_time

  sum = 0d0
  step = 1.d0/float(n)

  call cpu_time(start_time)
  ! Parallelize the following block of code
  do i = 0, n
     x = (i + 0.5d0) * step
     sum = sum + 4.d0 / (1.d0 + x ** 2)
  end do
  pi = step * sum
  call cpu_time(end_time)

  print '(a,f17.15)', "pi = ", pi
  print '(a,f9.6,a)', "time to compute =",end_time - start_time, " seconds"

end program pi_serial
