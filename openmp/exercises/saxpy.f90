program saxpy

  implicit none
  integer :: i,n
  real,dimension(:),allocatable :: x, y
  real :: a,start_time, end_time

  n=100000000
  allocate(x(n),y(n))
  ! Parallelize this block of code (optional)
  x = 1.0d0
  y = 2.0d0
  a = 2.0d0

  call cpu_time(start_time)
  ! Parallelize this block of code
  do i = 1, n
     y(i) = y(i) + a * x(i)
  end do
  call cpu_time(end_time)
  deallocate(x,y)
  
  print '(a,f8.6)', 'SAXPY Time: ', end_time - start_time
  
end program saxpy
    
