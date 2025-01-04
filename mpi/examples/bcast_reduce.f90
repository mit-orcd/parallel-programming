program main

  implicit none
  include 'mpif.h'

  integer errcode, my_rank, s, x

  call MPI_INIT(errcode)
  call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, errcode)

  if(my_rank==0) x=2 
  !if(my_rank==0) then
  !   x=2
  !endif

  call MPI_Bcast(x, 1, MPI_INT, 0, MPI_COMM_WORLD, errcode)

  x = x * my_rank

  call MPI_Reduce(x, s, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, errcode)

  if(my_rank==0) print *, 'The sum is:', s

  call MPI_FINALIZE(errcode)

end program main

