program hello

  include 'mpif.h'

  integer my_rank, my_size, errcode

  call MPI_INIT(errcode)

  call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, errcode)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, my_size, errcode)
  print *, 'Hello from ', my_rank, 'of', my_size, '.'

  call MPI_FINALIZE(errcode)

end program hello
