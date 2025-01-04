program sender

  implicit none
  include 'mpif.h'

  integer my_rank, errcode, numbertoreceive, numbertosend, status(MPI_STATUS_SIZE)
  
  call MPI_INIT(errcode)
  call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, errcode)

  if (my_rank.EQ.0) then
    numbertosend = 36
    call MPI_Send( numbertosend, 1,MPI_INTEGER, 1, 10, MPI_COMM_WORLD, errcode)
  elseif (my_rank.EQ.1) then
    call MPI_Recv( numbertoreceive, 1, MPI_INTEGER, 0, 10, MPI_COMM_WORLD, status, errcode)
    print *, 'Number received is:', numbertoreceive
  endif

  call MPI_FINALIZE(errcode)

end
