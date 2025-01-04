!*************************************************
! Laplace MPI Template Fortran Version
!
! Temperature is initially 0.0
! Boundaries are as follows:
!
!          T=0.   
!       _________                    ____|____|____|____  
!       |       | 0                 |    |    |    |    | 
!       |       |                   |    |    |    |    | 
! T=0.  | T=0.0 | T                 | 0  | 1  | 2  | 3  | 
!       |       |                   |    |    |    |    | 
!       |_______| 100               |    |    |    |    | 
!       0     100                   |____|____|____|____|
!                                        |    |    |
! Each Processor works on a sub grid and then sends its
! boundaries to neighbours
!
!*************************************************
program mpi
      implicit none

      include    'mpif.h'

      !Size of plate
      integer, parameter             :: columns_global=1000
      integer, parameter             :: rows=1000

      !these are the new parameters for parallel purposes
      integer, parameter             :: total_pes=4
      integer, parameter             :: columns=columns_global/total_pes
      integer, parameter             :: left=100, right=101

      !usual mpi variables
      integer                        :: mype, npes, ierr
      integer                        :: status(MPI_STATUS_SIZE)

      double precision, parameter    :: tolerance=0.01

      integer                        :: i, j, max_iterations, iteration=1
      double precision               :: dA, dA_global=100.0
      real                           :: start_time, stop_time

      double precision, dimension(0:rows+1,0:columns+1) :: A_new, A

      !usual mpi startup routines
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      !It is nice to verify that proper number of PEs are running
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      !Only one PE should prompt user
      if( mype == 0 ) then
         print*, 'Maximum iterations [100-4000]?'
         read*,   max_iterations
      endif
      
      !Other PEs need to recieve this information
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      call cpu_time(start_time)

      call initialize(A, npes, mype)

      !do until global error is minimal or until maximum steps
      do while ( dA_global > tolerance .and. iteration <= max_iterations)

         do j=1,columns
            do i=1,rows
               A_new(i,j)=0.25*(A(i+1,j)+A(i-1,j)+ &
                                      A(i,j+1)+A(i,j-1) )
            enddo
         enddo

         ! COMMUNICATION PHASE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

         dA=0.0

         do j=1,columns
            do i=1,rows
               dA = max( abs(A_new(i,j) - A(i,j)), dA )
               A(i,j) = A_new(i,j)
            enddo
         enddo

         !Need to determine and communicate maximum error
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

         !periodically print test values - only for PE in lower corner
         if( mod(iteration,100).eq.0 ) then
            if( mype == npes-1 ) then
               call track_progress(A_new, iteration)
            endif
         endif

         iteration = iteration+1

      enddo

      !Slightly more accurate timing and cleaner output
      call MPI_Barrier(MPI_COMM_WORLD, ierr)

      call cpu_time(stop_time)

      if( mype == 0 ) then
         print*, 'Max error at iteration ', iteration-1, ' was ',dA_global
         print*, 'Total time was ',stop_time-start_time, ' seconds.'
      endif

      call MPI_Finalize(ierr)

end program mpi

!Parallel version requires more attention to global coordinates
subroutine initialize(A, npes, mype )
      implicit none

      integer, parameter             :: columns_global=1000
      integer, parameter             :: rows=1000
      integer, parameter             :: total_pes=4
      integer, parameter             :: columns=columns_global/total_pes

      integer                        :: i,j
      integer                        :: npes,mype

      double precision, dimension(0:rows+1,0:columns+1) :: A
      double precision               :: tmin, tmax

      A = 0

      !Left and Right Boundaries
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      !Top and Bottom Boundaries
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

end subroutine initialize

subroutine track_progress(A_new, iteration)
      implicit none

      integer, parameter             :: columns_global=1000
      integer, parameter             :: rows=1000
      integer, parameter             :: total_pes=4
      integer, parameter             :: columns=columns_global/total_pes

      integer                        :: i,iteration

      double precision, dimension(0:rows+1,0:columns+1) :: A_new

!Parallel version uses global coordinate output so users don't need
!to understand decomposition
      print *, '---------- Iteration number: ', iteration, ' ---------------'
      do i=5,0,-1
         write (*,'("("i4,",",i4,"):",f6.2,"  ")',advance='no'), &
                   rows-i,columns_global-i,A_new(rows-i,columns-i)
      enddo
      print *
end subroutine track_progress
