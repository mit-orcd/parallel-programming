program main

      double precision  PI25DT
      parameter        (PI25DT = 3.141592653589793238462643d0)
      double precision  mypi, pi, h, sum, x, f, a
      integer*4 num_iters
      integer rank, size, i, rc
      character junk

!     Loop until finished
!      num_iters = 200000
      num_iters = 2000

!     Calculate the interval size
      h = 1.0d0 / num_iters
      sum  = 0.0d0
      do 10 i = 1, num_iters
         x = h * (dble(i) - 0.5d0)
         sum = sum + 4.d0 / (1.d0 + x * x)
 10   continue
      mypi = h * sum

!     All finished
      write(6, 97) num_iters, mypi, abs(mypi - PI25DT)
 97   format(i20, ' points: pi is approximately: ', F18.16, ' error is: ', F18.16)
      stop

end
