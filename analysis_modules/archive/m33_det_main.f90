!***********************************************************************************************************************************
!
!                                                       M 3 3 D E T _ M A I N
!
!  Program:      M33DET_MAIN
!
!  Programmer:   David G. Simpson
!                NASA Goddard Space Flight Center
!                Greenbelt, Maryland  20771
!
!  Date:         July 22, 2005
!
!  Language:     Fortran-90
!
!  Version:      1.00a
!
!  Description:  This program is a short "driver" to call function M33DET, which computes the determinant a 3x3 matrix.
!
!  Files:        Source files:
!
!                   m33det.f90                   Main program
!
!***********************************************************************************************************************************

      PROGRAM M33DET_MAIN

      IMPLICIT NONE

      INTEGER :: I, J
      DOUBLE PRECISION, DIMENSION(3,3) :: MAT
      DOUBLE PRECISION :: DET

      DOUBLE PRECISION :: M33DET

!-----------------------------------------------------------------------------------------------------------------------------------

!
!     get user input.
!

      write (unit=*, fmt='(/a/)') ' enter matrix:'

      do i = 1, 3
         do j = 1, 3
            write (unit=*, fmt='(a,i1,1h,,i1,a)', advance='no') ' a(', i, j, ') = '
            read (unit=*, fmt=*) mat(i,j)
         end do
      end do

!
!     compute the determinant of the input matrix.
!

      det = m33det (mat)

!
!     print the result.
!

      write (unit=*, fmt='(/a,es25.15)') ' det = ', det

      stop

      end program m33det_main



!***********************************************************************************************************************************
!  m33det  -  compute the determinant of a 3x3 matrix.
!***********************************************************************************************************************************

      function m33det (a) result (det)

      implicit none

      double precision, dimension(3,3), intent(in)  :: a

      double precision :: det


      det =   a(1,1)*a(2,2)*a(3,3)  &
            - a(1,1)*a(2,3)*a(3,2)  &
            - a(1,2)*a(2,1)*a(3,3)  &
            + a(1,2)*a(2,3)*a(3,1)  &
            + a(1,3)*a(2,1)*a(3,2)  &
            - a(1,3)*a(2,2)*a(3,1)

      return

      end function m33det
