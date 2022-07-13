subroutine test_arr (xx)

  implicit none

  real(kind = 8), dimension(:), intent(in) :: xx
  real(kind = 8), dimension(3) :: a

  a = 0.

  print *, 'test_arr: xx = ', xx
  
end subroutine test_arr


program testit

  
  implicit none

  interface
     subroutine test_arr(x)
       real(kind = 8), dimension(:), intent(in) :: x
     end subroutine test_arr
  end interface
  
  real(kind = 8), dimension(3) ::  a

  real(kind = 8), dimension(6) :: x, y

  x = (/1.,2.,3.,4.,5.,6./)
  y = (/2.,4.,6.,7., 8., 5./)

  print *, 'x, y = ', x, y
  
  ! a = fit_parabola(x, y)
  call test_arr(x)


end program testit
