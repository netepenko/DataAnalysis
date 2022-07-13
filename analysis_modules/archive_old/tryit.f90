
program testit
  implicit none

  real(kind = 8), dimension(6) :: x, y
  integer :: i

  character(len = 80) name

  x = (/1.,2.,3.,4.,5.,6./)
  y = (/2.,4.,6.,7., 8., 5./)

  print *,  'x = [ ', (x(i), ',' , i = 1, size(x)), ']'

  write(name, "(a7,i1)") 'length_', size(x)
  print *, 'name = ', trim(name)
end program testit
