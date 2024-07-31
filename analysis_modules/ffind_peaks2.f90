! find minima and maxiam in an array
!
! This is a function for f2py
! 
! Compilation: f2pi -c ffind_peaks2.f90 -m find_peaks2
!
! Usage:
!
!  input: val - Array of data to be seached
!         N  - Number of entries in val
!         Np - Max. number of minima and maxima expected 
!         delta - value by which val has changed after min or max befor it is classified as min or max
!        
! output:   n_min - number of minima found
!           min_pos - array of indices with minima
!           n_max - number of maxima found
!           max_pos - array of maxima found
!
!
!  example on how to used it
!
!  import ffind_peaks2 as FP2
!  x  = np.linspace(0., 20, 1000)
!  y = np.sin(5*x) 
!  nmin, minpo, nmax, maxpo = FP2.find_peaks(y.size, y.size, .1, y)
!  x_max = x[maxpo[:nmax]]
!  y_max = y[maxpo[:nmax]]
!  plot(x,y)
! plot(x_max, y_max, 'ro')



! f90 version



subroutine find_peaks(N, Np, delta, val, n_min, min_pos, n_max, max_pos)
      
  implicit none

  
  ! values can be returned via array pos      
  
  integer, intent(in):: N, Np
  real(kind = 8), intent(in):: delta
  real(kind = 8), dimension(N), intent(in) ::val
  ! integer, dimension(2), intent(out):: results
  integer, intent(out) :: n_max, n_min
  integer, dimension(Np), intent(out):: min_pos
  integer, dimension(Np), intent(out):: max_pos
  
  real(kind = 8):: inf = 1.0D30
  
  integer :: i, max_i, min_i
  
  
  logical :: look_for_max
  
  real(kind = 8) :: max_val, min_val, this
  real(kind = 8) ::frac, rn
  
  max_val = -inf
  min_val = inf
  
  n_max = 0
  n_min = 0
  
  look_for_max = .True.
  rn = real(N)
  
  
  print *, 'Searching : ', N, ' points'
  do i = 1, N
     this = val(i)
     frac = real(i)/rn
     if (this .gt. max_val) then
        max_val = this
        max_i = i-1 ! make sure the indices start at 0
     endif
     if (this .lt. min_val) then
        min_val = this
        min_i = i-1
     endif
     if (look_for_max) then
        if (this .lt. (max_val-delta)) then
           n_max = n_max + 1
           max_pos(n_max) = max_i
           min_val = this
           min_i = i-1
           look_for_max = .False.
        endif
     else
        if (this .gt. min_val+delta) then
           n_min = n_min + 1
           min_pos(n_min) = min_i
           max_val = this
           max_i = i-1
           look_for_max = .True.
        endif
     endif
  enddo
  print *, 'PD-found : ', n_max, ' peaks'
  
  return
  
end subroutine find_peaks


!----------------------------------------------------------------------
!  m33det  -  compute the determinant of a 3x3 matrix.
!----------------------------------------------------------------------

function m33det (a) result (det)

  implicit none

  interface
     subroutine print_real_matrix(M, name)
       real(kind = 8), dimension(:,:), intent(in) :: M
       character(len = *), intent(in) :: name
     end subroutine print_real_matrix
  end interface

  real(kind = 8), dimension(3,3), intent(in)  :: a
  real(kind = 8) :: det

  ! call print_real_matrix(a, 'M_m33det')

  det =    a(1,1)*a(2,2)*a(3,3) - a(1,1)*a(3,2)*a(2,3) &
         - a(2,1)*a(1,2)*a(3,3) + a(2,1)*a(3,2)*a(1,3) &
         + a(3,1)*a(1,2)*a(2,3) - a(3,1)*a(2,2)*a(1,3)

  return

end function m33det


subroutine print_real_array(x, name)

  implicit none

  
  real(kind = 8), dimension(:), intent(in) :: x
  character(len = *), intent(in) :: name

  integer::i

  print *,  name, ' = [ ', (x(i), ',' , i = 1, size(x)), ']'

end subroutine print_real_array


subroutine print_real_matrix(M, name)

  implicit none

  interface
     subroutine print_real_array(xx, name )
       real(kind = 8), dimension(:), intent(in) :: xx
       character(len = *), intent(in) :: name
     end subroutine print_real_array
  end interface

  
  real(kind = 8), dimension(:,:), intent(in) :: M
  character(len = *), intent(in) :: name

  character(len = 80) :: row_name
  
  integer::i

  print *, '---------', name, ' -------------------'

  do i = 1, size(M, 1)
     write(row_name, "(a4,i1)") 'row_', i
     call print_real_array(M(i,:), trim(row_name))
  enddo

  print *, '----------------------------------------------------'
  
end subroutine print_real_matrix


! fast function to fit a parabola to a set of points (x, y)
!
! This according to Bevington 2nd edition p. 118

! function fit_parabola (x, y, dy) result (a)

function fit_parabola (x, y) result (a)

  implicit none

  interface
     function m33det (a) result (det)
       real(kind = 8), dimension(3,3), intent(in)  :: a
       real(kind = 8) :: det
     end function m33det
     subroutine print_real_array(xx, name )
       real(kind = 8), dimension(:), intent(in) :: xx
       character(len = *), intent(in) :: name
     end subroutine print_real_array
     subroutine print_real_matrix(M, name)
       real(kind = 8), dimension(:,:), intent(in) :: M
       character(len = *), intent(in) :: name
     end subroutine print_real_matrix
  end interface
  
  ! real(kind = 8), dimension(:),  intent(in) :: x, y, dy
  real(kind = 8), dimension(:),  intent(in) :: x, y
  real(kind = 8), dimension(3) :: a

  real(kind = 8), dimension(3,3) :: M
  ! calculate the various matrices (seel Bevington, 2nd edition p 118)

  real(kind = 8), dimension(size(x)) :: w, x2, x3, x4 
  real(kind = 8), dimension(3) ::c1, c2, c3, b
  real(kind = 8) :: det

  w = 1.   ! inverse errors for the momentwe do not use errors

  x2 = x**2   ! powers of the x-values
  x3 = x**3
  x4 = x**4

  b  = (/sum(y*w), sum(y*x*w), sum(y*x2*w)/)
  c1 = (/sum(w),   sum(x*w),   sum(x2*w)/)
  c2 = (/c1(2),    c1(3),       sum(x3*w)/)
  c3 = (/c1(3),   c2(3),       sum(x4*w)/)

  !print *, '--- c1 ---'
  !call print_real_array(c1, 'c1')
  !print *, '--- c2 ---'
  !call print_real_array(c2, 'c2')
  !print *, '--- c3 ---'
  !call print_real_array(c3, 'c3')
  
  ! Delta
  M(:,1) = c1
  M(:,2) = c2
  M(:,3) = c3

  !call print_real_matrix(M, 'M_fit')
  
  det = m33det(M)

  if (det .eq. 0.) then
     a = 0.
     print *, '------------------------- Det = 0 ------------------'
     call print_real_array(x, 'x')
     call print_real_array(y, 'y')
     print *, '----------------------------------------------------'
     return
  endif
  

  ! parameter a1
  M(:,1) = b

  a(1) = m33det(M)/det

  ! parameter a2
  M(:,1) = c1
  M(:,2) = b
  M(:,3) = c3

  a(2) = m33det(M)/det

  ! parameter a3
  M(:,1) = c1
  M(:,2) = c2
  M(:,3) = b

  a(3) = m33det(M)/det
  
  
end function fit_parabola

  

!----------------------------------------------------------------------
! refine peak position found by find_peaks by fitting a parabola to a regions around the
! peak and using its maximum/minimum position
!----------------------------------------------------------------------

subroutine refine_positions(N, nm, n_close, i_m,  x, y, x_m, y_m, p_a)

  ! Input:
  ! N - number of data points in x and val
  ! nm - number of maxima/minima
  ! n_close - number of neigboring points to be included in fit : i +/- n_close
  ! x - array of e.g. time values for the data
  ! y - array of e.g. digitizer data values

  ! Output:
  ! x_m - array of fitted positions for the nm maxima/minima

  implicit none

  
  interface
     function fit_parabola(x,y) result(a)
       real(kind = 8), dimension(:),  intent(in) :: x
       real(kind = 8), dimension(:),  intent(in) :: y
       real(kind = 8), dimension(3) :: a
     end function fit_parabola
     subroutine print_real_array(xx, name )
       real(kind = 8), dimension(:), intent(in) :: xx
       character(len = *), intent(in) :: name
     end subroutine print_real_array
  end interface

  
  integer, intent(in) :: N, nm, n_close

  real(kind = 8), dimension(N), intent(in) :: x, y
  integer, dimension(nm), intent(in) :: i_m
  real(kind = 8), dimension(nm), intent(out) :: x_m, y_m
  real(kind = 8), dimension(nm,3), intent(out) :: p_a

  ! fitting formula
  real(kind = 8), dimension(3) :: a
  real(kind = 8) :: xf, yf, x0, y0
  integer :: i, ip, i_start, i_end
  
  ! loop over all positions and fit a parabola to the maximum(minimum) at index i_m +/- n_close

  ! open(16, file = 'test.data')
  
  do i = 1, nm
     ip = i_m(i) + 1
     i_start = max(1, ip - n_close)
     i_end = min(ip + n_close, N)
     x0 = x(ip)
     y0 = y(ip)
     a = fit_parabola( x(i_start:i_end)-x0, y(i_start:i_end) )
     
     ! calculate maxima/minima

     xf = -a(2)/(2.*a(3))
     yf = a(1) + ( a(2) + a(3)*xf )*xf

     ! write(16, *) a, ip, x0, y0, i_start, i_end, xf, yf, x(i_start:i_end)-x0, y(i_start:i_end)
     

     ! call print_real_array(a, 'a')
     ! print *, 'xf, yf = ', xf, yf
     x_m(i) = xf + x0
     y_m(i) = yf 
     p_a(i,:) = a
  enddo

  ! close(16)
  return
end subroutine refine_positions


! example program
!program testit
!  implicit none
!
!  interface
!     function fit_parabola(x,y) result(a)
!       real(kind = 8), dimension(:),  intent(in) :: x
!       real(kind = 8), dimension(:),  intent(in) :: y
!       real(kind = 8), dimension(3) :: a
!     end function fit_parabola
!  end interface
!  real(kind = 8), dimension(3) :: a
!
!  real(kind = 8), dimension(6) :: x, y
!
!  x = (/1.,2.,3.,4.,5.,6./)
!  y = (/2.,4.,6.,7., 8., 5./)
!  
!  a = fit_parabola(x, y)
  !a = test_arr(x, y)

!  print*, ' a = ', a

!end program testit
