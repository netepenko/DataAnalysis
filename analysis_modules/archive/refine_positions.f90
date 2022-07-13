!----------------------------------------------------------------------
! refine peak position found by find_peaks by fitting a parabola to a regions around the
! peak and using its maximum/minimum position
!----------------------------------------------------------------------

subroutine refine_positions(N, nm, n_close, i_m,  x, val, x_m)

  ! Input:
  ! N - number of data points in x and val
  ! nm - number of maxima/minima
  ! n_close - number of neigboring points to be included in fit : i +/- n_close
  ! x - array of e.g. time values for the data
  ! val - array of e.g. digitizer data values

  ! Output:
  ! x_m - array of fitted positions for the nm maxima/minima

  implicit none
  
  integer, intent(in) :: N, nm, n_close

  real(kind = 8), dimension(N), intent(in) :: x, val
  integer, dimension(nm), intent(in) :: i_m
  real(kind = 8), dimension(nm), intent(out) :: x_m

  ! fitting formula
  real(kind = 8), dimension(3) :: fit_parabola, a
  integer :: ip, i_start, i_end
  
  ! loop over all positions and fit a parabola to the maximum(minimum) at index i_m +/- n_close

  do ip = 1, nm
     i_start = max(1, ip - n_close)
     i_end = min(ip + n_close, N)

     a = fit_parabola( x(i_start:i_end), val(i_start:i_end) )

     ! calculate maxima/minima

     x_m(ip) = -a(2)/(2.*a(3))
  enddo
  return
end subroutine refine_positions
