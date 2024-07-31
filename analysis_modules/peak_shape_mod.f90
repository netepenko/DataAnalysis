! peak shape module for standard peak shape

module peak_shape_mod
    implicit NONE
    real*8, ap, bp
    real*8 x_0  ! position of eak maximum
    real*8 y_0  ! height of maximum
    
CONTAINS
    

subroutine calc_peak_shape(al, bl)

    real*8, intent(in) :: al, bl

    ap = al
    bp = bl

  ! calculate the fixed values of the peak shape
    x_0 = log((2.*bp)/ap-1.)/(2.*bp)
    y_0 = exp(-ap*(x_0))*(1.+tanh(bp*(x_0)))

end subroutine calc_peak_shape

function c_func(x)
