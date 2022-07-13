! f90 version
!
! the best in the end it to make it into a module
! 
! line shape fitting for proton peaks, fit multiple peaks


module lfitm1
  implicit none 
! fixed peak parameters, the position has already been defined (can be changed later)

    real*8 :: ap, bp
    real*8 :: x_0 = 0.
    real*8 :: y_0 = 1. 
   ! fit parameters
    
    ! defined by the calling program
    ! order of back ground
    integer :: norder 

    integer :: ma  
    integer :: ncvm 
    integer:: mfit

    ! number of peaks to be fitted
    integer :: np

    ! peak positions
    real*8, dimension(:), allocatable :: x_p
    

   ! fit control
    integer, dimension(:), allocatable :: lista

    
    ! x_0  internal peak position (calculated)
    ! a  internal alpha (decay time) 
    ! b  internal beta (rise time)
    ! y_0 internal norm 

    integer:: ndata
    integer, parameter:: mmax=60

    real*8, dimension(:), allocatable :: a
    
    real*8, dimension(:,:), allocatable::covar
    real*8, dimension(:,:), allocatable::mat, matinv
    real*8, dimension(:), allocatable ::beta,afunc
    integer, dimension(:), allocatable :: ipiv,indxr,indxc

contains

subroutine init_all(al, bl, xp, n_peaks, n_order, vary_code, istat)

  implicit none 
! fixed peak parameters, the positions have already been defined (can be changes later)

    real*8, intent(in) :: al, bl
    real*8, dimension(:), intent(in) :: xp
    integer, intent(in):: n_peaks, n_order
    integer, dimension(:), intent(in) :: vary_code

    integer,intent(out):: istat

    np = n_peaks
    ! order of polygon for background
    norder = n_order
    
!   get space for the parameter array
    ma = size(vary_code)

! allocate arrays
    allocate(x_p(np), STAT = istat)
    allocate(a(ma),  STAT = istat)

    allocate(lista(ma), STAT = istat)
    allocate(covar(ma,ma), STAT = istat)
    
    allocate(mat(ma,ma))
    allocate(matinv(ma,ma))
    
    allocate(beta(ma), STAT = istat)
    allocate(afunc(ma), STAT = istat)

    allocate(ipiv(ma), STAT = istat)
    allocate(indxr(ma), STAT = istat)
    allocate(indxc(ma), STAT = istat)


    if (istat .ne. 0) then
       print *, 'Cannot allocate arrays : ', istat
       return
    endif
    
! copy peak positions
    x_p = xp

    call calc_peak_shape(al, bl)
    call init_fit(vary_code)

end subroutine init_all

subroutine set_par(i, x)
  implicit none
  
  integer, intent(in) :: i
  real*8, intent(in) :: x

  a(i) = x
  return
end subroutine set_par

subroutine free_all
! deallocate arrays
    deallocate(a)
    deallocate(lista)
    deallocate(covar)
    deallocate(beta)
    deallocate(afunc)
    deallocate(x_p)
    
    deallocate(mat)
    deallocate(matinv)

    deallocate(ipiv)
    deallocate(indxr)
    deallocate(indxc)

    return

end subroutine free_all

subroutine init_fit(vary_code)

    integer, dimension(:), intent(in) :: vary_code
    integer :: i
    
    mfit = 0
    a = 0.
  ! set fit array
    do i = 1, size(vary_code)
       if (vary_code(i) .eq. 1) then
          mfit = mfit + 1
          lista(mfit) = i
       endif
    enddo

end subroutine init_fit


subroutine calc_peak_shape(al, bl)

    real*8, intent(in) :: al, bl

    ap = al
    bp = bl

  ! calculate the fixed values of the peak shape
    x_0 = log((2.*bp)/ap-1.)/(2.*bp)
    y_0 = exp(-ap*(x_0))*(1.+tanh(bp*(x_0)))

end subroutine calc_peak_shape

subroutine shapes(x)
! peak shape
    implicit none

    real*8, intent(in)::x

    real*8 :: x_pos

    integer j, nb, k
    

    ! qudratic background
    ! constant term
    nb = 1    
    afunc(nb)= 1.

    ! terms dependent on x
    do k = 1, norder
        nb = nb + 1
        afunc(nb) = x*afunc(nb - 1)
    enddo
    ! peak value for n_peak peaks

    nb = nb + 1
    do j = nb, ma
       x_pos = x_0 - x_p(j-(nb-1))
       if (-ap*(x+x_pos)>20) then
          afunc(j)= 0.
       else
          afunc(j) = 1./y_0*exp(-ap*(x+x_pos))*(1.+tanh(bp*(x+x_pos)))
       endif
    enddo
    return
end subroutine shapes

! function to calculate the peak shape for a give position
! and set of parameters

function line_shape (x)
    implicit none
! assemble the fitting function  
! total of ma fitting parameters

    real*8, intent(in):: x

    real*8 :: line_shape
    real*8 :: yfit
        
    integer:: i

    call shapes(x) ! calculate the value of the basis functions
                                  ! for x
    yfit = 0.
    do i = 1, ma
        yfit = yfit + a(i)*afunc(i)
    enddo

    line_shape = yfit

    return
end function line_shape

! the actual fitting routine, you need to init the peak shape before using this

function peak_fit(x, y, sig, ndata)
! the first peak is at 0. by default
    implicit none
    integer, intent(in) :: ndata
    real*8, dimension(:), intent(in) :: x, y, sig

    real*8 ::  peak_fit, chi_sq

    call pfit( x, y, sig, ndata, chi_sq) 
    peak_fit = chi_sq

end function peak_fit

! complete package for a linear least quare fit using Numerical Recipes 
! functions with Gauss-Jordan Matrix inversion
!

!subroutine linfit(x, y, sig, ndata, a, ma, lista, mfit , covar, ncvm, chisq, funcs)

subroutine pfit(x, y, sig, ndata, chi_sq) 

! replace with global parameters a, ma, lista, mfit , covar, ncvm, chisq, funcs)
!
!
! linear fit subroutine from NR.1st edition using gauss-jordan method
!
!      Given a set of NDATA points X(I), Y(I) with individual
!      std. dev. SIC(I), use Chi_sq minimization to determine MFIT of the 
!      MA coefficients A of a function that depends linearly on A, y =
!      SUM_i A_i AFUNC_i(x). 
!
!      The array LISTA renumbers the parameters so 
!      that the first MFIT elements correspond to the parameters actually 
!      being determined; the remaining MA-MFIT elements are held fixed at 
!      their input values. 
!
!      The program returns values for the MA fit parameters A, Chi_sq,
!      CHI2, and the covariance matrix COVAR(I,J). NVCM is the physical
!      dimension of COVAR(NCVM, NCVM) in the calling routine. 
!
!      The user supplies the subroutine FUNCS(X, AFUNC, MA) that returns
!      the MA basis functions evaluated at x=X in the array AFUNC.
!
    implicit none

    integer, intent(in) :: ndata

    real*8, dimension(ndata), intent(in) :: x, y, sig

    real*8, intent(out):: chi_sq
    
! locals
    integer:: i, j, k, kk
    integer:: ihit
    integer :: err
    real*8:: ym, sig2i, wt, chisq, sum

    chi_sq = -1.

      kk=mfit+1

! WB check data
!      write (6,*) x, y, sig, a, lista, ncvm


! check fit parameter list
      do j=1,ma
         ihit=0
         do k=1,mfit
            if (lista(k).eq.j) ihit=ihit+1
         enddo
         if (ihit.eq.0) then
            lista(kk)=j
            kk=kk+1
         else if (ihit.gt.1) then
            print*,  'linefit: improper set in lista ihit = ', ihit 
            return
        endif
      enddo
      if (kk.ne.(ma+1)) then
         print*,  'linefit: improper set in lista ', kk, ma+1
         return
      endif

! start fit here

      covar = 0.
      beta = 0. 
      mat = 0.
      matinv = 0.
      do i=1,ndata
         call shapes(x(i))
         ym=y(i)
         if(mfit.lt.ma) then
            do j=mfit+1,ma
               ym = ym - a(lista(j))*afunc(lista(j))
            enddo
         endif
         sig2i=1./sig(i)**2
         do j=1,mfit
            wt=afunc(lista(j))*sig2i
            do k=1,j
               covar(j,k)=covar(j,k)+wt*afunc(lista(k))
            enddo
            beta(j)=beta(j)+ym*wt
         enddo
      enddo

      if (mfit.gt.1) then
         do j=2,mfit
            do k=1,j-1
               covar(k,j)=covar(j,k)
            enddo
         enddo
      endif
      mat = covar
      call gaussj(covar,mfit,ma,beta,1,1, err)
      matinv = covar
      if (err .ne. 0) then
!        if there is an inverstion error return neg. chi sq.
        chi_sq = err
         return
      endif
      do j=1,mfit
         a(lista(j))=beta(j)
      enddo
      chisq=0.
      do i=1,ndata
         call shapes(x(i))
         sum=0.
         do j=1,ma
            sum=sum+a(j)*afunc(j)
         enddo
         chisq=chisq+((y(i)-sum)/sig(i))**2
      enddo
      call covsrt
      chi_sq =  chisq
end subroutine pfit


subroutine covsrt
!
! given the covariance matrix covar of a fit for mfit of ma total
! parameters, and their ordering lista(i), repack the covariance matrix
! to the true order of the parameters. elements associated with fixed 
! parameters will be zero. ncvm is the physical dimension of covar
! 
  implicit none
  
  integer:: i, j
  real*8 swap

  do j=1,ma-1               ! zero all elements below diagonal
     do i=j+1,ma
        covar(i,j)=0.
     enddo
  enddo
  do  i=1,mfit-1            ! repack off-diagonal elements of the fit into
     do  j=i+1,mfit         ! correct locations below diagonal
        
        if(lista(j).gt.lista(i)) then
           covar(lista(j),lista(i))=covar(i,j)
        else
           covar(lista(i),lista(j))=covar(i,j)
        endif
     enddo
  enddo
  swap=covar(1,1)    ! temporarily store original diagonal elements
  do j=1,ma                 ! in top row, and zero the diagonal
     covar(1,j)=covar(j,j)
     covar(j,j)=0.
  enddo
  covar(lista(1),lista(1))=swap
  do  j=2,mfit     ! now sort elements into proper order on diagonal 
     covar(lista(j),lista(j))=covar(1,j)
  enddo
  do  j=2,ma
     do  i=1,j-1            ! finally, fill in above diagonal by symmerty
        covar(i,j)=covar(j,i)
     enddo
  enddo
  return
end subroutine covsrt

subroutine gaussj(a,n,np,b,m,mp, err)
!
! linear equation solution by gauss-jordan elimination, equation (2.1.1)
! a is an input matrix of nxn elements, stored in an array of physical
! dimensions np by np. b is an input matrix of n by m containing the m 
! right hand side vectors, stored in an array  of physical dimensions
! np by mp. on output , a is replaced by its matrix inverse, and b is
! replaced by the corresponding set of solution vectors
!
  implicit none
  integer, parameter::nmax=50
  integer::n, np, m, mp, err
  
  real*8, dimension(np,np):: a
  real*8, dimension(np,mp):: b

  integer:: i, j, k, l, ll, irow, icol
  real*8:: big, pivinv, dum

  err = 0

!      do j=1,n
!         ipiv(j)=0
!      enddo
! f90
      ipiv = 0

      do  i=1,n         ! main loop over the columns to be reduced
         big=0.
         do  j=1,n              ! outer loop of the search of a pivot element
            if(ipiv(j).ne.1)then
               do  k=1,n
                  if (ipiv(k).eq.0) then
                     if (abs(a(j,k)).ge.big)then
                        big=abs(a(j,k))
                        irow=j
                        icol=k
                     endif
                  else if (ipiv(k).gt.1) then
                     ! print*,  'gaussj: singular matrix', k, ipiv(k)
                     err = -20
                     return
                  endif
               enddo
            endif
         enddo
         ipiv(icol)=ipiv(icol)+1
!
! we now have the pivot element, so we interchange rows, if needed, to put the
! pivot element on the diagonal. the columns are not physically interchanged,
! only relabeled: indxc(i), the column of the ith pivot element, is the ith 
! column that is reduced, while indxr(i) is the row in which that pivot element
! was originally located. if indxr(i) .ne. indxc(i) there is an implied 
! column interchange. with this form of bookkeeping, the solution b's will end
! up in the correct order, and the inverse matrix will me scramled by 
! columns.
!
         if (irow.ne.icol) then
            do l=1,n
               dum=a(irow,l)
               a(irow,l)=a(icol,l)
               a(icol,l)=dum
            enddo
            do l=1,m
               dum=b(irow,l)
               b(irow,l)=b(icol,l)
               b(icol,l)=dum
            enddo
         endif
         indxr(i)=irow          ! we are now ready to divide 
                                ! the pivot row by the pivot
         indxc(i)=icol          ! element, located at irow and icol

         if (a(icol,icol).eq.0.) then 
            ! print*, 'gaussj: singular matrix.'
            err = -21
            return
         endif
         pivinv=1./a(icol,icol)
         a(icol,icol)=1.
         do l=1,n
            a(icol,l)=a(icol,l)*pivinv
         enddo
         do l=1,m
            b(icol,l)=b(icol,l)*pivinv
         enddo
         do ll=1,n              ! next we reduce the rows ...
            if(ll.ne.icol)then  ! ...except fot the pivot one, of course ...
               dum=a(ll,icol)
               a(ll,icol)=0.
               do l=1,n
                  a(ll,l)=a(ll,l)-a(icol,l)*dum
               enddo
               do l=1,m
                  b(ll,l)=b(ll,l)-b(icol,l)*dum
               enddo
            endif
         enddo
      enddo                           ! end of the main loop over columns 
                                      ! of the reduction
      do l=n,1,-1                     ! it only remains to unscramble the 
                                      ! solution in view
         if(indxr(l).ne.indxc(l))then ! of the the column interchanges. we 
                                      ! do so by interchanging pairs
            do  k=1,n                 ! of columns in the reverse order
               dum=a(k,indxr(l))      !  that the permutation was built up.
               a(k,indxr(l))=a(k,indxc(l)) 
               a(k,indxc(l))=dum
            enddo
         endif
      enddo
      return
    end subroutine gaussj

end module lfitm1

