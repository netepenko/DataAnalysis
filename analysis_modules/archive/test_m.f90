program matrix
  implicit none
  real, dimension(3) ::cv1, cv2, cv3  ! column vectors
  real, dimension(3,3) :: M

  integer :: i, j
  
  cv1 = (/11., 21., 31./)
  cv2 = (/12., 22., 32./)
  cv3 = (/13., 23., 33./)
  
  print*, 'cv1 = ', cv1
  print*, 'cv2 = ', cv2
  print*, 'cv3 = ', cv3

  M(:,1) = cv1  ! set the matrix by column vectors
  M(:,2) = cv2
  M(:,3) = cv3

  do i = 1, 3
     do j = 1,3
        print*, 'i,j, M(i,j) = ', i, j, M(i,j)
     enddo
  enddo
  
  print *, 'M = ', M

  print *, ' sums = ', sum(cv1), sum(cv2), sum(cv3)
end program matrix
