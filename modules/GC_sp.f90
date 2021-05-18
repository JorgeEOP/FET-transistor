subroutine dost(energy, sigmal, sigmar, hc, sc, dos, trans, dimhc)
  use omp_lib
  !use mpi
  !!$ use mpi 
  !$ use omp_lib
  ! ============================================================== !
  ! Diese Subroutine rechnet G von das zentrales System und danach !
  ! dos und T(E) (Einzel Partikel)                                 !
  ! ============================================================== !

  implicit none

  ! Eingabe
  integer, intent(in) :: dimhc
  complex, intent(in) :: energy
  complex, intent(in), dimension(dimhc, dimhc) :: hc, sc
  complex, intent(in), dimension(dimhc, dimhc) :: sigmal, sigmar

  ! Ausgabe
  real, intent(out) :: dos, trans

  ! Lokal
  real :: pi, start, finish
  integer :: i, j
  complex :: traceb, tracet, invpi
  complex, parameter :: im = (0, 1)
  complex, dimension(dimhc, dimhc) :: hc_eff, gammal, gammar, a, b,&
                                      mdos, tmat, tmat1, tmat2

  !lapack Variablen
  integer :: n, nrhs, lda, ldb, info
  integer, dimension(dimhc) :: ipiv
  
  !mpi variablen
  !integer :: comm, rank, sizem, ierr
  
  !call MPI_Comm_size(comm, sizem, ierr)
  !call MPI_Comm_rank(comm, rank, ierr)
  !print *, "Aus F:", "rank", rank, ", ", "size", sizem 

  pi     = 4.d0 * datan(1.d0)
  invpi  = 1d0/pi + 0d0*im
  trans  = 0d0
  traceb = (0d0, 0d0)
  tracet = (0d0, 0d0)

  n    = dimhc
  lda  = max(1, n)
  ldb  = max(1, n)
  nrhs = dimhc

  !!$omp parallel 
  gammal = im * (sigmal - transpose(conjg(sigmal)))
  gammar = im * (sigmar - transpose(conjg(sigmar)))
  !!$omp end parallel

  a = (energy * sc) - hc - sigmal - sigmar

  b = (0,0)
  do i = 1, dimhc
    b(i,i) = (1,0)
  end do

  call cgesv(n, nrhs, a, lda, ipiv, b, ldb, info)

  if (info /= 0) then
    write (*,*) "Da war ein Fehler"
  endif

  !call cmatrix_mul_h('R', n, n, invpi, sc, b, (0,0), mdos)
  call cmatrix_mul(n, n, n, invpi, b, sc, (0,0), 'N', 'N', mdos)
  call cmatrix_mul(n, n, n, (1,0), gammal, b, (0,0), 'N', 'C', tmat1)
  call cmatrix_mul(n, n, n, (1,0), gammar, b, (0,0), 'N', 'N', tmat2)
  call cmatrix_mul(n, n, n, (1,0), tmat1, tmat2, (0,0), 'N', 'N', tmat)

  !mdos = matmul(b, sc)

  !print ("Time = ", f6.3, "seconds"), finish-start

  do j = 1, dimhc
    traceb = traceb + mdos(j,j)
    tracet = tracet + tmat(j,j) 
  end do

  dos   = -aimag(traceb)
  trans = real(tracet)

end subroutine dost

subroutine dost_h(energy, sigmal, sigmar, hc, sc, dos, trans, dimhc)
  use omp_lib
  ! ============================================================== !
  ! Diese Subroutine rechnet G von das zentrales System und danach !
  ! dos und T(E) (Einzel Partikel)                                 !
  ! ============================================================== !

  implicit none

  ! Eingabe
  integer, intent(in) :: dimhc
  complex, intent(in) :: energy
  complex, intent(in), dimension(dimhc, dimhc) :: hc, sc
  complex, intent(in), dimension(dimhc, dimhc) :: sigmal, sigmar

  ! Ausgabe
  real, intent(out) :: dos, trans

  ! Lokal
  real :: pi
  integer :: i, j
  complex :: traceb, tracet, invpi
  complex, parameter :: im = (0, 1)
  complex, dimension(dimhc, dimhc) :: hc_eff, gammal, gammar, a, b,&
                                      mdos, tmat, tmat1, tmat2

  !lapack Variablen
  integer :: n, nrhs, lda, ldb, info
  integer, dimension(dimhc) :: ipiv
  
  pi     = 4.d0 * datan(1.d0)
  invpi  = 1d0/pi + 0d0*im
  trans  = 0d0
  traceb = (0d0, 0d0)
  tracet = (0d0, 0d0)

  n    = dimhc
  lda  = max(1, n)
  ldb  = max(1, n)
  nrhs = dimhc

  gammal = im * (sigmal - transpose(conjg(sigmal)))
  gammar = im * (sigmar - transpose(conjg(sigmar)))

  hc_eff = hc + sigmal + sigmar
  
  a = (energy * sc) - hc_eff

  b = (0,0)
  do i = 1, dimhc
    b(i,i) = (1,0)
  end do

  call cgesv(n, nrhs, a, lda, ipiv, b, ldb, info)

  if (info /= 0) then
    write (*,*) "Da war ein Fehler"
  endif

  call cmatrix_mul_h('R', n, n, invpi, sc, b, (0,0), mdos)
  call cmatrix_mul_h('L', n, n, (1,0), gammar, b, (0,0), tmat2)
  call cmatrix_mul(n, n, n, (1,0), gammal, b, (0,0), 'N', 'C', tmat1)
  call cmatrix_mul(n, n, n, (1,0), tmat1, tmat2, (0,0), 'N', 'N', tmat)

  !call cmatrix_mul_h('R', n, n, invpi, sc, b, (0,0), mdos)

  do j = 1, dimhc
    traceb = traceb + mdos(j,j)
    tracet = tracet + tmat(j,j) 
  end do

  dos   = -aimag(traceb)
  trans = real(tracet)

end subroutine dost_h

subroutine cmatrix_mul(nrow_a, ncol_b, ncol_a, alpha_cnum, mat_a, mat_b, &
                       beta_cnum, transa, transb, mat_res)

  ! ====================================================================== !
  ! Dieses subroutine nimmt 2 Matrizen mat_a und mat_b zusammen mit den    !
  ! Komplexezahle alpha_cnum und beta_cnum, multipliziert mat_a mal mat_b  !
  ! und rueck die Matrix mat_c gibt.                                       !
  ! ====================================================================== !
  implicit none

  ! cgemm eingabe
  character, intent(in) :: transa, transb
  integer, intent(in) :: nrow_a, ncol_b, ncol_a
  complex, intent(in) :: alpha_cnum, beta_cnum
  complex, intent(in), dimension(nrow_a, ncol_a) :: mat_a
  complex, intent(in), dimension(ncol_a, ncol_b) :: mat_b
  complex, dimension(nrow_a, ncol_b) :: mat_c
  integer :: lda, ldb, ldc

  ! ausgabe
  complex, intent(out), dimension(nrow_a, ncol_b) :: mat_res

  lda = max(1, nrow_a)
  ldb = max(1, ncol_a)
  ldc = max(1, nrow_a)

  call cgemm(transa, transb, nrow_a, ncol_b, ncol_a, alpha_cnum, mat_a, lda, &
             mat_b, ldb, beta_cnum, mat_c, ldc)

  mat_res = mat_c

end subroutine cmatrix_mul

subroutine cmatrix_mul_h(side, nrow_c, ncol_c, alpha_cnum, mat_a, mat_b, &
                         beta_cnum, mat_res)

  ! ====================================================================== !
  ! Dieses subroutine nimmt 2 Matrizen mat_a und mat_b zusammen mit den    !
  ! Komplexezaehle alpha_cnum und beta_cnum, multipliziert mat_a mal mat_b !
  ! und rueck die Matrix mat_c gibt. (mat_a ist eine Hermitische Matrix)   !
  ! ====================================================================== !
  implicit none

  !chemm Eigabe
  character, intent(in) :: side
  character :: uplo
  integer, intent(in) :: nrow_c, ncol_c
  complex, intent(in) :: alpha_cnum, beta_cnum
  complex, intent(in), dimension(nrow_c, ncol_c) :: mat_a, mat_b
  complex, dimension(nrow_c, ncol_c) :: mat_c
  integer :: lda, ldb, ldc

  !Ausgabe
  complex, intent(out), dimension(nrow_c, ncol_c) :: mat_res

  uplo = 'U'
  
  if (side == 'L') then
    lda = max(1, nrow_c)
  else if (side == 'R') then
    lda = max(1, ncol_c)
  else 
    write (*,*) "SIDE option from chemm not found!"
  end if

  ldb = max(1, nrow_c)
  ldc = max(1, nrow_c)

  call chemm(side, uplo, nrow_c, ncol_c, alpha_cnum, mat_a, lda, mat_b, ldb, &
             beta_cnum, mat_c, ldc)
  
  mat_res = mat_c

  end subroutine cmatrix_mul_h
