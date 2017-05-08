 program main
 use dyn_tt
 integer :: d, rmax, nswp, verb, kickrank, Asize, Ysize
 real(8) :: tau
 integer, allocatable :: n(:), m(:), ra(:), ry(:)
 complex(8), allocatable ::  crA(:), crY(:)
 integer :: i
    open(unit=10,status='old',file='test_ksl.dat',form='unformatted',access='stream')
    !open(unit=10,status='old',file='test_eye_ksl.dat',form='unformatted',access='stream')
    read(10) d
    allocate(n(d))
    allocate(m(d))
    print *,'d=',d
    read(10) n(1:d)
    read(10) m(1:d)
    allocate(ra(d+1))
    allocate(ry(d+1))
    read(10) ra(1:d+1)
    read(10) ry(1:d+1)
    read(10) Asize
    allocate(crA(Asize))
    read(10) crA(1:Asize)
    allocate(crY(Ysize))
    read(10) Ysize
    read(10) crY(1:Ysize)
    read(10) tau,rmax,kickrank,nswp,verb
    close(10)
    
    !Test if we read all correctly
    print *,'n=',n(1:d)
    print *,'m=',m(1:d)
    print *,'ra=',ra(1:d+1)
    print *,'ry=',ry(1:d+1)
    print *,'tau=',tau, 'rmax=',rmax,'kickrank=',kickrank,'nswp=',nswp,'verb=',verb
    
    do while ( 1 > 0 )
        call ztt_ksl(d,n,m,ra,crA, crY, ry, tau, rmax, kickrank, nswp, verb)
    end do
    !Now we can call the main block
    
    !open(unit=10,status='replace',file='test_ksl.dat',form='unformatted')
    !write(10) d,n,m,ra,ry,pa(d+1)-1,crA(1:(pa(d+1)-1)),mm-1,crY0(1:mm-1),tau,rmax,kickrank,nswp,verb 
    !close(10)
    !return
 end program
