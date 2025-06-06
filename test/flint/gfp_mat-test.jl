@testset "fpMatrix.constructors" begin
  Z2 = Native.GF(2)
  Z3 = Native.GF(3)

  @test_throws ErrorException matrix_space(Z2, -1, 5)
  @test_throws ErrorException matrix_space(Z2, 0, -2)
  @test_throws ErrorException matrix_space(Z2, -3, -4)
  @test_throws ErrorException fpMatrixSpace(Z2, 2, -1)
  @test_throws ErrorException fpMatrixSpace(Z2, -1, 2)
  @test_throws ErrorException fpMatrixSpace(Z2, -1, -1)

  R = fpMatrixSpace(Z2, 2, 2)

  @test elem_type(R) == fpMatrix
  @test elem_type(fpMatrixSpace) == fpMatrix
  @test parent_type(fpMatrix) == fpMatrixSpace
  @test nrows(R) == 2
  @test ncols(R) == 2

  @test isa(R, fpMatrixSpace)

  @test base_ring(R) == Z2

  S = fpMatrixSpace(Z3, 2, 2)

  @test isa(S, fpMatrixSpace)

  RR = fpMatrixSpace(Z2, 2, 2)

  @test isa(RR, fpMatrixSpace)

  @test R == RR

  a = R()

  @test isa(a, fpMatrix)
  @test parent(a) == R

  ar = [ BigInt(1) BigInt(1); BigInt(1) BigInt(1) ]

  b = R(ar)

  @test isa(b, fpMatrix)
  @test parent(b) == R
  @test nrows(b) == 2 && ncols(b) == 2
  @test_throws ErrorConstrDimMismatch R(reshape(ar,1,4))
  @test b == R([BigInt(1), BigInt(1), BigInt(1), BigInt(1)])
  @test_throws ErrorConstrDimMismatch R([BigInt(1) BigInt(1)])
  @test_throws ErrorConstrDimMismatch R([BigInt(1) BigInt(1) ; BigInt(1) BigInt(1) ;
                                         BigInt(1) BigInt(1)])
  @test_throws ErrorConstrDimMismatch R([BigInt(1), BigInt(1), BigInt(1)])
  @test_throws ErrorConstrDimMismatch R([BigInt(1), BigInt(1),
                                         BigInt(1), BigInt(1), BigInt(1)])

  ar = [ ZZ(1) ZZ(1); ZZ(1) ZZ(1) ]

  c = R(ar)
  @test isa(c, fpMatrix)
  @test parent(c) == R
  @test nrows(c) == 2 && ncols(c) == 2
  @test_throws ErrorConstrDimMismatch R(reshape(ar,4,1))
  @test c == R([ ZZ(1), ZZ(1), ZZ(1), ZZ(1)])
  @test_throws ErrorConstrDimMismatch R([ZZ(1) ZZ(1)])
  @test_throws ErrorConstrDimMismatch R([ZZ(1) ZZ(1) ; ZZ(1) ZZ(1) ; ZZ(1) ZZ(1)])
  @test_throws ErrorConstrDimMismatch R([ZZ(1), ZZ(1), ZZ(1)])
  @test_throws ErrorConstrDimMismatch R([ZZ(1), ZZ(1), ZZ(1), ZZ(1), ZZ(1)])

  ar = [ 1 1; 1 1]

  d = R(ar)

  @test isa(d, fpMatrix)
  @test parent(d) == R
  @test nrows(d) == 2 && ncols(d) == 2
  @test_throws ErrorConstrDimMismatch R(reshape(ar,1,4))
  @test d == R([1,1,1,1])
  @test_throws ErrorConstrDimMismatch R([1 1 ])
  @test_throws ErrorConstrDimMismatch R([1 1 ; 1 1 ; 1 1 ])
  @test_throws ErrorConstrDimMismatch R([1, 1, 1])
  @test_throws ErrorConstrDimMismatch R([1, 1, 1, 1, 1])

  ar = [ 1 1; 1 1]

  d = R(ar)

  @test isa(d, fpMatrix)

  ar = matrix_space(ZZ, 2, 2)([ 1 1; 1 1])

  e = R(ar)

  @test isa(e, fpMatrix)
  @test parent(e) == R
  @test nrows(e) == 2 && ncols(e) == 2

  ar = matrix_space(ZZ, 3, 3)([ 1 1 1 ; 1 1 1; 1 1 1])

  @test_throws ErrorException R(ar)

  ar = [ Z2(1) Z2(1); Z2(1) Z2(1) ]

  f = R(ar)

  @test isa(f, fpMatrix)
  @test parent(f) == R
  @test nrows(f) == 2 && ncols(f) == 2
  @test_throws ErrorConstrDimMismatch R(reshape(ar,4,1))
  @test f == R([Z2(1), Z2(1), Z2(1), Z2(1)])
  @test_throws ErrorConstrDimMismatch R([Z2(1) Z2(1) ])
  @test_throws ErrorConstrDimMismatch R([Z2(1) Z2(1) ; Z2(1) Z2(1) ; Z2(1) Z2(1) ])
  @test_throws ErrorConstrDimMismatch R([Z2(1), Z2(1), Z2(1)])
  @test_throws ErrorConstrDimMismatch R([Z2(1), Z2(1), Z2(1), Z2(1), Z2(1)])

  @test isa(S(1), fpMatrix)

  @test isa(S(ZZRingElem(1)), fpMatrix)

  @test isa(S(Z3(1)), fpMatrix)

  g = deepcopy(e)

  @test b == c
  @test c == d
  @test d == e
  @test e == f
  @test g == e

  arr = [1 2; 3 4]
  arr2 = [1, 2, 3, 4, 5, 6]

  for T in [Z3, ZZRingElem, Int, BigInt]
    M = matrix(Z3, map(T, arr))
    @test isa(M, fpMatrix)
    @test base_ring(M) == Z3

    M2 = matrix(Z3, 2, 3, map(T, arr2))
    @test isa(M2, fpMatrix)
    @test base_ring(M2) == Z3
    @test nrows(M2) == 2
    @test ncols(M2) == 3
    @test_throws ErrorConstrDimMismatch matrix(Z3, 2, 2, map(T, arr2))
    @test_throws ErrorConstrDimMismatch matrix(Z3, 2, 4, map(T, arr2))
  end

  M3 = zero_matrix(Z3, 2, 3)

  @test isa(M3, fpMatrix)
  @test base_ring(M3) == Z3

  M4 = identity_matrix(Z3, 3)

  @test isa(M4, fpMatrix)
  @test base_ring(M4) == Z3

  a = zero_matrix(Z2, 2, 2)
  b = zero_matrix(Z2, 2, 3)
  @test a in [a, b]
  @test a in [b, a]
  @test !(a in [b])
  @test a in keys(Dict(a => 1))
  @test !(a in keys(Dict(b => 1)))

  a = zero_matrix(Z2, 2, 2)
  b = zero_matrix(Z3, 2, 2)
  @test a in [a, b]
  @test a in [b, a]
  @test !(a in [b])
  @test a in keys(Dict(a => 1))
  @test !(a in keys(Dict(b => 1)))

  M = matrix(Z3, Z3.([-1 -2; -3 -4]))
  @test M == matrix(Z3, [-1 -2; -3 -4])
  @test M == matrix(Z3, 2, 2, [-1, -2, -3, -4])
end

@testset "fpMatrix.similar" begin
  Z13 = Native.GF(13)
  S = fpMatrixSpace(Z13, 2, 2)
  s = S(ZZRingElem(3))

  t = similar(s)
  @test t isa fpMatrix
  @test size(t) == size(s)

  t = similar(s, 2, 3)
  @test t isa fpMatrix
  @test size(t) == (2, 3)

  for (R, M) in ring_to_mat
    t = similar(s, R)
    @test size(t) == size(s)

    t = similar(s, R, 2, 3)
    @test size(t) == (2, 3)
  end
end

@testset "fpMatrix.is_zero_entry" begin
  R = Native.GF(13)
  M = matrix(R, [1 2 3;4 0 6;0 8 9])
  for i in 1:3, j in 1:3
    @test is_zero_entry(M, i, j) == is_zero(M[i, j])
  end
end

@testset "fpMatrix.is_zero_row" begin
  R = Native.GF(13)
  M = matrix(R, [1 2 3;4 0 6;0 8 9;0 0 0])
  for i in 1:nrows(M)
    @test is_zero_row(M, i) == all(j -> is_zero(M[i, j]), 1:ncols(M))
  end
end

@testset "fpMatrix.printing" begin
  Z2 = Native.GF(2)
  R = fpMatrixSpace(Z2, 2, 2)

  a = R(1)

  # test that default Julia printing is not used
  @test !occursin(string(typeof(a)), string(a))
end

@testset "fpMatrix.manipulation" begin
  Z11 = Native.GF(11)
  R = fpMatrixSpace(Z11, 2, 2)
  Z23 = Native.GF(23)
  S = fpMatrixSpace(Z23, 2, 2)

  ar = [ 1 2; 3 4]

  a = R(ar)
  aa = S(ar)

  @test nrows(a) == 2
  @test ncols(a) == 2

  b = deepcopy(a)

  c = R([ 1 3; 2 4])

  @test a[1,1] == Z11(1)

  a[1,1] = UInt(2)

  @test a[1,1] == Z11(2)
  @test_throws BoundsError a[0,-1] = Z11(2)

  a[2,1] = ZZ(3)

  @test a[2,1] == Z11(ZZ(3))
  @test_throws BoundsError a[-10,-10] = ZZ(3)

  a[2,2] = Z11(4)

  @test a[2,2] == Z11(4)
  @test_throws BoundsError a[-2,2] = Z11(4)

  a[1,2] = 5

  @test a[1,2] == Z11(5)
  @test_throws BoundsError a[-2,2] = 5

  @test a != b

  d = one(R)

  @test isa(d, fpMatrix)

  e = zero(R)

  @test isa(e, fpMatrix)

  @test iszero(e)

  @test_throws DomainError one(matrix_space(Native.GF(2), 1, 2))

  @test is_square(a)

  @test a == a
  @test a == deepcopy(a)
  @test a != aa

  @test transpose(b) == c

  transpose!(b)

  @test b == c

  @test transpose(matrix_space(Z11,1,2)([ 1 2; ])) ==
  matrix_space(Z11,2,1)(reshape([ 1 ; 2],2,1))

  @test_throws ErrorConstrDimMismatch transpose!(R([ 1 2 ;]))

  C = Z23[1 2 3; 4 5 6; 7 8 9]
  C[3, :] = Z23[7 7 7]
  @test C == Z23[1 2 3; 4 5 6; 7 7 7]

  C[:, 3] = Z23[5; 5; 5]
  @test C == Z23[1 2 5; 4 5 5; 7 7 5]

  C[1:2, 2:3] = Z23[3 3; 3 3]
  @test C == Z23[1 3 3; 4 3 3; 7 7 5]

  @test_throws DimensionMismatch C[1:2, 2:3] = Z23[3 3]
  @test_throws BoundsError C[1:2, 3:4] = Z23[3 3; 3 3]
end

@testset "fpMatrix.unary_ops" begin
  Z17 = Native.GF(17)

  R = matrix_space(Z17, 3, 4)
  RR = matrix_space(Z17, 4, 3)
  Z2 = Native.GF(2)
  S = matrix_space(Z2, 3, 4)

  a = R([ 1 2 3 1; 3 2 1 2; 1 3 2 0])

  b = R([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])

  c = R()

  d = -b

  @test d == R([ 15 16 0 16; 0 0 0 0; 0 16 15 0])
end

@testset "fpMatrix.binary_ops" begin
  Z17 = Native.GF(17)

  R = matrix_space(Z17, 3, 4)
  a = R([ 1 2 3 1; 3 2 1 2; 1 3 2 0])
  b = R([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])
  c = R()
  d = a + b
  @test d == R([3 3 3 2; 3 2 1 2; 1 4 4 0])

  d = a - b
  @test d == R([ 16 1 3 0; 3 2 1 2; 1 2 0 0 ])

  d = a*transpose(a)
  @test d == matrix_space(Z17, 3, 3)([15 12 13; 12 1 11; 13 11 14])

  d = transpose(a)*a
  @test d == matrix_space(Z17, 4, 4)([11 11 8 7; 11 0 14 6; 8 14 14 5; 7 6 5 5])

  a = matrix(Z17, [1 2; 3 4])
  @test mul!(Vector{UInt}(undef, 2), a, UInt[1, 2]) == [Z17(5), Z17(11)]
  @test mul!(Vector{UInt}(undef, 2), UInt[1, 2], a) == [Z17(7), Z17(10)]
end

@testset "fpMatrix.row_col_swapping" begin
  R, = residue_field(ZZ, 17)

  a = matrix(R, [1 2; 3 4; 5 6])

  @test swap_rows(a, 1, 3) == matrix(R, [5 6; 3 4; 1 2])

  swap_rows!(a, 2, 3)

  @test a == matrix(R, [1 2; 5 6; 3 4])

  @test swap_cols(a, 1, 2) == matrix(R, [2 1; 6 5; 4 3])

  swap_cols!(a, 2, 1)

  @test a == matrix(R, [2 1; 6 5; 4 3])

  a = matrix(R, [1 2; 3 4])
  @test reverse_rows(a) == matrix(R, [3 4; 1 2])
  reverse_rows!(a)
  @test a == matrix(R, [3 4; 1 2])

  a = matrix(R, [1 2; 3 4])
  @test reverse_cols(a) == matrix(R, [2 1; 4 3])
  reverse_cols!(a)
  @test a == matrix(R, [2 1; 4 3])

  a = matrix(R, [1 2 3; 3 4 5; 5 6 7])

  @test reverse_rows(a) == matrix(R, [5 6 7; 3 4 5; 1 2 3])
  reverse_rows!(a)
  @test a == matrix(R, [5 6 7; 3 4 5; 1 2 3])

  a = matrix(R, [1 2 3; 3 4 5; 5 6 7])
  @test reverse_cols(a) == matrix(R, [3 2 1; 5 4 3; 7 6 5])
  reverse_cols!(a)
  @test a == matrix(R, [3 2 1; 5 4 3; 7 6 5])
end

@testset "fpMatrix.adhoc_binary" begin
  Z17 = Native.GF(17)

  R = matrix_space(Z17, 3, 4)
  Z2 = Native.GF(2)

  a = R([ 1 2 3 1; 3 2 1 2; 1 3 2 0])

  b = R([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])

  c = R()
  d = UInt(2)*a
  dd = a*UInt(2)

  @test d == R([ 2 4 6 2; 6 4 2 4; 2 6 4 0])
  @test dd == R([ 2 4 6 2; 6 4 2 4; 2 6 4 0])


  d = 2*a
  dd = a*2

  @test d == R([ 2 4 6 2; 6 4 2 4; 2 6 4 0])
  @test dd == R([ 2 4 6 2; 6 4 2 4; 2 6 4 0])

  d = ZZ(2)*a
  dd = a*ZZ(2)

  @test d == R([ 2 4 6 2; 6 4 2 4; 2 6 4 0])
  @test dd == R([ 2 4 6 2; 6 4 2 4; 2 6 4 0])

  d = Z17(2)*a
  dd = a*Z17(2)

  @test d == R([ 2 4 6 2; 6 4 2 4; 2 6 4 0])
  @test dd == R([ 2 4 6 2; 6 4 2 4; 2 6 4 0])

  @test_throws ErrorException Z2(1)*a
end

@testset "fpMatrix.comparison" begin
  Z17 = Native.GF(17)

  R = matrix_space(Z17, 3, 4)

  a = R([ 1 2 3 1; 3 2 1 2; 1 3 2 0])

  @test a == a

  @test deepcopy(a) == a

  @test a != R([0 1 3 1; 2 1 4 2; 1 1 1 1])
end

@testset "fpMatrix.adhoc_comparison" begin
  Z17 = Native.GF(17)

  R = matrix_space(Z17, 3, 4)

  @test R(5) == 5
  @test R(5) == ZZRingElem(5)
  @test R(5) == Z17(5)

  @test 5 == R(5)
  @test ZZRingElem(5) == R(5)
  @test Z17(5) == R(5)
end

@testset "fpMatrix.powering" begin
  Z17 = Native.GF(17)

  R = matrix_space(Z17, 3, 4)

  a = R([ 1 2 3 1; 3 2 1 2; 1 3 2 0])

  f = a*transpose(a)

  g = f^1000

  @test g == matrix_space(Z17, 3, 3)([1 2 2; 2 13 12; 2 12 15])

  g = f^ZZ(1000)

  @test g == matrix_space(Z17, 3, 3)([1 2 2; 2 13 12; 2 12 15])

  @test_throws ErrorException f^(ZZ(2)^1000)
end

@testset "fpMatrix.row_echelon_form" begin
  for iters = 1:100
    m = rand(0:100)
    n = rand(0:100)
    S = matrix_space(Native.GF(7), m, n)
    M = rand(S)
    r, N = rref(M)

    @test is_rref(N)
  end

  Z17 = Native.GF(17)
  R = matrix_space(Z17, 3, 4)
  RR = matrix_space(Z17, 4, 3)
  Z2 = Native.GF(2)
  S = matrix_space(Z2, 3, 4)

  a = R([ 1 2 3 1; 3 2 1 2; 1 3 2 0])

  b = R([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])

  b = transpose(b)

  c = a*transpose(a)

  r, d = rref(a)

  @test d == R([ 1 0 0 8; 0 1 0 15; 0 0 1 16])
  @test r == 3

  r = rref!(a)

  @test a == R([ 1 0 0 8; 0 1 0 15; 0 0 1 16])
  @test r == 3

  r, d = rref(b)

  @test d == parent(b)([ 1 0 0 ; 0 0 1; 0 0 0; 0 0 0])
  @test r == 2
end

@testset "fpMatrix.howell_form" begin
  Z17 = Native.GF(17)
  R = matrix_space(Z17, 3, 3)

  a = R([4 1 0; 0 0 5; 0 0 0 ])

  b = R([8 5 5; 0 9 8; 0 0 10])

  c = R([1 13 0; 0 0 1; 0 0 0])

  d = R([4 0 0; 0 0 1; 0 0 0])

  @test howell_form(a) == c
  @test howell_form(b) == 1
  @test strong_echelon_form(d) == R([1 0 0; 0 0 0; 0 0 1])

  a = matrix(Z17, [0 1 0 0; 0 0 1 0; 0 0 0 1; 0 0 0 0])
  @test strong_echelon_form(a) == matrix(Z17, [0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
end

@testset "fpMatrix.trace_det" begin
  Z17 = Native.GF(17)
  R = matrix_space(Z17, 3, 4)
  RR = matrix_space(Z17, 4, 3)
  Z2 = Native.GF(2)
  S = matrix_space(Z2, 3, 4)

  a = R([ 1 2 3 1; 3 2 1 2; 1 3 2 0])

  aa = matrix_space(Z17,3,3)([ 1 2 3; 3 2 1; 1 1 2])

  b = R([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])

  a = transpose(a)*a

  c = tr(a)

  @test c == Z17(13)

  @test_throws ErrorException tr(b)

  c = det(a)

  @test c == zero(Z17)

  @test_throws ErrorException det(b)

  c = det(aa)

  @test c == Z17(13)

  a = R([ 1 2 3 1; 3 2 1 2; 1 3 2 0])
end

@testset "fpMatrix.rank" begin
  Z17 = Native.GF(17)
  R = matrix_space(Z17, 3, 4)
  RR = matrix_space(Z17, 4, 3)
  Z2 = Native.GF(2)
  S = matrix_space(Z2, 3, 4)

  a = R([ 1 2 3 1; 3 2 1 2; 1 3 2 0])

  aa = matrix_space(Z17,3,3)([ 1 2 3; 3 2 1; 1 1 2])

  b = R([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])

  c = rank(a)

  @test c == 3

  c = rank(aa)

  @test c == 3

  c = rank(b)

  @test c == 2
end

@testset "fpMatrix.inv" begin
  Z17 = Native.GF(17)
  R = matrix_space(Z17, 3, 4)
  RR = matrix_space(Z17, 4, 3)
  Z2 = Native.GF(2)
  S = matrix_space(Z2, 3, 4)

  a = R([ 1 2 3 1; 3 2 1 2; 1 3 2 0])

  aa = matrix_space(Z17,3,3)([ 1 2 3; 3 2 1; 1 1 2])

  b = R([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])

  c = inv(aa)

  @test c == parent(aa)([12 13 1; 14 13 15; 4 4 1])

  @test_throws ErrorException inv(a)

  @test_throws ErrorException inv(transpose(a)*a)
end

@testset "fpMatrix.solve" begin
  Z17 = Native.GF(17)
  R = matrix_space(Z17, 3, 3)
  S = matrix_space(Z17, 3, 4)

  a = R([ 1 2 3 ; 3 2 1 ; 0 0 2 ])

  @test AbstractAlgebra.Solve.matrix_normal_form_type(Z17) === AbstractAlgebra.Solve.LUTrait()
  @test AbstractAlgebra.Solve.matrix_normal_form_type(a) === AbstractAlgebra.Solve.LUTrait()

  b = S([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])

  c = a*b

  d = solve(a, c, side = :right)

  @test d == b

  a = zero(R)

  @test_throws ArgumentError solve(a, c, side = :right)

  for i in 1:10
    m = rand(0:10)
    n = rand(0:10)
    k = rand(0:10)

    M = matrix_space(Z17, n, k)
    N = matrix_space(Z17, n, m)

    A = rand(M)
    B = rand(N)

    fl, X, K = can_solve_with_solution_and_kernel(A, B, side = :right)

    if fl
      @test A * X == B
      @test is_zero(A*K)
      @test rank(A) + ncols(K) == ncols(A)
    end
  end

  A = matrix(Z17, 2, 2, [1, 2, 2, 5])
  B = matrix(Z17, 2, 1, [1, 2])
  fl, X = can_solve_with_solution(A, B, side = :right)
  @test fl
  @test A * X == B
  @test can_solve(A, B, side = :right)

  A = matrix(Z17, 2, 2, [1, 2, 2, 4])
  B = matrix(Z17, 2, 1, [1, 2])
  fl, X = can_solve_with_solution(A, B, side = :right)
  @test fl
  @test A * X == B
  @test can_solve(A, B, side = :right)

  A = matrix(Z17, 2, 2, [1, 2, 2, 4])
  B = matrix(Z17, 2, 1, [1, 3])
  fl, X = can_solve_with_solution(A, B, side = :right)
  @test !fl
  @test !can_solve(A, B, side = :right)

  A = zero_matrix(Z17, 2, 3)
  B = identity_matrix(Z17, 3)
  @test_throws ErrorException can_solve_with_solution(A, B, side = :right)

  A = transpose(matrix(Z17, 2, 2, [1, 2, 2, 5]))
  B = transpose(matrix(Z17, 2, 1, [1, 2]))
  fl, X = can_solve_with_solution(A, B)
  @test fl
  @test X * A == B
  @test can_solve(A, B)

  A = transpose(matrix(Z17, 2, 2, [1, 2, 2, 4]))
  B = transpose(matrix(Z17, 2, 1, [1, 2]))
  fl, X = can_solve_with_solution(A, B)
  @test fl
  @test X * A == B
  @test can_solve(A, B)

  A = transpose(matrix(Z17, 2, 2, [1, 2, 2, 4]))
  B = transpose(matrix(Z17, 2, 1, [1, 3]))
  fl, X = can_solve_with_solution(A, B)
  @test !fl
  @test !can_solve(A, B)

  A = transpose(zero_matrix(Z17, 2, 3))
  B = transpose(identity_matrix(Z17, 3))
  @test_throws ErrorException can_solve_with_solution(A, B)

  @test_throws ArgumentError can_solve_with_solution(A, B, side = :garbage)
  @test_throws ArgumentError can_solve(A, B, side = :garbage)

  A = matrix(Z17, [ 1 2 3 ; 4 5 6 ])
  K = @inferred kernel(A, side = :right)
  @test is_zero(A*K)
  @test ncols(K) == 1

  K = @inferred kernel(A)
  @test is_zero(K*A)
  @test nrows(K) == 0

  A = transpose(A)
  K = @inferred kernel(A)
  @test is_zero(K*A)
  @test nrows(K) == 1
end

@testset "fpMatrix.solve_context" begin
  F = Native.GF(101)
  A = matrix(F, [1 2 3 4 5; 0 0 8 9 10; 0 0 0 14 15])
  C = solve_init(A)
  @test C isa AbstractAlgebra.solve_context_type(F)
  @test C isa AbstractAlgebra.solve_context_type(A)

  @test AbstractAlgebra.Solve.matrix_normal_form_type(C) === AbstractAlgebra.Solve.LUTrait()
  @test C isa AbstractAlgebra.solve_context_type(AbstractAlgebra.Solve.LUTrait(), fpFieldElem)
  @test C isa AbstractAlgebra.solve_context_type(AbstractAlgebra.Solve.LUTrait(), F())
  @test C isa AbstractAlgebra.solve_context_type(AbstractAlgebra.Solve.LUTrait(), Nemo.fpField)
  @test C isa AbstractAlgebra.solve_context_type(AbstractAlgebra.Solve.LUTrait(), F)
  @test C isa AbstractAlgebra.solve_context_type(AbstractAlgebra.Solve.LUTrait(), typeof(A))
  @test C isa AbstractAlgebra.solve_context_type(AbstractAlgebra.Solve.LUTrait(), A)

  @test_throws ErrorException solve(C, [ F(1) ])
  @test_throws ErrorException solve(C, [ F(1) ], side = :right)
  @test_throws ErrorException solve(C, matrix(F, 1, 1, [ F(1) ]))
  @test_throws ErrorException solve(C, matrix(F, 1, 1, [ F(1) ]), side = :right)
  @test_throws ArgumentError solve(C, [ F(1), F(2), F(3) ], side = :test)
  @test_throws ArgumentError solve(C, matrix(F, 3, 1, [ F(1), F(2), F(3) ]), side = :test)

  for b in [ [ F(1), F(2), F(3) ],
            matrix(F, 3, 1, [ F(1), F(2), F(3) ]),
            matrix(F, 3, 2, [ F(1), F(2), F(3), F(4), F(5), F(6) ]) ]
    @test @inferred can_solve(C, b, side = :right)
    x = @inferred solve(C, b, side = :right)
    @test A*x == b
    fl, x = @inferred can_solve_with_solution(C, b, side = :right)
    @test fl
    @test A*x == b
    fl, x, K = @inferred can_solve_with_solution_and_kernel(C, b, side = :right)
    @test fl
    @test A*x == b
    @test is_zero(A*K)
    @test ncols(K) == 2
    K = @inferred kernel(C, side = :right)
    @test is_zero(A*K)
    @test ncols(K) == 2
  end

  for b in [ [ F(1), F(1), F(1), F(1), F(1) ],
            matrix(F, 1, 5, [ F(1), F(1), F(1), F(1), F(1) ]),
            matrix(F, 2, 5, [ F(1), F(1), F(1), F(1), F(1),
                             F(1), F(1), F(1), F(1), F(1) ]) ]
    @test_throws ArgumentError solve(C, b)
    @test @inferred !can_solve(C, b)
    fl, x = @inferred can_solve_with_solution(C, b)
    @test !fl
    fl, x, K = @inferred can_solve_with_solution_and_kernel(C, b)
    @test !fl
  end

  for b in [ [ F(1), F(2), F(3), F(4), F(5) ],
            matrix(F, 1, 5, [ F(1), F(2), F(3), F(4), F(5)]),
            matrix(F, 2, 5, [ F(1), F(2), F(3), F(4), F(5),
                             F(0), F(0), F(8), F(9), F(10) ]) ]
    @test @inferred can_solve(C, b)
    x = @inferred solve(C, b)
    @test x*A == b
    fl, x = @inferred can_solve_with_solution(C, b)
    @test fl
    @test x*A == b
    fl, x, K = @inferred can_solve_with_solution_and_kernel(C, b)
    @test fl
    @test x*A == b
    @test is_zero(K*A)
    @test nrows(K) == 0
    K = @inferred kernel(C)
    @test is_zero(K*A)
    @test nrows(K) == 0
  end

  N = zero_matrix(F, 2, 1)
  C = solve_init(N)
  b = zeros(F, 2)
  fl, x, K = @inferred can_solve_with_solution_and_kernel(C, b, side = :right)
  @test fl
  @test N*x == b
  @test K == identity_matrix(F, 1)
  K = @inferred kernel(C, side = :right)
  @test K == identity_matrix(F, 1)

  N = zero_matrix(F, 1, 2)
  C = solve_init(N)
  b = zeros(F, 1)
  fl, x, K = @inferred can_solve_with_solution_and_kernel(C, b, side = :right)
  @test fl
  @test N*x == b
  @test K == identity_matrix(F, 2) || K == swap_cols!(identity_matrix(F, 2), 1, 2)
  K = @inferred kernel(C, side = :right)
  @test K == identity_matrix(F, 2) || K == swap_cols!(identity_matrix(F, 2), 1, 2)
end

@testset "fpMatrix.kernel" begin
  Z17 = Native.GF(17)
  A = matrix(Z17, [ 1 2 3 ; 4 5 6 ])
  K = @inferred kernel(A, side = :right)
  @test is_zero(A*K)
  @test ncols(K) == 1

  K = @inferred kernel(A)
  @test is_zero(K*A)
  @test nrows(K) == 0

  A = transpose(A)
  K = @inferred kernel(A)
  @test is_zero(K*A)
  @test nrows(K) == 1
end

@testset "fpMatrix.lu" begin

  Z17 = Native.GF(17)
  R = matrix_space(Z17, 3, 3)
  S = matrix_space(Z17, 3, 4)

  a = R([ 1 2 3 ; 3 2 1 ; 0 0 2 ])

  b = S([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])

  r, P, l, u = lu(a)

  @test l*u == P*a

  r, P, l, u = lu(b)

  @test l*u == S([ 2 1 0 1; 0 1 2 0; 0 0 0 0])

  @test l*u == P*b

  c = matrix(Z17, 6, 3, [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])

  r, P, l, u = lu(c)

  @test r == 3
  @test l*u == P*c
end

@testset "fpMatrix.swap_rows" begin
  Z17 = Native.GF(17)

  A = matrix(Z17, 5, 1, [1, 2, 3, 4, 5])

  B = swap_rows(A, 3, 4)
  @test B == matrix(Z17, 5, 1, [1, 2, 4, 3, 5])

  swap_rows!(A, 3, 4)
  @test A == matrix(Z17, 5, 1, [1, 2, 4, 3, 5])

  @test_throws BoundsError swap_rows(A, 0, 5)
  @test_throws BoundsError swap_rows(A, 4, 6)
end
@testset "fpMatrix.view" begin
  Z17 = Native.GF(17)
  R = matrix_space(Z17, 3, 3)
  S = matrix_space(Z17, 3, 4)

  a = R([ 1 2 3 ; 3 2 1 ; 0 0 2 ])

  b = S([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])

  t = view(a, 1:3, 1:3)

  @test t == a

  @test view(a, 1:3, 1:3) == view(a, 1:3, 1:3)
  @test view(a, 1:3, 1:3) == sub(a, 1, 1, 3, 3)
  @test view(a, 1:3, 1:3) == sub(a, 1:3, 1:3)

  t = view(a, 1:2, 1:2)

  @test t == Z17[1 2; 3 2]

  t = view(a, 2:3, 2:2)

  @test t == transpose(Z17[2 0])

  @test view(a, 2:3, 2:2) == view(a, 2:3,  2:2)
  @test view(a, 2:3, 2:2) == sub(a, 2, 2, 3, 2)
  @test view(a, 2:3, 2:2) == sub(a, 2:3, 2:2)

  @test_throws BoundsError view(a, 3:5, 3:5)

  S = matrix_space(Z17, 3, 3)

  A = S([1 2 3; 4 5 6; 7 8 9])

  B = @inferred view(A, 1:2, 1:2)

  @test typeof(B) == fpMatrix
  @test B == matrix_space(Z17, 2, 2)([1 2; 4 5])

  B[1, 1] = 10
  @test A[1, 1] == 10

  C = @inferred view(B, 1:2, 1:2)

  @test typeof(C) == fpMatrix
  @test C == matrix_space(Z17, 2, 2)([10 2; 4 5])

  C[1, 1] = 20
  @test B[1, 1] == 20
  @test A[1, 1] == 20

  A = 0
  GC.gc()
  @test B[1, 1] == 20

  A = S([1 2 3; 4 5 6; 7 8 9])
  v = @view A[2, :]
  @test v isa AbstractVector{elem_type(Z17)}
  @test length(v) == 3
  @test v[1] == 4
  @test collect(v) == [4, 5, 6]
  v[2] = 7
  @test A == S([1 2 3; 4 7 6; 7 8 9])
  A = S([1 2 3; 4 5 6; 7 8 9])
  v = @view A[:, 3]
  @test v isa AbstractVector{elem_type(Z17)}
  @test length(v) == 3
  @test v[3] == 9
  @test collect(v) == [3, 6, 9]
  v[1] = 1
  @test A == S([1 2 1; 4 5 6; 7 8 9])
end

@testset "fpMatrix.sub" begin
  Z17 = Native.GF(17)
  S = matrix_space(Z17, 3, 3)

  A = S([1 2 3; 4 5 6; 7 8 9])

  B = @inferred sub(A, 1, 1, 2, 2)

  @test typeof(B) == fpMatrix
  @test B == matrix_space(Z17, 2, 2)([1 2; 4 5])

  B[1, 1] = 10
  @test A == S([1 2 3; 4 5 6; 7 8 9])

  C = @inferred sub(B, 1:2, 1:2)

  @test typeof(C) == fpMatrix
  @test C == matrix_space(Z17, 2, 2)([10 2; 4 5])

  C[1, 1] = 20
  @test B == matrix_space(Z17, 2, 2)([10 2; 4 5])
  @test A == S([1 2 3; 4 5 6; 7 8 9])
end

@testset "fpMatrix.concatenation" begin
  Z17 = Native.GF(17)
  R = matrix_space(Z17, 3, 3)
  S = matrix_space(Z17, 3, 4)

  a = R([ 1 2 3 ; 3 2 1 ; 0 0 2 ])

  b = S([ 2 1 0 1; 0 0 0 0; 0 1 2 0 ])

  c = hcat(a,a)

  @test c == matrix_space(Z17, 3, 6)([1, 2, 3, 1, 2, 3,
                                      3, 2, 1, 3, 2, 1,
                                      0, 0, 2, 0, 0, 2])

  c = hcat(a,b)

  @test c == matrix_space(Z17, 3, 7)([1, 2, 3, 2, 1, 0, 1,
                                      3, 2, 1, 0, 0, 0, 0,
                                      0, 0, 2, 0, 1, 2, 0])

  @test_throws ErrorException c = hcat(a,transpose(b))

  c = vcat(a,transpose(b))

  @test c == matrix_space(Z17, 7, 3)([1, 2, 3,
                                      3, 2, 1,
                                      0, 0, 2,
                                      2, 0, 0,
                                      1, 0, 1,
                                      0, 0, 2,
                                      1, 0, 0])

  @test_throws ErrorException vcat(a,b)
end

@testset "fpMatrix.conversion" begin
  Z17 = Native.GF(17)
  R = matrix_space(Z17, 3, 3)
  S = matrix_space(ZZ, 3, 3)

  c = S()

  a = R([ 1 2 3 ; 3 2 1 ; 0 0 2 ])

  @test Array(a) == [Z17(1) Z17(2) Z17(3);
                     Z17(3) Z17(2) Z17(1);
                     Z17(0) Z17(0) Z17(2) ]

  b = lift(a)

  @test b == S([ 1 2 3; 3 2 1; 0 0 2])
  @test parent(b) == S

  lift!(c,a)

  @test c == S([ 1 2 3; 3 2 1; 0 0 2])
end

@testset "fpMatrix.charpoly" begin
  R = Native.GF(17)

  for dim = 0:5
    S = matrix_space(R, dim, dim)
    U, x = polynomial_ring(R, "x")

    for i = 1:10
      M = rand(S, -5:5)

      p1 = charpoly(U, M)
      p2 = charpoly_danilevsky!(U, M)

      @test p1 == p2
    end
  end
end

@testset "fpMatrix.rand" begin
  R = Native.GF(17)
  S = matrix_space(R, 3, 3)

  M = rand(S, 1:5)
  @test parent(M) == S

  for i=1:3, j=1:3
    @test M[i, j] in 1:5
  end

  M = rand(S)
  @test parent(M) == S
end

@testset "fpMatrix.add_one!" begin
  F = Native.GF(2)
  A = F[0 0; 0 0]
  Generic.add_one!(A, 1, 1)
  @test A == F[1 0; 0 0]
  # Make sure reduction works
  Generic.add_one!(A, 1, 1)
  @test A == F[0 0; 0 0]
  @test_throws BoundsError Generic.add_one!(A, 3, 1)
end

@testset "fpMatrix.unsafe" begin
  R = Native.GF(5)
  A = R[1 1; 0 1]
  b = R(2)
  C = similar(A)
  mul!(C, b, A)
  @test C == b * A
  mul!(C, A, b)
  @test C == b * A
end

