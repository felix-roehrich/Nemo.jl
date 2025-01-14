@testset "Poly.binary_ops_delayed_reduction" begin
   S, t = polynomial_ring(QQ, "t")
   K, a = number_field(t^3 + 3t + 1, "a")
   R, x = polynomial_ring(K, "x")

   for iter = 1:100
      f = rand(R, 0:10, -10:10)
      g = rand(R, 0:10, -10:10)
      h = rand(R, 0:10, -10:10)

      @test f*g == g*f
      @test f*(g + h) == f*g + f*h
      @test (f + g)*(f - g) == f*f - g*g
   end
end

@testset "Poly.truncation_delayed_reduction" begin
   S, t = polynomial_ring(QQ, "t")
   K, a = number_field(t^3 + 3t + 1, "a")
   R, x = polynomial_ring(K, "x")

   for iter = 1:300
      f = rand(R, 0:10, -10:10)
      g = rand(R, 0:10, -10:10)
      n = rand(0:20)

      @test truncate(f*g, n) == mullow(f, g, n)
   end
end

@testset "@polynomial_ring" begin
   # cf. AbstractAlgebra issue #274
   R, x = @polynomial_ring(ZZ, x)
   @test typeof(R) == ZZMPolyRing
   R, x = @polynomial_ring(QQ, x)
   @test typeof(R) == QQMPolyRing
end
