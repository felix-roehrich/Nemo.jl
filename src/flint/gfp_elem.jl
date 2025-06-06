###############################################################################
#
#   gfp_elem.jl : Nemo gfp_elem (integers modulo small n)
#
###############################################################################

###############################################################################
#
#   Type and parent object methods
#
###############################################################################

parent_type(::Type{fpFieldElem}) = fpField

elem_type(::Type{fpField}) = fpFieldElem

base_ring_type(::Type{fpField}) = typeof(Union{})

base_ring(a::fpField) = Union{}

parent(a::fpFieldElem) = a.parent

is_domain_type(::Type{fpFieldElem}) = true

###############################################################################
#
#   Basic manipulation
#
###############################################################################

function Base.hash(a::fpFieldElem, h::UInt)
  b = 0x749c75e438001387%UInt
  return xor(xor(hash(a.data), h), b)
end

data(a::fpFieldElem) = a.data

function coeff(x::fpFieldElem, n::Int)
  n < 0 && throw(DomainError(n, "Index must be non-negative"))
  n == 0 && return data(x)
  return UInt(0)
end


lift(a::fpFieldElem) = ZZRingElem(data(a))
lift(::ZZRing, x::fpFieldElem) = lift(x)

function zero(R::fpField)
  return fpFieldElem(UInt(0), R)
end

function one(R::fpField)
  return fpFieldElem(UInt(1), R)
end

iszero(a::fpFieldElem) = a.data == 0

isone(a::fpFieldElem) = a.data == 1

modulus(R::fpField) = R.n

function deepcopy_internal(a::fpFieldElem, dict::IdDict)
  R = parent(a)
  return fpFieldElem(deepcopy(a.data), R)
end

order(R::fpField) = ZZRingElem(R.n)

characteristic(R::fpField) = ZZRingElem(R.n)

degree(::fpField) = 1

###############################################################################
#
#   AbstractString I/O
#
###############################################################################

function show(io::IO, a::fpField)
  @show_name(io, a)
  @show_special(io, a)
  if is_terse(io)
    io = pretty(io)
    print(io, LowercaseOff(), "GF($(signed(widen(a.n))))")
  else
    print(io, "Finite field of characteristic ", signed(widen(a.n)))
  end
end

function expressify(a::fpFieldElem; context = nothing)
  return a.data
end

function show(io::IO, a::fpFieldElem)
  print(io, signed(widen(a.data)))
end

###############################################################################
#
#   Unary operations
#
###############################################################################

function -(x::fpFieldElem)
  if x.data == 0
    return deepcopy(x)
  else
    R = parent(x)
    return fpFieldElem(R.n - x.data, R)
  end
end

###############################################################################
#
#   Binary operations
#
###############################################################################

function +(x::fpFieldElem, y::fpFieldElem)
  check_parent(x, y)
  R = parent(x)
  n = modulus(R)
  d = x.data + y.data - n
  if d > x.data
    return fpFieldElem(d + n, R)
  else
    return fpFieldElem(d, R)
  end
end

function -(x::fpFieldElem, y::fpFieldElem)
  check_parent(x, y)
  R = parent(x)
  n = modulus(R)
  d = x.data - y.data
  if d > x.data
    return fpFieldElem(d + n, R)
  else
    return fpFieldElem(d, R)
  end
end

function *(x::fpFieldElem, y::fpFieldElem)
  check_parent(x, y)
  R = parent(x)
  d = mulmod(x.data, y.data, R.n, R.ninv)
  return fpFieldElem(d, R)
end

###############################################################################
#
#   Ad hoc binary operators
#
###############################################################################

function *(x::Integer, y::fpFieldElem)
  R = parent(y)
  return R(widen(x)*signed(widen(y.data)))
end

*(x::fpFieldElem, y::Integer) = y*x

function *(x::Int, y::fpFieldElem)
  R = parent(y)
  if x < 0
    d = mulmod(reinterpret(UInt, -x), y.data, R.n, R.ninv)
    return -fpFieldElem(d, R)
  else
    d = mulmod(UInt(x), y.data, R.n, R.ninv)
    return fpFieldElem(d, R)
  end
end

*(x::fpFieldElem, y::Int) = y*x

function *(x::UInt, y::fpFieldElem)
  R = parent(y)
  d = mulmod(x, y.data, R.n, R.ninv)
  return fpFieldElem(d, R)
end

*(x::fpFieldElem, y::UInt) = y*x

+(x::fpFieldElem, y::Integer) = x + parent(x)(y)

+(x::Integer, y::fpFieldElem) = y + x

-(x::fpFieldElem, y::Integer) = x - parent(x)(y)

-(x::Integer, y::fpFieldElem) = parent(y)(x) - y

*(x::ZZRingElem, y::fpFieldElem) = BigInt(x)*y

*(x::fpFieldElem, y::ZZRingElem) = y*x

+(x::fpFieldElem, y::ZZRingElem) = x + parent(x)(y)

+(x::ZZRingElem, y::fpFieldElem) = y + x

-(x::fpFieldElem, y::ZZRingElem) = x - parent(x)(y)

-(x::ZZRingElem, y::fpFieldElem) = parent(y)(x) - y

###############################################################################
#
#   Powering
#
###############################################################################

function ^(x::fpFieldElem, y::Int)
  R = parent(x)
  if y < 0
    x = inv(x)
    y = -y
  end
  d = @ccall libflint.n_powmod2_preinv(UInt(x.data)::UInt, y::Int, R.n::UInt, R.ninv::UInt)::UInt
  return fpFieldElem(d, R)
end

###############################################################################
#
#   Comparison
#
###############################################################################

function ==(x::fpFieldElem, y::fpFieldElem)
  check_parent(x, y)
  return x.data == y.data
end

###############################################################################
#
#   Ad hoc comparison
#
###############################################################################

==(x::fpFieldElem, y::Integer) = x == parent(x)(y)

==(x::Integer, y::fpFieldElem) = parent(y)(x) == y

==(x::fpFieldElem, y::ZZRingElem) = x == parent(x)(y)

==(x::ZZRingElem, y::fpFieldElem) = parent(y)(x) == y

###############################################################################
#
#   Inversion
#
###############################################################################

function inv(x::fpFieldElem)
  R = parent(x)
  iszero(x) && throw(DivideError())
  xinv = @ccall libflint.n_invmod(x.data::UInt, R.n::UInt)::UInt
  return fpFieldElem(xinv, R)
end

###############################################################################
#
#   Exact division
#
###############################################################################

function divexact(x::fpFieldElem, y::fpFieldElem; check::Bool=true)
  check_parent(x, y)
  y == 0 && throw(DivideError())
  R = parent(x)
  yinv = @ccall libflint.n_invmod(y.data::UInt, R.n::UInt)::UInt
  d = mulmod(x.data, yinv, R.n, R.ninv)
  return fpFieldElem(d, R)
end

function divides(a::fpFieldElem, b::fpFieldElem)
  check_parent(a, b)
  if iszero(a)
    return true, a
  end
  if iszero(b)
    return false, a
  end
  return true, divexact(a, b)
end

###############################################################################
#
#   Square root
#
###############################################################################

function Base.sqrt(a::fpFieldElem; check::Bool=true)
  R = parent(a)
  if iszero(a)
    return zero(R)
  end
  r = @ccall libflint.n_sqrtmod(a.data::UInt, R.n::UInt)::UInt
  check && iszero(r) && error("Not a square in sqrt")
  return fpFieldElem(r, R)
end

function is_square(a::fpFieldElem)
  R = parent(a)
  if iszero(a) || R.n == 2
    return true
  end
  r = @ccall libflint.n_jacobi(a.data::UInt, R.n::UInt)::Cint
  return isone(r)
end

function is_square_with_sqrt(a::fpFieldElem)
  R = parent(a)
  if iszero(a) || R.n == 2
    return true, a
  end
  r = @ccall libflint.n_sqrtmod(a.data::UInt, R.n::UInt)::UInt
  if iszero(r)
    return false, zero(R)
  end
  return true, fpFieldElem(r, R)
end

###############################################################################
#
#   Unsafe functions
#
###############################################################################

# Since this data type is immutable, we can not do better for the unsafe ops
# than their default implementations.

###############################################################################
#
#   Random functions
#
###############################################################################

# define rand(::fpField)

Random.Sampler(::Type{RNG}, R::fpField, n::Random.Repetition) where {RNG<:AbstractRNG} =
Random.SamplerSimple(R, Random.Sampler(RNG, UInt(0):R.n - 1, n))

rand(rng::AbstractRNG, R::Random.SamplerSimple{fpField}) =
fpFieldElem(rand(rng, R.data), R[])

# define rand(make(::fpField, arr)), where arr is any abstract array with integer or ZZRingElem entries

RandomExtensions.maketype(R::fpField, _) = elem_type(R)

rand(rng::AbstractRNG, sp::SamplerTrivial{<:Make2{fpFieldElem,fpField,<:AbstractArray{<:IntegerUnion}}}) =
sp[][1](rand(rng, sp[][2]))

# define rand(::fpField, arr), where arr is any abstract array with integer or ZZRingElem entries

rand(rng::AbstractRNG, R::fpField, b::AbstractArray) = rand(rng, make(R, b))

rand(R::fpField, b::AbstractArray) = rand(Random.default_rng(), R, b)

###############################################################################
#
#   Promotions
#
###############################################################################

promote_rule(::Type{fpFieldElem}, ::Type{T}) where T <: Integer = fpFieldElem

promote_rule(::Type{fpFieldElem}, ::Type{ZZRingElem}) = fpFieldElem

###############################################################################
#
#   Parent object call overload
#
###############################################################################

function (R::fpField)()
  return fpFieldElem(UInt(0), R)
end

function (R::fpField)(a::Integer)
  n = R.n
  d = a%signed(widen(n))
  if d < 0
    d += n
  end
  return fpFieldElem(UInt(d), R)
end

function (R::fpField)(a::Int)
  n = R.n
  ninv = R.ninv
  if reinterpret(Int, n) > 0 && a < 0
    a %= Int(n)
  end
  d = reinterpret(UInt, a)
  if a < 0
    d += n
  end
  if d >= n
    d = @ccall libflint.n_mod2_preinv(d::UInt, n::UInt, ninv::UInt)::UInt
  end
  return fpFieldElem(d, R)
end

function (R::fpField)(a::UInt)
  n = R.n
  ninv = R.ninv
  a = @ccall libflint.n_mod2_preinv(a::UInt, n::UInt, ninv::UInt)::UInt
  return fpFieldElem(a, R)
end

function (R::fpField)(a::ZZRingElem)
  d = @ccall libflint.fmpz_fdiv_ui(a::Ref{ZZRingElem}, R.n::UInt)::UInt
  return fpFieldElem(d, R)
end

function (R::fpField)(a::QQFieldElem)
  num = numerator(a, false)
  den = denominator(a, false)
  n = @ccall libflint.fmpz_fdiv_ui(num::Ref{ZZRingElem}, R.n::UInt)::UInt
  d = @ccall libflint.fmpz_fdiv_ui(den::Ref{ZZRingElem}, R.n::UInt)::UInt
  V = [UInt(0)]
  g = @ccall libflint.n_gcdinv(V::Ptr{UInt}, d::UInt, R.n::UInt)::UInt
  g != 1 && error("Unable to coerce")
  return R(n)*R(V[1])
end

function (R::fpField)(a::Union{fpFieldElem, zzModRingElem, FpFieldElem, ZZModRingElem})
  S = parent(a)
  if S === R
    return a
  else
    is_divisible_by(modulus(S), modulus(R)) || error("incompatible parents")
    return R(data(a))
  end
end

function (R::fpField)(a::Vector{<:IntegerUnion})
  is_one(length(a)) || error("Coercion impossible")
  return R(a[1])
end

###############################################################################
#
#   Representation matrix
#
###############################################################################

function representation_matrix(a::fpFieldElem)
  return matrix(parent(a), 1, 1, [a])
end

###############################################################################
#
#   Iterator interface
#
###############################################################################

Base.iterate(R::fpField) = (zero(R), zero(UInt))

function Base.iterate(R::fpField, st::UInt)
  if st == R.n - 1
    return nothing
  end

  return R(st + 1), st + 1
end

Base.IteratorEltype(::Type{fpField}) = Base.HasEltype()
Base.eltype(::Type{fpField}) = fpFieldElem

Base.IteratorSize(::Type{fpField}) = Base.HasLength()
Base.length(R::fpField) = R.n
