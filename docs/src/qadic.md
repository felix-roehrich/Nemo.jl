```@meta
CurrentModule = Nemo
DocTestSetup = quote
    using Nemo
end
```

# Qadics

Q-adic fields, that is, unramified extensions of p-adic fields, are provided in
Nemo by FLINT. This allows construction of $q$-adic fields for any prime power
$q$.

Q-adic fields are constructed using the `qadic_field` function.

The types of $q$-adic fields in Nemo are given in the following table, along
with the libraries that provide them and the associated types of the parent
objects.

 Library | Field            | Element type | Parent type
---------|----------------|----------------|---------------------
FLINT    | $\mathbb{Q}_q$ | `QadicFieldElem`        | `QadicField`

All the $q$-adic field types belong to the `Field` abstract type and the
$q$-adic field element types belong to the `FieldElem` abstract type.

## P-adic functionality

Q-adic fields in Nemo provide all the functionality described in AbstractAlgebra
for fields:.

<https://nemocas.github.io/AbstractAlgebra.jl/stable/field>

Below, we document all the additional function that is provide by Nemo for q-adic
fields.

### Constructors

In order to construct $q$-adic field elements in Nemo, one must first construct
the $q$-adic field itself. This is accomplished with one of the following
constructors.

```@docs
qadic_field
unramified_extension
```

Here are some examples of creating $q$-adic fields and making use of the
resulting parent objects to coerce various elements into those fields.

**Examples**

```jldoctest
julia> R, p = qadic_field(7, 1, precision = 30);

julia> S, _ = qadic_field(ZZ(65537), 1, precision = 30);

julia> a = R()
0

julia> b = S(1)
65537^0 + O(65537^30)

julia> c = S(ZZ(123))
123*65537^0 + O(65537^30)

julia> d = R(ZZ(1)//7^2)
7^-2 + O(7^28)
```

### Big-oh notation

Elements of p-adic fields can  be constructed using the big-oh notation. For this
purpose we define the following functions.

```@docs
O(::QadicField, ::Integer)
O(::QadicField, ::ZZRingElem)
O(::QadicField, ::QQFieldElem)
```

The $O(p^n)$ construction can be used to construct $q$-adic values of precision
$n$ by adding it to integer values representing the $q$-adic value modulo
$p^n$ as in the examples.

**Examples**

```jldoctest
julia> R, _ = qadic_field(7, 1, precision = 30);

julia> S, _ = qadic_field(ZZ(65537), 1, precision = 30);

julia> c = 1 + 2*7 + 4*7^2 + O(R, 7^3)
7^0 + 2*7^1 + 4*7^2 + O(7^3)

julia> d = 13 + 357*ZZ(65537) + O(S, ZZ(65537)^12)
13*65537^0 + 357*65537^1 + O(65537^12)

julia> f = ZZ(1)//7^2 + ZZ(2)//7 + 3 + 4*7 + O(R, 7^2)
7^-2 + 2*7^-1 + 3*7^0 + 4*7^1 + O(7^2)
```

Beware that the expression `1 + 2*p + 3*p^2 + O(R, p^n)` is actually computed
as a normal Julia expression. Therefore if `{Int}` values are used instead
of FLINT integers or Julia bignums, overflow may result in evaluating the
value.

### Basic manipulation

```@docs
prime(::QadicField)
```

```@docs
precision(::QadicFieldElem)
```

```@docs
valuation(::QadicFieldElem)
```

```@docs
lift(::QQPolyRing, ::QadicFieldElem)
lift(::ZZPolyRing, ::QadicFieldElem)
```

**Examples**

```julia
R, _ = qadic_field(7, 1, precision = 30);

a = 1 + 2*7 + 4*7^2 + O(R, 7^3)
b = 7^2 + 3*7^3 + O(R, 7^5)
c = R(2)

k = precision(a)
m = prime(R)
n = valuation(b)
Qx, x = QQ["x"]
p = lift(Qx, a)
Zy, y = ZZ["y"]
q = lift(Zy, divexact(a, b))
```

### Square root

```@docs
Base.sqrt(::QadicFieldElem)
```

**Examples**

```jldoctest
julia> R, _ = qadic_field(7, 1, precision = 30);

julia> a = 1 + 7 + 2*7^2 + O(R, 7^3)
7^0 + 7^1 + 2*7^2 + O(7^3)

julia> b = 2 + 3*7 + O(R, 7^5)
2*7^0 + 3*7^1 + O(7^5)

julia> c = 7^2 + 2*7^3 + O(R, 7^4)
7^2 + 2*7^3 + O(7^4)

julia> d = sqrt(a)
7^0 + 4*7^1 + 3*7^2 + O(7^3)

julia> f = sqrt(b)
4*7^0 + 7^1 + 5*7^2 + 5*7^3 + 6*7^4 + O(7^5)

julia> f = sqrt(c)
7^1 + 7^2 + O(7^3)

julia> g = sqrt(R(121))
4*7^0 + 7^1 + O(7^30)
```

### Special functions

```@docs
Base.exp(::QadicFieldElem)
```

```@docs
log(::QadicFieldElem)
```

```@docs
teichmuller(::QadicFieldElem)
```

```@docs
frobenius(::QadicFieldElem, ::Int)
```

**Examples**

```jldoctest
julia> R, _ = qadic_field(7, 1, precision = 30);

julia> a = 1 + 7 + 2*7^2 + O(R, 7^3)
7^0 + 7^1 + 2*7^2 + O(7^3)

julia> b = 2 + 5*7 + 3*7^2 + O(R, 7^3)
2*7^0 + 5*7^1 + 3*7^2 + O(7^3)

julia> c = 3*7 + 2*7^2 + O(R, 7^5)
3*7^1 + 2*7^2 + O(7^5)

julia> c = exp(c)
7^0 + 3*7^1 + 3*7^2 + 4*7^3 + 4*7^4 + O(7^5)

julia> d = log(a)
7^1 + 5*7^2 + O(7^3)

julia> c = exp(R(0))
7^0 + O(7^30)

julia> d = log(R(1))
0

julia> f = teichmuller(b)
2*7^0 + 4*7^1 + 6*7^2 + O(7^3)

julia> g = frobenius(a, 2)
7^0 + 7^1 + 2*7^2 + O(7^3)
``` 
