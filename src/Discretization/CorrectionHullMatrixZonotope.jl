module CorrectionHullMatrixZonotopeModule

using LazySets, Reexport
using MathematicalSystems
using ..DiscretizationModule
using LinearAlgebra: I

export CorrectionHullMatrixZonotope

@reexport import ..DiscretizationModule: discretize

const LPCS = LinearParametricContinuousSystem
const LPDS = LinearParametricDiscreteSystem

"""
    CorrectionHullMatrixZonotope{N, R} <: AbstractApproximationModel

Approximation model used to discretize linear parametric continuous systems with
matrix zonotopes via a correction-hull style approximation of the matrix exponential.

### Fields

- `taylor_order` -- (optional, default: `5`) order of the Taylor series expansion used
                    to approximate the matrix exponential.
- `recursive`    -- (optional, default: `true`) whether to compute each Taylor term
                    recursively (more accurate but more expensive) or to use a
                    non-recursive overapproximation (cheaper, potentially coarser).
- `ztol`         -- (optional, keyword, default: `0`) tolerance below which
                    generators are removed.
- `norm`         -- (optional, keyword, default: `Inf`) p-norm used to remove generators.

### Notes

If `recursive == true`, terms of the Taylor expansion are computed recursively
(e.g., `A^k * P = A * (A^{k-1} * P)`), which often improves tightness at the
cost of additional computation. If `recursive == false`, a single non-recursive
overapproximation of the exponential is produced; this reduces cost but makes
the quality of the result more dependent on `taylor_order`.
"""
struct CorrectionHullMatrixZonotope{N, R} <: AbstractApproximationModel
    taylor_order::Int
    recursive::R
    ztol::N
    norm::Real
end

function CorrectionHullMatrixZonotope(::Type{N}=Float64; taylor_order::Int=5,
                                     recursive::Bool=true, ztol::N=zero(N),
                                     norm::Real=Inf) where {N}
    return CorrectionHullMatrixZonotope{N, Val{recursive}}(taylor_order, Val(recursive), ztol, norm)
end

function discretize(ivp::IVP{<:LPCS, <:SparsePolynomialZonotope}, δ,
                    alg::CorrectionHullMatrixZonotope{N, Val{true}}) where {N}
    taylor_order = alg.taylor_order
    A = state_matrix(ivp)
    X0 = initial_state(ivp)
    n = dim(X0)

    IDₜ = ngens(A) > 0 ? maximum(indexvector(A)) + 1 : 1
    Tₜ = N(0.5) * δ * Matrix(N(1) * I, n, n)
    T = MatrixZonotope(Tₜ, [Tₜ], [IDₜ])
    expAT = MatrixZonotopeExp(A * T)

    # recursive logic
    em = ExponentialMap(expAT, X0)
    Ω0 = overapproximate(em, SparsePolynomialZonotope, taylor_order)
    Ω0 = _remove_small_generators(Ω0, alg.ztol, alg.norm)

    Sdis = LPDS(A)
    return InitialValueProblem(Sdis, Ω0)
end

function discretize(ivp::IVP{<:LPCS, <:SparsePolynomialZonotope}, δ,
                    alg::CorrectionHullMatrixZonotope{N, Val{false}}) where {N}
    taylor_order = alg.taylor_order
    A = state_matrix(ivp)
    X0 = initial_state(ivp)
    n = dim(X0)

    IDₜ = ngens(A) > 0 ? maximum(indexvector(A)) + 1 : 1
    Tₜ = N(0.5) * δ * Matrix(N(1) * I, n, n)
    T = MatrixZonotope(Tₜ, [Tₜ], [IDₜ])
    AT = overapproximate(A * T, MatrixZonotope)
    expAT = MatrixZonotopeExp(AT)

    # non-recursive logic
    expAT_approx = overapproximate(expAT, MatrixZonotope, taylor_order)
    expAT_approx = _remove_small_generators(expAT_approx, alg.ztol, alg.norm)
    Ω0 = overapproximate(expAT_approx * X0, SparsePolynomialZonotope)

    Sdis = LPDS(A)
    return InitialValueProblem(Sdis, Ω0)
end

end # module
