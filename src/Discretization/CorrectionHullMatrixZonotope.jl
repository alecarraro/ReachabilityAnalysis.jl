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

- `taylor_order` -- order of the Taylor series expansion used to approximate the matrix exponential.
- `recursive`    -- whether to compute each Taylor term recursively.
- `tol`          -- tolerance below which generators are removed.
- `norm`         -- p-norm used to remove generators.
"""
struct CorrectionHullMatrixZonotope{N, R} <: AbstractApproximationModel
    taylor_order::Int
    recursive::R
    tol::N
    norm::Real
end

# explicit constructor: require tol and norm keywords (defaults provided)
function CorrectionHullMatrixZonotope(::Type{N}=Float64;
                                     taylor_order::Int=5,
                                     recursive::Bool=true,
                                     tol::N=zero(N),
                                     norm::Real=Inf) where {N}
    return CorrectionHullMatrixZonotope{N, Val{recursive}}(taylor_order, Val(recursive), tol, norm)
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

    em = ExponentialMap(expAT, X0)
    Ω0 = overapproximate(em, SparsePolynomialZonotope, taylor_order)
    Ω0 = ReachabilityAnalysis._remove_small_generators(Ω0, alg.tol, alg.norm)

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

    expAT_approx = overapproximate(expAT, MatrixZonotope, taylor_order)
    expAT_approx = ReachabilityAnalysis._remove_small_generators(expAT_approx, alg.tol, alg.norm)
    Ω0 = overapproximate(expAT_approx * X0, SparsePolynomialZonotope)

    Sdis = LPDS(A)
    return InitialValueProblem(Sdis, Ω0)
end

end # module
