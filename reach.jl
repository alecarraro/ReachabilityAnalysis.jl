using ReachabilityAnalysis, LazySets, Plots, IntervalMatrices

const N = Float64

AS = MatrixZonotope(N[0 -1; 1 0], [zeros(N, 2, 2)], [5])

AS = MatrixZonotope(N[0 -1; 1 0], [N[0 0.05; 0 0]], [5])

X0 = SparsePolynomialZonotope(N[1.0, 1.0],         # center
                              N(0.1) * N[2.0 0.0 1.0; 1.0 2.0 1.0],  # generators
                              N(0.1) * reshape(N[1.0, 0.5], 2, 1), # dependent generators
                              [1 0 1; 0 1 3])      # exponents

prob = @ivp(x' = A * x, x(0) ∈ X0, A ∈ AS)
δ = N(2π) / 200

# Instantiate the HLBS25 algorithm.
# - δ: Step size.
# - approx_model: Discretization model. CorrectionHullMatrixZonotope is suitable for matrix zonotopes.
# - max_order: Maximum order for the Taylor series expansion of the matrix exponential.
# - taylor_order: Taylor series order for each step.
# - reduction_method: Method for reducing the order of matrix zonotopes.
# - recursive: Specifies whether to use a recursive method for the Taylor expansion.
alg = HLBS25(δ=δ,
             approx_model=CorrectionHullMatrixZonotope(taylor_order=10, recursive=false, tol= 1e-4, norm=Inf),
             max_order=10,
             taylor_order=10,
             reduction_method=LazySets.GIR05(),
             recursive=false,
             tol = 1e-4,
             norm = Inf)

T = 2 * N(π) / 200 * 150
sol = solve(prob, alg; T=T)

plot()
for (i,rset) in enumerate(sol)
     if i % 10 == 0
          plot!(set(rset), vars=(1, 2), lw=0.5, color=:blue, alpha=0.8,
               xlab="x₁", ylab="x₂",
               title="Flowpipe of the parametric linear system using HLBS25",
               legend=:bottomright)
     end
end

# Plot the initial set X0 for context.
plot!(X0, vars=(1, 2), color=:red, alpha=0.5, lab="X₀")
