using LinearAlgebra: norm

function reach_homog_HLBS25!(F::Vector{ReachSet{N,SparsePolynomialZonotope{N,VN,MN,MNI,VI}}},
                             Ω0::SparsePolynomialZonotope{N,VN,MN,MNI,VI},
                             Φ::MatrixZonotope{N,MN},
                             NSTEPS::Integer,
                             δ::N,
                             max_order::Integer,
                             taylor_order::Integer,
                             ::Val{false},
                             reduction_method::AbstractReductionMethod,
                             Δt0::IA.Interval,
                             ztol::N,
                             norm::Int) where {N,VN,MN,MNI,VI}
    # initial reach set
    Δt = (zero(N) .. δ) + Δt0
    @inbounds F[1] = ReachSet(Ω0, Δt)

    expΦδ = MatrixZonotopeExp(scale(δ, Φ))
    expΦδ_approx = overapproximate(expΦδ, MatrixZonotope, taylor_order)
    expΦδ_approx = _remove_small_generators(expΦδ_approx, ztol, norm)

    j = 1
    @inbounds while j < NSTEPS
        Zⱼ = set(F[j])
        Zⱼ₊₁ = overapproximate(expΦδ_approx * Zⱼ, SparsePolynomialZonotope)
        Zⱼ₊₁ = _remove_small_generators(Zⱼ₊₁, ztol, norm)
        Zⱼ₊₁ʳ = reduce_order(Zⱼ₊₁, max_order, reduction_method)

        j += 1
        Δt += δ
        F[j] = ReachSet(Zⱼ₊₁ʳ, Δt)
    end
    return F
end

function reach_homog_HLBS25!(F::Vector{ReachSet{N,SparsePolynomialZonotope{N,VN,MN,MNI,VI}}},
                             Ω0::SparsePolynomialZonotope{N,VN,MN,MNI,VI},
                             Φ::MatrixZonotope{N,MN},
                             NSTEPS::Integer,
                             δ::N,
                             max_order::Integer,
                             taylor_order::Integer,
                             ::Val{true},
                             reduction_method::AbstractReductionMethod,
                             Δt0::IA.Interval,
                             ztol::N,
                             norm::Int) where {N,VN,MN,MNI,VI}
    # initial reach set
    Δt = (zero(N) .. δ) + Δt0
    @inbounds F[1] = ReachSet(Ω0, Δt)

    expΦδ = MatrixZonotopeExp(scale(δ, Φ))

    j = 1
    @inbounds while j < NSTEPS
        Zⱼ = set(F[j])

        em = ExponentialMap(expΦδ, Zⱼ)

        Zⱼ₊₁ = overapproximate(em, SparsePolynomialZonotope, taylor_order)
        Zⱼ₊₁ = _remove_small_generators(Zⱼ₊₁, ztol, norm)
        Zⱼ₊₁ʳ = reduce_order(Zⱼ₊₁, max_order, reduction_method)

        j += 1
        Δt += δ
        F[j] = ReachSet(Zⱼ₊₁ʳ, Δt)
    end
    return F
end

function _remove_small_generators(M::MatrixZonotope, ztol::Real, p::Real=Inf)
    Ai = generators(M)
    idx = indexvector(M)

    if isempty(Ai)
        return M
    end

    norm_Ai = [norm(gen, p) for gen in Ai]
    filter_idx = findall(x -> x >= ztol, norm_Ai)

    if length(filter_idx) == length(Ai)
        return M
    end

    return MatrixZonotope(center(M), Ai[filter_idx], idx[filter_idx])
end

function _remove_small_generators(P::SparsePolynomialZonotope, tol::Real; p::Real=Inf)
    E = expmat(P)
    G = genmat_dep(P)
    GI = genmat_indep(P)

    ng_dep = ngens_dep(P)
    ng_ind = ngens_indep(P)
    keep_dep = falses(ng_dep)

    @inbounds for j = 1:ng_dep
        keep_dep[j] = norm(@view G[:, j], p) >= tol
    end

    keep_ind = falses(ng_ind)
    @inbounds for j = 1:ng_ind
        keep_ind[j] = norm(@view GI[:, j], p) >= tol
    end
    nd = count(keep_dep)
    ni = count(keep_ind)

    Gnew = Matrix{eltype(P.G)}(undef, dim(P), nd)
    Enew = Matrix{eltype(P.E)}(undef, nparams(P), nd)
    GInew = Matrix{eltype(P.GI)}(undef, dim(P), ni)
    
    idg = 1
    @inbounds for j = 1:ng_dep
        if keep_dep[j]
            Gnew[:, idg] = G[:, j]
            Enew[:, idg] = E[:, j]
            idg += 1
        end
    end
    
    idi = 1
    @inbounds for j = 1:ng_ind
        if keep_ind[j]
            GInew[:, idi] = GI[:, j]
            idi += 1
        end
    end
    return SparsePolynomialZonotope(center(P), Gnew, GInew, Enew, indexvector(P))
end
