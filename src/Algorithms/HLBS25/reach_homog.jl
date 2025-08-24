using LinearAlgebra: norm

function _remove_small_generators(M::MatrixZonotope, ztol::Real, p::Int)
    A0 = center(M)
    Ai = genmats(M)
    idx = M.idx

    if isempty(Ai)
        return M
    end

    norm_Ai = [norm(gen, p) for gen in Ai]
    filter_idx = findall(x -> x >= ztol, norm_Ai)

    if length(filter_idx) == length(Ai)
        return M
    end

    return MatrixZonotope(A0, Ai[filter_idx], idx[filter_idx])
end

function _remove_small_generators(spz::SparsePolynomialZonotope, ztol::Real, p::Int)
    c = center(spz)
    G = genmat(spz)
    GI = depgenmat(spz)
    E = expmat(spz)
    idx = indexvector(spz)

    # independent generators
    norm_G = [norm(view(G, :, i), p) for i in 1:size(G, 2)]
    filter_idx_G = findall(x -> x >= ztol, norm_G)

    # dependent generators
    norm_GI = [norm(view(GI, :, i), p) for i in 1:size(GI, 2)]
    filter_idx_GI = findall(x -> x >= ztol, norm_GI)

    if length(filter_idx_G) == size(G, 2) && length(filter_idx_GI) == size(GI, 2)
        return spz
    end

    G_new = G[:, filter_idx_G]
    GI_new = GI[:, filter_idx_GI]
    E_new = E[filter_idx_GI, filter_idx_G]
    idx_new = idx[filter_idx_GI]

    return SparsePolynomialZonotope(c, G_new, GI_new, E_new, idx_new)
end

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
    expΦδ = _remove_small_generators(expΦδ, ztol, norm)

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
