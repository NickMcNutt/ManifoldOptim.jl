function truncated_conjugate_gradient{N}(M::Manifold, F::ManifoldFunction, x::Array{Float64, N}, ∇f::Array{Float64, N}, Δ::Float64;
    θ = 1.0,
    κ = 0.1,
    min_iters = 1,
    max_iters = dim(M),
    verbosity = 1)

    η = zerovec(M, x)
    Hη = zerovec(M, x)
    r = ∇f
    δ = -r
    
    r_r = inner(M, x, r, r)
    δ_δ = r_r
    η_η = 0.0
    η_δ = 0.0
    
    norm_r = sqrt(r_r)
    norm_r₀ = norm_r
    
    m = 0.0
    
    stop_flag = :max_iters
    
    for j in 1:max_iters
        Hδ = F.rhess(x, δ)
        δ_Hδ = inner(M, x, δ, Hδ)
        α = r_r / δ_Hδ
        η₊_η₊ = η_η + 2⋅α⋅η_δ + α^2⋅δ_δ
        
        if verbosity > 1
            @printf("  tCG                                                  %10.2e   %10.2e   %10.2e\n", r_r, δ_Hδ, α)
            flush(STDOUT)
        end
        
        if δ_Hδ <= 0 || η₊_η₊ >= Δ^2
            τ = (-η_δ + √(η_δ^2 + (Δ^2 - η_η)⋅δ_δ)) / δ_δ
            η = η + τ * δ
            Hη = Hη + τ * Hδ
            
            if verbosity > 1
                @printf("  tCG                                                                                         %10.2e\n", τ)
                flush(STDOUT)
            end
            
            stop_flag = δ_Hδ <= 0 ? :negative_curvature : :trust_region_exceeded
            break
        end
        
        η_η = η₊_η₊
        η₊ = η + α * δ
        Hη₊ = Hη + α * Hδ
        
        m₊ = inner(M, x, η₊, ∇f) + 0.5⋅inner(M, x, η₊, Hη₊)
        if m₊ >= m
            stop_flag = :model_increased
            break
        end
        
        η = η₊
        Hη = Hη₊
        m = m₊
        
        r = r + α * Hδ
        r₊_r₊ = inner(M, x, r, r)
        β = r₊_r₊ / r_r
        r_r = r₊_r₊
        norm_r = sqrt(r_r)
                
        if j >= min_iters && norm_r <= norm_r₀⋅min(norm_r₀^θ, κ)
            stop_flag = κ < norm_r₀^θ ? :linear_convergence : :superlinear_convergence
            break
        end
        
        δ = -r + β * δ
        η_δ = β⋅(η_δ + α⋅δ_δ)
        δ_δ = r_r + β^2⋅δ_δ
    end
    
    return η, Hη, stop_flag
end
