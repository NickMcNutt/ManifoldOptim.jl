function trust_regions(M::Manifold, F::ManifoldFunction, x₀ = rand(M);
    verbosity = 0,
    tol_grad = 1e-6,
    max_time = Inf,
    min_iters = 3,
    max_iters = 1000,
    θ = 1.0,
    κ = 0.1,
    min_tcg_iters = 1,
    max_tcg_iters = dim(M),
    Δ̄ = typicaldist(M),
    Δ₀ = Δ̄ / 8,
    ρ′ = 0.1,
    ρ_regularization = 1e3)
    
    @assert ρ′ < 1/4
    @assert Δ̄ > Δ₀ > 0
    
    x = x₀
    Δ = Δ₀
    f = F.f(x)
    
    if verbosity == 1
        @printf("  Step    Accept?  TR change     f(x)         |∇fₓ|\n")
        @printf("-----------------------------------------------------\n")
    elseif verbosity == 2
        @printf("  Step    Accept?  TR change     f(x)         |∇fₓ|       ⟨r, r⟩ₓ      ⟨δ, Hδ⟩ₓ         α            τ\n")
        @printf("-------------------------------------------------------------------------------------------------------------\n")
    end
    
    for k in 1:max_iters
        ∇f = F.rgrad(x)
        norm_∇f = norm(M, x, ∇f)
        
        if norm_∇f < tol_grad
            stop_flag = :tol_grad_reached
            break
        end
        
        η, Hη, tcg_flag = truncated_conjugate_gradient(M, F, x, ∇f, Δ, θ = θ, κ = κ, min_iters = min_tcg_iters, max_iters = max_tcg_iters, verbosity = verbosity)
        x₊ = retr(M, x, η)
        f₊ = F.f(x₊)
        
        ρ_num = f - f₊
        ρ_den = -inner(M, x, η, ∇f) - 0.5⋅inner(M, x, η, Hη)
        
        ρ_reg = max(1.0, abs(f)) * eps() * ρ_regularization
        ρ = (ρ_num + ρ_reg) / (ρ_den + ρ_reg)

        if ρ < 1/4
            Δ₊ = Δ / 4
            Δ_change = '-'
        elseif ρ > 3/4 && (tcg_flag == :negative_curvature || tcg_flag == :trust_region_exceeded)
            Δ₊ = min(2Δ, Δ̄)
            Δ_change = '+'
        else
            Δ₊ = Δ
            Δ_change = ' '
        end
        
        accept_step = false
        if ρ > ρ′
            accept_step = true
            x = x₊
            f = f₊
        end
        
        if verbosity > 0
            @printf("  RTR       %s        (%s)     %10.2e   %10.2e\n", ifelse(accept_step, '✓', ' '), Δ_change, f, norm_∇f)
        end
        
        Δ = Δ₊
    end
    
    return x, f
end
