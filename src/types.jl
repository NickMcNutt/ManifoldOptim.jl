abstract Manifold

immutable ManifoldFunction
    f::Function
    egrad::Function
    ehess::Function
    rgrad::Function
    rhess::Function
end

ManifoldFunction(M::Manifold, f, ∇f, ∇²f) = ManifoldFunction(f, ∇f, ∇²f, egrad2rgrad(M, ∇f), ehess2rhess(M, ∇f, ∇²f))
