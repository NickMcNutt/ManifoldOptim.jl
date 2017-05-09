module ManifoldOptim

export
    # Types
    Manifold,
    ManifoldFunction,

    # Solvers
    truncated_conjugate_gradient,
    trust_regions,

    # Manifold methods
    dim,
    inner,
    norm,
    dist,
    typicaldist,
    proj,
    egrad2rgrad,
    ehess2rhess,
    exp,
    retr,
    log,
    rand,
    randvec,
    zerovec,
    transp,
    pairmean,

    # Stacked Stiefel helper methods
    project_to_stiefel,

    # Manifolds
    Orthogonal,
    Sphere,
    StackedStiefel,
    Stiefel


include("types.jl")

include("helpers.jl")

include("solvers/trust_regions/truncated_conjugate_gradient.jl")
include("solvers/trust_regions/trust_regions.jl")

include("manifolds/sphere.jl")
include("manifolds/stacked_stiefel.jl")
include("manifolds/stiefel.jl")

end
