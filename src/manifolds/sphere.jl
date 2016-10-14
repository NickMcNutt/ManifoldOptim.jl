import Base: norm, exp, log, rand

type Sphere <: Manifold
    n::Int
    m::Int
end

Sphere(n::Int) = Sphere(n, 1)

dim(M::Sphere) = M.n * M.m - 1

inner{T}(M::Sphere, X::Matrix{T}, U::Matrix{T}, V::Matrix{T}) = trace(U' * V)

norm{T}(M::Sphere, X::Matrix{T}, U::Matrix{T}) = sqrt(inner(M, X, U, U))

dist{T}(M::Sphere, X::Matrix{T}, Y::Matrix{T}) = acos(trace(X' * Y))

typicaldist(M::Sphere) = 1π

proj{T}(M::Sphere, X::Matrix{T}, H::Matrix{T}) = H - trace(X' * H) * X

egrad2rgrad(M::Sphere, egrad::Function) = X::Matrix -> proj(M, X, egrad(X))

ehess2rhess(M::Sphere, egrad::Function, ehess::Function) = (X::Matrix, U::Matrix) -> proj(M, X, ehess(X, U)) - trace(X' * egrad(X)) * U

exp{T}(M::Sphere, X::Matrix{T}, U::Matrix{T}, t::T) = cos(vecnorm(t * U)) * X + (sin(vecnorm(t * U)) / vecnorm(U)) * U

exp{T}(M::Sphere, X::Matrix{T}, U::Matrix{T}) = exp(M, X, U, one(T))

retr{T}(M::Sphere, X::Matrix{T}, U::Matrix{T}, t::T) = (X + t * U) / vecnorm(X + t * U)

retr{T}(M::Sphere, X::Matrix{T}, U::Matrix{T}) = retr(M, X, U, one(T))

log{T}(M::Sphere, X::Matrix{T}, Y::Matrix{T}) = (dist(M, X, Y) / vecnorm(proj(M, X, Y - X))) * proj(M, X, Y - X)

rand(M::Sphere) = reshape(normalize!(randn(M.n * M.m)), M.n, M.m)

# randvec{T}(M::Sphere, X::Matrix{T}) = 

zerovec{T}(M::Sphere, X::Matrix{T}) = zeros(T, M.n, M.m)

transp{T}(M::Sphere, X::Matrix{T}, Y::Matrix{T}, U::Matrix{T}) = proj(M, Y, U)

pairmean{T}(M::Sphere, X::Matrix{T}, Y::Matrix{T}) = (X + Y) / vecnorm(X + Y);
