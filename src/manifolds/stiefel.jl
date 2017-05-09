import Base: norm, exp, log, rand

type Stiefel <: Manifold
    n::Int
    p::Int
    k::Int
end

Stiefel(n::Int, p::Int) = Stiefel(n, p, 1)
Orthogonal(n::Int) = Stiefel(n, n, 1)
Orthogonal(n::Int, k::Int) = Stiefel(n, n, k)

# Manifold methods

dim(M::Stiefel) = M.k * (M.n * M.p - 0.5 * M.p * (M.p + 1))

inner{T}(M::Stiefel, X::Array{T, 3}, U::Array{T, 3}, V::Array{T, 3}) = dot(U, V)

norm{T}(M::Stiefel, X::Array{T, 3}, U::Array{T, 3}) = sqrt(inner(M, X, U, U))

dist{T}(M::Stiefel, X::Array{T, 3}, Y::Array{T, 3}) = error("dist() for Stiefel not yet implemented")

typicaldist(M::Stiefel) = sqrt(M.p * M.k)

function proj{T}(M::Stiefel, X::Array{T, 3}, U::Array{T, 3})
    XtU = multiprod(multitransp(X), U)
    symXtU = multisym(XtU)
    Up = U - multiprod(X, symXtU)

    return Up
end

egrad2rgrad(M::Stiefel, egrad::Function) = X::Array -> proj(M, X, egrad(X))

function ehess2rhess(M::Stiefel, egrad::Function, ehess::Function)
    return function (X::Array, Ẋ::Array)
        XtG = multiprod(multitransp(X), egrad(X))
        symXtG = multisym(XtG)
        ẊsymXtG = multiprod(Ẋ, symXtG)
        proj(M, X, ehess(X, Ẋ) - ẊsymXtG)
    end
end

function exp{T}(M::Stiefel, X::Array{T, 3}, U::Array{T, 3}, t::T)
    tU = t * U
    Y = zeros(X)
    for i in 1:M.k
        Xᵢ = X[:, :, i]
        Uᵢ = tu[:, :, i]
        Y[:, :, i] = [Xᵢ Uᵢ] * expm([Xᵢ'*Uᵢ -Uᵢ'*Uᵢ ; eye(M.p) Xᵢ'*Uᵢ]) * [expm(-Xᵢ'*Uᵢ) ; zeros(M.p)]
    end

    return Y
end

exp{T}(M::Stiefel, X::Array{T, 3}, U::Array{T, 3}) = exp(M, X, U, one(T))

function retr{T}(M::Stiefel, X::Array{T, 3}, U::Array{T, 3}, t::T)
    Y = X + t * U
    for i in 1:M.k
        W, S, V = svd(Y[:, :, i])
        Y[:, :, i] = W * V'
    end

    return Y
end

retr{T}(M::Stiefel, X::Array{T, 3}, U::Array{T, 3}) = retr(M, X, U, one(T))

log{T}(M::Stiefel, X::Array{T, 3}, Y::Array{T, 3}) = error("log() for Stiefel not yet implemented")

function rand(M::Stiefel)
    X = zeros(M.n, M.p, M.k)

    for i in 1:M.k
        X[:, :, i] = qr(randn(M.n, M.p))[1]
    end

    return X
end

function randvec{T}(M::Stiefel, X::Array{T, 3})
    U = proj(M, X, randn(M.n, M.p, M.k))

    return U / norm(M, X, U)
end

zerovec{T}(M::Stiefel, X::Array{T, 3}) = zeros(T, M.n, M.p, M.k)

transp{T}(M::Stiefel, X::Array{T, 3}, Y::Array{T, 3}, U::Array{T, 3}) = proj(M, Y, U)

pairmean{T}(M::Stiefel, X::Array{T, 3}, Y::Array{T, 3}) = error("pairmean() for Stiefel not yet implemented")
