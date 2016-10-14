import Base: norm, exp, log, rand

type StackedStiefel <: Manifold
    m::Int
    d::Int
    k::Int
    n::Int

    function StackedStiefel(m::Int, d::Int, k::Int)
        @assert k >= d "k must be at least as large as d."
        new(m, d, k, m * d)
    end
end

# Manifold methods

dim(M::StackedStiefel) = M.m * (M.k * M.d - 0.5 * M.d * (M.d + 1))

inner{T}(M::StackedStiefel, X::Matrix{T}, U::Matrix{T}, V::Matrix{T}) = trace(U' * V)

norm{T}(M::StackedStiefel, X::Matrix{T}, U::Matrix{T}) = sqrt(inner(M, X, U, U))

dist{T}(M::StackedStiefel, X::Matrix{T}, Y::Matrix{T}) = error("dist() for StackedStiefel not yet implemented")

typicaldist(M::StackedStiefel) = sqrt(dim(M))

function proj{T}(M::StackedStiefel, Y::Matrix{T}, Z::Matrix{T})
    Y3 = to3D(M, Y)
    Z3 = to3D(M, Z)
    λ = symbdiag(Y3, Z3)
    Zt3 = Z3 - multiprod(λ, Y3)

    return to2D(M, Zt3)
end

egrad2rgrad(M::StackedStiefel, egrad::Function) = X::Matrix -> proj(M, X, egrad(X))

function ehess2rhess(M::StackedStiefel, egrad::Function, ehess::Function)
    return function (Y::Matrix, Ẏ::Matrix)
        Y3 = to3D(M, Y)
        Ẏ3 = to3D(M, Ẏ)
        egrad3 = to3D(M, egrad(Y))
        C = symbdiag(Y3, egrad3)
        CẎ = to2D(M, multiprod(C, Ẏ3))
        proj(M, Y, ehess(Y, Ẏ) - CẎ)
    end
end

function exp{T}(M::StackedStiefel, Y::Matrix{T}, U::Matrix{T}, t::T)
    tU3 = permutedims(to3D(M, t * U), (2, 1, 3))
    Y3 = permutedims(to3D(M, Y), (2, 1, 3))

    for i in 1:M.m
        X = Y3[:, :, i]
        Z = tU3[:, :, i]
        Y3[:, :, i] = [X Z] * expm([X'*Z -Z'*Z ; eye(M.d) X'*Z]) * [expm(-X'*Z) ; zeros(M.d)]
        W, S, V = svd(Y3[:, :, i])
        Y3[:, :, i] = W * V'
    end

    return to2D(M, permutedims(Y3, (2, 1, 3)))
end

exp{T}(M::StackedStiefel, Y::Matrix{T}, U::Matrix{T}) = exp(M, Y, U, one(T))

function retr{T}(M::StackedStiefel, Y::Matrix{T}, U::Matrix{T}, t::T)
    Y = Y + t * U
    Y3 = to3D(M, Y)

    for i in 1:M.m
        W, S, V = svd(Y3[:, :, i])
        Y3[:, :, i] = W * V'
    end

    return to2D(M, Y3)
end

retr{T}(M::StackedStiefel, Y::Matrix{T}, U::Matrix{T}) = retr(M, Y, U, one(T))

log{T}(M::StackedStiefel, X::Matrix{T}, Y::Matrix{T}) = error("log() for StackedStiefel not yet implemented")

function rand(M::StackedStiefel)
    Y3 = zeros(M.d, M.k, M.m)

    for i in 1:M.m
        Q = qr(randn(M.k, M.d))[1]
        Y3[:, :, i] = Q'
    end

    return to2D(M, Y3)
end

function randvec{T}(M::StackedStiefel, Y::Matrix{T})
    U = proj(M, Y, randn(M.n, M.k))

    return U / norm(M, Y, U)
end

zerovec{T}(M::StackedStiefel, X::Matrix{T}) = zeros(T, M.n, M.k)

transp{T}(M::StackedStiefel, X::Matrix{T}, Y::Matrix{T}, U::Matrix{T}) = proj(M, Y, U)

pairmean{T}(M::StackedStiefel, X::Matrix{T}, Y::Matrix{T}) = error("pairmean() for StackedStiefel not yet implemented")

# Helper methods

to2D(M::StackedStiefel, A::Array) = reshape(permutedims(A, (1, 3, 2)), M.m * M.d, M.k)

to3D(M::StackedStiefel, A::Matrix) = permutedims(reshape(A, M.d, M.m, M.k), (1, 3, 2))

multitransp(A) = permutedims(A, (2, 1, 3))

function multiprod(A, B)
    C = similar(A, size(A, 1), size(B, 2), size(B, 3))
    
    for i in 1:size(A, 3)
        C[:, :, i] = A[:, :, i] * B[:, :, i]
    end

    return C
end

multisym(A) = 0.5 * (A + multitransp(A))

symbdiag(A, B) = multisym(multiprod(A, multitransp(B)))

function project_to_stiefel{T}(M::StackedStiefel, Y::Matrix{T}, k::Int)
    M_proj = StackedStiefel(M.m, M.d, k)
    
    U, S, V = svd(Y)
    Y3 = to3D(M_proj, (U * diagm(S))[:, 1:k])
    
    for i in 1:M.m
        W, S, V = svd(Y3[:, :, i])
        Y3[:, :, i] = W * V'
    end

    return to2D(M_proj, Y3)
end

