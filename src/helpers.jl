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
