function A = tril2(A)
% tril2
% Obtains the lower portion of the input matrix A on or below the diagonal
% of 2 x 1 blocks (that is, entries of A_{i,j} for which i >= 2 * j).
    [R, C] = size(A);
    for cc = 1:C
        A(1:R < 2 * cc - 1, cc) = 0;
    end
end