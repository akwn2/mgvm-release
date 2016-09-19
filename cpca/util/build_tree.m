function [A, B, V, L] = build_tree(Y, D, n_nodes)

    [N, M] = size(Y);
    D_Y = M/2;
    
    V = inf * ones(D_Y, D_Y);
    L = zeros(D_Y, D_Y);
    
    A = zeros(M, D);
    B = zeros(M, D);
    
    % Build length variance matrix
    R = zeros(D_Y, D_Y, N);
    for ii = 1:D_Y
        for jj = 1:ii-1
            for nn = 1:N
                u = [Y(nn,2 * ii - 1), Y(nn, 2 * ii)];
                v = [Y(nn,2 * jj - 1), Y(nn, 2 * jj)];
                R(ii, jj, nn) = norm(u - v);
            end
            
            L(ii, jj) = mean(R(ii, jj, :));
            V(ii, jj) = var(R(ii, jj, :));
        end
    end
    
    % Find variance treshold and filter the lengths by it
    threshold = sort(V(:));
    threshold = threshold(n_nodes);
    
    L = (V <= threshold) .* L;
    
    for mm = 1:M
        if mod(mm,2)
            A(mm, :) = L((mm + 1) / 2, :);
            B(mm, :) = 0;
        else
            A(mm, :) = 0;
            B(mm, :) = L(mm / 2, :);
        end
    end

end