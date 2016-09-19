function [A, B, V, L] = build_structure(Y, D, n_edges)

    Y = Y';
    [N, M] = size(Y);
    DP1 = M/2 + 1;
    
    V = inf * ones(DP1, DP1);
    L = zeros(DP1, DP1);
    
    A = zeros(M, D);
    B = zeros(M, D);
    
    % Build length variance matrix
    R = zeros(DP1, DP1, N);
    for ii = 1:DP1
        for jj = 1:ii-1
            for nn = 1:N
                if ii == DP1
                    u = [0, 0];
                else
                    u = [Y(nn,2 * ii - 1), Y(nn, 2 * ii)];
                end
                v = [Y(nn,2 * jj - 1), Y(nn, 2 * jj)];
                R(ii, jj, nn) = norm(u - v);
            end
            
            L(ii, jj) = mean(R(ii, jj, :));
            V(ii, jj) = var(R(ii, jj, :));
        end
    end
    V = V ./ L; %scale variance
    
    rot_x = mean(Y(:,1:2:end), 1);
    rot_y = mean(Y(:,2:2:end), 1);
    
    rot = angle(rot_x + 1.i * rot_y);
    
    % Find variance treshold and filter the lengths by it
    threshold = sort(V(:));
    threshold = threshold(n_edges);
    
    L = (V <= threshold) .* L;
    
    for dd = 1:D
        A(2 * dd - 1, :) = L(dd + 1, 1:end-1);
        B(2 * dd, :) = L(dd + 1, 1:end-1);
    end
    A = A * diag(cos(rot)) + B * diag(sin(rot));
    B = A * diag(sin(rot)) - B * diag(cos(rot));
end