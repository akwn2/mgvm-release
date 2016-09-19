function [A, B, V, L] = connectEntries(Y, nClusters, max_var)

    M = size(Y, 1);
    D = floor(M/2);
    
    V = zeros(D, D);
    L = zeros(D, D);
    
    A = zeros(M, D);
    B = zeros(M, D);
    
    % Build length variance matrix
    for ii = 1:D
        for jj = 1:D
            % column-wise norm
            R = sum((Y(2 * ii - 1: 2 * ii, :)...
                   - Y(2 * jj - 1: 2 * jj, :)).^2, 1) .^ 0.5;
            L(ii, jj) = mean(R);
            V(ii, jj) = var(R);
        end
    end
    
    % Set all upper triangular parts of V to be too big
    INF = 10 * max(V(:));
    if nargin < 3
        max_var = INF;
    end
    for ii = 1:D
        V(ii, ii:end) = INF;
    end
    
    % Finds the greatest entry to be considered in clustering
    threshold = sort(V(:));
    threshold = threshold(nClusters);
    
    % Finds the entries that need to be connected
    [rows, cols] = find(V <= min(threshold, max_var));
    
    for ii = 1:length(rows)
        A(2 * rows(ii) - 1, cols(ii)) = L(rows(ii), cols(ii));
        B(2 * rows(ii), cols(ii)) = L(rows(ii), cols(ii));
    end
    
    % Always fill in the main diagonal
    for ii = 1:D
        A(2 * ii - 1, ii) = mean(L(:));
        B(2 * ii, ii) = mean(L(:));
    end

end