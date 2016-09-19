function T = gvm_euler(r, m1, m2, k1, k2, weighting, n_pts)
% gvm_euler
% Calculates the moments of a Generalized von Mises distribution of order 2
% with a grid using euler's method

    if nargin < 6
        weighting = false;
    end
    if nargin < 7
        n_pts = 1E6;
    end
    
    D = size(m1, 1);
    
    th = repmat(linspace(-pi, pi, n_pts), [D, 1]);
    step = th(1, 2) - th(1, 1);
    
    M1 = repmat(m1, [1, n_pts]);
    M2 = repmat(m2, [1, n_pts]);
    K1 = repmat(k1, [1, n_pts]);
    K2 = repmat(k2, [1, n_pts]);
    % Change the definition of the integration depending on the exponential
    % weighting
    if weighting
        gvm_r = @(th, r) exp(K1 .* cos(th - M1) + ...
                             K2 .* cos(2 .* th - 2 .* M2) +...
                             1.i .* r .* th - K1 - K2);
    else
        gvm_r = @(th, r) exp(K1 .* cos(th - M1) + ...
                             K2 .* cos(2 .* th - 2 .* M2) +...
                             1.i .* r .* th - K1 - K2);
    end
    
    % Calculate the r-th trigonometric moment
    T = sum(gvm_r(th, r), 2) .* step;
end