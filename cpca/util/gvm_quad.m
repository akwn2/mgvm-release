function T = gvm_quad(r, m1, m2, k1, k2, weighting)
% gvm_quad
% Calculates the moments of a Generalized von Mises distribution of order 2
% with a quadrature-based integration method

    if nargin < 6
        weighting = false;
    end
       
    % Change the definition of the integration depending on the exponential
    % weighting
    if weighting
        gvm_r = @(th, r) exp(k1 .* cos(th - m1) + ...
                             k2 .* cos(2 .* th - 2 .* m2) +...
                             1.i .* r .* th - k1 - k2);
    else
        gvm_r = @(th, r) exp(k1 .* cos(th - m1) + ...
                             k2 .* cos(2 .* th - 2 .* m2) +...
                             1.i .* r .* th - k1 - k2);
    end
    
    % Calculate the r-th trigonometric moment
    T = quadv(@(th)gvm_r(th, r), -pi, pi);
end