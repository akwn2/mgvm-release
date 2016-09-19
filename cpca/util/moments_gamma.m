function [m, dm_a, dm_b] = moments_gamma(a, b)
% moments_gamma
% Calculates the moments of a gamma distribution and their derivatives
% analytically.

    m = a ./ b;
    
    if nargout > 1
        dm_a = 1 ./ b;
        dm_b = - a .* b .^ -2;
    end
end
