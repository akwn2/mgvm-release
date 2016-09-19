function [h, dh_a, dh_b] = entropy_gamma(a, b)
% entropy_gamma
% Entropy of a Gamma distribution with parameters a and b and its partial
% derivatives with respect to the gamma parameters a and b

    h = a - log(b) + gammaln(a) + (1 - a) .* polygamma(0, a);

    if nargout > 1
        dh_a = 1 + (1 - a) .* polygamma(1, a);
        dh_b = - 1 ./ b;
    end
end
