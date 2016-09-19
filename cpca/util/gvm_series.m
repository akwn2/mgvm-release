function T = gvm_series(r, m1, m2, k1, k2, weighting, n_terms)
% gvm_series
% calculates the rth moment of the Generalised von Mises distribution
% by series approximations in complex number form.

    if nargin < 6
        weighting = false;
    end
    if nargin < 7
        n_terms = 25;
    end
    [R, C] = size(m1);
    
    m1 = repmat(m1(:), [1, n_terms]);
    m2 = repmat(m2(:), [1, n_terms]);
    k1 = repmat(k1(:), [1, n_terms]);
    k2 = repmat(k2(:), [1, n_terms]);

    delta = m2 - m1;
    
    % Calculate the r-th trigonometric moment
    if weighting
        varphi_r = ive(0, k2(:, 1)) .* ive(r, k1(:, 1));

        JJ = repmat(1:n_terms, [R * C, 1]);

        term = ive(JJ, k2) .* ...
               (exp(+2.i .* JJ .* delta) .* ive(2. .* JJ + r, k1) + ...
                exp(-2.i .* JJ .* delta) .* ive(abs(2. .* JJ - r), k1));
    else
        varphi_r = iv(0, k2(:, 1)) .* iv(r, k1(:, 1));

        JJ = repmat(1:n_terms, [R * C, 1]);

        term = iv(JJ, k2) .* ...
               (exp(+2.i .* JJ .* delta) .* iv(2. .* JJ + r, k1) + ...
                exp(-2.i .* JJ .* delta) .* iv(abs(2. .* JJ - r), k1));
    end
    varphi_r = varphi_r + sum(term, 2);

    T = 2 .* pi .* exp(1.i .* r .* m1(:, 1) ) .* varphi_r;
    T = reshape(T, R, C);
end     