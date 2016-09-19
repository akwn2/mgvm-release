function [h, dh_mu, dh_K] = entropy_gaussian(mu, K)
% entropy_gaussian
% Entropy of a (multivariate) Gaussian distribution with parameters mu and
% covariance K. The partial derivatives with respect to the parameters are
% also provided.

    % Get multivariate Gaussian dimensions
    D = size(mu, 1);
    
    % Calculate the cholesky decomposition
    L = rchol(K);
    
    % Calculate the entropy
    h = 0.5 * D * (1 + log(2 * pi)) + log(det(L));

    % Get the derivatives
    if nargout > 1
        dh_mu = zeros(D, 1);
        
        % Get the inverse of K
        L_inv = L \ eye(D);
        K_inv = L_inv * L_inv';
        
        dh_K = K_inv';
    end

end