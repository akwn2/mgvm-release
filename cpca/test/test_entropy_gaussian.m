function [error_mu, error_K] = test_entropy_gaussian()
% test_entropy_gaussian
% tests the entropy for the Gaussian distribution using grid integration
    
    K = rand(2);
    K = K * K';
    mu = rand(2,1);

    [~, dh_mu, dh_K] = entropy_gaussian(mu, K);
        
    dx = 1E-8;
    % dh_mu
    fd_h_mu = zeros(2,1);
    for ii = 1:2
        e = zeros(2, 1);
        e(ii) = dx;
        
        h_f = entropy_gaussian(mu + e * dx, K);
        h_b = entropy_gaussian(mu - e * dx, K);
        
        fd_h_mu(ii) = (h_f - h_b) ./ (2 .* dx);
    end
    error_mu = norm(fd_h_mu - dh_mu);
    fprintf('Error in mu derivative: %1.4e\n', error_mu);
    
    
    % dh_K
    fd_h_K = zeros(2);
    for ii = 1:4
        e = zeros(2);
        e(ii) = dx;
        
        h_f(ii) = entropy_gaussian(mu, K + e);
        h_b(ii) = entropy_gaussian(mu, K - e);
        
        fd_h_K(ii) = (h_f(ii) - h_b(ii)) ./ (2 .* dx);
    end
    error_K = norm(fd_h_K(:) - dh_K(:));
    
    fprintf('Error in K derivative: %1.4e\n', error_K);
end