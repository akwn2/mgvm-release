function phi = sample_gvm(mu1, mu2, kappa1, kappa2, max_rejections)
% sample_gvm
% Generated random variates from a Generalised von Mises Distribution.
%
% The implementation follows the von Neumann Acceptance-Rejection algorithm
% from [1] determining the mode by analytic solution of quartic equations.
%
% Usage:
%
%   kappa = gvmrand(kappa1, kappa2, mu1, mu2)
%
%   where:
%       kappa1 is a M x N matrix
%       kappa2 is a M x N matrix
%       mu1 is a M x N matrix
%       mu2 is a M x N matrix
%
%   and phi will be a M x N matrix of the sampled variates.
%
% References
%
% [1] Gatto, R. (2008) Some computational aspects of the generalised von 
% Mises distribution. Stat. Comp. v. 18. n. 3. pp 321-331. Springer.
% DOI: 10.1007/s11222-008-9060-4.
%
%------------------------------------------------------------------------

    %% Precomputable relations
    if nargin < 5
        max_rejections = 1E4;
    end
    
    [R, C] = size(kappa1); % assumes all vectors are the same dimension.
    RC = R * C;
    
    % Flatten kappa and mu
    kappa1 = kappa1(:); 
    kappa2 = kappa2(:);
    mu1 = mu1(:);
    mu2 = mu2(:);
    
    % Finding the maximum of the g-function
    max_g = zeros(R, C);

    Ng = 1E3;
    for rr = 1:R % loop to avoid memory issues
        theta = linspace(-pi, pi, Ng);
        
        k1 = kappa1(rr,1);
        k2 = kappa2(rr,1);
        m1 = mu1(rr,1);
        m2 = mu2(rr,1);

        g = k1 .* cos(theta - m1) + k2 .* cos(2 * (theta - m2));

        max1 = max(g, [], 2);
        max2 = max(fliplr(g), [], 2);

        max_g(rr, :) = max([max1, max2], [], 2);
    end
    exp_max_g = exp(max_g(:));
    
    %% Rejection sampling loop
    %---------------------------------------------------------------------
    U = zeros(RC, 1);
    V = zeros(RC, 1);
    exp_g_U = zeros(RC, 1);
    
    notAccepted = ones(RC, 1);
    rejections = 0;
    while any(notAccepted) && rejections < max_rejections
        % We introduced the index variable idx so that we only update the
        % variables that haven't been accepted yet.
        idx = find(notAccepted);
        rejections = rejections + 1;
        
        U(idx) = 2 .* pi .* rand(length(idx), 1);
        V(idx) = exp_max_g(idx) .* rand(length(idx), 1);
        
        exp_g_U(idx) = exp(kappa1(idx) .* cos(U(idx) - mu1(idx)) ...
                         + kappa2(idx) .* cos(2 * (U(idx) - mu2(idx))));
             
        % Update the accepted list
        notAccepted(idx) = V(idx) > exp_g_U(idx);
    end
    
    % Reshape to output
    phi = reshape(U, [R, C]);
end