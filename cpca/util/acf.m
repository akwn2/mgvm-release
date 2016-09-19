function rho = acf(x, t)
% acf.m
% Autocorrelation function to determine effective sample size of MCMC
% sampler.
%
% usage: y = acf(x, t);
%
% where y is acf coefficient, x are the samples and t is the lag order
% (which defaults to 1)

    if nargin < 2
        t = 1;
    end
    
    [D, N]= size(x);
    
    mu = mean(x, 2);
    
    num = zeros(D, 1);
    den = zeros(D, 1);
    
    for n = 1:N - t
        num = num + (x(:, n) - mu) .* (x(:, n + 1) - mu);
        den = den + (x(:, n) - mu) .^ 2;
    end
    
    for n = N - t:N
        den = den + (x(:, n) - mu) .^ 2;
    end
    num = num ./ (N - t);
    den = den ./ (N - 1);
    rho = num ./ den;

end