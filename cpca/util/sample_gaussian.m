function x = sample_gaussian(mu, K)
% sample_gaussian
% Wrapper for sampling a (multivariate) Gaussian with mean mu and
% covariance K (basically a rename for standardization).
    x = mvnrnd(mu, K);
end