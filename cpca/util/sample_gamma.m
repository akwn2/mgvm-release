function x = sample_gamma(a, b)
% sample_gamma
% Wrapper for sampling a gamma distribution with parameters a and b
% (basically a rename for standardization).
    x = gamrnd(a, b);
end