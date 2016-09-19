function y = log_iv(n, x)
% log_iv.m
% Log of modified bessel function of first kind.
    y = x + log(ive(n, x));
end
