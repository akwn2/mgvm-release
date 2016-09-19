function y = ive(n, x)
% ive.m
% Wrapper function for the Matlab implementation of the exponentially
% weighted modified bessel function of first kind besseli (basically a
% rename)
    y = besseli(n, x, 1);
end