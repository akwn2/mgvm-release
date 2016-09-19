function y = modpi(x)
% modpi.m
% Calculates the equivalent angle of x in [-pi, pi[.
    y = mod(x + pi, 2 * pi) - pi;
end