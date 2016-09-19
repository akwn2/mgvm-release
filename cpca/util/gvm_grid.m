function [max_g, t0, t1, t2, t3, t4] = gvm_grid(m1, m2, k1, k2, N)
% gvm_grid Obtains the first five trigonometric moments of the GvM, where
% the zeroth moment is exponentiall weighted.
%
%   Usage: [max_g, t0 t1, t2, t3, t4] = gvmz(m1, m2, k1, k2, N)
%
% where
%   max_g    is the exponential weighting factor (t0 = exp(max_g) * t0)
%   t0       is the 0th trigonometric moment (exponentially weighted)
%   t1       is the 1st trigonometric moment
%   t2       is the 2nd trigonometric moment
%   t3       is the 3rd trigonometric moment
%   t4       is the 4th trigonometric moment
% and
%   m1       is the first location parameter
%   m2       is the second location parameter
%   k1       is the first concentration parameter
%   k2       is the second concentration parameter
%   N        is the number of points in the integration grid
%
%   This implementation uses a for loop over the dimensions of the
%   parameters to avoid memory problems

    if nargin < 5
        N = 1E4;
    end
    
    theta = linspace(-pi, pi, N);
    
    cos_th = cos(theta);
    sin_th = sin(theta);
    
    cos_2th = cos(2 * theta);
    sin_2th = sin(2 * theta);
        
    delta = theta(2) - theta(1);

    J = length(m1);
    max_g = zeros(J, 1);
    t0 = zeros(J, 1);
    t1 = zeros(J, 1);
    t2 = zeros(J, 1);
    
    if nargout > 3
        cos_3th = cos(3 * theta);
        sin_3th = sin(3 * theta);

        cos_4th = cos(4 * theta);
        sin_4th = sin(4 * theta);
        
        t3 = zeros(J, 1);
        t4 = zeros(J, 1);
    end
    
    for jj = 1:J
        
        % Evaluate function at gridpoints
        g = k1(jj) .* cos(theta - m1(jj)) ...
          + k2(jj) .* cos(2 .* (theta - m2(jj)));

        % Find the mode of the distribution
        max1 = max(g, [], 2);
        max2 = max(fliplr(g), [], 2);
        max_g(jj) = max([max1, max2], [], 2);
        exp_g = exp(g - max_g(jj));

        % 0th trigonometric moment
        t0(jj) = sum(sort(exp_g)) .* delta;

        % 1st trigonometric moment
        t1(jj) = (sum(sort(cos_th .* exp_g)) ...
                + 1.i * sum(sort(sin_th .* exp_g))) * delta;
            
        % 2nd trigonometric moment
        t2(jj) = (sum(sort(cos_2th .* exp_g)) ...
                + 1.i * sum(sort(sin_2th .* exp_g))) * delta;
            
        if nargout > 3
            % 3rd trigonometric moment
            t3(jj) = (sum(sort(cos_3th .* exp_g)) ...
                    + 1.i * sum(sort(sin_3th .* exp_g))) * delta;

            % 4th trigonometric moment
            t4(jj) = (sum(sort(cos_4th .* exp_g)) ...
                    + 1.i * sum(sort(sin_4th .* exp_g))) * delta;
        end
            
    end
    
end

