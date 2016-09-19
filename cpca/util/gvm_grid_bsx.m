function [max_g, t0, t1, t2, t3, t4] = gvm_grid_bsx(m1, m2, k1, k2, N, useGPU)
% gvm_grid Obtains the first five trigonometric moments of the GvM, where
% the zeroth moment is exponentiall weighted.
%
%   Usage: [max_g, t0 t1, t2, t3, t4] = gvmz(m1, m2, k1, k2, N, useGPU)
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
%   useGPU   is an option parameter for choosing GPU computation.

    
    if nargin < 5
        N = 1E4;
    end
    
    if nargin < 6
        useGPU = false;
    end
    
    
    theta = linspace(-pi, pi, N);
    % Enable GPU computation
    if useGPU
        theta = gpuArray(theta);
        k1 = gpuArray(k1);
        k2 = gpuArray(k2);
        m1 = gpuArray(m1);
        m2 = gpuArray(m2);
    end
    
    th1 = -bsxfun(@minus, m1, theta);
    th2 = -2 .* bsxfun(@minus, m2, theta);
%     clear m1 m2; % Free more memory
    
    kc_th1 = bsxfun(@times, cos(th1), k1);
    kc_th2 = bsxfun(@times, cos(th2), k2);
%     clear th1 th2 k1 k2; % Free more memory
    
    U = bsxfun(@plus, kc_th1, kc_th2);
%     clear kc_th1 kc_th2; % Free more memory
    
    max_g_l = max(U,[],2);
    max_g_r = max(fliplr(U),[],2);
    max_g = max([max_g_l, max_g_r],[],2);
    
    U = -bsxfun(@minus, max_g, U);
    expU = exp(U);
    
    t0 = sum(expU, 2) * (theta(2) - theta(1));
    if useGPU
        t0 = gather(t0);
    end
    
    if nargout > 2
        t1 = bsxfun(@times, exp(1.i .* theta), expU);
        t1 = sum(t1, 2) * (theta(2) - theta(1));
%         t1 = sum(sort(t1, 2), 2);
%         t1 = t1 ./ abs(t1);
%         t1 = exp(1.i * angle(t1));
        if useGPU
            t1 = gather(t1);
        end
    end
    if nargout > 3
        t2 = bsxfun(@times, exp(2.i .* theta), expU);
        t2 = sum(t2, 2) * (theta(2) - theta(1));
%         t2 = sum(sort(t2, 2), 2);
%         t2 = t2 ./ abs(t2);
%         t2 = exp(2.i * angle(t2));
        if useGPU
            t2 = gather(t2);
        end
    end
    if nargout > 4
        t3 = bsxfun(@times, exp(3.i .* theta), expU);
        t3 = sum(t3, 2) * (theta(2) - theta(1));
        if useGPU
            t3 = gather(t3);
        end
%         t3 = sum(sort(t3, 2), 2);
%         t3 = t3 ./ abs(t3);
%         t3 = exp(3.i * angle(t3));
    end
    if nargout > 5
        t4 = bsxfun(@times, exp(4.i .* theta), expU);
        t4 = sum(t4, 2) * (theta(2) - theta(1));
        if useGPU
            t4 = gather(t4);
        end
%         t4 = sum(sort(t4, 2), 2);
%         t4 = t4 ./ abs(t4);
%         t4 = exp(4.i * angle(t4));
    end
end