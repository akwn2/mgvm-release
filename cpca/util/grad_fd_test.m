function [error, gradf, fd_grad] = grad_fd_test(obj, f, df, X, h, verbose)
%GRAD_FD_TEST Compares gradients by finite differences for a model object
% Inputs:
%   obj - model object
%   f   - function that has df as its gradient
%   df  - gradient of the function f
%   X   - point at which the gradient is to be evaluated
%   h   - step-size for finite differences.
%   verbose - extra option for verbose output

    D = size(X, 1);
    if ~exist('verbose', 'var')
        verbose = false;
    end
    
    % Gradient calculation
    gradf = eval(['obj.', df, '(X)']);
    fd_grad = zeros(size(X));

    % Finite differences calculation
    for dd = 1:D
      dX = zeros(D, 1);
      dX(dd) = dX(dd) + h;
      f1 = eval(['obj.', f, '(X + dX)']);
      f2 = eval(['obj.', f, '(X - dX)']);
      fd_grad(dd) = (f1 - f2) / (2 * h);
    end

    % Error calculation
    error = norm(fd_grad - gradf) / norm(fd_grad + gradf);
    
    % Outputting
    if verbose
        fprintf('%7s %12s %15s\n', 'grad', 'fd', 'grad - fd');
        for dd = 1:D
            fprintf('%+6.4e %+12.4e %+12.4e \n', ...
                    gradf(dd), fd_grad(dd), gradf(dd) - fd_grad(dd));
        end
        fprintf('Total Relative Error: %1.4e\n', error)
    end
end
