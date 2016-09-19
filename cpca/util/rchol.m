function [L, jitter] = rchol(K, max_tries)
% rchol
% Cholesky factorisation robust to numerical error by adding isotropic
% jitter to the main diagonal

    if nargin < 2
        max_tries = 5;
    end
    jitter = 0;
    D = size(K,1);
    [L, p] = chol(K, 'lower');
    tries = 0;
    % Attemp jitter if it wasn't possible to decompose K
    if p ~= 0
        fprintf('Warning: \n');
        
        jitter = mean(diag(K)) * 1E-6;      
        while p ~= 0 && tries < max_tries
            [L, p] = chol(K + jitter * eye(D), 'lower');
            jitter = jitter * 10;
            tries = tries + 1;
        end
        
    end
    
    % Throw error if could not perform decomposition
    if tries >= max_tries && p ~= 0
        ME = MException('rchol:notPositiveDefinite', ...
                        'Could not decompose the K supplied.');
        throw(ME)
    end
end