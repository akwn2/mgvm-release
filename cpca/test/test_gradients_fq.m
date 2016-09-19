function error = test_gradients_fq(model, mf_type)
% test_gradients_fq.m
% Unit test for gradients of the negative free energy objective function 
% the implemented models

    fix_seed(0);    % Fix the seed for random number generation
    N = 5;          % Number of data points to be created
    D = 2;          % Number of dimensions of the hidden space
    M = 4;          % Number of dimensions of the hidden space
    noise = 01;     % Noise level to be added to the data
    tol = 1E-7;     % Check gradients

    % Generate data
    kappa = 2 * rand(D, 1);
    mu = - pi + 2 * pi * rand(D, 1);
    phi = vmrand(repmat(mu, [1, N]), repmat(kappa, [1, N]));
    
    A = rand(M, D);
    B = rand(M, D);
    
    Y_true = A * cos(phi) + B * sin(phi);
    Y = Y_true + noise * randn(size(Y_true));
    
    clc
    if strcmp(model, 'Point')
        % Point model
        model = Cpca(Y, D);
        if strcmp(mf_type, 'von Mises')
            model = model.preset_opt_free_point_vm();
        elseif strcmp(mf_type, 'GvM')
            model = model.preset_opt_free_point_gvm();
        end

        error = grad_fd_test(model,...
                            'obj_free_energy_point', ...
                            'grad_free_energy_point', ...
                             model.var_old, tol, true);
    elseif strcmp(model, 'Bayes')
        % Bayes model
        model = Cpca(Y, D);
        if strcmp(mf_type, 'von Mises')
            model = model.preset_opt_free_bayes_vm();
        elseif strcmp(mf_type, 'GvM')
            model = model.preset_opt_free_bayes_gvm();
        end

        error = grad_fd_test(model,...
                            'obj_free_energy_bayes', ...
                            'grad_free_energy_bayes', ...
                             model.var_old, tol, true);
    end
    
    if error > tol
        fprintf('Failed test.\n');
    else
        fprintf('Passed test.\n');
    end
end