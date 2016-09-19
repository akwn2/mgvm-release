function self = optimize_free_energy(self)
% optimize_free_energy
% Function that optimises the selected variational free energy function

    % Model selection
    %----------------------------------------------
    funcs.objective = self.obj_free;
    funcs.gradient = self.grad_free;
    options.lb = self.lb_array;
    options.ub = self.ub_array;
    
    % Optimisation section
    %----------------------------------------------
    fprintf('----------------------------------\n');
    fprintf('Starting Free Energy optimisation.\n');
    fprintf('Variational distribution: %s \n', self.q_phi_type);
    fprintf('Solver to be used: %s \n', self.opt_solver);
    fprintf('----------------------------------\n');
    
    
    if strcmp(self.opt_solver, 'IPOPT') % Set IPOPT options
        
        options.ipopt.hessian_approximation = 'limited-memory';
        options.ipopt.mu_strategy = 'monotone';
        options.ipopt.acceptable_tol = self.opt_tol;
        options.ipopt.max_iter = self.opt_max_iter;

        self.var_old = self.var_new;
        [self.var_new, self.opt_info] = ...
            ipopt(self.var_new, funcs, options);
                                          
    elseif strcmp(self.opt_solver, 'U-BFGS') % Set BFGS options

        self.var_old = self.var_new;

        [self.var_new, ~, ~] = minimize_lbfgsb2(self,...
            self.var_new, 'bfgs_free_energy', -self.opt_max_iter);
                                          
    elseif strcmp(self.opt_solver, 'C-BFGS') % CBFGS options

        opts.x0 = self.var_new;
        opts.m  = self.opt_mem;
        opts.pgtol = self.opt_tol;
        opts.maxIts = self.opt_max_iter;
        self.var_old = self.var_new;
        
        [self.var_new, ~, self.opt_info] = ...
            constrained_lbfgsb(@(x)self.bfgs_free_energy(x), ...
                               options.lb, options.ub, opts);
    end
    
    self.fq = -funcs.objective(self.var_new);

    % Unpack to update internal model state
    self = self.unpack_from_array(self.var_new,'variables','Fq');

    fprintf('----------------------------------\n');
    fprintf('Finished Free Energy optimisation.\n');
    fprintf('Final Free Energy = %1.4e.\n', self.fq);
    fprintf('----------------------------------\n');
end