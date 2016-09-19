function self = preset_opt_free_point_gvm(self)
% preset_opt_free_point_gvm
% Sets the cpca to learn all variables under the point-estimate model for
% all variables using GvM distributions for the latent angle mean field
% distributions.

    self.q_phi_type = 'GvM';

    % Initialise all randomly and low precision
    %--------------------------------------------------------
    self.p_A.xv = tril2(rand(self.M, self.D));
    self.p_B.xv = tril2(rand(self.M, self.D));
    self.p_kappa.xv = rand(self.D, 1);
    self.p_prc2.xv = 1.E-3 * rand();

    % Initialise the mean field random values
    %--------------------------------------------------------
    self.q_phi_k1.xv = repmat(self.p_kappa.xv, [1, self.N]);
    self.q_phi_k2.xv = rand(self.D, self.N);
    self.q_phi_m1.xv = rand(self.D, self.N);
    self.q_phi_m2.xv = rand(self.D, self.N);

    % Gradients
    %--------------------------------------------------------
    self.p_kappa.dx = rand(size(self.p_kappa.xv));
    self.p_A.dx = rand(size(self.p_A.xv));
    self.p_B.dx = rand(size(self.p_B.xv));
    self.p_prc2.dx = rand(size(self.p_prc2.xv));

    self.q_phi_k1.dx = rand(size(self.q_phi_k1.xv));
    self.q_phi_k2.dx = rand(size(self.q_phi_k2.xv));
    self.q_phi_m1.dx = rand(size(self.q_phi_m1.xv));
    self.q_phi_m2.dx = rand(size(self.q_phi_m2.xv));

    % Lower bounds
    %--------------------------------------------------------
    self.p_kappa.lb = zeros(size(self.p_kappa.xv));
    self.p_A.lb = -inf * ones(size(self.p_A.xv));
    self.p_B.lb = -inf * ones(size(self.p_B.xv));
    self.p_prc2.lb = 1E-6;

    self.q_phi_k1.lb = zeros(size(self.q_phi_k1.xv));
    self.q_phi_k2.lb = zeros(size(self.q_phi_k2.xv));
    self.q_phi_m1.lb = -pi * ones(size(self.q_phi_m1.xv));
    self.q_phi_m2.lb = -pi * ones(size(self.q_phi_m2.xv));

    % Upper bounds
    %--------------------------------------------------------
    self.p_kappa.ub = inf * ones(size(self.p_kappa.xv));
    self.p_A.ub = inf * ones(size(self.p_A.xv));
    self.p_B.ub = inf * ones(size(self.p_B.xv));
    self.p_prc2.ub = inf;

    self.q_phi_k1.ub = inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_k2.ub = inf * ones(size(self.q_phi_k2.xv));
    self.q_phi_m1.ub = pi * ones(size(self.q_phi_m1.xv));
    self.q_phi_m2.ub = pi * ones(size(self.q_phi_m2.xv));

    % Moments
    %--------------------------------------------------------
    self.T0 = rand(size(self.q_phi_k1.xv));
    self.T1 = rand(size(self.q_phi_k1.xv));
    self.T2 = rand(size(self.q_phi_k1.xv));
    self.T3 = rand(size(self.q_phi_k1.xv));
    self.T4 = rand(size(self.q_phi_k1.xv));

    % Fix the model variables
    self.name_map('logP_variables') = {'p_kappa', ...
                                      'p_A', ...
                                      'p_B', ...
                                      'p_prc2'};

    self.name_map('logP_subset_ini') = {1,...
                                       [1, 1],...
                                       [1, 1],...
                                       1,...
                                       1};

    self.name_map('logP_subset_end') = {self.D,...
                                       [self.M, self.D],...
                                       [self.M, self.D],...
                                       2,...
                                       1};

    % Find split point
    self.fqVarSplit = size(self.pack_as_array('variables', 'logP'), 1);

    % Add mean field parameters
    self.name_map('h_variables') = {'q_phi_k1',...
                                   'q_phi_k2',...
                                   'q_phi_m1',...
                                   'q_phi_m2'};

    self.name_map('h_subset_ini') = {[1, 1],...
                                    [1, 1],...
                                    [1, 1],...
                                    [1, 1]};

    self.name_map('h_subset_end') = {[self.D, self.N],...
                                    [self.D, self.N],...
                                    [self.D, self.N],...
                                    [self.D, self.N]};

    self.lb_array= self.pack_as_array('lower_bounds', 'Fq');
    self.ub_array= self.pack_as_array('upper_bounds', 'Fq');
    self.var_new = self.pack_as_array('variables', 'Fq');
    self.var_old = rand(size(self.var_new));
    
    self.obj_free = @self.obj_free_energy_point;
    self.grad_free = @self.grad_free_energy_point;
end