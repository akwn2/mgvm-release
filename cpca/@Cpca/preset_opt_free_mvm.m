function self = preset_opt_free_mvm(self)
% preset_opt_free_mvm
% Sets the cpca to learn all variables under the GP-kernel model for
% all variables using GvM distributions for the latent angle mean field
% distributions.
    self.q_phi_type = 'von Mises';

    % Model parameters
    %--------------------------------------------------------

    % Kernels
    self.p_Ucc.xv = triu(rand(self.D, self.D));
    self.p_Ucs.xv = rand(self.D, self.D);
    self.p_Uss.xv = triu(rand(self.D, self.D));

    self.p_Ucc.xv = self.p_Ucc.xv + self.p_Ucc.xv';
    self.p_Uss.xv = self.p_Uss.xv + self.p_Uss.xv';

    % Concentration on likelihood
    self.p_kappa.xv = rand(self.D, 1);

    % Model parameter gradients
    %--------------------------------------------------------

    % Kernels
    self.p_Ucc.dx = triu(rand(size(self.p_Ucc.xv)));
    self.p_Ucs.dx = rand(size(self.p_Ucs.xv));
    self.p_Uss.dx = triu(rand(size(self.p_Uss.xv)));

    self.p_Ucc.dx = self.p_Ucc.dx + self.p_Ucc.dx';
    self.p_Uss.dx = self.p_Uss.dx + self.p_Uss.dx';

    % Prior latent angles
    self.p_kappa.dx = rand(size(self.p_kappa.xv));

    % Model parameter lower bounds
    %--------------------------------------------------------

    % Kernels
    self.p_Ucc.lb = 1E-6 * triu(ones(size(self.p_Ucc.xv)));
    self.p_Ucs.lb = 1E-6 * ones(size(self.p_Ucs.xv));
    self.p_Uss.lb = 1E-6 * triu(ones(size(self.p_Uss.xv)));

    % Prior for concentration of latent angles
%     self.p_kappa.lb = 1E-6 * ones(size(self.p_kappa.xv));
    self.p_kappa.lb = -inf * ones(size(self.p_kappa.xv));
    self.p_kappa.et = true;
    

    % Model parameter upper bounds
    %--------------------------------------------------------

    % Kernels
    self.p_Ucc.ub = inf * triu(ones(size(self.p_Ucc.xv)));
    self.p_Ucs.ub = inf * ones(size(self.p_Ucs.xv));
    self.p_Uss.ub = inf * triu(ones(size(self.p_Uss.xv)));

    % Prior for latent angles
    self.p_kappa.ub = inf * ones(size(self.p_kappa.xv));

    % Mean field parameters
    %--------------------------------------------------------

    % Latent angles
    self.q_phi_k1.xv = rand(self.D, self.N);
    self.q_phi_m1.xv = rand(self.D, self.N);

    self.m_sin_phi.xv = rand(self.D, self.N);
    self.m_cos_phi.xv = rand(self.D, self.N);
    self.m_sin2_phi.xv = rand(self.D, self.N);
    self.m_sincos_phi.xv = rand(self.D, self.N);
    self.m_cos2_phi.xv = rand(self.D, self.N);

    % Mean field parameters gradients
    %--------------------------------------------------------

    % Latent angles
    self.q_phi_k1.dx = rand(size(self.q_phi_k1.xv));
    self.q_phi_m1.dx = rand(size(self.q_phi_m1.xv));

    % Mean field parameter lower bounds
    %--------------------------------------------------------

    % Latent angles
%     self.q_phi_k1.lb = 1e-6 * ones(size(self.q_phi_k1.xv));
    self.q_phi_k1.lb = -inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_k1.et = true;
    
    self.q_phi_m1.lb = -pi * ones(size(self.q_phi_m1.xv));

    % Mean field parameter upper bounds
    %--------------------------------------------------------

    % Latent angles
    self.q_phi_k1.ub = +inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_m1.ub = +pi * ones(size(self.q_phi_m1.xv));

    % FIXING THE MODEL VARIABLES
    %--------------------------------------------
    self.name_map('Fq_variables') = ...
                                  {'p_Ucc', ...
                                   'p_Ucs', ...
                                   'p_Uss', ...
                                   'p_kappa', ...
                                   'q_phi_k1', ...
                                   'q_phi_m1', ...
                                  };

    self.name_map('Fq_subset_ini') = ...
                                  {[1, 1], ... %'p_Ucc', ...
                                   [1, 1], ... %'p_Ucs', ...
                                   [1, 1], ... %'p_Uss', ...
                                   1, ... %'p_kappa'
                                   [1, 1], ... %'q_phi_k1'
                                   [1, 1], ... %'q_phi_m1'
                                  };

    self.name_map('Fq_subset_end') = ...
                                  {[self.D, self.D], ... %'p_Ucc', ...
                                   [self.D, self.D], ... %'p_Ucs', ...
                                   [self.D, self.D], ... %'p_Uss', ...
                                   self.D, ... %'p_kappa'
                                   [self.D, self.N], ... %'q_phi_k1'
                                   [self.D, self.N], ... %'q_phi_m1'
                                  };

    self.lb_array= self.pack_as_array('lower_bounds', 'Fq');
    self.ub_array= self.pack_as_array('upper_bounds', 'Fq');
    self.var_new = self.pack_as_array('variables', 'Fq');
    self.var_old = rand(size(self.var_new));
    
    self.obj_free = @self.obj_free_energy_mvm;
    self.grad_free = @self.grad_free_energy_mvm;
end
