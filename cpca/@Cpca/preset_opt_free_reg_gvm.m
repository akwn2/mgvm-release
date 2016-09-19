function self = preset_opt_free_reg_gvm(self)
% preset_opt_free_reg_gvm
% Sets the mgvm for the regression model with GvM mean field.

    self.q_phi_type = 'GvM';

    % Model parameters
    %--------------------------------------------------------
    % Concentration on likelihood
    self.p_kappa.xv = rand(self.D, 1);

    % Model parameter gradients
    %--------------------------------------------------------
    % Prior latent angles
    self.p_kappa.dx = rand(size(self.p_kappa.xv));

    % Model parameter lower bounds
    %--------------------------------------------------------
    % Prior for concentration of latent angles
    self.p_kappa.lb = zeros(size(self.p_kappa.xv));

    % Model parameter upper bounds
    %--------------------------------------------------------
    % Prior for latent angles
    self.p_kappa.ub = inf * ones(size(self.p_kappa.xv));

    % Mean field parameters
    %--------------------------------------------------------
    % Latent angles
    self.q_phi_k1.xv = rand(self.D, self.N);
    self.q_phi_k2.xv = rand(self.D, self.N);
    self.q_phi_m1.xv = rand(self.D, self.N);
    self.q_phi_m2.xv = rand(self.D, self.N);

    self.m_sin_phi.xv = rand(self.D, self.N);
    self.m_cos_phi.xv = rand(self.D, self.N);
    self.m_sin2_phi.xv = rand(self.D, self.N);
    self.m_sincos_phi.xv = rand(self.D, self.N);
    self.m_cos2_phi.xv = rand(self.D, self.N);

    % Mean field parameters gradients
    %--------------------------------------------------------

    % Latent angles
    self.q_phi_k1.dx = rand(size(self.q_phi_k1.xv));
    self.q_phi_k2.dx = rand(size(self.q_phi_k2.xv));
    self.q_phi_m1.dx = rand(size(self.q_phi_m1.xv));
    self.q_phi_m2.dx = rand(size(self.q_phi_m2.xv));

    % Mean field parameter lower bounds
    %--------------------------------------------------------

    % Latent angles
    self.q_phi_k1.lb = 1e-6 * ones(size(self.q_phi_k1.xv));
    self.q_phi_k2.lb = 1e-6 * ones(size(self.q_phi_k2.xv));
    self.q_phi_m1.lb = -pi * ones(size(self.q_phi_m1.xv));
    self.q_phi_m2.lb = -pi / 2 * ones(size(self.q_phi_m2.xv));

    % Mean field parameter upper bounds
    %--------------------------------------------------------

    % Latent angles
    self.q_phi_k1.ub = +inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_k2.ub = +inf * ones(size(self.q_phi_k2.xv));
    self.q_phi_m1.ub = +pi * ones(size(self.q_phi_m1.xv));
    self.q_phi_m2.ub = +pi / 2 * ones(size(self.q_phi_m2.xv));

    % FIXING THE MODEL VARIABLES
    %--------------------------------------------
    self.name_map('Fq_variables') = ...
                                  {'p_kappa', ...
                                   'q_phi_k1', ...
                                   'q_phi_m1', ...
                                   'q_phi_k2', ...
                                   'q_phi_m2', ...
                                  };

    self.name_map('Fq_subset_ini') = ...
                                  {1, ... %'p_kappa'
                                   [1, 1], ... %'q_phi_k1'
                                   [1, 1], ... %'q_phi_m1'
                                   [1, 1], ... %'q_phi_k2'
                                   [1, 1], ... %'q_phi_m2'
                                  };

    self.name_map('Fq_subset_end') = ...
                                  {self.D, ... %'p_kappa'
                                   [self.D, self.N], ... %'q_phi_k1'
                                   [self.D, self.N], ... %'q_phi_m1'
                                   [self.D, self.N], ... %'q_phi_k2'
                                   [self.D, self.N], ... %'q_phi_m2'
                                  };

    self.lb_array= self.pack_as_array('lower_bounds', 'Fq');
    self.ub_array= self.pack_as_array('upper_bounds', 'Fq');
    self.var_new = self.pack_as_array('variables', 'Fq');
    self.var_old = rand(size(self.var_new));
    
    self.obj_free = @self.obj_free_energy_reg;
    self.grad_free = @self.grad_free_energy_reg;
end
