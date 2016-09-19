function self = set_var_bayes_vm(self)
% set_var_bayes_vm
% Sets the cpca to learn all variables under the variational bayes model
% for all variables using vM distributions for the latent angle mean field
% distributions.
    % FIXING THE MODEL VARIABLES
    %--------------------------------------------
    self.name_map('Fq_variables') = ...
                                  {'p_a_alph2', ...%'p_b_alph2', ...
                                   'q_alph2_a', ...%'q_alph2_b', ...
                                   'p_a_beta2', ...%'p_b_beta2', ...
                                   'q_beta2_a', ...%'q_beta2_b', ...
                                   'p_a_prc2', ... %'p_b_prc2', ...
                                   'q_prc2_a', ... %'q_prc2_b', ...
                                   'p_kappa', ...
                                   'q_phi_k1', ...
                                   'q_phi_m1', ...
                                   'q_A_mu', ...
                                   'q_B_mu', ...
                                   'q_AA_cov', ...
                                   'q_BB_cov', ...
                                   'q_AB_cov', ...
                                  };

    self.name_map('Fq_subset_ini') = ...
                                  {1, ... %'p_a_alph2' %1, ... %'p_b_alph2'
                                   1, ... %'q_alph2_a' %1, ... %'q_alph2_b'
                                   1, ... %'p_a_beta2' %1, ... %'p_b_beta2'
                                   1, ... %'q_beta2_a' %1, ... %'q_beta2_b'
                                   1, ... %'p_a_prc2'  %1, ... %'p_b_prc2'
                                   1, ... %'q_prc2_a'  %1, ... %'q_prc2_b'
                                   1, ... %'p_kappa'
                                   [1, 1], ... %'q_phi_k1'
                                   [1, 1], ... %'q_phi_m1'
                                   [1, 1], ... %'q_A_mu'
                                   [1, 1], ... %'q_B_mu'
                                   1, ... %'q_AA_cov'
                                   1, ... %'q_BB_cov'
                                   1, ... %'q_AB_cov'
                                  };

    self.name_map('Fq_subset_end') = ...
                                  {self.D, ... %'p_a_alph2' %self.D, ... %'p_b_alph2'
                                   self.D, ... %'q_alph2_a' %self.D, ... %'q_alph2_b'
                                   self.D, ... %'p_a_beta2' %self.D, ... %'p_b_beta2'
                                   self.D, ... %'q_beta2_a' %self.D, ... %'q_beta2_b'
                                   1, ... %'p_a_prc2' %1, ... %'p_b_prc2'
                                   1, ... %'q_prc2_a' %1, ... %'q_prc2_b'
                                   self.D, ... %'p_kappa'
                                   [self.D, self.N], ... %'q_phi_k1'
                                   [self.D, self.N], ... %'q_phi_m1'
                                   [self.M, self.D], ... %'q_A_mu'
                                   [self.M, self.D], ... %'q_B_mu'
                                   self.D * self.D * self.M, ... %'q_AA_cov'
                                   self.D * self.D * self.M, ... %'q_BB_cov'
                                   self.D * self.D * self.M, ... %'q_AB_cov'
                                  };

    self.lb_array= self.pack_as_array('lower_bounds', 'Fq');
    self.ub_array= self.pack_as_array('upper_bounds', 'Fq');
    self.var_new = self.pack_as_array('variables', 'Fq');
    self.var_old = rand(size(self.var_new));
    
    self.obj_free = @self.obj_free_energy_bayes;
    self.grad_free = @self.grad_free_energy_bayes;
end