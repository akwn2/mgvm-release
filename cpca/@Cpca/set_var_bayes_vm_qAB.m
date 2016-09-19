function self = set_var_bayes_vm_qAB(self)
% set_var_bayes_vm_qAB()
% Sets the cpca to learn all variables under the variational bayes model
% for all variables using vM distributions for the latent angle mean field
% distributions.

    % FIXING THE MODEL VARIABLES
    %--------------------------------------------
    self.name_map('Fq_variables') = {...
                                   'q_A_mu', ...
                                   'q_B_mu', ...
                                   'q_AA_cov', ...
                                   'q_BB_cov', ...
                                   'q_AB_cov', ...
                                  };

    self.name_map('Fq_subset_ini') = {...
                                   [1, 1], ... %'q_A_mu'
                                   [1, 1], ... %'q_B_mu'
                                   1, ... %'q_AA_cov'
                                   1, ... %'q_BB_cov'
                                   1, ... %'q_AB_cov'
                                  };

    self.name_map('Fq_subset_end') = {...
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