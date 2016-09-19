function self = set_var_bayes_vm_phi_mf(self)
% set_var_bayes_vm_phi_mf
% Sets the cpca to learn all variables under the variational bayes model
% for all variables using vM distributions for the latent angle mean field
% distributions.

    % FIXING THE MODEL VARIABLES
    %--------------------------------------------
    self.name_map('Fq_variables') = ...
                                  {'q_phi_k1', ...
                                   'q_phi_m1', ...
                                   };

    self.name_map('Fq_subset_ini') = ...
                                  {[1, 1], ... %'q_phi_k1'
                                   [1, 1], ... %'q_phi_m1'
                                   };

    self.name_map('Fq_subset_end') = ...
                                  {[self.D, self.N], ... %'q_phi_k1'
                                   [self.D, self.N], ... %'q_phi_m1'
                                   };

    self.lb_array= self.pack_as_array('lower_bounds', 'Fq');
    self.ub_array= self.pack_as_array('upper_bounds', 'Fq');
    self.var_new = self.pack_as_array('variables', 'Fq');
    self.var_old = rand(size(self.var_new));
    
    self.obj_free = @self.obj_free_energy_bayes;
    self.grad_free = @self.grad_free_energy_bayes;
end