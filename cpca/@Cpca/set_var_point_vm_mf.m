function self = set_var_point_vm_mf(self)
% preset_opt_free_point_vm_mf.m
% Sets the cpca to learn only the mean field under the point-estimate model
% using vM distributions for the latent angle mean field distributions.
    self.q_phi_type = 'von Mises';

    % FIXING THE MODEL VARIABLES
    %--------------------------------------------
    self.name_map('Fq_variables') = {...
                                    'q_phi_k1',...
                                    'q_phi_m1',...
                                    };

    self.name_map('Fq_subset_ini') = {...
                                      [1, 1],...
                                      [1, 1],...
                                     };

    self.name_map('Fq_subset_end') = {...
                                      [self.D, self.N],...
                                      [self.D, self.N],...
                                     };

    self.lb_array= self.pack_as_array('lower_bounds', 'Fq');
    self.ub_array= self.pack_as_array('upper_bounds', 'Fq');
    self.var_new = self.pack_as_array('variables', 'Fq');
    self.var_old = rand(size(self.var_new));
    
    self.obj_free = @self.obj_free_energy_point;
    self.grad_free = @self.grad_free_energy_point;
end