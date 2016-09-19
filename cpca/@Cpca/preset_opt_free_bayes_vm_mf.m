function self = preset_opt_free_bayes_vm_mf(self)
% preset_opt_free_bayes_vm_mf
% Sets the cpca to learn all variables under the variational bayes model
% for all variables using vM distributions for the latent angle mean field
% distributions.

    self.q_phi_type = 'von Mises';

    % Model parameters
    %--------------------------------------------------------
    % Priors for ARD parameters
    self.p_a_alph2.xv = rand(self.D, 1);
    self.p_b_alph2.xv = ones(self.D, 1);

    self.p_a_beta2.xv = rand(self.D, 1);
    self.p_b_beta2.xv = ones(self.D, 1);

    % Prior for model precision
    self.p_a_prc2.xv = rand();
    self.p_b_prc2.xv = 1;

    % Prior for latent angles
    self.p_kappa.xv = rand(self.D, 1);

    % Model parameter gradients
    %--------------------------------------------------------

    % Priors for ARD parameters
    self.p_a_alph2.dx = rand(size(self.p_a_alph2.xv));
    self.p_a_beta2.dx = rand(size(self.p_a_beta2.xv));
    
    % Prior for model precision
    self.p_a_prc2.dx = rand(size(self.p_a_prc2.xv));
    
    % Prior latent angles
    self.p_kappa.dx = rand(size(self.p_kappa.xv));

    % Model parameter lower bounds
    %--------------------------------------------------------

    % Priors for ARD parameters
    self.p_a_alph2.lb = 1E-6 * ones(size(self.p_a_alph2.xv));
    self.p_a_beta2.lb = 1E-6 * ones(size(self.p_a_beta2.xv));
%     self.p_a_alph2.lb = -inf * ones(size(self.p_a_alph2.xv));
%     self.p_a_beta2.lb = -inf * ones(size(self.p_a_beta2.xv));
%     self.p_a_alph2.et = true;
%     self.p_a_beta2.et = true;
    
    % Prior for model precision
    self.p_a_prc2.lb = 1E-6 * ones(size(self.p_a_prc2.xv));
%     self.p_a_prc2.lb = -inf * ones(size(self.p_a_prc2.xv));
%     self.p_a_prc2.et = true;
    
    % Prior for concentration of latent angles
    self.p_kappa.lb = zeros(size(self.p_kappa.xv));
%     self.p_kappa.lb = -inf * ones(size(self.p_kappa.xv));
%     self.p_kappa.et = true;

    % Model parameter upper bounds
    %--------------------------------------------------------

    % Priors for ARD parameters
    self.p_a_alph2.ub = inf * ones(size(self.p_a_alph2.xv));
    self.p_a_beta2.ub = inf * ones(size(self.p_a_beta2.xv));

    % Prior for model precision
    self.p_a_prc2.ub = inf * ones(size(self.p_a_prc2.xv));

    % Prior for latent angles
    self.p_kappa.ub = inf * ones(size(self.p_kappa.xv));

    
    
    % Mean field parameters
    %--------------------------------------------------------

    % ARD priors
    self.q_alph2_a.xv = rand(self.D, 1);
    self.q_alph2_b.xv = ones(self.D, 1);
    self.m_alph2.xv = rand(self.D, 1);

    self.q_beta2_a.xv = rand(self.D, 1);
    self.q_beta2_b.xv = ones(self.D, 1);
    self.m_beta2.xv = rand(self.D, 1);

    % Prior for model precision
    self.q_prc2_a.xv = rand();
    self.q_prc2_b.xv = 1;
    self.m_prc2.xv = rand();

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

    % Mean field for coefficient matrices and offset
    self.q_A_mu.xv = tril2(rand(self.M, self.D));
    self.q_B_mu.xv = tril2(rand(self.M, self.D));
    self.q_AA_cov.xv = rand(self.D, self.D, self.M);
    self.q_BB_cov.xv = rand(self.D, self.D, self.M);
    self.q_AB_cov.xv = rand(self.D, self.D, self.M);
    for mm = 1:self.M
        self.q_AA_cov.xv(:,:,mm) = triu(self.q_AA_cov.xv(:,:,mm));
        self.q_BB_cov.xv(:,:,mm) = triu(self.q_BB_cov.xv(:,:,mm));
    end
    self.q_AA_cov.xv = self.q_AA_cov.xv(:);
    self.q_BB_cov.xv = self.q_BB_cov.xv(:);
    self.q_AB_cov.xv = self.q_AB_cov.xv(:);

    self.m_A.xv = tril2(rand(self.M, self.D));
    self.m_B.xv = tril2(rand(self.M, self.D));
    self.m_AA.xv = rand(self.M, self.D);
    self.m_BB.xv = rand(self.M, self.D);
    self.m_AB.xv = rand(self.M, self.D);

    % Mean field parameters gradients
    %--------------------------------------------------------

    % ARD priors
    self.q_alph2_a.dx = rand(size(self.q_alph2_a.xv));
    self.q_beta2_a.dx = rand(size(self.q_beta2_a.xv));

    % Prior for model precision
    self.q_prc2_a.dx = rand(size(self.q_prc2_a.xv));

    % Latent angles
    self.q_phi_k1.dx = rand(size(self.q_phi_k1.xv));
    self.q_phi_k2.dx = rand(size(self.q_phi_k2.xv));
    self.q_phi_m1.dx = rand(size(self.q_phi_m1.xv));
    self.q_phi_m2.dx = rand(size(self.q_phi_m2.xv));

    % Mean field for coefficient matrices and offset
    self.q_A_mu.dx = tril2(rand(size(self.q_A_mu.xv)));
    self.q_B_mu.dx = tril2(rand(size(self.q_B_mu.xv)));
    self.q_AA_cov.dx = rand(size(self.q_AA_cov.xv));
    self.q_BB_cov.dx = rand(size(self.q_BB_cov.xv));
    self.q_AB_cov.dx = rand(size(self.q_AB_cov.xv));

    % Mean field parameter lower bounds
    %--------------------------------------------------------

    % ARD priors
%     self.q_alph2_a.lb = 1e-6 * ones(size(self.q_alph2_a.xv));
%     self.q_beta2_a.lb = 1e-6 * ones(size(self.q_beta2_a.xv));
    self.q_alph2_a.lb = -inf * ones(size(self.q_alph2_a.xv));
    self.q_beta2_a.lb = -inf * ones(size(self.q_beta2_a.xv));
    self.q_alph2_a.et = true;
    self.q_beta2_a.et = true;
        
    % Prior for model precision
%     self.q_prc2_a.lb = 1e-6 * ones(size(self.q_prc2_a.xv));
    self.q_prc2_a.lb = -inf * ones(size(self.q_prc2_a.xv));
    self.q_prc2_a.et = true;
  
    % Latent angles
%     self.q_phi_k1.lb = 1e-6 * ones(size(self.q_phi_k1.xv));
%     self.q_phi_k2.lb = 1e-6 * ones(size(self.q_phi_k2.xv));
    self.q_phi_k1.lb = -inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_k2.lb = -inf * ones(size(self.q_phi_k2.xv));
    self.q_phi_k1.et = true;
    self.q_phi_k2.et = true;
    
    self.q_phi_m1.lb = -pi * ones(size(self.q_phi_m1.xv));
    self.q_phi_m2.lb = -pi * ones(size(self.q_phi_m2.xv));
%     self.q_phi_m1.lb = -inf * ones(size(self.q_phi_m1.xv));
%     self.q_phi_m2.lb = -inf * ones(size(self.q_phi_m2.xv));

    % Mean field for coefficient matrices and offset
    self.q_A_mu.lb = tril2(-inf * ones(size(self.q_A_mu.xv)));
    self.q_B_mu.lb = tril2(-inf * ones(size(self.q_B_mu.xv)));
    
    self.q_AA_cov.lb = -inf * ones(self.D, self.D, self.M);
    self.q_BB_cov.lb = -inf * ones(self.D, self.D, self.M);
    self.q_AB_cov.lb = -inf * ones(self.D, self.D, self.M);
    for mm = 1:self.M 
        self.q_AA_cov.lb(:,:,mm) = 1E-6 * eye(self.D) ...
            + triu(self.q_AA_cov.lb(:,:,mm), 1);

        self.q_BB_cov.lb(:,:,mm) = 1E-6 * eye(self.D) ...
            + triu(self.q_BB_cov.lb(:,:,mm), 1);
    end
    self.q_AA_cov.lb = self.q_AA_cov.lb(:);
    self.q_BB_cov.lb = self.q_BB_cov.lb(:);
    self.q_AB_cov.lb = self.q_AB_cov.lb(:);

    % Mean field parameter upper bounds
    %--------------------------------------------------------

    % ARD priors
    self.q_alph2_a.ub = +inf * ones(size(self.q_alph2_a.xv));
    self.q_beta2_a.ub = +inf * ones(size(self.q_beta2_a.xv));

    % Prior for model precision
    self.q_prc2_a.ub = +inf * ones(size(self.q_prc2_a.xv));

    % Latent angles
    self.q_phi_k1.ub = +inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_k2.ub = +inf * ones(size(self.q_phi_k2.xv));
    self.q_phi_m1.ub = +pi * ones(size(self.q_phi_m1.xv));
    self.q_phi_m2.ub = +pi * ones(size(self.q_phi_m2.xv));
%     self.q_phi_m1.ub = +inf * ones(size(self.q_phi_m1.xv));
%     self.q_phi_m2.ub = +inf * ones(size(self.q_phi_m2.xv));

    % Mean field for coefficient matrices and offset
    self.q_A_mu.ub = tril2(+inf * ones(size(self.q_A_mu.xv)));
    self.q_B_mu.ub = tril2(+inf * ones(size(self.q_B_mu.xv)));
    self.q_AA_cov.ub = +inf * ones(self.D, self.D, self.M);
    self.q_BB_cov.ub = +inf * ones(self.D, self.D, self.M);
    self.q_AB_cov.ub = +inf * ones(self.D, self.D, self.M);
    
    for mm = 1:self.M
        self.q_AA_cov.ub(:,:,mm) = triu(self.q_AA_cov.ub(:,:,mm));
        self.q_BB_cov.ub(:,:,mm) = triu(self.q_BB_cov.ub(:,:,mm));
    end
    
    self.q_AA_cov.ub = self.q_AA_cov.ub(:);
    self.q_BB_cov.ub = self.q_BB_cov.ub(:);
    self.q_AB_cov.ub = self.q_AB_cov.ub(:);

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