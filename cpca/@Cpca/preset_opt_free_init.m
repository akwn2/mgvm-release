function self = preset_opt_free_init(self, ...
                                     M_ini, M_end,...
                                     D_ini, D_end, ...
                                     N_ini, N_end)
% preset_opt_free_init
% Sets the cpca to learn all variables under the point-estimate model for
% all variables using vM distributions for the latent angle mean field
% distributions.
%
% This function is used for initialisation and has the precision value
% fixed so that it can be used in annealing.


    self.q_phi_type = 'von Mises';

    if nargin < 2
        M_ini = 1;
        M_end = self.M;
    end
    if nargin < 4
        D_ini = 1;
        D_end = self.D;
    end
    if nargin < 6
        N_ini = 1;
        N_end = self.N;
    end

    self.M_ini = M_ini;
    self.M_end = M_end;
    self.D_ini = D_ini;
    self.D_end = D_end;
    self.N_ini = N_ini;
    self.N_end = N_end;

    % Initialise all randomly and low precision
    %--------------------------------------------------------
    self.p_A.xv = tril2(rand(self.M, self.D));
    self.p_B.xv = tril2(rand(self.M, self.D));
    self.p_kappa.xv = 2.5 .* ones(self.D, 1);%rand(self.D, 1);
    self.p_prc2.xv = 1.E-3 * rand();
    
    self.p_alph2.xv = zeros(self.D, 1);
    self.p_beta2.xv = zeros(self.D, 1);
    
    % Initialise the mean field random values
    %--------------------------------------------------------
    self.q_phi_k1.xv = repmat(self.p_kappa.xv, [1, self.N]);
    self.q_phi_m1.xv = rand(self.D, self.N);

    % Gradients
    %--------------------------------------------------------
    self.p_kappa.dx = rand(size(self.p_kappa.xv));
    self.p_A.dx = rand(size(self.p_A.xv));
    self.p_B.dx = rand(size(self.p_B.xv));
    self.p_prc2.dx = rand(size(self.p_prc2.xv));

    self.q_phi_k1.dx = rand(size(self.q_phi_k1.xv));
    self.q_phi_m1.dx = rand(size(self.q_phi_m1.xv));

    % Lower bounds
    %--------------------------------------------------------
    self.p_kappa.lb = zeros(size(self.p_kappa.xv));
    self.p_A.lb = -inf * ones(size(self.p_A.xv));
    self.p_B.lb = -inf * ones(size(self.p_B.xv));
%     self.p_prc2.lb = 1E-6;
    self.p_prc2.lb = -inf;

%     self.q_phi_k1.lb = zeros(size(self.q_phi_k1.xv));
    self.q_phi_k1.lb = -inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_m1.lb = -pi * ones(size(self.q_phi_m1.xv));

    self.p_prc2.et = true;
    self.q_phi_k1.et = true;

    % Upper bounds
    %--------------------------------------------------------
    self.p_kappa.ub = inf * ones(size(self.p_kappa.xv));
    self.p_A.ub = inf * ones(size(self.p_A.xv));
    self.p_B.ub = inf * ones(size(self.p_B.xv));
    self.p_prc2.ub = inf;

    self.q_phi_k1.ub = inf * ones(size(self.q_phi_k1.xv));
    self.q_phi_m1.ub = pi * ones(size(self.q_phi_m1.xv));

    % FIXING THE MODEL VARIABLES
    %--------------------------------------------
    self.name_map('Fq_variables') = {'p_kappa', ...
                                    'p_A', ...
                                    'p_B', ...
                                    'p_prc2'...
                                    'q_phi_k1',...
                                    'q_phi_m1',...
                                    };

    self.name_map('Fq_subset_ini') = { self.D_ini, ...
                                      [self.M_ini, self.D_ini],...
                                      [self.M_ini, self.D_ini],...
                                      1,...
                                      [self.D_ini, self.N_ini],...
                                      [self.D_ini, self.N_ini],...
                                     };

    self.name_map('Fq_subset_end') = { self.D_end, ...
                                      [self.M_end, self.D_end],...
                                      [self.M_end, self.D_end],...
                                      1,...
                                      [self.D_end, self.N_end],...
                                      [self.D_end, self.N_end],...
                                     };

    self.lb_array= self.pack_as_array('lower_bounds', 'Fq');
    self.ub_array= self.pack_as_array('upper_bounds', 'Fq');
    self.var_new = self.pack_as_array('variables', 'Fq');
    self.var_old = rand(size(self.var_new));
    
    self.obj_free = @self.obj_free_energy_point;
    self.grad_free = @self.grad_free_energy_point;
end