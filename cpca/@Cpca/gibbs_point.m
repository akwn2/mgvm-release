function self = gibbs_point(self)
% gibbs_point
% Performs Gibbs sampling on the cpca model based on point estimates of
% matrices A and B.

    if self.verbose
        fprintf('Starting Gibbs sampling for CPCA model\n');
    end
    % Priming Gibbs variates
    self.s_phi = rand(self.D, self.N);

    self.m_cos_phi = cos(self.s_phi);
    self.m_sin_phi = sin(self.s_phi);

    self.p_kappa = rand(self.D, 1);
    
    ee = 1;
    
    for ss = 1:self.S
        
        % Sample conditionals
        self = self.sample_phi_point();
        self = self.sample_A_point();
        self = self.sample_B_point();
        self = self.sample_prc2_point();

        if self.verbose
            fprintf('Completed sample %d of %d.\n', ss, self.S);
        end
        
        % Store samples (using thinning parameter E)
        if mod(ss, self.E) == 0
            self.samples{ee} = self.s_phi;
            ee = ee + 1;
        end
        
    end
end
