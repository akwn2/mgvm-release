function self = sample_kappa(self)
% sample_kappa
% Sample the prior concentrations conditioned on all other variables under
% the point-estimate model. This has the form of a Bessel Exponential
% distribution.
    if self.verbose 
        fprintf('\tSampling kappa.\n');
    end

    cos_term = sum(cos(self.s_phi), 2);
    ardB0 = -1 / self.N * cos_term;
    eta = self.N * ones(self.D, 1);
    self.p_kappa = berand(eta, ardB0);
end
