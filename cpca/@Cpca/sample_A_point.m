function self = sample_A_point(self)
% sample_A_point
% Sample the coefficient matrix A conditioned on all other variables for
% the point estimate model. This has the form of a Normal distribution.

    if self.verbose 
        fprintf('\tSampling A.\n');
    end

    for dd = 1:self.D
        sum_cos2 = sum(self.m_cos_phi(dd,:) .^ 2);
        ndd = 1:self.D ~= dd;

        a_prc2 = self.p_alph2(dd) + self.p_prc2 * sum_cos2;

        mu_hat = (-2 * self.Y + self.p_A(:, ndd) * self.m_cos_phi(ndd,:) ...
                  +2 * self.p_B(:, ndd) * self.m_sin_phi(ndd,:)) * self.m_cos_phi(dd,:)'; 

        mu_a = 0.5 * mu_hat ./ a_prc2;

        self.p_A(:, dd) = mu_a + randn(self.M, 1) / a_prc2;
    end
end