function self = sample_B_point(self)
% sample_B_point
% Sample the coefficient matrix B conditioned on all other variables for
% the point estimate model. This has the form of a Normal distribution.

    if self.verbose 
        fprintf('\tSampling B.\n');
    end

    for dd = 1:self.D
        sum_sin2 = sum(self.m_sin_phi(dd, :) .^ 2);
        ndd = 1:self.D ~= dd;

        b_prc2 = self.p_beta2(dd) + self.p_prc2 * sum_sin2;

        mu_hat = (-2 * self.Y ...
                + 2 * self.p_A(:, ndd) * self.m_cos_phi(ndd,:) ...
                    + self.p_B(:, ndd) * self.m_sin_phi(ndd,:)) * self.m_sin_phi(dd,:)'; 

        mu_b = 0.5 * mu_hat ./ b_prc2;

        self.p_B(:, dd) = mu_b + randn(self.M, 1) / b_prc2;
    end
end