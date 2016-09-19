function self = sample_prc2_point(self)
% sample_prc2_point
% Samples the precision conditioned on all other variables of the point
% estimate model. This has the form of a inverse gamma distribution.

    if self.verbose 
        fprintf('\tSampling prc2.\n');
    end

    ATA = self.p_A' * self.p_A;
    ATB = self.p_A' * self.p_B;
    BTB = self.p_B' * self.p_B;

    % Norm term within the Likelihood
    ccT = self.m_cos_phi * self.m_cos_phi';
    scT = self.m_sin_phi * self.m_cos_phi';
    ssT = self.m_sin_phi * self.m_sin_phi';

    Y_term = (self.Y - 2 * (self.p_A * self.m_cos_phi + ...
        self.p_B * self.m_sin_phi)) * self.Y';

    AB_term = ATA * ccT + 2 * ATB * scT + BTB * ssT;

    normTerm = trace(Y_term) + trace(AB_term);

    % Sample      
    self.p_prc2 = gamrnd(0.5 * self.M * self.N + 1, 2 / normTerm);
end