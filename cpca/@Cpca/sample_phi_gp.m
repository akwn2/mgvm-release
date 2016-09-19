function self = sample_phi_gp(self)
% sample_phi_gp
% Samples the angles phi using a Generalised von Mises distribution under
% the GP model.

    % Trigonometric terms
    self.m_cos_phi = cos(self.s_phi);
    self.m_sin_phi = sin(self.s_phi);

    for dd = 1:self.D

        rhs1 = self.p_kappa.xv(dd) * self.cos_y(dd, :) ...
               - (Kcc(dd, 1:self.D ~= dd) * ...
                        self.m_cos_phi(1:self.D ~= dd, :) ...
                + Kcs(dd, 1:self.D ~= dd) * ...
                        self.m_sin_phi(1:self.D ~= dd, :) );

        rhs2 = self.p_kappa.xv(dd) * self.sin_y(dd, :) ...
           - (Kss(dd, 1:self.D ~= dd) * ...
                    self.m_sin_phi(1:self.D ~= dd, :) ...
            + Kcs(1:self.D ~= dd, dd)' * ...
                    self.m_cos_phi(1:self.D ~= dd, :) );

        rhs3 = - 0.5 * (Kcc(dd, dd) - Kss(dd, dd));

        rhs4 = - Kcs(dd, dd);

        % Calculate distribution parameters
        z1 = rhs1 + 1.0i * rhs2;
        z2 = rhs3 + 1.0i * rhs4;

        self.q_phi_k1.xv(dd, :) = abs(z1);
        self.q_phi_k2.xv(dd, :) = abs(z2) * ones(1, self.N);
        self.q_phi_m1.xv(dd, :) = angle(z1);
        self.q_phi_m2.xv(dd, :) = 0.5 * angle(z2) * ones(1, self.N);

        K1 = self.q_phi_k1.xv(dd, :)';
        K2 = self.q_phi_k2.xv(dd, :)';
        M1 = self.q_phi_m1.xv(dd, :)';
        M2 = self.q_phi_m2.xv(dd, :)';

        % Sample
        self.s_phi(dd, :) = sample_gvm(M1, M2, K1, K2);

        % Update sine and cosine terms
        self.m_cos_phi(dd, :) = cos(self.s_phi(dd, :));
        self.m_sin_phi(dd, :) = sin(self.s_phi(dd, :));
    end
end
