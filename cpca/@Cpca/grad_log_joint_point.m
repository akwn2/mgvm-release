function ndlogP = grad_log_joint_point(self, vararray)
% grad_log_joint_point
% Calculates the gradient of the average log-joint distribution for the
% point-estimate model.

    self.unpack_from_array(vararray,'variables', 'Fq');

    if self.set_matrices_tril2
        self.p_A.xv = tril2(self.p_A.xv);
        self.p_B.xv = tril2(self.p_B.xv);
    end

    % PRIOR TERMS
    %-----------------------------------------------
    b_ratio = ive(1, self.p_kappa.xv) ./ ive(0, self.p_kappa.xv);

    self.p_kappa.dx = sum(self.m_cos_phi.xv, 2) - self.N * b_ratio;


    % LIKELIHOOD TERMS
    %-----------------------------------------------
    % pre-computable relations
    AA = self.p_A.xv' * self.p_A.xv;
    BB = self.p_B.xv' * self.p_B.xv;
    AB = self.p_A.xv' * self.p_B.xv;

    ccT = (self.m_cos_phi.xv * self.m_cos_phi.xv') ...
        + diag(- diag(self.m_cos_phi.xv * self.m_cos_phi.xv') ...
               + sum(self.m_cos2_phi.xv, 2));

    scT = (self.m_sin_phi.xv * self.m_cos_phi.xv') ...
        + diag(- diag(self.m_sin_phi.xv * self.m_cos_phi.xv') ...
               + sum(self.m_sincos_phi.xv, 2));

    ssT = (self.m_sin_phi.xv * self.m_sin_phi.xv') ...
        + diag(- diag(self.m_sin_phi.xv * self.m_sin_phi.xv') ...
               + sum(self.m_sin2_phi.xv, 2));

    modelPred = self.p_A.xv * self.m_cos_phi.xv ...
              + self.p_B.xv * self.m_sin_phi.xv;

    normTerm = self.trYYT ... 
            - 2 * trace(modelPred * self.Y')...
            + trace(AA * ccT + 2 * AB * scT + BB * ssT);

    % Matrix derivatives
    self.p_A.dx = -self.p_prc2.xv * (-self.Y * self.m_cos_phi.xv' ...
                                + self.p_A.xv * ccT ...
                                + self.p_B.xv * scT)...
              - self.p_A.xv * diag(self.p_alph2.xv);

    self.p_B.dx = -self.p_prc2.xv * (-self.Y * self.m_sin_phi.xv' ...
                                + self.p_B.xv * ssT ...
                                + self.p_A.xv * scT')...
              - self.p_B.xv * diag(self.p_beta2.xv);

    if self.set_matrices_tril2
        self.p_A.dx = tril2(self.p_A.dx);
        self.p_B.dx = tril2(self.p_B.dx);
    end

    % Precision Derivatives (used in chain rule)
    self.p_prc2.dx = + 0.5 * self.N * self.M / self.p_prc2.xv ...
                   - 0.5 * normTerm;

    % GRADIENTS W.R.T. VARIATIONAL PARAMETERS
    %-----------------------------------------------------------

    % 2. Derivatives of the average log joint with respect to
    % the mean field parameters

    % 2.1 Derivatives of the log joint with respect to the
    % average trigonometric quadratic form terms
    dlogp_mc = zeros(self.D, self.N);
    dlogp_ms = zeros(self.D, self.N);
    dlogp_mc2 = zeros(self.D, self.N);
    dlogp_ms2 = zeros(self.D, self.N);
    dlogp_msc = zeros(self.D, self.N);

    for dd = 1:self.D
        dlogp_mc(dd, :) = self.p_kappa.xv(dd) ...
            - self.p_prc2.xv * self.p_A.xv(:, dd)' * (- self.Y ...
               + self.p_A.xv(:, 1:self.D ~= dd) * self.m_cos_phi.xv(1:self.D ~= dd, :) ...
               + self.p_B.xv(:, 1:self.D ~= dd) * self.m_sin_phi.xv(1:self.D ~= dd, :));

        dlogp_ms(dd, :) = - self.p_prc2.xv * self.p_B.xv(:, dd)' *(- self.Y ...
           + self.p_B.xv(:, 1:self.D ~= dd) * self.m_sin_phi.xv(1:self.D ~= dd, :) ...
           + self.p_A.xv(:, 1:self.D ~= dd) * self.m_cos_phi.xv(1:self.D ~= dd, :));

        dlogp_mc2(dd, :) = - 0.5 * self.p_prc2.xv * AA(dd, dd);

        dlogp_ms2(dd, :) = - 0.5 * self.p_prc2.xv * BB(dd, dd);

        dlogp_msc(dd, :) = - self.p_prc2.xv * AB(dd, dd);
    end

    % 3. Assemble the derivatives by the chain rule
    %----------------------------------------------
    self.q_phi_k1.dx = dlogp_mc .* self.g_cos_phi_k1 ...
             + dlogp_ms .* self.g_sin_phi_k1 ...
             + dlogp_mc2 .* self.g_cos2_phi_k1 ...
             + dlogp_ms2 .* self.g_sin2_phi_k1 ...
             + dlogp_msc .* self.g_sincos_phi_k1;

    self.q_phi_m1.dx = dlogp_mc .* self.g_cos_phi_m1 ...
             + dlogp_ms .* self.g_sin_phi_m1 ...
             + dlogp_mc2 .* self.g_cos2_phi_m1 ...
             + dlogp_ms2 .* self.g_sin2_phi_m1 ...
             + dlogp_msc .* self.g_sincos_phi_m1;

         
    if strcmp(self.q_phi_type, 'GvM')
        self.q_phi_k2.dx = dlogp_mc .* self.g_cos_phi_k2 ...
                 + dlogp_ms .* self.g_sin_phi_k2 ...
                 + dlogp_mc2 .* self.g_cos2_phi_k2 ...
                 + dlogp_ms2 .* self.g_sin2_phi_k2 ...
                 + dlogp_msc .* self.g_sincos_phi_k2;

        self.q_phi_m2.dx = dlogp_mc .* self.g_cos_phi_m2 ...
                 + dlogp_ms .* self.g_sin_phi_m2 ...
                 + dlogp_mc2 .* self.g_cos2_phi_m2 ...
                 + dlogp_ms2 .* self.g_sin2_phi_m2 ...
                 + dlogp_msc .* self.g_sincos_phi_m2;
    end

    ndlogP = - self.pack_as_array('gradients', 'Fq');

    % Check for numerical errors
    assert_real(ndlogP);
end