function self = get_moments_q_AB(self, get_gradients)
% get_moments_q_AB
% Function to get the moments of the A and B matrices joint distribution.
%
% *** Warning: This function will be substituted in later versions.

    % Get full matrix from blocks
    cov = zeros(2 * self.D);

    AA_cov = reshape(self.q_AA_cov.xv(:), [self.D, self.D, self.M]);
    BB_cov = reshape(self.q_BB_cov.xv(:), [self.D, self.D, self.M]);
    AB_cov = reshape(self.q_AB_cov.xv(:), [self.D, self.D, self.M]);

    for mm = 1:self.M     
        U = [AA_cov(:,:,mm), AB_cov(:,:,mm); ...
             zeros(self.D),  BB_cov(:,:,mm)];
        cov = cov + U' * U;
    end
    % Mean matrices
%     self.m_A.xv = tril2(self.q_A_mu.xv);
%     self.m_B.xv = tril2(self.q_B_mu.xv);
    self.m_A.xv = self.q_A_mu.xv;
    self.m_B.xv = self.q_B_mu.xv;

    % Mean squared matrices
    self.m_AA.xv = self.q_A_mu.xv' * self.q_A_mu.xv ...
              + cov(1:self.D, 1:self.D);

    self.m_BB.xv = self.q_B_mu.xv' * self.q_B_mu.xv ...
              + cov(self.D + 1:2 * self.D, self.D + 1:2 * self.D);

    self.m_AB.xv = self.q_A_mu.xv' * self.q_B_mu.xv ...
              + cov(1:self.D, self.D + 1:2 * self.D);

    if get_gradients
%         self.m_A.dx = tril2(ones(self.M, self.D));
%         self.m_B.dx = tril2(ones(self.M, self.D));
        self.m_A.dx = ones(self.M, self.D);
        self.m_B.dx = ones(self.M, self.D);
    end
end
