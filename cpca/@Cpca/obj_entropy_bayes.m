function nH = obj_entropy_bayes(self, vararray)
% obj_entropy_bayes
% Negative entropy function for the free energy maximisation under the
% variational Bayes model.

    % Unpack
    self = self.unpack_from_array(vararray, 'variables', 'Fq');

    % entropy of mean field terms
    %-----------------------------------------------------------

    % Mean field distributions for the ARD parameters
    self = self.get_entropy_q_alph2(false);

    self = self.get_entropy_q_beta2(false);

    % Mean field for the model precision
    self = self.get_entropy_q_prc2(false);

    % Mean field for the coefficient matrices and offsets
    self = self.get_entropy_q_AB(false);

    % Mean field for the latent angles
    self = self.get_entropy_q_phi(false);

    % AGREGATE ENTROPY TERMS
    %----------------------------------------------------------
    nH = - (self.h_q_alph2 + self.h_q_beta2 + self.h_q_prc2 ...
          + self.h_q_AB + self.h_q_phi);

    if isinf(nH) || isnan(nH) || ~isreal(nH)
        fprintf('Something went wrong in obj_entropy\n');
        keyboard;
    end
end