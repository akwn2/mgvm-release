function nH = obj_entropy_reg(self, vararray)
% obj_entropy_reg
% Negative entropy function for the free energy maximisation under the
% regression model.

    % Unpack
    self = self.unpack_from_array(vararray, 'variables', 'Fq');

    % entropy of mean field terms
    %-----------------------------------------------------------

    % Mean field for the latent angles
    self = self.get_entropy_q_phi(false);

    % AGREGATE ENTROPY TERMS
    %----------------------------------------------------------
    nH = - self.h_q_phi;

    assert_real(nH);
end