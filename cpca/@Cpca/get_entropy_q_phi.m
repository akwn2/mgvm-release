function self = get_entropy_q_phi(self, get_gradients)
% get_entropy_q_phi
% Calculates the entropy of the latent angles using either a von Mises or a
% Generalised von Mises distribution.

    if strcmp(self.q_phi_type, 'von Mises')
        
        if get_gradients
            [h, dh_m1, dh_k1] = entropy_von_mises(self.q_phi_m1.xv, ...
                                                  self.q_phi_k1.xv);
            self.q_phi_m1.dx = dh_m1;         
            self.q_phi_k1.dx = dh_k1;
        else
            h = entropy_von_mises(self.q_phi_m1.xv, self.q_phi_k1.xv);
        end
    else % strcmp(self.q_phi_type, 'GvM')
        if get_gradients
            [h, dh_m1, dh_m2, dh_k1, dh_k2] = ...
                entropy_gvm(self.q_phi_m1.xv, self.q_phi_m2.xv,...
                            self.q_phi_k1.xv, self.q_phi_k2.xv,...
                            self.T0, self.T1, self.T2, self.T3, self.T4);
                            
            self.q_phi_m1.dx = dh_m1;
            self.q_phi_m2.dx = dh_m2;
            self.q_phi_k1.dx = dh_k1;
            self.q_phi_k2.dx = dh_k2;
        else
            h = entropy_gvm(self.q_phi_m1.xv, self.q_phi_m2.xv,...
                            self.q_phi_k1.xv, self.q_phi_k2.xv,...
                            self.T0, self.T1, self.T2);
        end
    end
    
    self.h_q_phi = sum(h(:));
end
