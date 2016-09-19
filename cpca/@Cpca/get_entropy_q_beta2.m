function self = get_entropy_q_beta2(self, get_gradients)
% get_entropy_q_beta2.m
% Entropy for the mean field distribution for beta2

    if get_gradients
        [h, dh_a, dh_b] = entropy_gamma(self.q_beta2_a.xv, ...
                                        self.q_beta2_b.xv);
                                    
        self.h_q_beta2 = sum(h(:));
        self.q_beta2_a.dx = dh_a;
        self.q_beta2_b.dx = dh_b;
    else
        h = entropy_gamma(self.q_beta2_a.xv, self.q_beta2_b.xv);
        
        self.h_q_beta2 = sum(h(:));
    end
end