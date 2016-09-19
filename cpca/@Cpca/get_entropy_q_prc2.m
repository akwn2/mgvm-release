function self = get_entropy_q_prc2(self, get_gradients)
% get_entropy_q_prc2
% Entropy for the mean field distribution for prc2

    if get_gradients
        [h, dh_a, dh_b] = entropy_gamma(self.q_prc2_a.xv, ...
                                        self.q_prc2_b.xv);
                                    
        self.h_q_prc2 = sum(h(:));
        self.q_prc2_a.dx = dh_a;
        self.q_prc2_b.dx = dh_b;
    else
        h = entropy_gamma(self.q_prc2_a.xv, self.q_prc2_b.xv);
        
        self.h_q_prc2 = sum(h(:));
    end
end
