function self = get_entropy_q_AB(self, get_gradients)
% get_entropy_q_AB.m
% function to get the moments of the A and B matrices joint distribution.
%
% *** Warning: This function will be substituted in later versions.

    self.h_q_AB = self.M * (self.D + 1) * (1 + log(2 * pi));

    AA_cov = reshape(self.q_AA_cov.xv(:), [self.D, self.D, self.M]);
    BB_cov = reshape(self.q_BB_cov.xv(:), [self.D, self.D, self.M]);

    if get_gradients
        grad_h_AA_cov = zeros(size(AA_cov));
        grad_h_BB_cov = zeros(size(BB_cov));
    end

    for mm = 1:self.M
        diag_AA_cov = diag(AA_cov(:,:, mm));
        diag_BB_cov = diag(BB_cov(:,:, mm));

        % Calculate entropy
        self.h_q_AB = self.h_q_AB ...
                    + sum(log(diag_AA_cov) + log(diag_BB_cov));

        if get_gradients
            grad_h_AA_cov(:,:,mm) = diag(1 ./ diag_AA_cov);
            grad_h_BB_cov(:,:,mm) = diag(1 ./ diag_BB_cov);
        end
    end

    if get_gradients
        self.q_AA_cov.dx = grad_h_AA_cov(:);

        self.q_BB_cov.dx = grad_h_BB_cov(:);

        self.q_AB_cov.dx = zeros(size(self.q_AB_cov.xv));
    end

end
