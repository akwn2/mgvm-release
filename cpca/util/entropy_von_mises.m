function [h, dh_m1, dh_k1] = entropy_von_mises(m1, k1)
% entropy_von_mises
% Entropy of a von mises distribution with parameters m1 and k1. The
% partial derivatives with respect to the parameters are also provided.

%     h = k1 + log(2 .* pi .* ive(0, k1)) - k1 .* ive(1, k1) ./ ive(0, k1);
% 
%     % Get the derivatives
%     if nargout > 1
% 
%         dh_m1 = zeros(size(m1));
% 
%         dh_k1 = - k1 .* (0.5 .* ive(2, k1) ./ ive(0, k1) + 1) - ...
%                 (ive(1, k1) ./ ive(0, k1)) .^ 2;
% 
%     end

        ive0 = ive(0, k1);
        ive1 = ive(1, k1);

        h = k1 + log(2 .* pi .* ive0) - k1 .* ive1 ./ ive0;

        if nargout > 1

            ive2 = ive(2, k1);
            dive1 = (ive2 + ive0) ./ 2;

            dh_k1 = - k1 .* (dive1 .* ive0 - ive1 .^ 2) ./ (ive0 .^ 2);

            dh_m1 = zeros(size(m1));

        end
end