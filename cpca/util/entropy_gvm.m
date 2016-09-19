function [h, dh_m1, dh_m2, dh_k1, dh_k2] = entropy_gvm(m1, m2, k1, k2, ...
                                                       T0, T1, T2, T3, T4)
% entropy_gvm
% Entropy of a Generalised von Mises distribution with parameters m1, m2,
% k1 and k2. The partial derivatives with respect to the parameters are
% also provided.

    % If moment estimates are not provided, obtain them
    if nargin < 5
        
        weighting = true;
        n_pts = 1E6;
        
        T0 = gvm_euler(0, m1, m2, k1, k2, weighting, n_pts);
        T1 = gvm_euler(1, m1, m2, k1, k2, weighting, n_pts);
        T2 = gvm_euler(2, m1, m2, k1, k2, weighting, n_pts);
        
        if nargout > 1
            T3 = gvm_euler(3, m1, m2, k1, k2, weighting, n_pts);
            T4 = gvm_euler(4, m1, m2, k1, k2, weighting, n_pts);
        end
        
    end
    
    % Calculate the entropy
    k1_term = k1 .* (cos(m1) .* real(T1) + sin(m1) .* imag(T1));

    k2_term = k2 .* (cos(2 .* m2) .* real(T2) + sin(2 .* m2) .* imag(T2));

    h = log(T0) + k1 + k2 - (k1_term + k2_term) ./ T0;
                 
    % Get the derivatives
    if nargout > 1
        
        % We assemble the derivatives by chain rule
        %------------------------------------------
        
        % Precomputable relations
        
        e_p1im1 = exp(+1.i .* m1);
        e_m1im1 = exp(-1.i .* m1);
        e_p2im2 = exp(+2.i .* m2);
        e_m2im2 = exp(-2.i .* m2);
            
        g = 0.5 .* (k1 .* (e_m1im1 .* T1 + e_p1im1 .* conj(T1)) + ...
                    k2 .* (e_m2im2 .* T2 + e_p2im2 .* conj(T2)));

        g = g ./ 2;
        % Derivatives of the zeroth trigonometric moment
        
        dT0_k1 = +0.5 .* (e_m1im1 .* T1 + e_p1im1 .* conj(T1));
        dT0_k2 = +0.5 .* (e_m2im2 .* T2 + e_p2im2 .* conj(T2));
        dT0_m1 = -0.5i .* k1 .* (e_m1im1 .* T1 - e_p1im1 .* conj(T1));
        dT0_m2 = -1.0i .* k2 .* (e_m2im2 .* T2 - e_p2im2 .* conj(T2));


        % Derivatives of the first trigonometric moment
        
        dT1_k1 = +0.5 .* (e_m1im1 .* T2 + e_p1im1 .* T0);
        dT1_k2 = +0.5 .* (e_m2im2 .* T3 + e_p2im2 .* conj(T1));
        dT1_m1 = -0.5i .* k1 .* (e_m1im1 .* T2 - e_p1im1 .* T0);
        dT1_m2 = -1.0i .* k2 .* (e_m2im2 .* T3 - e_p2im2 .* conj(T1));


        % Derivatives of the second trigonometric moment
        
        dT2_k1 = +0.5 .* (e_m1im1 .* T3 + e_p1im1 .* T1);
        dT2_k2 = +0.5 .* (e_m2im2 .* T4 + e_p2im2 .* T0);
        dT2_m1 = -0.5i .* k1 .* (e_m1im1 .* T3 - e_p1im1 .* T1);
        dT2_m2 = -1.0i .* k2 .* (e_m2im2 .* T4 - e_p2im2 .* T0);
        
        
        % Derivatives with respect to the average log unnormalised density

        dg_k1 = (k1 .* (e_m1im1 .* dT1_k1 + e_p1im1 .* conj(dT1_k1)) + ...
                 k2 .* (e_m2im2 .* dT2_k1 + e_p2im2 .* conj(dT2_k1)) + ...
                        e_m1im1 .* T1 + e_p1im1 .* conj(T1)) ./ 2;


        dg_k2 = (k1 .* (e_m1im1 .* dT1_k2 + e_p1im1 .* conj(dT1_k2)) + ...
                 k2 .* (e_m2im2 .* dT2_k2 + e_p2im2 .* conj(dT2_k2)) + ...
                        e_m2im2 .* T2 + e_p2im2 .* conj(T2)) ./ 2;


        dg_m1 = (k1 .* (e_m1im1 .* (dT1_m1- 1.i .* T1) + ...
                        e_p1im1 .* (conj(dT1_m1) + 1.i .* conj(T1))) + ...
                 k2 .* (e_m2im2 .* dT2_m1 + e_p2im2 .* conj(dT2_m1))) ./ 2;


        dg_m2 = (k1 .* (e_m1im1 .* dT1_m2 + e_p1im1 .* conj(dT1_m2)) + ...
                 k2 .* (e_m2im2 .* (dT2_m2 - 2.i .* T2) + ...
                        e_p2im2 .* (conj(dT2_m2) + 2.i .* conj(T2)))) ./ 2;

        % Compose the derivatives through chain rule
        
        dh_k1 = dT0_k1 ./ T0 - (dg_k1 .* T0 - g .* dT0_k1) ./ (T0 .^ 2);
        dh_k2 = dT0_k2 ./ T0 - (dg_k2 .* T0 - g .* dT0_k2) ./ (T0 .^ 2);
        dh_m1 = dT0_m1 ./ T0 - (dg_m1 .* T0 - g .* dT0_m1) ./ (T0 .^ 2);
        dh_m2 = dT0_m2 ./ T0 - (dg_m2 .* T0 - g .* dT0_m2) ./ (T0 .^ 2);
        
    end

end