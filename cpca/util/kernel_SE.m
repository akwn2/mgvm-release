function [K, dK]= kernel_SE(x1, x2, s2, ell2)
% kernel_SE.m
% implements the squared exponential kernel.

    aux = bsxfun(@minus,sum(x1.^2,2),(2*x1)*x2.');
    norm_sq = bsxfun(@plus,aux.',sum(x2.^2,2));
    
    aux = exp(- 0.5 .* norm_sq ./ ell2);
    K = s2 .* aux;
    
    if any(isnan(K(:))) || any(isinf(K(:)))
        keyboard;
    end
    
    if nargout > 1
        dK = cell(2, 1);
        dK{1} = aux;
        dK{2} = 0.5 .* norm_sq ./ (ell2 .^ 2) .* K;
        
        if any(isnan(dK{1}(:))) || any(isinf(dK{1}(:)))
            keyboard;
        elseif any(isnan(dK{2}(:))) || any(isinf(dK{2}(:)))
            keyboard;
        end
    end
end