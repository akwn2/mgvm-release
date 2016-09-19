function argout = image2pi(x, spacing)
% image2pi.m
% imagesc for plots from histograms between -pi and pi. Useful for
% creating errorbar plots in heatmap form from circular data

    if nargin < 2
        spacing = 0.1;
    end
    
    edges = -pi:spacing:pi;
    dims = size(x, 2);
    boxes = length(edges);
    
    y = zeros(boxes, dims);
    for ii = 1:dims
        y(:, ii) = hist(x(:, ii), edges);
    end
    imagesc([1, dims], [-pi, pi], y)
    
    if nargout == 1
        argout = y;
    end
end