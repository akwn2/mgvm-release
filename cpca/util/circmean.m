function phi = circmean(th, idx)
% circmean.m
% obtains the circular mean
    if nargin < 2
         idx = 1;
    end
    z = exp(1.j * th);
    phi = angle(mean(z, idx));
end