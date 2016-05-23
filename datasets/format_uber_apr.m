%format_uber_apr.m
clear
clc
load('uber_apr.mat');

day_t = [1, 2, 3, 4,  9, 10, 14, 15, 16, 17, 21, 22, 23, 29, 30];
day_p = [5, 6, 7, 8, 11, 12, 13, 18, 19, 20, 24, 25, 26, 27, 28];

idx_t = zeros(size(day));
for dd = day_t
   idx_t = or(idx_t, day == dd); 
end
idx_v = ~idx_t;

phi = minute + 24 * 60 * hour;

% lon_t = lon(idx_t);
% lat_t = lat(idx_t);
% x_t = [lon_t(:), lat_t(:)];
y_t = phi(idx_t);
x_t = day(idx_t);

% lon_v = lon(idx_v);
% lat_v = lat(idx_v);
% x_v = [lon_v(:), lat_v(:)];
y_v = phi(idx_v);
x_v = day(idx_v);

% lon_p = unique(lon_v);
% lat_p = unique(lat_v);
% x_p = [lon_p(:), lat_p(:)];
x_p = unique(day_t);

save('uber_apr_2014.mat', 'y_t', 'x_t', 'y_v', 'x_v', 'x_p')