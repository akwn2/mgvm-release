clear;
clc;
table = load('easytide.mat');

pid = 0:1:50;
lat = table.easytide(2, :);
lon = table.easytide(3, :);
data = table.easytide(4:end, :);
clear table

[n_data, n_ports] = size(data);

rlon = repmat(lon, [n_data, 1]);
rlat = repmat(lat, [n_data, 1]);
rpid = repmat(pid, [n_data, 1]);

train_ports = 1 + [0, 1, 2, 7, 8, 9, 12, 13, 14, 15, 16, 22, 23, 24, 25,...
                   28, 29, 30, 32, 33, 37, 39, 40, 45, 49, 50];
valid_ports = 1 + [3, 4, 5, 6, 10, 11, 17, 18, 19, 20, 21, 26, 27, 31, ...
                   34, 35, 36, 38, 41, 42, 43, 44, 46, 47, 48];
% All points
y_all = data(:);
x_all = [rlon(:), rlat(:)];
pid_all = rpid(:);

% Training set
phi_train = data(1:2:end, train_ports);
lon_train = rlon(1:2:end, train_ports);
lat_train = rlat(1:2:end, train_ports);
pid_train = rpid(1:2:end, train_ports);

y_t = phi_train(:);
x_t = [lon_train(:), lat_train(:)];
pid_t = pid_train(:);

% Validation set
phi_valid = data(1:2:end, valid_ports);
lon_valid = rlon(1:2:end, valid_ports);
lat_valid = rlat(1:2:end, valid_ports);
pid_valid = rpid(1:2:end, valid_ports);

y_v = phi_valid(:);
x_v = [lon_valid(:), lat_valid(:)];
pid_v  = pid_valid(:);

% Prediction inputs
x_p = [lon(:), lat(:)];
pid_p = pid(:);

% port identifier

% obs = (1:2:N)';
% hop = (1:2:N)';

% y_train_1 = data(obs, train_ports);
% lon_train_1 = rlon(obs, train_ports);
% lat_train_1 = rlat(obs, train_ports);

% First prediction source: held out data from tide
% y_pred_1 = data(hop, train_ports);
% lon_pred_1 = rlon(hop, train_ports);
% lat_pred_1 = rlat(hop, train_ports);

% Second prediction source: held out data from ports
% y_pred_2 = data(:, pred_ports);
% lon_pred_2 = rlon(:, pred_ports);
% lat_pred_2 = rlat(:, pred_ports);

save('firsttide.mat')