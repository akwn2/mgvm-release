function k = inv_log_iv0(x, alpha)
% inv_log_iv0
% Finds the argument of the log of a Modified Bessel Function of First Kind
% and zeroth order weighted by exp(-alpha) using binary search and 
% interpolation.

    assert_real(x); % prevent from numerical errors
    if log(x) < log(ive(0, 1E7)) + 1E7 - alpha
        
        % First localise kappa in the coarser grid using binary search
        grid_ini = 0;
        grid_end = 1E7;
        
        for ii = 1:50
            midpoint = grid_ini + (grid_end - grid_ini) / 2;
            
            if x >= log(ive(0, midpoint)) + midpoint - alpha
                grid_ini = midpoint;
            else
                grid_end = midpoint;
            end
        end

        % Interpolate if the grid is still too coarse.
        if grid_end - grid_ini > 1E7
            grid = linspace(grid_ini, grid_end, 50);
            vals = log(ive(0, grid)) + grid - alpha;
            k = interp1(vals, grid, x);
        else
            k = grid_ini;
        end
        
    else
        fprintf(['!!! Warning: Kappa over 1E7.',...
                'Limiting it to 1E7.\n']);
        k = 1E7;
    end

end
