function k = inv_iv0(x)
% inv_iv0
% Finds the argument of a Modified Bessel Function of First Kind for zeroth
% order and produces the input value x using interpolation.

    assert_real(x); % prevent from numerical errors
    if x < iv(0, 700)
        
        % First localise kappa in the coarser grid using binary search
        grid_ini = 0;
        grid_end = 700;
        
        for ii = 1:8 % Results in (grid_end - grid_ini) roughly 5.5
            midpoint = grid_ini + (grid_end - grid_ini) / 2;
            
            if x >= iv(0, midpoint)
                grid_ini = midpoint;
            else
                grid_end = midpoint;
            end
        end
        
        % Find k by interpolation
        grid = linspace(grid_ini, grid_end, 50);
        vals = iv(0, grid);
        k = interp1(vals, grid, x);
        
    else
        fprintf(['!!! Warning: Kappa over 700 required.',...
                'Limiting it to 700.\n']);
        k = 700;
    end

end