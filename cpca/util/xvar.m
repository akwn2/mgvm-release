classdef xvar
% xvar
% Class for extended variables to facilitate bookkeeping by grouping the
% variable value, derivatives, bounds and if the variable should be
% exponentially transformed for positivity.
    
    properties
        xv  % value
        dx  % derivative
        lb  % lower bound
        ub  % upper bound
        et  % exponential transformation for positivity
        shape  % dimensions
        s_idx  % subset indices
    end
    
    methods
        function self = xvar()
            % xvar
            % constructor method for extended variable (xvar) class
            self.lb = -inf;
            self.ub = +inf;
            self.et = false;
        end
    end
end