function [f, df] = bfgs_free_energy(self,vararray)
% bfgs_free_energy
% Wrapper function for L-BFGS-B-C solver that encapsulates the objective
% and the gradient function (this is not accounted for in ) 

    f = feval(self.obj_free, vararray);
    if nargout > 1
        df = feval(self.grad_free, vararray);
    end
end