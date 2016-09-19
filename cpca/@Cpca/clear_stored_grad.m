function self = clear_stored_grad(self)
    self = self.unpack_from_array(zeros(size(self.var_new)),...
                                  'gradients', 'Fq');
end