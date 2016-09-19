function assert_real(x)
% assert_real
% Asserts that the argument is composed only of real numbers and has no
% numerical error tokens inf and nan.
    assert( ~(any(isinf(x)) || any(isnan(x)) || any(~isreal(x))) )
end