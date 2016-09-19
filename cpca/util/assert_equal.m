function assert_equal(A, B, msg)
% assert_equal
% Wrapper for asserting equality of two objects A and B using matlab
% built-in functions
    if nargin < 3
        assert(isequal(A, B));
    else
        assert(isequal(A, B), msg);
    end
end