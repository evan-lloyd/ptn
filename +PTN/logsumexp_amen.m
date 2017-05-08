function [ s ] = logsumexp_amen( x )
% logexpsum Returns log of the sum of the exponentiation of each of x's
% rows
%   Appropriate for use in amen_cross, with the tensor trains to exp-sum as
%   arguments.
    off = max(x, [], 2);
    s = log(sum(exp(x - repmat(off, 1, size(x,2))), 2)) + off;
end

