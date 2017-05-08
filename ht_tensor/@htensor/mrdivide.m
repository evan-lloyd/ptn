function x = mrdivide(x, a)
%MRDIVIDE Division by a scalar for htensor.
% 
%   X = MRDIVIDE(X,A) is called for the syntax 'X / A' when X is an
%   htensor and A is a scalar.
% 
%   For htensor-vector multiplication, use TTV.
%   For htensor-matrix multiplication, use TTM.
%   For htensor-htensor multiplication, use TTT.
% 
%   Examples
%   X = htenrandn([3,4,2])
%   W = 5 * X
%
%   See also HTENSOR, HTENSOR/MTIMES

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(~isa(x, 'htensor'))
  error('First argument X must be of class htensor.');
end

if( isscalar(a) )
  x.B{1} = x.B{1}/a;
else
  error('An htensor can only be divided by a scalar.')
end
