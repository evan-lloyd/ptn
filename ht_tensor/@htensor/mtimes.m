function x = mtimes(a, b)
%MTIMES Multiplication by a scalar for htensor.
% 
%   C = MTIMES(A,B) is called for the syntax 'A * B' when A or B is an
%   htensor and the other argument is a scalar.
% 
%   For htensor-vector multiplication, use TTV.
%   For htensor-matrix multiplication, use TTM.
%   For htensor-tensor multiplication, use TTT.
% 
%   Examples
%   X = htenrandn([3,4,2])
%   W = 5 * X
%
%   See also HTENSOR, HTENSOR/TTV, HTENSOR/TTM, HTENSOR/TTT

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Find which argument is a scalar
if( isscalar(a) )
  x = b;
  x.B{1} = a*x.B{1};
elseif( isscalar(b) )
  x = a;
  x.B{1} = b*x.B{1};
else
  error('One of A and B must be a scalar.')
end
