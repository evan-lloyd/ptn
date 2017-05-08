function x = uminus(x)
%UMINUS Unary minus (-) for htensor.
%
%   See also HTENSOR.
%
% 

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Argument checking unnecessary, as x must be of class htensor if
% this function is called

% Change sign of root node B
x.B{1} = -x.B{1};