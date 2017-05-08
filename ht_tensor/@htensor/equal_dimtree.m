function comp = equal_dimtree(x1, x2)
%EQUAL_DIMTREE Compares the dimension trees of two htensors.
%
%   EQUAL_DIMTREE(X1, X2) returns true if the tensors have equal
%   dimension trees, false otherwise.
%
%   Equal trees are needed, e.g., for addition of two htensors.
%
%   Trees that are equal after reordering of node indexes are
%   interpreted as different trees.
%
%   See also HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

if(~isa(x1, 'htensor') || ~isa(x2, 'htensor'))
  error('X1 and X2 must be of class htensor.');
end

comp = isequal(x1.children, x2.children) & ...
       isequal(x1.dims, x2.dims);
