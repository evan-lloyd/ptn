function [ndofs, ndofsB] = ndofs(x)
%NDOFS Returns number of degrees of freedom in htensor x.
%
%   [NDOFS, NDOFSB] = NDOFS(X) returns the number of degrees of
%   freedom (i.e. the number of doubles) used in all nodes, and
%   used only in the non-leaf nodes.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

ndofsB = sum(cellfun(@numel, x.B));
ndofsU = sum(cellfun(@numel, x.U));

ndofs = ndofsU + ndofsB;