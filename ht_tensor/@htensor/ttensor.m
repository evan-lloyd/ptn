function x_tucker = ttensor(x)
%TTENSOR Converts an htensor to a Tensor Toolbox ttensor.
%
%   Y = TTENSOR(X) returns a Tucker tensor in the Tensor Toolbox
%   format. This function requires the Tensor Toolbox to be installed.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Setting U{ii} to the identity matrix reduces x to its core
% tensor.
U = x.U(x.dim2ind);

for ii=find(x.is_leaf)
  x.U{ii} = eye(x.rank(ii));
end

% Calculate full core tensor
core = tensor(full(x));

% Construct ttensor
x_tucker = ttensor(core, U);