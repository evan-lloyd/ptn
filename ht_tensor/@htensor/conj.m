function x = conj(x)
%CONJ Complex conjugate of an htensor.
%
%  Returns the complex conjugate of htensor x.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

for ii=1:x.nr_nodes
  if(x.is_leaf(ii))
    x.U{ii} = conj(x.U{ii});
  else
    x.B{ii} = conj(x.B{ii});
  end
end