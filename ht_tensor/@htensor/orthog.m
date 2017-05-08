function x = orthog(x)
%ORTHOG Orthogonalizes an htensor.
%
%   X = ORTHOG(X) returns an equivalent tensor, where all matrices U{ii}
%   and mode(2, 3)-unfoldings of B{ii} are column-orthogonal (except for
%   the root node).
%
%   Sets the flag IS_ORTHOG of X to true.
%
%   See also HTENSOR
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

% Check if tensor is already orthogonal
if(x.is_orthog == true)
  return;
end

% Go through all nodes except root node, starting from the leafs
for ii=x.nr_nodes:-1:2

  % Calculate QR decomposition of U{ii} or matricized B{ii}, set
  % U{ii} or B{ii} to Q.
  
  if(x.is_leaf(ii))
    
    % Calculate QR-decomposition
    [Q, R] = qr(x.U{ii}, 0);
    
    % Make sure rank doesn't become zero
    if(size(R, 1) == 0)
      Q = ones(size(Q, 1), 1);
      R = ones(1, size(R, 2));
    end
    
    % Set U{ii} to Q
    x.U{ii} = Q;

  else
    % Matricize B{ii}
    B_mat = matricize(x.B{ii}, [1 2], 3);
    
    % Calculate QR-decomposition
    [Q, R] = qr(B_mat, 0);
    
    % Calculate dimensions of "tensor" Q
    tsize_new = size(x.B{ii});
    tsize_new(3) = size(Q, 2);
    
    % Reshape Q to tensor B{ii}
    x.B{ii} = dematricize(Q, tsize_new, [1 2], 3);
    
  end
  
  % Index of parent node
  ii_par = x.parent(ii);
  
  % Multiply R to the parent node's B.
  if(x.is_left(ii))
    % left child of parent node
    x.B{ii_par} = ttm(x.B{ii_par}, R, 1);
  else
    % right child of parent node
    x.B{ii_par} = ttm(x.B{ii_par}, R, 2);
  end

end

% Set flag ORTHOG to true
x.is_orthog = true;