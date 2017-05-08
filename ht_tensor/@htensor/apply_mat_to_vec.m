function Ax = apply_mat_to_vec(A, x)
%APPLY_MAT_TO_VEC Applies A in operator-HTD to htensor X.
%
%   AX = APPLY_MAT_TO_VEC(A, X) applies the htensor A, which
%   represents an operator, to htensor X. The ranks of A and X are
%   multiplied.
%
%   X is an htensor of size n_1 x ... x n_d, while A is an htensor
%   of size m_1 n_1 x ... x m_d n_d. The result Ax is an htensor of
%   size m_1 x ... x m_d.
%
%   Explanation of operator-HTD: A Kronecker-structured matrix
%     sum_i A_(d,i) x ... x A_(1,i), (m1...md x n1...nd)
%   is permuted to form the vector
%     sum_i vec(A_(d,i)) x ... x vec(A_(1,i)), (m1n1...mdnd)
%   which is the vectorization of a CP tensor.
%
%   Examples: See examples/example_operator.m

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

Ax = x;

n = size(x);

nm = size(A);
m = nm./n;

if( any(floor(m) ~= ceil(m)) )
  error('Size mismatch between A and X.')
end

for ii=1:Ax.nr_nodes
  
  if(Ax.is_leaf(ii))
    k_x = size(x.U{ii}, 2);
    k_A = size(A.U{ii}, 2);
    dim = x.dims{ii};
    
    AxU = cell(1, k_A);
    for jj=1:k_A
      Ajj = reshape(A.U{ii}(:, jj), m(dim), n(dim));
      AxU{jj} = Ajj * x.U{ii};
    end
    Ax.U{ii} = cell2mat(AxU);
  else
    sz_A = size(A.B{ii});
    sz_A(end+1:3) = 1;
    sz_x = size(x.B{ii});
    sz_x(end+1:3) = 1;

    % "3-D Kronecker product"
    Ax.B{ii} = zeros(sz_A.*sz_x);
    for jj=1:sz_A(3)
      for kk=1:sz_x(3)      
	Ax.B{ii}(:, :, kk+(jj-1)*sz_x(3)) = ...
	    kron(A.B{ii}(:, :, jj), x.B{ii}(:, :, kk));
      end
    end
  end
  
end

Ax.is_orthog = false;