function AB = apply_mat_to_mat(A, B, p)
%APPLY_MAT_TO_MAT Applies A in operator-HTD to B in operator-HTD.
%
%   AB = APPLY_MAT_TO_MAt(A, B, P) applies the htensor A, which
%   represents an operator, to htensor B, which represents another
%   operator. The ranks of A and B are multiplied.
%
%   A is an htensor of size m_1 p_1 x ... x m_d p_d, while B is an
%   htensor of size p_1 n_1 x ... x p_d n_d. The result AB is
%   an htensor of size m_1 n_1 x ... x m_d n_d. The vector
%   containing (p_1, ..., p_d) is required from the user.
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

AB = A;

np = size(A);
n = np./p;

mp = size(B);
m = mp./p;

if( any(floor(n) ~= ceil(n)) )
  error('Size mismatch between A and P.');
end

if( any(floor(m) ~= ceil(m)) )
  error('Size mismatch between B and P.');
end

for ii=1:AB.nr_nodes
  
  if(AB.is_leaf(ii))
    k_A = size(A.U{ii}, 2);
    k_B = size(B.U{ii}, 2);
    dim = A.dims{ii};
    
    for jj=1:k_A
      A_mat{jj} = reshape(A.U{ii}(:, jj), n(dim), p(dim));
    end
    
    for jj=1:k_A
      B_mat{jj} = reshape(B.U{ii}(:, jj), p(dim), m(dim));
    end
    
    AB_U = cell(k_A, k_B);
    for jj=1:k_A
      for ll=1:k_B
	AB_U{jj, ll} = A_mat{jj} * B_mat{ll};
	AB_U{jj, ll} = AB_U{jj, ll}(:);
      end
    end
    AB.U{ii} = cell2mat(transpose(AB_U(:)));
  else
    sz_A = size(A.B{ii});
    sz_A(end+1:3) = 1;
    sz_B = size(B.B{ii});
    sz_B(end+1:3) = 1;
    
    AB.B{ii} = zeros(sz_A.*sz_B);
    for jj=1:sz_A(3)
      for kk=1:sz_B(3)      
	AB.B{ii}(:, :, kk+(jj-1)*sz_B(3)) = ...
	    kron(A.B{ii}(:, :, jj), B.B{ii}(:, :, kk));
      end
    end
  end
  
end

AB.is_orthog = false;