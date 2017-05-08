%   Hierarchical Tucker Toolbox, Version 0.8.1
%   -------------------------------------------
%   @htensor:
%
%   apply_mat_to_mat   - Applies A in operator-HTD to B in operator-HTD.
%   apply_mat_to_vec   - Applies A in operator-HTD to htensor X.
%   change_root        - Changes the root of the dimension tree.
%   check_htensor      - Check internal consistency of htensor.
%   conj               - Complex conjugate of an htensor.
%   ctranspose         - Not defined for htensors.
%   define_tree        - Define a dimension tree.
%   disp               - Command window display of an htensor.
%   disp_tree          - Command window display of an htensor's structure.
%   display            - Command window display of an htensor.
%   elem_mult          - Element-by-element multiplication of htensors, with truncation.
%   end                - Last index in one dimension of an htensor.
%   equal_dimtree      - Compares the dimension trees of two htensors.
%   full               - Convert an htensor to a (dense) tensor.
%   full_block         - Return block of htensor as a (dense) tensor.
%   full_leafs         - Converts leaf matrices U to dense matrices.
%   gramians           - Gramians of matricization at each node of an htensor.
%   gramians_cp        - Gramians of matricization at each node of a CP tensor.
%   gramians_nonorthog - Gramians of matricization at each node of an htensor.
%   gramians_sum       - Gramians of matricization at each node of a sum of htensors.
%   htensor            - - Hierarchical Tucker tensor.
%   innerprod          - Efficient inner product for htensor.
%   ipermute           - Inverse permute dimensions of an htensor.
%   left_svd_gramian   - Left singular vectors and singular values from Gramian.
%   left_svd_qr        - Left singular vectors and singular values.
%   minus              - Binary subtraction for htensor.
%   mrdivide           - Division by a scalar for htensor.
%   mtimes             - Multiplication by a scalar for htensor.
%   mttkrp             - Matricized tensor times Khatri-Rao product for htensor.
%   ndims              - Order (number of dimensions) of an htensor.
%   ndofs              - Returns number of degrees of freedom in htensor x.
%   norm               - Tensor norm of an htensor.
%   norm_diff          - Norm of difference between htensor and full tensor.
%   nvecs              - Computes leading mode-n vectors of htensor.
%   orthog             - Orthogonalizes an htensor.
%   permute            - Permute dimensions of an htensor.
%   plot_sv            - Plot singular value tree of htensor.
%   plus               - Binary addition for htensor.
%   plus               - Elementwise square of tensor X.
%   rank               - Hierarchical ranks of an htensor.
%   singular_values    - Singular values of matricization at each node of htensor.
%   size               - Dimensions of htensor.
%   sparse_leafs       - Converts leaf matrices U to sparse matrices.
%   spy                - Show the nonzero structure of the nodes of an htensor.
%   squeeze            - Remove singleton dimensions from an htensor.
%   subsasgn           - Subscripted assignment for an htensor.
%   subsref            - Subscripted reference for htensor.
%   subtree            - Returns all nodes in the subtree of a node.
%   times              - Element-by-element multiplication of htensors.
%   transpose          - Not defined on htensors.
%   trunc_rank         - Rank to truncate at to fulfill requirements.
%   truncate_cp        - Truncates a CP tensor to a lower-rank htensor.
%   truncate_ltr       - Truncate a tensor to an htensor, Leaves-to-Root method.
%   truncate_nonorthog - Truncates an htensor to a lower-rank htensor.
%   truncate_rtl       - Truncate a tensor to an htensor, Root-to-Leaves method.
%   truncate_std       - Truncates an htensor to a lower-rank htensor.
%   truncate_sum       - Truncates a sum of htensors to a low-rank htensor.
%   ttensor            - Converts an htensor to a Tensor Toolbox ttensor.
%   ttm                - Tensor-times-matrix for htensor.
%   ttt                - Tensor-times-tensor for two htensors.
%   ttv                - Tensor-times-vector for htensor.
%   uminus             - Unary minus (-) for htensor.
%   uplus              - Unary plus for htensor.
%
%
% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------
