function example_operator()
%EXAMPLE_OPERATOR Demonstration of operator-HTD format.
%
%   See also HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp('n = 10; d = 4;'); disp(' ');
n = 10; d = 4;

disp(['Lapl = spdiags([-ones(n, 1), 2*ones(n, 1), -ones(n, 1)], ...' ...
      '[-1 0 1], n, n)*(n+1)^2;']);
disp('I = speye(n);'); disp(' ');
Lapl = spdiags([-ones(n, 1), 2*ones(n, 1), -ones(n, 1)], ...
	    [-1 0 1], n, n)*(n+1)^2;
I = speye(n);

disp('x = htenrandn(n*ones(1, d));');
x = htenrandn(n*ones(1, d));

disp('Ax = ttm(x, Lapl, 1);');
disp('for ii=2:d');
disp('  Ax = Ax + ttm(x, Lapl, ii);');
disp('end'); disp(' ');
Ax = ttm(x, Lapl, 1);
for ii=2:d
  Ax = Ax + ttm(x, Lapl, ii);
end

disp('spy(Ax, struct(''title'', ''A*x''));');
spy(Ax, struct('title', 'A*x'));
keyboard;

disp('A_kt = cell(d, 1);');
disp('for ii=1:d');
disp('  A_kt{ii} = repmat(I(:), 1, d);');
disp('  A_kt{ii}(:, ii) = Lapl(:);');
disp('end');
disp('A_htd = htensor(A_kt);'); disp(' ');
A_kt = cell(d, 1);
for ii=1:d
  A_kt{ii} = repmat(I(:), 1, d);
  A_kt{ii}(:, ii) = Lapl(:);
end
A_htd = htensor(A_kt);

disp('spy(A_htd); spy(apply_mat_to_vec(A_htd, x))');
spy(A_htd, struct('title', 'A_htd'));
spy(apply_mat_to_vec(A_htd, x), struct('title', 'A_htd*x'));
keyboard;

disp('plot_sv(A_htd);');
plot_sv(A_htd, struct('title', 'A_htd'));

disp('opts.max_rank = 2;');
disp('A_trunc = truncate(A_htd, opts);');
opts.max_rank = 2;
A_trunc = truncate(A_htd, opts);

fprintf('|| A_htd - A_trunc ||/||A_htd|| = %e\n', ...
	norm(orthog(A_htd - A_trunc))/norm(A_htd));

disp('rank(A_htd)')
rank(A_htd)
disp('rank(A_trunc)')
rank(A_trunc)

disp(['This is a more efficient way of applying the Laplacian to' ...
      ' an htensor. How can we calculate A_trunc directly?']);
keyboard;

disp('Consider only the core tensor: U = [A(:), I(:), ..., I(:)]')
disp(' = [A(:), I(:)]*[1 0 ... 0; 0 1 ... 1].'); disp(' ');

disp('for ii=1:d');
disp('  core_kt{ii} = [ones(1, d); zeros(1, d)];');
disp('  core_kt{ii}(:, ii) = [0; 1];');
disp('end');
for ii=1:d
  core_kt{ii} = [ones(1, d); zeros(1, d)];
  core_kt{ii}(:, ii) = [0; 1];
end
disp('core_htd = htensor(core_kt);'); disp(' ');
core_htd = htensor(core_kt);

disp('core_trunc = truncate(core_htd, opts);');
core_trunc = truncate(core_htd, opts);

disp('spy(core_trunc)');
spy(core_trunc, struct('title', 'core_trunc'))

fprintf('|| core_htd - core_trunc || = %e\n', ...
	norm(orthog(core_htd - core_trunc)));

keyboard;

disp('Consider core_trunc and guess a general decomposition:');
disp(' ');

disp('for ii=2:core_htd.nr_nodes');
disp('  if(core_htd.is_leaf(ii))');
disp('    U{ii} = eye(2);');
disp('  else');
disp('    B{ii} = dematricize([1 0; 0 1; 0 1; 0 0], [2 2 2], [1 2]);');
disp('  end');
disp('end');
disp('B{1} = [0 1; 1 0];');
for ii=2:core_htd.nr_nodes
  if(core_htd.is_leaf(ii))
    U{ii} = eye(2);
  else
    B{ii} = dematricize([1 0; 0 1; 0 1; 0 0], [2 2 2], [1 2]);
  end
end
B{1} = [0 1; 1 0];

disp('core_exact = htensor(core_htd.children, core_htd.dims, U, B);');
core_exact = htensor(core_htd.children, core_htd.dims, U, B);

fprintf('\n|| core_htd - core_exact || = %e\n', ...
	norm(core_htd - core_exact));
keyboard;

disp('for ii=2:A_htd.nr_nodes');
disp('  if(A_htd.is_leaf(ii))');
disp('    U{ii} = [I(:), Lapl(:)];');
disp('  end');
disp('end');
for ii=2:A_htd.nr_nodes
  if(A_htd.is_leaf(ii))
    U{ii} = [I(:), Lapl(:)];
  end
end

disp('A_exact = htensor(A_htd.children, A_htd.dims, U, B);');
A_exact = htensor(A_htd.children, A_htd.dims, U, B);

fprintf('\n|| A_htd - A_exact || = %e\n', ...
	norm(orthog(A_htd - A_exact)));

disp('spy(A_exact)');
spy(A_exact, struct('title', 'A_exact'));

disp('spy(apply_mat_to_vec(A_exact, x))');
spy(apply_mat_to_vec(A_exact, x), struct('title', 'A_exact*x'));

disp(['Other examples can be found in example_spins.m and' ...
      ' example_inv_laplacian.m.'])

keyboard;
