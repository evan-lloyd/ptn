function example_constructor(x)
%EXAMPLE_CONSTRUCTOR Demonstration of htensor constructors.
%
%   See also HTENSOR.
%

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp('Enter return to continue the demo.')
keyboard

disp('Construct random htensor:')
disp('> x = htenrandn([3 5 6 8])');
x = htenrandn([3 5 6 8])
keyboard

disp('> norm(x - x)/norm(x)')
nrm = norm(x - x)/norm(x)
disp('> norm(orthog(x - x))/norm(x)')
nrm = norm(orthog(x - x))/norm(x)
disp('> norm_nd(full(x) - full(x))/norm(x)')
norm_nd(full(x) - full(x))/norm(x)
keyboard

if(exist('ktensor'))
  disp('Construct from CP decomposition (using ktensor from Tensor Toolbox):')
  disp(['> kt = ktensor(rand(3, 1), {rand(4, 3), rand(5, 3), rand(2, 3), rand(3, 3)})']);
  kt = ktensor(rand(3, 1), {rand(4, 3), rand(5, 3), rand(2, 3), rand(3, 3)})
  keyboard
  disp('> x = htensor(kt)')
  x = htensor(kt)
  keyboard
  disp('> norm(full(x) - full(kt))')
  nrm = norm(full(x) - full(kt))
  keyboard
else
  disp('Construct from CP decomposition:')
  disp(['> cp_decomp = {rand(4, 3), rand(5, 3), rand(2, 3), rand(3, 3)}']);
  cp_decomp = {rand(4, 3), rand(5, 3), rand(2, 3), rand(3, 3)}
  keyboard
  disp('> x = htensor(cp_decomp)')
  x = htensor(cp_decomp)
  keyboard
end

disp('Construct by argument:')
disp('> x2 = htensor(x.children, x.dims, x.U, x.B, x.is_orthog);');
x2 = htensor(x.children, x.dims, x.U, x.B, x.is_orthog);
disp('> norm_nd(full(x) - full(x2))/norm(x)')
norm_nd(full(x) - full(x2))/norm(x)
keyboard

disp('Construct zero tensor:')
disp('> x = htensor([5 2 3 9 5 4])');
x = htensor([5 2 3 9 5 4])
keyboard

disp('disp_tree(x);');
disp_tree(x);
disp('> x2 = htensor([5 2 3 9 5 4], ''first_separate'');');
x2 = htensor([5 2 3 9 5 4], 'first_separate');
disp('disp_tree(x2);');
disp_tree(x2);
disp('> x3 = htensor([5 2 3 9 5 4], ''TT'');');
x3 = htensor([5 2 3 9 5 4], 'TT');
disp('disp_tree(x3);');
disp_tree(x3);
keyboard
