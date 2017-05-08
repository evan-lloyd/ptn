 x = htenrandn([2 3 4 2]);
y = htenrandn([2 3 4 2]);
A = {rand(2), rand(3), rand(4), rand(2)};
v = {rand(2, 1), rand(3, 1), rand(4, 1), rand(2, 1)};
M = {rand(2, 3), rand(3, 3), rand(4, 3), rand(2, 3)};

check_htensor(x);

x.U

x.is_leaf

x.U{4}

x.U{4} = rand(2, 2);

x{4}

x

disp(x, 'text');

disp_tree(x);

full(x)

full_block(x, [1 2; 2 3; 2 2; 1 2])

x(2, 1, :, 2)

x(2, 3, 1, 2)

x(2, 3, 1, end)

equal_dimtree(x, y)

x_orthog = orthog(x);
disp_tree(x_orthog);

G = gramians(x_orthog)

s = singular_values(x_orthog)

s = innerprod(x, y)

norm(x)
norm(x_orthog)

size(x)
ndims(x)

disp_tree(x + y)
disp_tree(x - y)
disp_tree(-x)

disp_tree(x/2)

disp_tree(2*x)

disp_tree(x.*y)

norm_diff(x, full(x), 100)

disp_tree(permute(x, [3 1 2 4]))

disp_tree(ipermute(permute(x, [3 1 2 4]), [3 1 2 4]))

disp_tree(ttm(x, A))

disp_tree(ttm(x, {A{1}, A{3}}, [1 3]))

disp_tree(ttt(x, y, [1 2 3], [1 2 3]))

disp_tree(change_root(x, 4))

disp_tree(change_root(x, 1))

htensor.subtree(x.children, 2)

disp_tree(ttt(x, y, [1 2]))

mttkrp(x, M, 2)

squeeze(x(2, 1, :, 2))

ttv(x, v)

ttv(x, {v{4}, v{3}}, [4 3])

spy(x)

opts.portrait = false;
opts.TickLabels = true;
opts.style = 'b-';

plot_sv(x, opts)
