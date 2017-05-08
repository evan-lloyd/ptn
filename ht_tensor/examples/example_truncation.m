function example_truncation()
%EXAMPLE_TRUNCATION Comparison of truncation speed of different methods.
%
%   See also HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp(['This example compares the run-times of different methods of' ...
      ' truncation. This can take some time; alternatively, run-times' ...
      ' from a previous run are plotted.']);

usr_str = input('Load run-times from previous run (Y/n)? ', 's');

if(numel(usr_str) == 0)
  usr_str = 'y';
end

figure;

if(usr_str(1) == 'y' || usr_str(1) == 'Y')
  load('example_truncation_output.mat');
elseif(usr_str(1) ~= 'n' && usr_str(1) ~= 'n')
  disp('Invalid input, loading run-times from previous run');
  load('example_truncation_output.mat');
else
  nr_summands = [2 3 4 7 10];
  
  opts.max_rank = 20;
  opts.rel_eps  = 1e-7;
  
  % Initialize summands
  x_cell = cell(1, 10);
  for ii=1:10
    x_cell{ii} = htenrandn(500*ones(5, 1), 'orthog', '', 20*ones(9, 1));
  end
  
  for ii=numel(nr_summands):-1:1
    
    n = nr_summands(ii)
    
    x = x_cell{1};
    for jj=2:n
      x = x + x_cell{jj};
    end
    
    tic; x_ = truncate(x, opts); time_truncate_std(ii) = toc;
    tic; x_add = htensor.truncate_sum(x_cell(1:n), opts); 
    time_truncate_sum(ii) = toc;
    
    tic;
    x_succ = x_cell{1};
    for jj=2:n
      x_succ = x_succ + x_cell{jj};
      x_succ = truncate(x_succ, opts);
    end
    time_truncate_succ(ii) = toc;
    
    nrm_x = norm(orthog(x));
    err_truncate_std(ii) = norm(orthog(x - x_))/nrm_x;
    err_truncate_sum(ii) = norm(orthog(x - x_add))/nrm_x;
    err_truncate_succ(ii) = norm(orthog(x - x_succ))/nrm_x;
    
    loglog(nr_summands, time_truncate_std , 'bx-', ...
	       nr_summands, time_truncate_sum , 'rx-', ...
	       nr_summands, time_truncate_succ, 'mx-');
    
    x.rank
    x_.rank
    x_add.rank
    x_succ.rank
    
  end
end

loglog(nr_summands, time_truncate_std , 'bx-', ...
       nr_summands, time_truncate_sum , 'rx-', ...
       nr_summands, time_truncate_succ, 'gx-');
hold on;

loglog(nr_summands(end-1:end), 1.2*nr_summands(end-1:end).^4* ...
       time_truncate_std(end)/nr_summands(end)^4, 'b--', ...
       nr_summands(end-1:end), 1.2*nr_summands(end-1:end).^2* ...
       time_truncate_sum(end)/nr_summands(end)^2, 'r--', ...
       nr_summands(end-1:end), 1.2*nr_summands(end-1:end)* ...
       time_truncate_succ(end)/nr_summands(end), 'g--')
       
legend('time truncate std', 'time truncate sum', 'time truncate succ.', ...
       'O(t^4)', 'O(t^2)', 'O(t)', 'Location', 'NorthWest');

xlabel('Number of summands');
ylabel('Runtime [s]');

save('example_truncation_output.mat', 'nr_summands', ...
     'time_truncate_std', 'time_truncate_sum', 'time_truncate_succ', ...
     'err_truncate_std', 'err_truncate_sum', 'err_truncate_succ');