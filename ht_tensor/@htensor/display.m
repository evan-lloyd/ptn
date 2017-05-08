function display(x)
%DISPLAY Command window display of an htensor.
%
%   DISPLAY(X) displays a hierarchical tensor with its name.
%
%   See also DISPLAY, HTENSOR/DISP, HTENSOR.

% ----------------------------------------------------------
% Hierarchical Tucker Toolbox
% Copyright (c) 2011, C. Tobler and D. Kressner, ETH Zurich
% FreeBSD License, see COPYRIGHT.txt
% email: ctobler@math.ethz.ch
% ----------------------------------------------------------

disp(x,inputname(1));