function C = flatten(A)
% This function downloaded from MathWorks File Exchange on August 31, 2016, from https://www.mathworks.com/matlabcentral/fileexchange/27009-flatten-nested-cell-arrays?focused=6787739&tab=function
% It was modified to include this line, the above note, and a copy of the license information from https://www.mathworks.com/matlabcentral/fileexchange/27009-flatten-nested-cell-arrays?focused=6787739&tab=function#license_modal
% ---------------------------------------------------------------------------
% License information: Copyright (c) 2016, The MathWorks, Inc. 
% All rights reserved.

% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:

% * Redistributions of source code must retain the above copyright 
% notice, this list of conditions and the following disclaimer. 
% * Redistributions in binary form must reproduce the above copyright 
% notice, this list of conditions and the following disclaimer in 
% the documentation and/or other materials provided with the distribution. 
% * In all cases, the software is, and all modifications and derivatives 
% of the software shall be, licensed to you solely for use in conjunction 
% with MathWorks products and service offerings.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.
% ----------------------------------------------------------------------------
% C1 = flatten({{1 {2 3}} {4 5} 6})
% C2 = flatten({{'a' {'b','c'}} {'d' 'e'} 'f'})
% 
% Outputs:
% C1 = 
%     [1]    [2]    [3]    [4]    [5]    [6]
% C2 = 
%     'a'    'b'    'c'    'd'    'e'    'f'
%
% Copyright 2010  The MathWorks, Inc.
C = {};
for i=1:numel(A)  
    if(~iscell(A{i}))
        C = [C,A{i}];
    else
       Ctemp = PTN.flatten(A{i});
       C = [C,Ctemp{:}];
       
    end
end