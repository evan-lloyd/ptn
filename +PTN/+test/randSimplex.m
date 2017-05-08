function [s] = randSimplex(varargin)
    if numel(varargin) == 1
        varargin = num2cell(varargin{1});
    end
    n = prod([varargin{:}]);
    x = rand(n, 1);
    x = -log(x);
    z = sum(x);
    if numel(varargin) > 1
        s = reshape(x / z, varargin{:});
    else
        s = x / z;
    end
end