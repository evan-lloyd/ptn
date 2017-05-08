function [ net, parms ] = tn_eval(net,parms,order)
% Evaluate the given tensor network in the given order.
% Returns a one-node network containing the evaluation of the network.
  if numel(order) == 1
      return;
  end
  
  [net,parms] = PTN.tn_eval(net, parms, order{1});
  v1 = min(flat(order(1)));
  v1removed = flat(order(1));
  v1removed(v1removed == v1) = [];
  order = updateOrder(order(2:end), v1removed);
      
  [net, parms] = PTN.tn_eval(net, parms, order{1});
  v2 = min(flat(order{1}));
  v2removed = flat(order(1));
  v2removed(v2removed == v2) = [];
  v1 = update(v1, v2removed);
      
  t = [];
  v = [];
  for edge=net.edge'
      if edge(1) == v1 && edge(3) == v2
          t(end+1) = edge(2);
          v(end+1) = edge(4);
      end
  end

  parms{min(v1,v2)} = ttt(parms{v1}, parms{v2}, t, v);
  net = merge(net, v1, v2);
  parms(max(v1,v2)) = [];

  if numel(net.size{min(v1,v2)}) > 1
    parms{min(v1,v2)} = reshape(parms{min(v1,v2)}, net.size{min(v1,v2)});
  elseif numel(net.size{min(v1,v2)}) == 1
    parms{min(v1,v2)} = reshape(parms{min(v1,v2)}, [net.size{min(v1,v2)}, 1]);
  end

  order = updateOrder(order, [v2removed max(v1,v2)]);
  
  order{1} = {min(v1,v2)};
  
  [net,parms] = PTN.tn_eval(net, parms, order);

end

function [net] = merge(in, v1, v2)
    t = [];
    v = [];
    for edge=in.edge'
        if edge(1) == v1 && edge(3) == v2
            t(end+1) = edge(2);
            v(end+1) = edge(4);
        end
    end
    
    net.n = in.n - 1;
    newA = in.arity(v1) + in.arity(v2) - 2 * numel(t);
    net.arity = in.arity;
    net.arity(min(v1,v2)) = newA;
    net.arity(max(v1,v2)) = [];
    d1 = in.size{v1};
    d2 = in.size{v2};
    d1(t) = [];
    d2(v) = [];
    newD = [d1 d2];
    net.size = in.size;
    net.size{min(v1,v2)} = newD;
    net.size(max(v1,v2)) = [];
    net.edge = [];
    
    for edge=in.edge'
        x = edge(1);
        w1 = edge(2);
        y = edge(3);
        w2 = edge(4);
        
        if x == v1
            if any(t == w1)
                continue;
            end
            w1 = update(w1, t);
        end
        
        if y == v1
            if any(t == w2)
                continue;
            end
            w2 = update(w2, t);
        end
        
        % update v2's ways
        if x == v2
            if any(v == w1)
                continue;
            end
            w1 = update(w1, v) + in.arity(v1) - numel(t);
        end
        
        if y == v2
            if any(v == w2)
                continue;
            end
            w2 = update(w2, v) + in.arity(v1) - numel(t);
        end
        
        if x == max(v1,v2)
            x = min(v1,v2);
        end
        
        if y == max(v1,v2)
            y = min(v1,v2);
        end
        
        % update node names coming after removed node (v2)
        if x > max(v1,v2)
            x = x - 1;
        end
        
        if y > max(v1,v2)
            y = y - 1;
        end
        
        net.edge(end+1, :) = [x w1 y w2];
    end
end

function [f] = flat(f)
    f = cell2mat(flatten(f));
end

function [f] = update(f, x)
    f = f - arrayfun(@(y) sum(y > x), f);
end

function [f] = updateOrder(f, x)
    for i=1:numel(f)
        if(~iscell(f{i}))
            f{i} = update(f{i},x);
        else
            f{i} = updateOrder(f{i}, x);
        end
    end
end