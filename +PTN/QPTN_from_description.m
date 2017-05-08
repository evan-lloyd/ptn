function [ Q ] = QPTN_from_description( desc, res )
    % Create a QPTN from a QPTN description structure, at the given
    % resolution
    if isfield(desc, 'fixedParameters')
        fixedParameters = desc.fixedParameters;
    else
        fixedParameters = {};
    end
    
    Q = PTN.qptn(desc.arity, desc.size, desc.edge, desc.stoch, fixedParameters, res);
end