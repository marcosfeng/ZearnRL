function [loglik] = wrapper_function_4(parameters, subj)
    % Action variable: NNDSVD_teacher4
    subj.action = subj.NNDSVD_teacher4;
    loglik = baseline_model(parameters, subj);
end
