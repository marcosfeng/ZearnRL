function [loglik] = wrapper_function_11(parameters, subj)
    % Action variable: NNDSVD_teacher3
    subj.action = subj.NNDSVD_teacher3;
    loglik = baseline_model(parameters, subj);
end
