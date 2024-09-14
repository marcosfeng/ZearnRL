function [loglik] = wrapper_function_14(parameters, subj)
    % Action variable: NNDSVD_teacher2
    subj.action = subj.NNDSVD_teacher2;
    loglik = baseline_model(parameters, subj);
end
