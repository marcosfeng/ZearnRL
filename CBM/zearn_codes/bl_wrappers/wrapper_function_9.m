function [loglik] = wrapper_function_9(parameters, subj)
    % Action variable: NNDSVD_teacher1
    subj.action = subj.NNDSVD_teacher1;
    loglik = baseline_model(parameters, subj);
end
