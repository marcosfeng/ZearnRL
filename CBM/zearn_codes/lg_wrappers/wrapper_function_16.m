function [loglik] = wrapper_function_16(parameters, subj)
    % Outcome variable: NNDSVD_student4
    subj.outcome = subj.NNDSVD_student4;
    % Action variable: NNDSVD_teacher4
    subj.action = subj.NNDSVD_teacher4;
    loglik = logit_model(parameters, subj);
end