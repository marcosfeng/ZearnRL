function [loglik] = wrapper_function_12(parameters, subj)
    % Outcome variable: NNDSVD_student3
    subj.outcome = subj.NNDSVD_student3;
    % Action variable: NNDSVD_teacher4
    subj.action = subj.NNDSVD_teacher4;
    loglik = logit_model(parameters, subj);
end
