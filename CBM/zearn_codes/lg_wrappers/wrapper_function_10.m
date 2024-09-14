function [loglik] = wrapper_function_10(parameters, subj)
    % Outcome variable: NNDSVD_student3
    subj.outcome = subj.NNDSVD_student3;
    % Action variable: NNDSVD_teacher2
    subj.action = subj.NNDSVD_teacher2;
    loglik = logit_model(parameters, subj);
end
