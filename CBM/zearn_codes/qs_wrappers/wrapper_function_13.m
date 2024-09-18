function [loglik] = wrapper_function_13(parameters, subj)
    % Outcome variable: NNDSVD_student4
    subj.outcome = subj.NNDSVD_student4;
    % Action variable: NNDSVD_teacher1
    subj.action = subj.NNDSVD_teacher1;
    loglik = q_model_simple(parameters, subj);
end
