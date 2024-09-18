function [loglik] = wrapper_function_11(parameters, subj)
    % Outcome variable: NNDSVD_student3
    subj.outcome = subj.NNDSVD_student3;
    % Action variable: NNDSVD_teacher3
    subj.action = subj.NNDSVD_teacher3;
    loglik = q_model_simple(parameters, subj);
end
