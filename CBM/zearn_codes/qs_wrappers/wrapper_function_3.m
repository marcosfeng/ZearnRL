function [loglik] = wrapper_function_3(parameters, subj)
    % Outcome variable: NNDSVD_student1
    subj.outcome = subj.NNDSVD_student1;
    % Action variable: NNDSVD_teacher3
    subj.action = subj.NNDSVD_teacher3;
    loglik = q_model_simple(parameters, subj);
end