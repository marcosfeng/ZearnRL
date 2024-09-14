function [loglik] = wrapper_function_9(parameters, subj)
    % Outcome variable: NNDSVD_student3
    subj.outcome = subj.NNDSVD_student3;
    % Action variable: NNDSVD_teacher1
    subj.action = subj.NNDSVD_teacher1;
    loglik = q_model(parameters, subj);
end
