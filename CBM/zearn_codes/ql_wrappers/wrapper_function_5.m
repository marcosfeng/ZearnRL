function [loglik] = wrapper_function_5(parameters, subj)
    % Outcome variable: NNDSVD_student2
    subj.outcome = subj.NNDSVD_student2;
    % Action variable: NNDSVD_teacher1
    subj.action = subj.NNDSVD_teacher1;
    loglik = q_model(parameters, subj);
end
