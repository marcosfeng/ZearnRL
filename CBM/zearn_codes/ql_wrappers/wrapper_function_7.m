function [loglik] = wrapper_function_7(parameters, subj)
    % Outcome variable: NNDSVD_student2
    subj.outcome = subj.NNDSVD_student2;
    % Action variable: NNDSVD_teacher3
    subj.action = subj.NNDSVD_teacher3;
    loglik = q_model(parameters, subj);
end
