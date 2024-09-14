function [loglik] = wrapper_function_8(parameters, subj)
    % Outcome variable: NNDSVD_student2
    subj.outcome = subj.NNDSVD_student2;
    % Action variable: NNDSVD_teacher4
    subj.action = subj.NNDSVD_teacher4;
    loglik = logit_model(parameters, subj);
end
