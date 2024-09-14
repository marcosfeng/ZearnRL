function [loglik] = wrapper_function_2(parameters, subj)
    % Outcome variable: NNDSVD_student1
    subj.outcome = subj.NNDSVD_student1;
    % Action variable: NNDSVD_teacher2
    subj.action = subj.NNDSVD_teacher2;
    loglik = logit_model(parameters, subj);
end
