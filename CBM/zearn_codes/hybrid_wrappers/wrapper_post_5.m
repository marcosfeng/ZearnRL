function [loglik, prob, choice, q_values] = wrapper_post_5(parameters, subj)
    % Outcome variable: NNDSVD_student2
    subj.outcome = subj.NNDSVD_student2;
    % Action variable: NNDSVD_teacher2
    subj.action = subj.NNDSVD_teacher2;
    [loglik, prob, choice, q_values] = q_posterior(parameters, subj);
end
