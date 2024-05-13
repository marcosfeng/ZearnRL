function [loglik, prob, choice, q_values] = wrapper_post_1(parameters, subj)
    % Outcome variable: NNDSVD_student1
    subj.outcome = subj.NNDSVD_student1;
    % Action variable: NNDSVD_teacher2
    subj.action = subj.NNDSVD_teacher2;
    [loglik, prob, choice, q_values] = q_posterior(parameters, subj);
end
