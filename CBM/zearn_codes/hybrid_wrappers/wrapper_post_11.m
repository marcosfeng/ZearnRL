function [loglik, prob, choice] = wrapper_post_11(parameters, subj)
% Outcome variable: NNDSVD_student4
subj.outcome = subj.NNDSVD_student4;
% Action variable: NNDSVD_teacher2
subj.action = subj.NNDSVD_teacher2;
% State variables: NNDSVD_student2 NNDSVD_student3 
subj.state = [subj.NNDSVD_student2 , subj.NNDSVD_student3 ];
[loglik, prob, choice] = hybrid_ac_logit_posterior(parameters, subj);
end