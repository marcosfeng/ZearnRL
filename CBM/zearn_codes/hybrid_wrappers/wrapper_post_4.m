function [loglik, prob, choice] = wrapper_post_4(parameters, subj)
subj.outcome = subj.NNDSVD_student1;
subj.action = subj.NNDSVD_teacher2;
subj.state = [subj.NNDSVD_student2 , subj.NNDSVD_student3 , subj.NNDSVD_student4 ];
[loglik, prob, choice] = logit_states_posterior(parameters, subj);
end