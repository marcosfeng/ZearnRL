function [loglik,prob,choice,theta,w] = wrapper_post_22(parameters, subj)
    % Outcome variable: NNDSVD_student2
    subj.outcome = subj.NNDSVD_student2;
    % Action variable: NNDSVD_teacher2
    subj.action = subj.NNDSVD_teacher2;
    % State variables: NNDSVD_student1 NNDSVD_student4 
    subj.state = [subj.NNDSVD_student1 , subj.NNDSVD_student4 ];
    [loglik,prob,choice,theta,w] = actor_critic_posterior(parameters, subj);
end
