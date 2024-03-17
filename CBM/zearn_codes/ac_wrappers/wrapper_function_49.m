function [loglik] = wrapper_function_49(parameters, subj)
    % Outcome variable: NNDSVD_student4
    subj.outcome = subj.NNDSVD_student4;
    % Action variable: NNDSVD_teacher2
    subj.action = subj.NNDSVD_teacher2;
    % State variables: NNDSVD_student1 NNDSVD_student2 
    subj.state = [subj.NNDSVD_student1 , subj.NNDSVD_student2 ];
    loglik = actor_critic_model(parameters, subj);
end
