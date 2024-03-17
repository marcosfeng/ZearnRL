function [loglik] = wrapper_function_5(parameters, subj)
    % Outcome variable: NNDSVD_student1
    subj.outcome = subj.NNDSVD_student1;
    % Action variable: NNDSVD_teacher3
    subj.action = subj.NNDSVD_teacher3;
    % State variables: NNDSVD_student3 
    subj.state = [subj.NNDSVD_student3 ];
    loglik = actor_critic_model(parameters, subj);
end
