%% Initialization
%  Initialize the world, Q-table, and hyperparameters
%Initialize world
world = 1;
s = gwinit(world);
Q = rand(s.ysize,s.xsize, 4);

%setting positions outside the world (illegal moves) to minus infinity
%limit up
Q(1,:,2) = -inf;

%limit down
Q(end,:,1) = -inf;

%limit left
Q(:,1,4) = -inf;

%limit right
Q(:,end,3) = -inf;

%initialize hyperparameters
episodes = 3000; 
a = [1,2,3,4];
a_prob = [1,1,1,1];
eps = 1.0;
eta  = 0.5;
gamma = 0.9;

%% Training loop
%  Train the agent using the Q-learning algorithm.

for i=1:episodes
  
    while s.isterminal==0
        
        %choose and take action
        y = s.pos(1);
        x = s.pos(2);
        [action, oa] = chooseaction(Q, y, x, a, a_prob, eps);
        s = gwaction(action);
        
        %observe new state
        r = s.feedback;
        new_y = s.pos(1);
        new_x = s.pos(2);
        
        %update Q
        Q_max = getvalue(Q);
        Q(y,x,action) = (1-eta)*Q(y,x,action)+eta*(r+gamma*Q_max(new_y, new_x));
        
        %gwdraw()
        
    end
    %if i >2990
     %   gwdraw()
    %end
    %reset robot at new random position
    eps = eps - 1/(episodes);
    s = gwinit(world);
end

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.

s=gwinit(world);

while s.isterminal==0
        
        %choose and take action
        y = s.pos(1);
        x = s.pos(2);
        [action, oa] = chooseaction(Q, y, x, a, a_prob, eps);
        s = gwaction(oa);

        gwdraw()
        
end
    %reset 
%%

P = getpolicy(Q);
gwdrawpolicy(P)