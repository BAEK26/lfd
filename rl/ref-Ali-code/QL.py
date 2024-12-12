def QL(CorGoal3D, CorObst3D, Corstart3D, DomXleft, DomXright, DomYdown, DomYup, DomZbackward, DomZforward, CellNoX, CellNoY, CellNoZ, PathLink):

    import numpy as np
    import random
        
    # Workspace
    WTablemax=DomYup
    WTablemin=DomYdown
    WTable=WTablemax-WTablemin
    W = CellNoX #8
    WCell=WTable/W

    LTablemax=DomXright
    LTablemin=DomXleft
    LTable=LTablemax-LTablemin
    L = CellNoY #8
    LCell=LTable/L

    DTablemax=DomZforward
    DTablemin=DomZbackward
    DTable=DTablemax-DTablemin
    D = CellNoZ #8
    DCell=DTable/D

    # Action
    Up=0
    Down=1
    Left=2
    Right=3
    Forward=4
    Backward=5
    Actions = [Up,Down,Left,Right,Forward,Backward]

    # Probability of Exploration
    Eps = 0.1
    
    # Step Size
    Alph = 0.9
    
    # Reward
    Reward = -1

    # Cells
    print('Corstart3D :',Corstart3D)
    Xstart= int(WTablemax/WCell-Corstart3D[0]/WCell-0.5) #int(H//2-(Corstart3D[1]//LCell)-1) #int(H//2-(Cstart3D[1]//LCell)-1)
    Ystart= int(Corstart3D[1]/LCell-LTablemin/LCell-0.5) #int(((Corstart3D[0]-WTablemin)//WCell)) #int(((Cstart3D[0]-WTablemin)//WCell))
    Zstart= int(Corstart3D[2]/DCell-DTablemin/DCell-0.5) #int(((Corstart3D[2]-DTablemin)//DCell)+2) #int(((Cstart3D[2]-DTablemin)//DCell))
    Start = [Xstart,Ystart,Zstart]
    print('Start: ',Start)

    print('CorObst3D :',CorObst3D)
    Xobs= int(WTablemax/WCell-CorObst3D[0]/WCell-0.5) #int(H//2-(CorObst3D[1]//LCell)-1) #int(H//2-(CObst3D[1]//LCell)-1)
    Yobs= int(CorObst3D[1]/LCell-LTablemin/LCell-0.5)#int(((CorObst3D[0]-WTablemin)//WCell)) #int(((CObst3D[0]-WTablemin)//WCell))
    Zobs= int(CorObst3D[2]/DCell-DTablemin/DCell-0.5) #int(((CorObst3D[2]-DTablemin)//DCell)+2) #int(((CObst3D[2]-DTablemin)//DCell))
    Obs=   [Xobs,Yobs,Zobs]
    print('Obs: ',Obs)

    print('CorGoal3D :',CorGoal3D)
    Xgoal= int(WTablemax/WCell-CorGoal3D[0]/WCell-0.5) #int(H//2-(CorGoal3D[1]//LCell)) #int(H//2-(CGoal3D[1]//LCell)-1)
    Ygoal= int(CorGoal3D[1]/LCell-LTablemin/LCell-0.5)#int(((CorGoal3D[0]-WTablemin)//WCell)) #int(((CGoal3D[0]-WTablemin)//WCell))
    Zgoal= int(CorGoal3D[2]/DCell-DTablemin/DCell-0.5) #int(((CorGoal3D[2]-DTablemin)//DCell)+2) #int(((CGoal3D[2]-DTablemin)//DCell))
    Goal=  [Xgoal,Ygoal,Zgoal]
    print('Goal: ',Goal)

    def step(state , action):
        i,j,k=state
        if action == Up:
            return [max(i-1,0),j,k]
        elif action == Down:
            return [min(i+1,L-1),j,k]
        elif action == Left:
            return [i,max(j-1,0),k]
        elif action == Right:
            return [i,min(j+1,W-1),k]
        elif action == Forward:
            return [i,j,min(k+1,D-1)]
        elif action == Backward:
            return [i,j,max(k-1,0)]
    
    # Algorithm:
    Q = np.zeros((L,W,D, len(Actions)))                 #[[[ Initialize Q ]]]
    def episode (Q):                       #[[[ Loop for each episode ]]]
        time = 0
        state = Start
    
        # countinue untill reaching Goal  #[[[ Loop for each step of episode ]]]
        while state != Goal:
    
            # punishment define
            if state == Obs :
                PUNISHMENT= -100.0
            else:
                PUNISHMENT= 0
            
        # Explor vs. Exploy               #[[[ Choose A from S of episode ]]]
            if random.randint(0,10)/10 <= Eps :
                action = np.random.choice(Actions)
            else:
                values_ = Q[state[0], state[1],  state[2], :]
                action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

            # Take action                #[[[ Take action A, observe R,S' ]]]
            next_state = step(state, action)
    
            # Update Q                   #[[[ Updating Q ]]]
            Q[state[0], state[1],  state[2], action] = Q[state[0], state[1],  state[2], action] + Alph * (PUNISHMENT + Reward + np.max(Q[next_state[0], next_state[1], next_state[2], :]) - Q[state[0], state[1], state[2], action])
                                         #[[[ S  <---  S' ]]]
            state = next_state
            time += 1
        return time



    Q = np.zeros((L, W, D, len(Actions)))
    episode_limit = 2000
    
    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(Q))
        # time = episode(q_value)
        # episodes.extend([ep] * time)
        ep += 1
    
    steps = np.add.accumulate(steps)



    # Test by numpy
    optimal_policy = np.full(shape=(D,L,W),fill_value='n',dtype=np.str)
    for k in range (0, D):
        for i in range(0, L):
            for j in range(0, W):
                if [i, j, k] == Goal:
                    optimal_policy[k][i][j]='G'
                    continue
                if [i, j, k] == Obs:
                    optimal_policy[k][i][j]='O'
                    continue
        
                bestAction = np.argmax(Q[i, j, k, :])
                if bestAction == Up:
                    optimal_policy[k][i][j]='U'
                elif bestAction == Down:
                    optimal_policy[k][i][j]='D'
                elif bestAction == Left:
                    optimal_policy[k][i][j]='L'
                elif bestAction == Right:
                    optimal_policy[k][i][j]='R'
                elif bestAction == Forward:
                    optimal_policy[k][i][j]='F'
                elif bestAction == Backward:
                    optimal_policy[k][i][j]='B'
                
    '''
    # display the optimal policy
    optimal_policy = []
    for k in range (0, D):
        #optimal_policy.append(['----------------'])
        for i in range(0, H):
            optimal_policy.append([])
            for j in range(0, W):
                if [i, j, k] == Goal:
                    optimal_policy[-1].append('G')
                    continue
                if [i, j, k] == Obs:
                    optimal_policy[-1].append('O')
                    continue
        
                bestAction = np.argmax(Q[i, j, k, :])
                if bestAction == Up:
                    optimal_policy[-1].append('U')
                elif bestAction == Down:
                    optimal_policy[-1].append('D')
                elif bestAction == Left:
                    optimal_policy[-1].append('L')
                elif bestAction == Right:
                    optimal_policy[-1].append('R')
                elif bestAction == Forward:
                    optimal_policy[-1].append('F')
                elif bestAction == Backward:
                    optimal_policy[-1].append('B')
    '''    
    
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)

    ## Pick Up Actions
    Up=1
    Down=2
    Left=3
    Right=4
    Forward=5
    Backward=6

    con=0;
    MPath=[]
    MAction=[]
    i=Start[0]; #Row
    j=Start[1]; #Coloumn
    k=Start[2]; #Depth
    m=0; #index of Path Matrix
    # pick up the start point:
          
    
    while con==0:
        if optimal_policy[k][i][j]=='U':
            MPath.append([i-1+1,j+1,k+1,Up])
            i=i-1;
            j=j;
            k=k;
            m=m+1;
        elif optimal_policy[k][i][j]=='R':
            MPath.append([i+1,j+1+1,k+1,Right])
            i=i;
            j=j+1;
            k=k;
            m=m+1;
        elif optimal_policy[k][i][j]=='D':
            MPath.append([i+1+1,j+1,k+1,Down])
            i=i+1;
            j=j;
            k=k;
            m=m+1;
        elif optimal_policy[k][i][j]=='L':
            MPath.append([i+1,j-1+1,k+1,Left])
            i=i;
            j=j-1;
            k=k;
            m=m+1;
        elif optimal_policy[k][i][j]=='F':
            MPath.append([i+1,j+1,k+1+1,Forward])
            i=i;
            j=j;
            k=k+1;
            m=m+1;
        elif optimal_policy[k][i][j]=='B':
            MPath.append([i+1,j+1,k-1+1,Backward])
            i=i;
            j=j;
            k=k-1;
            m=m+1;
            
            
        else:
            break
               
        
    return MPath, Xstart, Ystart, Zstart
