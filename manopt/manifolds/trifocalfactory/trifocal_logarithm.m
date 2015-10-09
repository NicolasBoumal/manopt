function Log= trifocal_logarithm(Xa,Xb)
% function Log= trifocal_logarithm(Xa,Xb)
%
% Computes the logarithm for the trifocal manifold from the logarithm
% of SO(3)^3 x S_2^3. Finds closest representative of [Xb] to Xa.
% 
% See also:  trifocalfactory

% parameters
depth_limit = 12;  
tol = 10^(-18); 
maxiter = 100;

%
Ta.R1 = Xa(:,1:3);
Ta.R2 = Xa(:,4:6);
Ta.R3 = Xa(:,7:9);
Ta.X  = Xa(:,10:11);

Tb.R1 = Xb(:,1:3);
Tb.R2 = Xb(:,4:6);
Tb.R3 = Xb(:,7:9);
Tb.X  = Xb(:,10:11);

% s = 1
topt1 = FindOptimalT(Ta,Tb);
Rz1 = [cos(topt1) -sin(topt1) 0 ;...
       sin(topt1)  cos(topt1) 0 ; 
       0       0     1];
cost1 = acos((trace(Ta.R1'*Rz1*Tb.R1)-1)/2)^2+...
        acos((trace(Ta.R2'*Rz1*Tb.R2)-1)/2)^2+...
        acos((trace(Ta.R3'*Rz1*Tb.R3)-1)/2)^2+...
        acos(trace(Ta.X'*Rz1*Tb.X))^2;
cost1 = real(cost1);

% s = 2
Rxpi = [1 0 0;0 -1 0; 0 0 -1];

Tc.R1  = Rxpi*Tb.R1;
Tc.R2  = Rxpi*Tb.R2;
Tc.R3  = Rxpi*Tb.R3;
Tc.X   = Rxpi*Tb.X;

topt2 = FindOptimalT(Ta,Tc);
Rz2 = [cos(topt2) -sin(topt2) 0 ;...
       sin(topt2)  cos(topt2) 0;
       0 0 1];
cost2 = acos((trace(Ta.R1'*Rz2*Tc.R1)-1)/2)^2+...
        acos((trace(Ta.R2'*Rz2*Tc.R2)-1)/2)^2+...
        acos((trace(Ta.R3'*Rz2*Tc.R3)-1)/2)^2+...
        acos(trace(Ta.X'*Rz2*Tc.X))^2;
cost2 = real(cost2);

if cost1<cost2
    Topt = Tb;
    topt = topt1;
    cost = cost1;
else
    Topt = Tc;
    topt = topt2;
    cost = cost2;
end


Rzopt = [cos(topt) -sin(topt) 0 ;...
         sin(topt)  cos(topt) 0 ; 
         0          0         1 ];

Xbopt = Rzopt*[Topt.R1 Topt.R2 Topt.R3 Topt.X];

LogT.xi1 =  LogMapRotation(Ta.R1,Rzopt*Topt.R1);
LogT.xi2 =  LogMapRotation(Ta.R2,Rzopt*Topt.R2);
LogT.xi3 =  LogMapRotation(Ta.R3,Rzopt*Topt.R3);
LogT.xi  =  LogMapSphere(Ta.X,Rzopt*Topt.X);
   
Log = [LogT.xi1 LogT.xi2 LogT.xi3 LogT.xi];    
     
 %%
 function current_best_location = FindOptimalT(Ta,Tb)


    [te1, te2] = FindConvexAndConcaveIntervals(Ta,Tb); 
    % [te1,te2]: f_s^2(t) is convex
    % [te2,te1+2*pi] f_s^2(t) is concave

    Q1 = Tb.R1*Ta.R1';
    Q2 = Tb.R2*Ta.R2';
    Q3 = Tb.R3*Ta.R3';

    td = zeros(3,1);
    td(1) = 3*pi/2 - atan2(Q1(1,1)+Q1(2,2),Q1(1,2)-Q1(2,1));
    td(2) = 3*pi/2 - atan2(Q2(1,1)+Q2(2,2),Q2(1,2)-Q2(2,1));
    td(3) = 3*pi/2 - atan2(Q3(1,1)+Q3(2,2),Q3(1,2)-Q3(2,1));


    for i=1:3
       if td(i)<te1 
          td(i)=td(i)+2*pi; 
       end

       if td(i)>te1+2*pi
           td(i)=td(i)-2*pi; 
       end
    end
    
    td = sort(td);

 % solve for the convex region first

    counter = 0;

    if td(1) < te2
        xmin = te1;
        xmax = td(1);
        counter = counter + 1;

        [xopt(counter), fopt(counter)] = NewtonProjected(xmin,xmax,'fTsquared','dotfTsquared','ddotfTsquared',Ta,Tb);

        if td(2) < te2
            xmin = td(1);
            xmax = td(2);
            counter = counter + 1;
            [xopt(counter), fopt(counter)] = NewtonProjected(xmin,xmax,'fTsquared','dotfTsquared','ddotfTsquared',Ta,Tb);

            if td(3) < te2
                xmin = td(2);
                xmax = td(3);
                counter = counter + 1;
                [xopt(counter), fopt(counter)] = NewtonProjected(xmin,xmax,'fTsquared','dotfTsquared','ddotfTsquared',Ta,Tb);

                xmin = td(3);
                xmax = te2;
                counter = counter + 1;
                [xopt(counter), fopt(counter)] = NewtonProjected(xmin,xmax,'fTsquared','dotfTsquared','ddotfTsquared',Ta,Tb);
            else
                xmin = td(2);
                xmax = te2;
                counter = counter + 1;
                [xopt(counter), fopt(counter)] = NewtonProjected(xmin,xmax,'fTsquared','dotfTsquared','ddotfTsquared',Ta,Tb);
            end
        else
            xmin = td(1);
            xmax = te2;
            counter = counter + 1;
            [xopt(counter), fopt(counter)] = NewtonProjected(xmin,xmax,'fTsquared','dotfTsquared','ddotfTsquared',Ta,Tb);
        end

    else
        xmin = te1;
        xmax = te2;
        counter = counter + 1;
        [xopt(counter), fopt(counter)] = NewtonProjected(xmin,xmax,'fTsquared','dotfTsquared','ddotfTsquared',Ta,Tb);
     end   


    [current_best, idx] = min(fopt);
    current_best_location = xopt(idx);

    % branch and bound for the nonconvex region

    if counter==1
       counter = counter + 1;
       % branch and bound in region (te2,td(1))


        xmin = te2;
        xmax = td(1);
        [root, current_best , current_best_location] = split_node(xmin,xmax,0, depth_limit,current_best,current_best_location,Ta,Tb); 

        counter = counter + 1;
        % branch and bound in region (td(1),td(2))
        xmin = td(1);
        xmax = td(2);
        [root, current_best , current_best_location] = split_node(xmin,xmax,0, depth_limit,current_best,current_best_location,Ta,Tb);

        counter = counter + 1;
        % branch and bound in region (td(2),td(3))
        xmin = td(2);
        xmax = td(3);
        [root, current_best , current_best_location] = split_node(xmin,xmax,0, depth_limit,current_best,current_best_location,Ta,Tb);

       counter = counter + 1;
        % branch and bound in region (td(3),te1+2*pi)
         xmin = td(3);
        xmax = te1+2*pi;
        [root, current_best , current_best_location] = split_node(xmin,xmax,0, depth_limit,current_best,current_best_location,Ta,Tb);


    elseif counter==2

        counter = counter + 1;
        xmin = te2;
        xmax = td(2);
        [root, current_best , current_best_location] = split_node(xmin,xmax,0, depth_limit,current_best,current_best_location,Ta,Tb); 
        % branch and bound in region (te2,td(2))

        counter = counter + 1;
        xmin = td(2);
        xmax = td(3);
        [root, current_best , current_best_location] = split_node(xmin,xmax,0, depth_limit,current_best,current_best_location,Ta,Tb); 
        % branch and bound in region (td(2),td(3))

         counter = counter + 1;
        xmin = td(3);
        xmax = te1+2*pi;
        [root, current_best , current_best_location] = split_node(xmin,xmax,0, depth_limit,current_best,current_best_location,Ta,Tb); 
        % branch and bound in region (td(3),te1+2*pi)

    elseif counter==3
        counter = counter + 1;
            xmin = te2;
        xmax = td(3);
        [root, current_best , current_best_location] = split_node(xmin,xmax,0, depth_limit,current_best,current_best_location,Ta,Tb); 
        % branch and bound in region (te2,td(3))

         counter = counter + 1;
            xmin = td(3);
        xmax = te1+2*pi;
        [root, current_best , current_best_location] = split_node(xmin,xmax,0, depth_limit,current_best,current_best_location,Ta,Tb); 
        % branch and bound in region (td(3),te1+2*pi)    
    elseif counter==4

         counter = counter + 1;
        xmin = te2;
        xmax = te1+2*pi;
        [root, current_best , current_best_location] = split_node(xmin,xmax,0, depth_limit,current_best,current_best_location,Ta,Tb); 
        % branch and bound in region (te2,te1+2*pi)       
    end


     end

    %%
    function [tt1, tt2] = FindConvexAndConcaveIntervals(Ta,Tb)

    %a1 = trace(Ta.X'*ezhat*Tb.X);
    a1 = Tb.X(1,1)*Ta.X(2,1) +Tb.X(1,2)*Ta.X(2,2)-...
         Ta.X(1,1)*Tb.X(2,1) -Ta.X(1,2)*Tb.X(2,2);

    %a2 = trace(Ta.X'*Tb.X);
    a2 = Ta.X(1,1)*Tb.X(1,1) +Ta.X(1,2)*Tb.X(1,2)+...
         Ta.X(2,1)*Tb.X(2,1) +Ta.X(2,2)*Tb.X(2,2);

    t1 = atan(a1/a2)+pi; % from pi/2 to 3*pi/2

    t2 = t1 + pi;

    if fSsquared(Ta.X,Tb.X,t1)>fSsquared(Ta.X,Tb.X,t2)
        tmin = t2;
        tmax = t1;
    else
        tmin = t1;
        tmax = t2-2*pi;
    end


    t11= tmax;
    t12 = tmin;

    if ddotfTsquared(Ta,Tb,tmax) >=0 
        tt1 = t11;
        tt2 = t11 + 2*pi;
    else

                for ii=1:maxiter
                    tmiddle = .5*(t11+t12);
                    if ddotfTsquared(Ta,Tb,tmiddle)<0
                        t11 = tmiddle;
                    else
                        t12= tmiddle;
                    end

                    if abs(t11-t12)<tol
                        break;
                    end
                end


                tt1 = t11; % first point where ddot{f}_s(t) = 0


                t11= tmin;
                t12 = tmax+2*pi;
                for ii=1:maxiter
                    tmiddle = .5*(t11+t12);

                    if ddotfTsquared(Ta,Tb,tmiddle)<0
                        t12 = tmiddle;
                    else
                        t11= tmiddle;
                    end

                    if abs(t11-t12)<tol
                        break;
                    end

                end

                tt2 = t11; % second point where ddot{f}_s(t) = 0

    end

    % [tt1,tt2]: f_s(t) is convex
    % [tt2,tt1+2*pi] f_s(t) is concave

end

%%  
function [node ,current_best , current_best_location] = split_node(xmin,xmax,depth, depth_limit,current_best,current_best_location,Ta,Tb)


if (depth == depth_limit)||(xmax-xmin < tol)
    node.terminal = true;
    node.value = 1;
    node.left = [];
    node.right = [];
    return;
end




middlepoint = .5*(xmin + xmax) ; 
node.value = fTsquared(Ta,Tb,middlepoint);

if current_best >  node.value
    current_best = node.value;
    current_best_location = middlepoint;
end



node.terminal = false;
[~, lowerfxleft]   = NewtonProjected(xmin,(xmax+xmin)/2,'costflower','dotflower','dotdotflower',Ta,Tb);
[~, lowerfxright]  = NewtonProjected((xmax+xmin)/2+eps,xmax,'costflower','dotflower','dotdotflower',Ta,Tb);




if lowerfxleft <= lowerfxright

    if lowerfxleft>=current_best 
        node.left = [];
        node.right = [];
        node.terminal = true;
        return;
    end 
   
    [node.left,  current_best, current_best_location] = split_node( xmin,(xmax+xmin)/2,    depth+1, depth_limit,current_best,current_best_location,Ta,Tb);
    [node.right, current_best, current_best_location] = split_node((xmax+xmin)/2+eps,xmax, depth+1, depth_limit,current_best,current_best_location,Ta,Tb);


else
    
    if lowerfxright>=current_best 
        node.left = [];
        node.right = [];
        node.terminal = true;
        return;
    end 
    
    [node.right, current_best, current_best_location] = split_node((xmax+xmin)/2+eps,xmax, depth+1, depth_limit,current_best,current_best_location,Ta,Tb);  
    [node.left,  current_best, current_best_location] = split_node( xmin,(xmax+xmin)/2,    depth+1, depth_limit,current_best,current_best_location,Ta,Tb);


    
    
end

end
%%
function [xnew,fxnew] = NewtonProjected(xmin,xmax,costf,dotf,dotdotf,Ta,Tb)


x0 = (xmin+xmax)/2;

xold = x0;


if strcmp(costf,'fTsquared') 
    
        for i=1:maxiter
           Df = dotfTsquared(Ta,Tb,xold,xmin,xmax);
           DDf = ddotfTsquared(Ta,Tb,xold);            
           xnew = xold - Df/(DDf+tol);      
           xnew = max( min(xmax,xnew), xmin);

           if (abs(xold-xnew) < tol), break; end
           xold = xnew;
        end
        
    fxnew = fTsquared(Ta,Tb,xnew,xmin,xmax);
else

    for i=1:maxiter
           Df = dotflower(Ta,Tb,xold,xmin,xmax);
           DDf = dotdotflower(Ta,Tb,xold);
           xnew = xold - Df/(DDf+tol);
           xnew = max( min(xmax,xnew), xmin);
           if (abs(xold-xnew) < tol), break; end
           xold = xnew;
    end
     
    fxnew = costflower(Ta,Tb,xnew,xmin,xmax);
    
end

end
%%
 function f = fTsquared(Ta,Tb,t,~,~)
      
  f =  fRsquared(Ta.R1,Tb.R1,t) + ...
       fRsquared(Ta.R2,Tb.R2,t) + ...
       fRsquared(Ta.R3,Tb.R3,t) + ...
       fSsquared(Ta.X, Tb.X, t);

end
%%
function  f = costflower(Ta,Tb,t,xmin,xmax)

  f =  fRsquared(Ta.R1,Tb.R1,t) + ...
       fRsquared(Ta.R2,Tb.R2,t) + ...
       fRsquared(Ta.R3,Tb.R3,t) ;
   
   a = ( fSsquared(Ta.X,Tb.X,xmax)- fSsquared(Ta.X,Tb.X,xmin))/(xmax-xmin+tol);
   b = fSsquared(Ta.X,Tb.X,xmin) - a*xmin;
   
   f = f +a*t+b;
   
end
%%
function ddf = ddotfTsquared(Ta,Tb,t)
      
  ddf =  ddotfRsquared(Ta.R1,Tb.R1,t) + ...
         ddotfRsquared(Ta.R2,Tb.R2,t) + ...
         ddotfRsquared(Ta.R3,Tb.R3,t) + ...
         ddotfSsquared(Ta.X,Tb.X,t);


end
%%
function ddf = ddotfSsquared(X,Y,t)

    Rz = [cos(t) -sin(t) 0 ;...
          sin(t)  cos(t) 0 ; 
           0       0     1];

    Q =   Y* X'*Rz;
    theta = real(acos(Q(1,1)+Q(2,2)+Q(3,3)));
    theta = theta+tol;
    alpha = (Q(1,2)-Q(2,1))/(sin(theta));
    as = alpha^2;
    ddf = as + theta*cot(theta)*(1-as);

end
%%
function ddf = dotdotflower(Ta,Tb,t)

 ddf =   ddotfRsquared(Ta.R1,Tb.R1,t) + ...
         ddotfRsquared(Ta.R2,Tb.R2,t) + ...
         ddotfRsquared(Ta.R3,Tb.R3,t);

end

%%
function ddf = ddotfRsquared(Ra,Rb,t)

    Rz = [cos(t) -sin(t) 0 ;...
          sin(t)  cos(t) 0 ;...
           0      0      1];

    Q =  Rb *Ra'*Rz;
    theta = real(acos(.5*(Q(1,1)+Q(2,2)+Q(3,3)-1))) ;

    x = .5 *(Q(1,2)-Q(2,1))/(sin(theta)+tol);
    xs = x^2;
    ddf = xs + .5*(theta*cot(.5*theta+tol))*(1-xs);

end

%%
function df = dotflower(Ta,Tb,t,xmin,xmax)

  df =  dotfRsquared(Ta.R1,Tb.R1,t) + ...
       dotfRsquared(Ta.R2,Tb.R2,t) + ...
       dotfRsquared(Ta.R3,Tb.R3,t);
   
   a = ( fSsquared(Ta.X,Tb.X,xmax)- fSsquared(Ta.X,Tb.X,xmin))/(xmax-xmin+tol);
   
   df = df +a;

end

%%
function dotf= dotfRsquared(Ra,Rb,t)

    Rz = [cos(t) -sin(t) 0;...
          sin(t)  cos(t) 0; 
          0       0      1];

     Q =  Rb*Ra'*Rz;    
     theta = real(acos((Q(1,1)+Q(2,2)+Q(3,3)-1)/2));
     dotf = -.5*(theta/(sin(theta)+tol))*(Q(1,2)-Q(2,1));

end

%%
function dotf= dotfSsquared(X,Y,t)

    Rz = [  cos(t) -sin(t) 0;...
            sin(t)  cos(t) 0; 
            0       0      1];

    Q  =Y*X'*Rz;   
    theta = real(acos(Q(1,1) + Q(2,2) + Q(3,3)));
    dotf = - (theta/(sin(theta)+tol))*(Q(1,2)-Q(2,1));

end

%%
function df = dotfTsquared(Ta,Tb,t,~,~)

	df =       dotfRsquared(Ta.R1,Tb.R1,t);
    df = df +  dotfRsquared(Ta.R2,Tb.R2,t) ;
    df = df +  dotfRsquared(Ta.R3,Tb.R3,t) ;
    df = df +  dotfSsquared(Ta.X,Tb.X,t);
    
end

%%
function f= fSsquared(x,y,t)

   Rz = [cos(t) -sin(t) 0;...
         sin(t)  cos(t) 0; 
          0      0      1];
      
  f = (1/2)*real(acos(trace(x'*Rz*y)))^2;
  

end

%%
function f= fRsquared(Ra,Rb,t)

  Rz = [cos(t) -sin(t) 0; sin(t)  cos(t) 0;0 0 1];
  f = (1/2)*real(acos((trace(Ra'*Rz*Rb)-1)/2))^2;
  
end

%%
function B = LogOfRotation(R) 
    B = real(logm(R));  
end

%%
function v = LogMapSphere(x1, x2)
        xi = x2-x1;
        v = xi - trace(x1'*xi)*x1;
        di = real(acos(trace(x1'*x2)));
        % If the two points are "far apart", correct the norm.
        if di > 1e-6
            nv = norm(v, 'fro');
            v = v * (di / nv);
        end
end
%%
function mylog = LogMapRotation(R1,R2)
    mylog = R1 *  LogOfRotation(R1'*R2);
end

end