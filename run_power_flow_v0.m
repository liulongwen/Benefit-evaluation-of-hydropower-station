clc;
clear;
clear all;
warning off;
%add the locations of examples and base programs
addpath('D:\Program Files\Polyspace\R2020b\case_30');
addpath('F:\yy的学习资料\win64\matlab');
alltime = tic;
tic
%%
% read-in
casename = 'case_30';
k_safe = 0.95;                                                             %？
n_T = 24;%define the number of optimization periods
%perform read data processing
initialize;
PD = bus(:, BUS_PD)/baseMVA;
QD = bus(:, BUS_QD)/baseMVA;
% 24-hour load data
Q_factor = QD/sum(QD);
P_factor = PD/sum(PD);
P_sum = mpc.PD'/baseMVA;
QD = Q_factor*sum(QD)*P_sum/sum(PD);
PD = P_factor*P_sum;
spnningReserve = 1.02*P_sum;
    
%%
%admittance matrix calculation
[Ybus, Yf, Yt] = makeYbus(baseMVA, bus, branch);  
%%
%hydropower station related data input
%water inflow data of hydropower station
water_data = [0.680374781	0.679953352	0.679473583	0.678940557	0.678476958	0.678473858	0.678273191	0.678030654	0.677792964	0.677568915	0.676809327	0.676140807	0.675811592	0.675591596	0.67524445	0.675025408	0.674896236	0.67487849	0.674642292	0.674753522	0.675039368	0.67528455	0.67541038	0.675693828;
3.370570916	3.369689379	3.36882039	3.366646275	3.36542144	3.364124709	3.362873389	3.361486914	3.362188933	3.362938939	3.361849458	3.360044543	3.360155775	3.359311028	3.359430835	3.360726749	3.362290093	3.365130954	3.367744383	3.368201885	3.36750003	3.368031627	3.36722979	3.367900685;
2.316E-11	2.31419E-11	2.31127E-11	2.30806E-11	2.3061E-11	2.3037E-11	2.30277E-11	2.30017E-11	2.29876E-11	2.29746E-11	2.2977E-11	2.29646E-11	2.29666E-11	2.29572E-11	2.29594E-11	2.29516E-11	2.29391E-11	2.29319E-11	2.29335E-11	2.29345E-11	2.29354E-11	2.29398E-11	2.29922E-11	2.29937E-11;
0.036286896	0.036201592	0.036095759	0.035993319	0.035893766	0.035798592	0.035716912	0.035672441	0.035662858	0.035657703	0.035653551	0.035622099	0.035602688	0.035542967	0.035529626	0.035470774	0.035387758	0.035337958	0.035305336	0.035239312	0.035178913	0.035132879	0.035095674	0.035023446;
0.094470519	0.09435343	0.094108395	0.093930005	0.093779044	0.093643215	0.093539807	0.093346451	0.093213762	0.093147797	0.093048838	0.092863888	0.092758885	0.092549909	0.09274862	0.092761167	0.092520695	0.092406389	0.092382239	0.092292444	0.092214753	0.092171858	0.092379174	0.092226312;
];

I = 5;%Number of hydropower stations
T = size(water_data,2);%Define the number of scheduled time periods
dt = 60;%Define time interval
Z_up_max = [570;325;203;577;633];  % Upper water level
Z_up_min = [535;298;202.3;569.7;628];         % Lower water level
Z_up_begin = [570;325;203;577;633];       % Initial control water level
Z_up_end = [570;325;203;577;633];         % End control water level                                          
H_max = [114;95;22.7;65;44.5];   
H_min = [30;30;14;40;27];
V_max=[12539300;113600000;3643000;603500;692550];%Upper reservoir storage
Q_max = [10.6;75.694;64.66;5.2;7.6];   % Upper discharge
Q_min = [0;0;0;0;0];            % Lower discharge          
P_hydro_max = [8.6;49.8;11.4;1.26;2.5];      % Upper power output
P_hydro_min = [0;0;0;0;0];     % Lower power output
A_hydro=[8.3;8.7;8.7;8;8];    %Hydropower output coefficient
H_loss_a=[0;0;0;0;0];  %Coefficient of water head loss
H_loss_b=[1;1;1;0.5;1];        %Water head loss constant
num_ZV = size(Z_up_data, 2);
num_ZQ = size(Z_down_data, 2);
Price_gw_store=1.3;%Unit is yuan /kw.d
Sale_gw=[0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307 0.307];
Price_gw = [0.3502 	0.3502 	0.3502 	0.3502 	0.3502 	0.3502 	0.3502 	0.3502 	0.6589 	0.6589 	0.9602 	0.9602 	0.9602 	0.9602 	0.9602 	0.9602 	0.6589 	0.6589 	0.6589 	0.6589 	0.6589 	1.1491 	1.1491 	0.6589];
%Upstream water level curve
Z_up_data = [
    510	535	540	545	550	555	560	565	570;
    278	285	292	299	306	313	320	327	333;
    180.5 184	188	192	196	200	202	204	206;
    569.7 570 571 572 573 574 575 576 577;
    610 614 618 622 626 630 634 638 642
   ];
%Reservoir storage curve
V_hydro_data = [839400	3619000	4484200	5453400	6563400	7859100	9278800	10831500	12539300;
    19090000	27400000	37260000	49370000	63620000	79550000	98000000	121040000 	146970000 ;
    0	35500	129800	345000	810000	1965000	2991000	4425000	6407000;
    181800 192000 236000 280000 334500 389000 456500 524000 603500; 
    48200 92300 155200 247600 381700 546900 744100 985000 1292400;
   ];
%Tail water level curve
Z_down_data = [456	457	458	459	460	461	462	463	464;
    230	230.5	231	231.5	232	232.5	233	233.5	234;
    180.5	182	183.5	185	186.5	188	189.5	191	192.5;
    512	512.5	513	513.5	514	514.5	515	515.5	516;
    587.5 588.5	590.5	591	591.5	592	592.5	593	593.5
    ];
%Outflow curve
Q_out_data = [0	12	70.9	167	298	464	667	907	1189;
    0	22.2	55	182	390	745	1300	2070	3010;
    0.1	62.5	335	740	1200	1710	2280	3070	3990;
    0	9	37	51	182	425	743	1126	1568;
    0 3.8	17.15	30	77	154	280	437	629
    ];
%%
%Hydropower station related decision variables
P_hydro = sdpvar(I, T);        % Power Output of hydropower station
V = sdpvar(I, T);            % Reservoir storage of hydropower station
Q_in = sdpvar(I, T);           % Inflow of hydropower station
Q_out = sdpvar(I, T);          % Outflow of hydropower station
Q_prod = sdpvar(I, T);         % Discharge of hydropower station
Q_spill = sdpvar(I, T);        % Spillage of hydropower station
Z_up = sdpvar(I, T);         % Upstream water level of hydropower station
Z_down = sdpvar(I, T);       % Tail water level of hydropower station
H_prod=sdpvar(I, T);        % Water head of hydropower station
d_ZV = binvar(I,T, num_ZV - 1);  % Upstream water level - reservoir storage range indicator variable
w_ZV = sdpvar(I, T,num_ZV);  % Upstream water level  - reservoir storage interval weight variable
d_ZQ = binvar(I, T,num_ZQ - 1);  % Tail water level - outflow range indicator variable
w_ZQ = sdpvar(I, T,num_ZQ);  % Upstream water level  - reservoir storage interval weight variable
z = sdpvar(I, T); %The product of the water head of a hydropower station and the discharge
%Add power purchase and sale variables
P_grid_state_gw = binvar(1,T);
P_grid_state_sale = binvar(1,T);

%Define power purchased from the grid
P_grid_gw = sdpvar(1,T);%Purchase electricity from the grid
P_grid_gw_max = sdpvar(1,1);
P_grid_sale = sdpvar(1,T);%Sell electricity to the grid


%%
%Create decision variable
gen_P = sdpvar(n_bus, n_T); 
gen_Q = sdpvar(n_bus, n_T);
% Slack variable substitution
x_i = sdpvar(n_bus, n_T);                        
h_ij = sdpvar(n_branch, n_T);
Va = sdpvar(n_bus, n_T);      %Phase angle
% Each branch power flow
PF_D = sdpvar(n_branch, n_T);    
QF_D = sdpvar(n_branch, n_T);   
PF_R = sdpvar(n_branch, n_T);    
QF_R = sdpvar(n_branch, n_T);     
%Branch breaking variable (0-1)
l_ij = binvar(n_branch,n_T); 
%Define the auxiliary variables of the objective function
y_gw = binvar(1, 1); %Add power grid capacity price
C = [];     %Constraints
assign(x_i, 1);
%%
%Relevant hydraulic constraints of hydropower stations
%Add water balance constraints
for t = 1:T-1
    for i = 1:I
        C = [C, V(i, t+1) == V(i, t) + dt*60* (Q_in(i, t) - Q_out(i, t))/10000];
    end
end
% Add upstream and tail water level constraints
for t = 1:T
    for i = 1:I
        C = [C, Z_up(i, t) <= Z_up_max(i)];
        C = [C, Z_up(i, t) >= Z_up_min(i)];
        C = [C, Z_down(i, t) <= Z_up(i, t)];
    end
end
% Added output constraint
for t = 1:T
    for i = 1:I
        C = [C, P_hydro(i, t) >= P_hydro_min(i)];
        C = [C, P_hydro(i, t) <= P_hydro_max(i)];
        C = [C,P_hydro(i,t) == A_hydro(i)*z(i,t)/1000];
        C = [C,P_hydro(i,t)==A_hydro(i)*H_prod(i,t)*Q_prod(i, t)/1000];  
    end

end

%Add power climbing constraints
for t = 1:T-1
    for i = 1:I
        C = [C, (P_hydro(i, t+1)-P_hydro(i, t))<=P_hydro_max(i)*0.2];
        C = [C, (P_hydro(i, t)-P_hydro(i, t+1))<=P_hydro_max(i)*0.2];
    end
end

% Add outflow constraints   P1,P2,P3 are one basin; P3,P4 are one basin
for t = 1:T
    for i = 1:I
        C = [C, Q_out(i, t) == Q_prod(i, t) + Q_spill(i, t)];
        C = [C, Q_out(i, t) >= Q_min(i)];
        C = [C, Q_out(i, t) <= Q_max(i)];
        C = [C, Q_spill(i, t) >= 0];
     
        if((i>=2&&i<=3)||i==5)
            C = [C, Q_in(i, t) == Q_out(i-1, t) + water_data(i, t)];
        else
            C = [C, Q_in(i, t) == water_data(i, t)];
        end
    end

end

% Upstream water level - reservoir storage constraints
for t=1:T
    for i = 1:I
        C = [C, sum(d_ZV(i,t, :)) == 1, sum(w_ZV(i,t,:)) == 1, w_ZV(i, t, :) >= 0];
        for j = 1:num_ZV
            if j == 1
                C = [C, w_ZV(i,t,  j) <= d_ZV(i,t,  j)];
            elseif j == num_ZV
                C = [C, w_ZV(i,t,  j) <= d_ZV(i,t,  j-1)];
            else
                C = [C, w_ZV(i, t, j) <= d_ZV(i,t,  j-1) + d_ZV(i, t, j)];
            end
        end
        C = [C, V(i,t) == sum(w_ZV(i,t,  :) .* V_hydro_data(i, :))];
        C = [C, Z_up(i,t)==sum(w_ZV(i,t,  :) .* Z_up_data(i, :))];
          
    end
end

% Tail water level - outflow constraints
for t=1:T
    for i = 1:I
        a=Z_down_data(i,2)-Z_down_data(i,1);
        b=Q_out_data(i,2)-Q_out_data(i,1);
        C = [C,(Q_out(i,t)-Q_out_data(i,1))*a==b*(Z_down(i,t)-Z_down_data(i,1))];    
    end
end
% % Add start and end upstream water level constraints
for i = 1:I
    C = [C, Z_up(i, 1) == Z_up_begin(i)];
    C = [C, Z_up(i, T) == Z_up_end(i)];
end
%Add water head constraints
for t = 1:T
    for i = 1:I
        if(t==1)
            C = [C, H_prod(i,t)==Z_up(i,t)-Z_down(i,t)-H_loss_b(i)];
        else
            C = [C, H_prod(i,t)==(Z_up(i,t)+Z_up(i,t-1))/2-Z_down(i,t)-H_loss_b(i)];
        end
    end
end

%%
%Add power balance constraints

for t =1:n_T
    C = [C,(P_hydro(1, t)+P_hydro(2, t)+P_hydro(3, t)+P_hydro(4, t)+P_hydro(5, t)+(gen_P(4,t)+gen_P(6,t)+...
        gen_P(7,t)+ gen_P(12,t))*baseMVA+P_grid_gw(t)) == (sum(PD(:,t))*baseMVA + sum(h_ij(:,t).*mpc.branch(:,BR_R))*baseMVA)];
end

% Linked node load to the unit
for t = 1:n_T
    C = [C,P_grid_gw(t) == gen_P(1,t)*baseMVA];
end

%%
%Generator nodes are linked to hydropower stations
for t = 1: n_T
     C = [C,gen_P(10,t) == P_hydro(1,t)/baseMVA];
     C = [C,gen_P(2,t) == P_hydro(2,t)/baseMVA];
     C = [C,gen_P(5,t) == P_hydro(3,t)/baseMVA];
     C = [C,gen_P(8,t) == P_hydro(4,t)/baseMVA];
     C = [C,gen_P(9,t) == P_hydro(5,t)/baseMVA];
end
%%
New_Br_temp = 1: n_bus;
New_Br_temp(gen(:, GEN_BUS)) = [];%Get the non-generator node
C = [C,
    gen_P(New_Br_temp, :) == 0,
    gen_Q(New_Br_temp, :) == 0,
    ];  %The active and reactive power of a non-generator node is 0

%%
%The relationship between voltage amplitude and power flow
for i = 1: n_branch
    for t = 1:n_T
        f_bus = branch_f_bus(i);            % Start bus of branch i  
        t_bus = branch_t_bus(i);            % End bus of branch i
        C = [C,
            (x_i(t_bus,t) - x_i(f_bus,t) + 2*(mpc.branch(i,3)*PF_D(i,t)+mpc.branch(i,4)*QF_D(i,t))-(mpc.branch(i,3)^2+mpc.branch(i,4)^2)*h_ij(i,t)) >= 0 ,...
            (x_i(t_bus,t) - x_i(f_bus,t) + 2*(mpc.branch(i,3)*PF_D(i,t)+mpc.branch(i,4)*QF_D(i,t))-(mpc.branch(i,3)^2+mpc.branch(i,4)^2)*h_ij(i,t)) <= 0
        ];
%Quadratic constraint
        C = [C,
            h_ij(i,t).*x_i(f_bus,t) - (PF_D(i,t).*PF_D(i,t)+QF_D(i,t).*QF_D(i,t)) >=0
            ];
%Relationship between phase Angle of node and branch power flow
        C = [C,
            (Va(f_bus,t)-Va(t_bus,t)) - (mpc.branch(i,BR_X)*PF_D(i,t)-mpc.branch(i,BR_R)*QF_D(i,t))  >= 0,...
            (Va(f_bus,t)-Va(t_bus,t)) - (mpc.branch(i,BR_X)*PF_D(i,t)-mpc.branch(i,BR_R)*QF_D(i,t))  <= 0
            ];
    end
end
%%
%Node power balance constraints
for i = 1: n_bus
    for t = 1: n_T
        upstream = find((branch(:,T_BUS)==i)~=0);
        downstream = find((branch(:,F_BUS)==i)~=0);
         if isempty(upstream)
            C = [C,(gen_P(i,t)-PD(i,t) - sum(PF_D(downstream,t)) + x_i(i,t)*bus(i, BUS_GS)/baseMVA) ==0];
            C = [C,(gen_Q(i,t) -QD(i,t) - sum(QF_D(downstream,t)) + x_i(i,t)*bus(i, BUS_BS)/baseMVA) == 0 ];  %Node power balance equation
         elseif isempty(downstream)
            C = [C,(gen_P(i,t)+sum(PF_D(upstream,t)-mpc.branch(upstream,BR_R).*h_ij(upstream,t)) -...
            PD(i,t) + x_i(i,t)*bus(i, BUS_GS)/baseMVA) ==0];
            C = [C,(gen_Q(i,t)+ sum(QF_D(upstream,t)-mpc.branch(upstream,BR_X).*h_ij(upstream,t)) -...
            QD(i,t) + x_i(i,t)*bus(i, BUS_BS)/baseMVA) == 0 ];  %Node power balance equation
        else
            C = [C,(gen_P(i,t)+sum(PF_D(upstream,t)-mpc.branch(upstream,BR_R).*h_ij(upstream,t)) -...
            PD(i,t) - sum(PF_D(downstream,t)) + x_i(i,t)*bus(i, BUS_GS)/baseMVA) ==0];
            C = [C,(gen_Q(i,t)+ sum(QF_D(upstream,t)-mpc.branch(upstream,BR_X).*h_ij(upstream,t)) -...
            QD(i,t) - sum(QF_D(downstream,t)) + x_i(i,t)*bus(i, BUS_BS)/baseMVA) == 0 ];  %Node power balance equation
         end
    end
end
%%
%Active power output constraint
for t =1:n_T
    for i = 1:n_gen
        C = [C,gen_P(gen(i,GEN_BUS),t) >= gen(i,GEN_PMIN)/baseMVA,...
            gen_P(gen(i,GEN_BUS),t) <=gen(i,GEN_PMAX)/baseMVA
            ];
    end
end
%%
% Reactive power output constraint 
for t = 1: n_T
    for i = 1: n_gen
        C = [C,
            gen(i, GEN_QMAX)/baseMVA >= gen_Q(gen(i, GEN_BUS),t),...
            gen_Q(gen(i, GEN_BUS),t) >= gen(i, GEN_QMIN)/baseMVA
            ];
    end
end
%%
%Constraints on the voltage amplitude of each node
for t = 1: n_T
    C = [C,
        bus(:, BUS_Vmax).^2 >= x_i(bus(:, BUS_I), t),x_i(bus(:, BUS_I), t) >= bus(:, BUS_Vmin).^2
        ];        
end
%%
%Constraints on the voltage phase Angle of each node
for t = 1:n_T
    C = [C,Va(:,t) <= 2*pi,Va(:,t) >=-2*pi,...
        Va(1,t) == 0,...
        ];  
end

%%
for t = 1:T
    C = [C,y_gw >= P_grid_state_gw(t)];
    C = [C,P_grid_gw_max >= P_grid_gw(1,t)];
    C = [C,P_grid_gw_max >= 0];
end
%Non-simultaneous constraints on power purchase and sale in the grid
for t = 1:n_T
    C = [C,...
        P_grid_gw(:,t) <= 500*P_grid_state_gw(:,t),...
        P_grid_gw(:,t)>= -500*(1-P_grid_state_gw(:,t))
        ];
end
%%
%Objective function
obj_value = (sum(1000*Price_gw.*P_grid_state_gw.*P_grid_gw) + sum(1000*(1-P_grid_state_gw).*Sale_gw.*P_grid_gw)+ 1000*Price_gw_store*y_gw.*P_grid_gw_max) ;

%% , 'gurobi.MIPGap', 0.01 
ops = sdpsettings('solver','gurobi','verbose',2,'usex0',1); 
ops.gurobi.FeasibilityTol = 1e-5;
ops.gurobi.OptimalityTol = 1e-5; 
params.IntFeasTol=1e-5;
%%
toc
%Solve
result = optimize(C, obj_value, ops);
toc(alltime)
if result.problem ==1
    disp('Infeasible');
    disp('Infeasible information:');
    disp(result.info); %Display solution information
elseif result.problem ==2
    disp('Unbounded');
elseif result.problem ~= 0
    disp(['Other questions:', result.info]);
else
    disp('Model solving success');
    optimal_P_hydro = value(P_hydro);
    optional_gen_P = value(gen_P)*baseMVA;
    optional_gen_Q = value(gen_Q)*baseMVA;
    optimal_V = value(V);
    optimal_Q_in = value(Q_in);
    optimal_Q_out = value(Q_out);
    optimal_Q_prod = value(Q_prod);
    optimal_Q_spill = value(Q_spill);
    optimal_Z_up = value(Z_up);
    optimal_Z_down = value(Z_down);
    optimal_obj_value = value(obj_value);

end
Va = value(Va);
x_i = value(x_i);
PF_D = value(PF_D);
QF_D = value(QF_D);
PF_R = value(PF_R);
QF_R = value(QF_R);
h_ij = value(h_ij);
P_sum = sum(optimal_P_hydro);