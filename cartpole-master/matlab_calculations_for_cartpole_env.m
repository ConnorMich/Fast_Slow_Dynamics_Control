syms s F Dth th
M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;
q = (M+m)*(I+m*l^2)-(m*l)^2;
k=I*(M+m)+M*m*l^2;
TL2_2=0;
TL2_3=m*l/M;
TL4_2=0;
TL4_3=-(M+m)*g/(m*l);
T2_2=0;
T4_2=0;
T2_3=(g*m^2*l^2)/k;
T4_3=-g*(M+m)/k;
UL2=1/M;
UL4=-1/(M*l);
U2=(I+m*l^2)/k;
U4=-m*l/k;
B=[0;U2;0;U4];
A=[0 1 0 0;
   0 TL2_2 TL2_3 0;
   0 0 0 1;
   0 TL4_2 TL4_3 0];
C=[1 0 0 0;0 0 1 0];
TF=C*inv(s*eye(4)-A)*B
xdd=((I+m*l^2)*(F+m*l*Dth^2 *sin(th))-g*m^2*l^2*sin(th)*cos(th))/(I*(M+m)+m*l^2*(M+m*(sin(th)^2)))
thdd=-(m*l*(F*cos(th)+m*l*Dth^2*sin(th)*cos(th)-(M+m)*g*sin(th)))/(I*(M+m)+m*l^2*(M+m*(sin(th)^2)))

