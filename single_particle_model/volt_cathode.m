function [eta] = volt_cathode(I,kr,cmax,c,U,T,R_cei)
%   Cathode Voltage Calculation

b=(5.8035e+03)/(T);

A=(6.1032e+06)*kr.*sqrt((cmax-c).*c); 

iter=0;
V1=U;
dV1=1E-14;

while abs(dV1)>=1E-15 || iter<100
     iter=iter+1;
    Iint=A*sinh(b*(V1-U-I*R_cei));
    f=I-Iint;
    dIint=A*b*cosh(b*(V1-U-I*R_cei));
    df=-(dIint);
    dV1=-f/df;
    dV1=real(dV1);
    V1=V1+dV1;
end
eta=real(V1);

end

