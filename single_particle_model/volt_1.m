function [eta] = volt_1(I,kr,cmax,c,U,T,R_film)
%   Voltage Calculation during discharge
b= (5.8035e+03/T);
A=(6.1032e+06)*kr.*sqrt((cmax-c).*c);
iter=0;
V=U;
dV=1E-14;
while abs(real(dV))>=1E-15 || iter<100
    iter=iter+1;
    Iint=A*sinh(b*(V-U-I*R_film));
    f=I-Iint;
    dIint=A*b*cosh(b*(V-U-I*R_film));
    df=-dIint;
    dV=-f/df;
      V=V+dV;
 end
eta=(V);
end

