function [Ise,eta,Istrp] = volt_rest(I,kr,cmax,c,U,Cecs,k_sei,k_pl,R_fim,mpl,mass_pl1,T)
%Voltage Calculation during Rest
b=( 5.8035e+03)/(T);
A=(6.1032e+06)*kr.*sqrt((cmax-c).*c); 
i0=96500*k_sei*Cecs; 
ist=6.1032e+06*k_pl; 
iter=0;
V1=U;
dV1=1E-14;
while abs(dV1)>=1E-15 || iter<100
   iter=iter+1;
   Iint=A*sinh(b*(V1-U-(I)*R_fim));
   Ise=-i0*exp(-b*(V1-0.1-(I)*R_fim)); 
if mass_pl1<=0
    Istrp=0; 
else
   if V1>0 && (abs(mpl)/mass_pl1)>=0 && (abs(mpl)/mass_pl1)<=1 
    Istrp=max(0,ist*sinh(b*(V1-(I)*R_fim))*abs(mpl)/mass_pl1);
   else
    Istrp=0;
   end
end  
   f=I-(Iint+Ise+Istrp);
 dIint=A*b*cosh(b*(V1-U-(I)*R_fim));
  dIse=(i0*b*exp(-b*(V1-0.1-(I)*R_fim)));
 if mass_pl1<=0
    dIstrp=0;
 else
    if V1>0 && (mpl)/mass_pl1>=0 && (abs(mpl)/mass_pl1)<=1 
    dIstrp=(ist*b*cosh(b*(V1-(I)*R_fim))*abs(mpl)/mass_pl1);
    else
    dIstrp=0;
     end
 end
    df=-(dIint+dIse+dIstrp);
    dV1=-f/df;
    V1=V1+dV1;
end
   eta=real(V1);
end


