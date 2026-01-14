function [Iplt,eta,Ise] = volt_cccv_chg(I,kr,cmax,c,U,Cecs,k_sei,k_pl,R_film,icra,T)
b=5.8035e+03/(T);
A=6.1032e+06*kr*sqrt((cmax-c).*c); 
i0=96500*k_sei*Cecs; 
ipl=(6.1032e+06)*k_pl; 
iter=0;
V1=U;
dV1=1E-14;

while abs(dV1)>=1E-15 || iter<100
   
   iter=iter+1;
   Iint=A*sinh(b*(V1-U-(I)*R_film));
   Ise=-i0*exp(-b*(V1-0.1-(I)*R_film));
  
   if V1<0 
    Iplt=min(0,ipl*sinh(b*(V1-(I)*R_film)));
   else
    Iplt=0;
   end
    
   f=I-(Iint+Ise+Iplt+icra) ;
   dIint=A*b*cosh(b*(V1-U-(I)*R_film));
  dIse=(i0*b*exp(-b*(V1-0.1-(I)*R_film)));
  dicra=0;

   if V1<0 && Iplt<0 
       dIplt=(ipl*b*cosh(b*(V1-(I)*R_film)));
   else
    dIplt=0;
   end
    df=-(dIint+dIse+dIplt+dicra);
    dV1=-f/df;
    V1=V1+dV1;
     
end
  eta=real(V1);
  
end


