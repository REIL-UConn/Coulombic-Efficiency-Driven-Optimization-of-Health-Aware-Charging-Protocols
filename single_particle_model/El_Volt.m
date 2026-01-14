function [Ve1] = EI_Volt(ce1,ce2,tp1,sigel1,I,Ln,Ls,Lp,l1)
%Euler Implicit electrolyte diffusion

F=96500;
kf=0.01;
dx=l(2)-l(1);
loc_n=find(l==round(Ln*1E5)/1E5);
loc_p=find(l==round((Ln+Ls)*1E5)/1E5);

for x=1:numel(l)
    if x<=loc_n-1
        Bd1(x)=((-I/Ln)*x*dx);%Change I
        Iel1(x)=integral(Bd1(x)*x*dx);
    elseif x>=loc_n+1 && x<=loc_p
        Bd1(x)=I;
       
    elseif x>=loc_p+1
        Bd1(x)=((I/Lp)*(l1-(x*dx)));%Change I 
    end
end


Ve(n-1)=(Ln+2*Ls+Lp)*abs(Iel)/(2*sigel1)+kf*2*8.314*298/F*(1-tp1)*(log(ce1)-log(ce2));





outputArg1 = inputArg1;
outputArg2 = inputArg2;
end