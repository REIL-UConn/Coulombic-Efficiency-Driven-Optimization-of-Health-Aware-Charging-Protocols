function [cout] = EI_el(ce,l,dt,De,I,en,es,ep,Ln,Ls,l1)
%Euler Implicit electrolyte diffusion

F=96500;
dx=l(2)-l(1);
loc_n=find(l==round(Ln*1E5)/1E5);
loc_p=find(l==round((Ln+Ls)*1E5)/1E5);

for x=1:numel(l)
    if x<=loc_n-1
        Bd(x)=5.0932e+03*((-I)*(x-1)*dx)*dt;   
        Def(x)=De*en^1.5;
    elseif x>=loc_n+1 && x<=loc_p
        Bd(x)=0;
        Def(x)=De*es^1.5;
    elseif x>=loc_p+1
        Bd(x)=4.7316e+03*((+I)*(l1-(x-1)*dx))*dt;
        Def(x)=De*ep^1.5;
    end
end

for i=2:numel(l)-1
    
    b(i)=1+2*Def(i)*dt/dx^2;
    a(i)=-Def(i)*dt/dx^2;
    c(i-1)=-Def(i)*dt/dx^2;
    
    if i==loc_n+1
        b(i)=1;
        a(i)=-es/(es+en);
        c(i-1)=-en/(es+en);
    elseif i==loc_p
        b(i)=1;
        a(i)=-ep/(es+ep);
        c(i-1)=-es/(es+ep);
    end
end

b(1)=-1;
b(end+1)=1;
c(end+1)=-1;
a(1)=1;

M=(diag(b,0)+diag(c,-1)+diag(a,1));

uinp=transpose(ce+Bd);
    
for i=1:numel(l)
    if i==1 || i==loc_n+1 || i==loc_p || i==numel(l)
        uinp(i)=0;
    end
end

cout(:)=mtimes(inv(M),uinp);

end

