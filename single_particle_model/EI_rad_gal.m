function [uout,I] = EI_rad_gal(u,dr,D,J)
% for i=2:numel(r)-1
%     b(i)=1+2*dt/dr^2*D;
%     a(i)=D*(-dt/dr^2-dt/(r(i)*dr));
%     c(i-1)=D*(-dt/dr^2+dt/(r(i)*dr));
% end
% 
% a(1)=1;
% c(end+1)=0;
% b(1)=-1;
% b(end+1)=1;
% M=(diag(b,0)+diag(c,-1)+diag(a,1));
uinp=u';
uinp(1)=0;
uout(:)=mtimes(J,uinp);
I=-96500*D*(u(end)-u(end-1))/dr;
end

