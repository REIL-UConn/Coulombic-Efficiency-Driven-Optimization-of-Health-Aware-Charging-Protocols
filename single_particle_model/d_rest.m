function [del_s,del_p,R_film,d_film,cec,mpl_strip,dpl_rev] = d_rest(is,dt,istr,K_sei,K_pl,cec0,D_ec,Rn,del_si,dpl_revi,dpl_irrevi)
%Film growth during rest


d1=(-is)*dt*9.9335e-10;
del_s=del_si+dt*(-is*9.9335e-10);
del_p_cal=dpl_revi+dt*(-istr*1.3468e-10);
if del_p_cal<=0
  dpl_rev=0;
else
  dpl_rev=dpl_revi+dt*(-istr*1.3468e-10);
end
del_p=dpl_irrevi+dpl_rev;
d2=(-istr*dt*1.3468e-10);
d_film=(del_s)+del_p;
K_film=K_sei+abs(del_p/d_film)/((1/(K_sei-K_pl))+((abs((del_s)/d_film))/(3*K_sei)));
R_film=d_film/K_film;
cec=cec0+(is*(d_film)/(96500*D_ec));
if cec<0
cec=0;
end
if del_p_cal<=0
 mpl_strip=0;
else
mpl_strip=dpl_rev*Rn^2*6.7104e+03;
end
end

