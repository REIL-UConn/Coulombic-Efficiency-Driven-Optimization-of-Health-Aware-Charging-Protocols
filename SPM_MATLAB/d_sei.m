function [del_s,R_film,d_film,cec,del_pirev,del_prev] = d_sei(is,dt,ipl,K_sei,K_pl,cec0,zita,del_previ,del_si,del_pirevi)

del_s=del_si+dt*(-is* 9.9335e-10); 
d2=(-ipl*dt*1.3468e-10); 
del_pirev=del_pirevi+zita*d2; 
del_prev=del_previ+(1-zita)*d2;
del_p=del_pirev+del_prev;

 d_film=(del_s)+del_p;
 K_film=K_sei+abs(del_p/d_film)/((1/(K_sei-K_pl))+((abs((del_s)/d_film))/(3*K_sei)));
 R_film=d_film/K_film;

cec=cec0+(is*(d_film)/(9.6500e-12));

if cec<0 && cec>cec0
cec=0;
end


end

