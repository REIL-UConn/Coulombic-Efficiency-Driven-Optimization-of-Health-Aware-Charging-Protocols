function [i_cr] = film(nr,r,vm,c,R,dr,c_ini,del_s,d_film,E_sei,del_p,E_pl,v_sei,v_pl,sigUTS_sei,gamma_n,W_cr_m)
%Film growth during charging
E_film=1/(((del_s/d_film)/E_sei)+((del_p/d_film)/E_pl));
v_film=1/(((del_s/d_film)/v_sei)+((del_p/d_film)/v_pl));

film_d=(R:dr:R+d_film);

for i=1:(nr+1)

    if i==1
C2(i)=(1/R^3)*dr*(c(i)*((r(i))^2)-c_ini(i)*((r(i))^2));
    else
C2(i)=C2(i-1)+(1/R^3)*dr*(c(i)*((r(i))^2)-c_ini(i)*((r(i))^2));
    end   
end

sr_film=zeros(numel(film_d),1);

for j=1:numel(film_d)
sr_film(j)=(vm/3)*(E_film/(1-v_film))*C2(nr+1-(j-1)); 

end

G_film=1.9757*d_film/(E_film);
sr_film_cri=(1-(sqrt(del_p/d_film)-(del_p/d_film))*(1-(E_sei/E_film))*sigUTS_sei);
G_film_cri=5.3344e-07*sr_film_cri/(E_film);

if G_film>G_film_cri

    length_cr=(G_film*4*pi*R^2)/(2*gamma_n*d_film);
else
    length_cr=0;
end

A_cr=length_cr*W_cr_m;

i_cr=-A_cr*d_film*8.0110e+07/(R^2);

end