function Vn_p_1 = eSPM(C_rate,Current_SOH)
 
    %% Geomaterical Properties
    Ln=50e-6; %Anode length [m] 
    Ls=15e-6; %Separator length [m]
    Lp=45e-6; %Cathode length [m] 
    Rn=32e-6; 
    Rp=12e-6; 
    L=Ln+Ls+Lp; 
    width=2*11.5e-3;
    length=2.58*75.635e-3; 
    Area=width*length; 
    R=8.314; %Gas constant
    F=96500; 
    lamda_el=0.225;
    lamda_pl=0.0044; 
    csmaxn=31507; 
    csmaxp=49900; %Refence Lithium Concentration
    epsp=0.297; %Electrode porosity cathode
    eplp=1-epsp; %Electrolyte porosity cathode
    epsn=0.471; %Electrode porosity anode
    epln=1-epsn; %Electrolyte porosity anode
    epsep=0.55; %Electrolyte porosity separator
    cec0=4541;
    cl0=1000; 
    a=0.5; 
    iapp=7.8; 
    ICS=iapp*0.93*Area; 
    Resis=4.15; 
    cp0=0.98*csmaxp;   
    cn0=0.2*csmaxn;
    nr=40; 
    nl=round(2*(Ln+Ls+Lp)*1E5); 
    dnl=(Ln+Lp+Ls)/nl;
    Vp(1)=U_LCO(cp0/csmaxp);
    Vn(1)=Un(cn0/csmaxn);
    del1=0; 
    M_sei=0.162; %Molar Mass
    M_pl=0.00694;  %Molar Mass
    Ipl=0;
    Zita_c=0.65175*1.22899; %0.93   0.79
    zita=Zita_c*exp(Ipl*0.56681); 
    Z(1,1)=zita;
    Q2=0;
    Q3=0;
    QC=0;
    cec=cec0;
    v_sei=0.2; 
    v_pl=0.38; 
    delfilm_crti=270e-9; 
    gamma_n=13; 
    W_cr_m= 10e-9; 
    dt_cr=1; 
    delfilm_cr_t=1; 
    dpl=0;
    Isei=0;
    Icei_1(1)=0;
    D_ec=10e-17; 
    irr_Q=0;
    irr_Q1=0;
    Q4=0;
    v_LCO=0.3;
    du_OCV_T_pos=0.25e-3; 
    du_OCV_T_neg=0.14e-3; 
    del_c=0;
    fitv=0.03; 
    SEI_Parametrized=0.275;   %0.125
    
    cycle=1:10; %Number of cycles
    dpl_irrev=0;
    
    SOH=Current_SOH; %In put RPT SOH after 10 cyles
    Resist_SEI_fac=-54.30463+243.28757*exp(-(1-SOH)*100/(-24.90151));%((950.09936+(-948.35273)*exp(-((1-SOH)*100)/10.65538))); 
    SEI_Factor=1; 
    del_s=SEI_Factor*SOH_LCO_No_cyc(SOH*100);
    nomCap=SOH*(36/1000);
    del=del_s;
    
    %% Cycle iterations
    
    for k=1:numel(cycle)
            
    
            Cr_1st=C_rate(1); %1st Step C rate
            Cr_2nd=C_rate(2); %2nd Step C rate
            Cr_3rd=C_rate(3); %3rd Step C rate
            Cr_4th=C_rate(4);  %4th Step C rate
            
           
            Max_Voltage=4.9;  %  maximum C rate
     
    Q21=0;
    Q31=0;
    QC1=0;
    
    
          Cr_1=min(Cr_1st,Cr_2nd);
          Cr_2=min(Cr_3rd,Cr_4th);
          
         
    
          Cr=min( Cr_1,Cr_2);%C_rates{k, 1}; %Rate of Charging
       
    soc_1st=0.2; %1st Step SOC
    soc_2nd=0.4; %2nd Step SOC
    soc_3rd=0.6; %3rd Step SOC
    soc_4th=0.8; %4th Step SOC
    
    
    %Column A Anodic Potential in V
    %Column B State of Charge
    
        T=298; %Temperature
       
       Vm=8.9e-6; 
       
        ap=(3*epsp/Rp); 
        an=(3*epsn/(Rn)); 
        drn=Rn/nr;
        drp=Rp/nr;
        rneg=(0:drn:Rn)';
        rpos=(0:drp:Rp)';
        df=0.3/Cr;
        nt=360/Cr; %No of Time step 
        tmax=(3600*4)/Cr;   
        dt=4; %TimeSteps in Sec
    I=[ones(1,ceil(3600/(dt*Cr))) 0*ones(1,ceil(3600/(dt*Cr))) -1*ones(1,ceil(3600/(dt*df*Cr))) 0*ones(1,ceil(3600/(dt*Cr)))]; 
    if k==1
    cp=cp0*ones(numel(I),nr+1)*(SOH); %*(1-Damage);
    cn=cn0*ones(numel(I),nr+1); %*(2-SOH); %*(2-SOH);
    [timelen,radilen]=size(cp);
    Lt=0:dnl:(Ln+Lp+Ls);
    ce=cl0*ones(numel(I),nl+1);  %1 Mol Lithium
    Qs_cy=zeros(numel(cycle),1);
    Qs_rel=zeros(numel(cycle),1);
    discap=zeros(k,1);
    time_rest_d=zeros(ceil(timelen/5),1);
    Discharge=zeros(ceil(timelen/5),1);
    full_cell=zeros(timelen,k);
    Qs_irPl=zeros(k,1);
    qs_pl_1=zeros(k,1);
    Qs_pl=zeros(k,1);
    end
    
    if k>1
    D1=(((100-eff(k-1))/100));
    cp=cp1*ones(numel(I),nr+1)*D1;
    %cn=cn0*ones(numel(I),nr+1)*(2-SOH);
    %eff;
    end

    
    time=zeros(timelen,k);
    time_cccv=zeros(ceil(timelen/6),1);
    time_cc=zeros(ceil(timelen/4),1);
    time_pl=zeros(ceil(timelen/4.5),1);
    Vp=zeros(timelen,1);
    Ve=zeros(timelen,1);
    Vn=zeros(timelen,1);
    
    Vn_p_1=zeros(ceil(timelen/6),1);
    Vp_cc=zeros(ceil(timelen/6),1);
    Vn_cc=zeros(ceil(timelen/6),1);
    Ve_cc=zeros(ceil(timelen/6),1);
    Volt_cccv_cell=zeros(timelen,1);
    SOC=zeros(timelen,1);
    Volt_cell_output=zeros(timelen,1);
    Ipl_2=zeros(timelen,1);
    Volt_cell=zeros(timelen,k);
    Charge=zeros(ceil(timelen/4),1);
    
    Vn_sp_1=zeros(ceil(timelen/4),1);
     time_rest_c=zeros(ceil(timelen/4),1);
     Vp_r=zeros(ceil(timelen/4),1);
     Ve_r=zeros(ceil(timelen/4),1);
     Volt_rest_cell=zeros(ceil(timelen/4),1);
    time_dccv=zeros(ceil(timelen/4),1);
    time_dc=zeros(ceil(timelen/4),1);
    Vp_d=zeros(ceil(timelen/4),1);
    Vn_d=zeros(ceil(timelen/4),1);
    Ve_d=zeros(ceil(timelen/4),1);
    Volt_dis_cell=zeros(ceil(timelen/4),1);
    time_dcv=zeros(ceil(timelen/4),1);
    Ip_cv_d=zeros(ceil(timelen/4),1);
    In_cv_d=zeros(ceil(timelen/4),1);
    Vn_r=zeros(ceil(timelen/4),1);
    Ea_pos=50*10^3; %Activation energy of Cathode J/mol
    Ea_neg=36*10^3;  %Activation energy of Anode J/mol
    
    Ea_pos_kel=35*10^3; %Activation energy of Cathode J/mol
    Ea_neg_kel=45*10^3;  %Activation energy of Anode J/mol
    
    T_ref_neg=298; 
    T_ref_pos=298; 
    T_ref=298; 
    Arrh_p=exp((Ea_pos/R)*((1/T_ref_pos)-(1/T)));
    Arrh_n=exp((Ea_neg/R)*((1/T_ref_neg)-(1/T)));
    Arrh_p_kel=exp((Ea_pos_kel/R)*((1/T_ref_pos)-(1/T)));
    Arrh_n_kel=exp((Ea_neg_kel/R)*((1/T_ref_neg)-(1/T)));
    Dl0=10^-4*(1+(lamda_el*(T-T_ref))); %Diffusion coefficient electrolyte [m^2/s] 
    Dsn0=30e-14*Arrh_n; %Diffusion coefficient Anode [m^2/s] % Comsol
    Dsp0=2e-13*Arrh_p;  %Diffusion coefficient cathode [m^2/s] 
    sigsn0=100*Arrh_n; %Electronic conductivity anode [S/m] 
    sigsp0=0.34*Arrh_p; %Electronic conductivity cathode [S/m]
    sigel=1*(1+(lamda_el*(T-T_ref))); %Ionic conductivity electrolyte separator [S/m]
    kr=2.43e-11*Arrh_n_kel; %Butler volmer reaction rate of Anode  
    kp=1.02e-10*Arrh_p_kel;%Butler volmer reaction rate of Cathode 
    tp=0.35; 
    kf=((1-0.363)*(0.601-0.24*(0.001*cl0)^0.5))/((1-0.399)*(1-tp))+(0.982*(1-0.0052*(T-294))*(0.001*cl0)^1.5)-1;
    Dsp=Dsp0*epsp^0;
    Dsn=Dsn0*epsn^0;
    sigsp=sigsp0*epsp^0;
    sigsn=sigsn0*epsn^0;
    
    k_sei=SEI_Parametrized*1.35e-12*Arrh_n_kel; 
    k_pl=44.73e-7*Arrh_n_kel; 
    rho_pl=534*Arrh_n_kel; %[kg/m^3] density of plated lithium 
    rho_sei=1690*Arrh_n_kel; %[kg/m^3] density of SEI
    K_pl=1.1e7/(1+(lamda_pl)*(T-T_ref)); %Electrical Conductivity of pl
    E_sei=0.43e9*Arrh_n_kel; 
    E_pl=7.82e9*Arrh_n_kel; 
    sigUTS_sei=9e6*Arrh_n_kel; 
    sigUTS_pl=15e6*Arrh_n_kel; 
    E_Gra=1.28e9*Arrh_p; 
        if k==1
        K_sei=Resist_SEI_fac*6.15E-7*Arrh_n_kel;   %2.15
        K_film=K_sei;
        R_film=del/K_film;
        K_cei=5.25E-6*Arrh_p_kel; %Electrical Conductivity
        R_cei=del_c/K_cei;
        del_film=zeros(cycle(end),1);
        end
    k_cei=1e-12*Arrh_p; 
    rho_cei=1690*Arrh_p; 
    
    Ir_drop=Cr*(iapp*Area)*(0.15*Resis); 
    mass_pl=0;
        Is(1)=0;
        Iv=iapp*Cr;
        var=0;
        Itot=-Iv;
        Iel=((((abs(Itot))/2)*Ln)+((abs(Itot))*Ls)+(((abs(Itot))/2)*Lp))/(Ln+Ls+Lp);
        Iel2=((((abs(Itot)*df)/2)*Ln)+((abs(Itot)*df)*Ls)+(((abs(Itot)*df)/2)*Lp))/(Ln+Ls+Lp);
        E1=0;
        E2=0;   
        E3=0;
        E4=0;
        E5=0;
        E6=0;
        E1_1= 0;
        E1_2= 0;
        E1_3= 0;
        E1_4= 0;
        E1_5= 0;
        E1_6= 0;
        E1_7= 0;
        E1_8= 0;
        E1_9= 0;
        E1_10= 0;
        s=1;
        s1=1;
        Lt=fix(Lt.*1E6)./1E6;
        n=1;
        q=1;
        y=1;
        p1=1;
        z=1;
        p=1;
        q1=1;
        m=1;
        m1=1;
        m2=1;
        dpl_rev=0; 
        time_cccv(1,1)=0;
        time_cc(1)=0;
        Ic_cv(1,1)=0;
        time_pl(1)=0;
        time_rest_d(1)=0;
        time_dcv(1)=0;
        time_dccv(1)=0;
        time_dc(1)=0;
        Ic_cv(1,1)=0;
        time_rest_c(1)=0;
        time_rest_d(1)=0;
        icra(1,1)=0; 
        del_film(k,1)=del; 
        soc_p=0;
        clear SOC
        clear Vn_p_1
        clear Ipl_2
    
        %0: Charge CC (Constant Current)
        %1: Charge CV (Constant Voltage)
        %2: Rest (Zero Current)
        %3: Discharge (Constant Current)
        %4: Discharge (Constant Voltage
        %5: Rest (Zero Current)
        
            if k==1
                for i=2:numel(rpos)-1
                b_2(i)=1+2*dt*Dsp/drp^2;
                a_2(i)=Dsp*(-dt/drp^2-dt/(rpos(i)*drp));
                c_2(i-1)=Dsp*(-dt/drp^2+dt/(rpos(i)*drp));
                 end
              a_2(1)=1;
              c_2(end+1)=1;
              b_2(1)=-1;
              b_2(end+1)=-1;
    
              M_2=(diag(b_2,0)+diag(c_2,-1)+diag(a_2,1));
              J_LCO=inv(M_2);
    
            end
    
            if k==1
                for i=2:numel(rneg)-1
                b_1(i)=1+2*dt*Dsn/drn^2;
                a_1(i)=Dsn*(-dt/drn^2-dt/(rneg(i)*drn));
                c_1(i-1)=Dsn*(-dt/drn^2+dt/(rneg(i)*drn));
                 end
              a_1(1)=1;
              c_1(end+1)=1;
              b_1(1)=-1;
              b_1(end+1)=-1;
    
              M_1=(diag(b_1,0)+diag(c_1,-1)+diag(a_1,1));
              J_C=inv(M_1);
    
            end
    
            if k==1
                for i=2:numel(rpos)-1
                b_3(i)=1+2*dt*Dsp/drp^2;
                a_3(i)=Dsp*(-dt/drp^2-dt/(rpos(i)*drp));
                c_3(i-1)=Dsp*(-dt/drp^2+dt/(rpos(i)*drp));
                 end
              a_3(1)=1;
              c_3(end+1)=0;
              b_3(1)=-1;
              b_3(end+1)=1;
    
              M_3=(diag(b_3,0)+diag(c_3,-1)+diag(a_3,1));
              J_LCO_CV=inv(M_3);
    
            end
    
            if k==1
                for i=2:numel(rneg)-1
                b_4(i)=1+2*dt*Dsn/drn^2;
                a_4(i)=Dsn*(-dt/drn^2-dt/(rneg(i)*drn));
                c_4(i-1)=Dsn*(-dt/drn^2+dt/(rneg(i)*drn));
                 end
              a_4(1)=1;
              c_4(end+1)=0;
              b_4(1)=-1;
              b_4(end+1)=1;
    
              M_4=(diag(b_4,0)+diag(c_4,-1)+diag(a_4,1));
              J_C_CV=inv(M_4);
    
            end
    
        while var<10
            
           n=n+1;
          
           Ir_drop=Cr*(iapp*Area)*(0.15*Resis); 
         Iel=((((abs(Itot))/2)*Ln)+((abs(Itot))*Ls)+(((abs(Itot))/2)*Lp))/(Ln+Ls+Lp);
            Iel2=((((abs(Itot)*df)/2)*Ln)+((abs(Itot)*df)*Ls)+(((abs(Itot)*df)/2)*Lp))/(Ln+Ls+Lp);
           Ip=-Itot/(ap*Lp); 
           In=Itot/(an*Ln); 
           
          time(n-1,k)=dt*(n-1);
      
       switch var
           
        case 0
        
         if n==2
         Cr=Cr_1st;
         Ir_drop=Cr*(iapp*Area)*(0.15*Resis); 
         Iv=iapp*Cr;
         Itot=-Iv;
         Iel=((((abs(Itot))/2)*Ln)+((abs(Itot))*Ls)+(((abs(Itot))/2)*Lp))/(Ln+Ls+Lp);
         Iel2=((((abs(Itot)*df)/2)*Ln)+((abs(Itot)*df)*Ls)+(((abs(Itot)*df)/2)*Lp))/(Ln+Ls+Lp);
         Ip=-Itot/(ap*Lp); 
         In=Itot/(an*Ln); 
         end
          z=z+1;   
            time_cccv(z)=time_cccv(z-1)+dt;
            E1=E1+1;
            E1_1= E1_1+1;
            s=s+1;
            time_cc(z)=time_cc(z-1)+dt;
            cp(n,:)=EI_rad(cp(n-1,:),drp,(Cr*1.2096e+08)/(Arrh_p),J_LCO);
            [Vp_cal]=volt_cathode((Ip),kp,csmaxp,cp(n,end),U_LCO(cp(n,end)/csmaxp)+du_OCV_T_pos*(T-298.15),T,R_cei);
            Vp(n-1)=real(Vp_cal);
            ce(n,:)=EI_el(ce(n-1,:),Lt,dt,Dl0,Itot,epln,epsep,eplp,Ln,Ls,L);
            Ve(n-1,1)=(6.2500e-05*(Iel)/(1+(lamda_el*(T-T_ref))))+(0.0184*(T/298)*(log(ce(n,end))-log(ce(n,1))));
            cn(n,:)=EI_rad(cn(n-1,:),drn,(-Cr*9.9154e+07)/(Arrh_n),J_C);
             if k>1
                 [icra]=film(nr,rneg,Vm,cn(n,:),Rn,drn,cn_level(1,:),del_s,del,E_sei,dpl,E_pl,v_sei,v_pl,sigUTS_sei,gamma_n,W_cr_m);
                 icra_n(s-1,k)=icra;
             end
            [Ipl,Vn_p,Isei]=volt_cccv_chg(In,kr,csmaxn,cn(n,end),Un(cn(n,end)/csmaxn)+du_OCV_T_neg*(T-298.15),cec,k_sei,k_pl,R_film,icra,T);
            Ipl_1(s,k)=real(Ipl);
            Ipl_2(s-1)=real(Ipl);
            Isei_1(s,1)=real(Isei);
            Vn_p_1(s-1,1)=real(Vn_p);
            Vn(n-1,1)=real(Vn_p);
              if Ipl_1(s,1) ~= 0
                 q1=q1+1;
                 time_pl(q1)=time_pl(q1-1)+dt;         
              end         
           zita=Zita_c*exp(Ipl*0.56681);  
           Q2=Q2+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q3=Q3+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC=QC+((abs(In))*an*Ln*dt)*Area;


           Q21=Q21+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q31=Q31+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC1=QC1+((abs(In))*an*Ln*dt)*Area;
           
            [del_s,R_film,del,cec,dpl_irrev,dpl_rev]=d_sei(Isei,dt,Ipl,K_sei,K_pl,cec0,zita,dpl_rev,del_s,dpl_irrev); 
            Ic_cv(s,1)=Ic_cv(s-1,1)+((abs(Itot)*Area)*dt);

            Vp_cc(n-1,1)=real(Vp(n-1));
            Vn_cc(n-1,1)=real(Vn(n-1));
            Ve_cc(n-1,1)=real(Ve(n-1));

            Volt_cccv_cell(n-1)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            Volt_cell(n-1,k)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            Volt_cell_output(s-1)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            soc_p=soc_p+(abs(Ip)*ap*Lp*dt*Area);
            SOC(s-1)=soc_p/(nomCap*3600);
            if (Volt_cccv_cell(n-1))>=Max_Voltage || SOC(s-1)>=soc_1st
            var=1;   
            Charge_1(1,1)=abs((Itot*Area)*time_cc(E1_1+1));
        Cr=Cr_2nd;
        Ir_drop=Cr*(iapp*Area)*(0.15*Resis); 
        Iv=iapp*Cr;
        Itot=-Iv;
        Iel=((((abs(Itot))/2)*Ln)+((abs(Itot))*Ls)+(((abs(Itot))/2)*Lp))/(Ln+Ls+Lp);
        Ip=-Itot/(ap*Lp); 
        In=Itot/(an*Ln); 
            end

         case 1
           
            z=z+1;   
            time_cccv(z)=time_cccv(z-1)+dt;
            E1=E1+1;
            E1_2= E1_2+1;
            s=s+1;
            time_cc(z)=time_cc(z-1)+dt;
            cp(n,:)=EI_rad(cp(n-1,:),drp,(Cr*1.2096e+08)/(Arrh_p),J_LCO);
            [Vp_cal]=volt_cathode((Ip),kp,csmaxp,cp(n,end),U_LCO(cp(n,end)/csmaxp)+du_OCV_T_pos*(T-298.15),T,R_cei);
            Vp(n-1)=real(Vp_cal);
            ce(n,:)=EI_el(ce(n-1,:),Lt,dt,Dl0,Itot,epln,epsep,eplp,Ln,Ls,L);
            Ve(n-1,1)=(6.2500e-05*(Iel)/(1+(lamda_el*(T-T_ref))))+(0.0184*(T/298)*(log(ce(n,end))-log(ce(n,1))));
            cn(n,:)=EI_rad(cn(n-1,:),drn,(-Cr*9.9154e+07)/(Arrh_n),J_C);
             if k>1
                 [icra]=film(nr,rneg,Vm,cn(n,:),Rn,drn,cn_level(1,:),del_s,del,E_sei,dpl,E_pl,v_sei,v_pl,sigUTS_sei,gamma_n,W_cr_m);
                 icra_n(s-1,k)=icra;
             end
            [Ipl,Vn_p,Isei]=volt_cccv_chg(In,kr,csmaxn,cn(n,end),Un(cn(n,end)/csmaxn)+du_OCV_T_neg*(T-298.15),cec,k_sei,k_pl,R_film,icra,T);
            Ipl_1(s,k)=real(Ipl);
            Ipl_2(s-1)=real(Ipl);
            Isei_1(s,1)=real(Isei);
            Vn_p_1(s-1,1)=real(Vn_p);
            Vn(n-1,1)=real(Vn_p);
              if Ipl_1(s,1) ~= 0
                 q1=q1+1;
                 time_pl(q1)=time_pl(q1-1)+dt;         
              end         
           zita=Zita_c*exp(Ipl*0.56681);  
           Q2=Q2+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q3=Q3+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC=QC+((abs(In))*an*Ln*dt)*Area;

           Q21=Q21+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q31=Q31+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC1=QC1+((abs(In))*an*Ln*dt)*Area;
           
            [del_s,R_film,del,cec,dpl_irrev,dpl_rev]=d_sei(Isei,dt,Ipl,K_sei,K_pl,cec0,zita,dpl_rev,del_s,dpl_irrev); 
            Ic_cv(s,1)=Ic_cv(s-1,1)+((abs(Itot)*Area)*dt);

            Vp_cc(n-1,1)=real(Vp(n-1));
            Vn_cc(n-1,1)=real(Vn(n-1));
            Ve_cc(n-1,1)=real(Ve(n-1));

            Volt_cccv_cell(n-1)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            Volt_cell(n-1,k)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            Volt_cell_output(s-1)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            soc_p=soc_p+(abs(Ip)*ap*Lp*dt*Area);
            SOC(s-1)=soc_p/(nomCap*3600);
            if (Volt_cccv_cell(n-1))>=Max_Voltage || SOC(s-1)>=soc_2nd
            var=2;   
            Charge_2(1,1)= Charge_1+abs((Itot*Area)*time_cc(E1_2+1));
            Cr=Cr_3rd;
             Ir_drop=Cr*(iapp*Area)*(0.15*Resis); 
         Iv=iapp*Cr;
         Itot=-Iv;
         Iel=((((abs(Itot))/2)*Ln)+((abs(Itot))*Ls)+(((abs(Itot))/2)*Lp))/(Ln+Ls+Lp);
         Ip=-Itot/(ap*Lp); 
         In=Itot/(an*Ln); 
            end
        
          case 2
           
            z=z+1;   
            time_cccv(z)=time_cccv(z-1)+dt;
            E1=E1+1;
            E1_3= E1_3+1;
            s=s+1;
            time_cc(z)=time_cc(z-1)+dt;
            cp(n,:)=EI_rad(cp(n-1,:),drp,(Cr*1.2096e+08)/(Arrh_p),J_LCO);
            [Vp_cal]=volt_cathode((Ip),kp,csmaxp,cp(n,end),U_LCO(cp(n,end)/csmaxp)+du_OCV_T_pos*(T-298.15),T,R_cei);
            Vp(n-1)=real(Vp_cal);
            ce(n,:)=EI_el(ce(n-1,:),Lt,dt,Dl0,Itot,epln,epsep,eplp,Ln,Ls,L);
            Ve(n-1,1)=(6.2500e-05*(Iel)/(1+(lamda_el*(T-T_ref))))+(0.0184*(T/298)*(log(ce(n,end))-log(ce(n,1))));
            cn(n,:)=EI_rad(cn(n-1,:),drn,(-Cr*9.9154e+07)/(Arrh_n),J_C);
             if k>1
                 [icra]=film(nr,rneg,Vm,cn(n,:),Rn,drn,cn_level(1,:),del_s,del,E_sei,dpl,E_pl,v_sei,v_pl,sigUTS_sei,gamma_n,W_cr_m);
                 icra_n(s-1,k)=icra;
             end
            [Ipl,Vn_p,Isei]=volt_cccv_chg(In,kr,csmaxn,cn(n,end),Un(cn(n,end)/csmaxn)+du_OCV_T_neg*(T-298.15),cec,k_sei,k_pl,R_film,icra,T);
            Ipl_1(s,k)=real(Ipl);
            Ipl_2(s-1)=real(Ipl);
            Isei_1(s,1)=real(Isei);
            Vn_p_1(s-1,1)=real(Vn_p);
            Vn(n-1,1)=real(Vn_p);
              if Ipl_1(s,1) ~= 0
                 q1=q1+1;
                 time_pl(q1)=time_pl(q1-1)+dt;         
              end         
           zita=Zita_c*exp(Ipl*0.56681);  
           Q2=Q2+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q3=Q3+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC=QC+((abs(In))*an*Ln*dt)*Area;


           Q21=Q21+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q31=Q31+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC1=QC1+((abs(In))*an*Ln*dt)*Area;
           
            [del_s,R_film,del,cec,dpl_irrev,dpl_rev]=d_sei(Isei,dt,Ipl,K_sei,K_pl,cec0,zita,dpl_rev,del_s,dpl_irrev); 
            Ic_cv(s,1)=Ic_cv(s-1,1)+((abs(Itot)*Area)*dt);

            Vp_cc(n-1,1)=real(Vp(n-1));
            Vn_cc(n-1,1)=real(Vn(n-1));
            Ve_cc(n-1,1)=real(Ve(n-1));

            Volt_cccv_cell(n-1)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            Volt_cell(n-1,k)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            Volt_cell_output(s-1)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            soc_p=soc_p+(abs(Ip)*ap*Lp*dt*Area);
            SOC(s-1)=soc_p/(nomCap*3600);
            if (Volt_cccv_cell(n-1))>=Max_Voltage || SOC(s-1)>=soc_3rd
            var=3;   
            Charge_3(1,1)= Charge_2+abs((Itot*Area)*time_cc(E1_3+1));
            Cr=Cr_4th;
            Ir_drop=Cr*(iapp*Area)*(0.15*Resis); 
            Iv=iapp*Cr;
            Itot=-Iv;
            Iel=((((abs(Itot))/2)*Ln)+((abs(Itot))*Ls)+(((abs(Itot))/2)*Lp))/(Ln+Ls+Lp);
            Ip=-Itot/(ap*Lp); 
            In=Itot/(an*Ln); 
            end
       
           case 3
           
            z=z+1;   
            time_cccv(z)=time_cccv(z-1)+dt;
            E1=E1+1;
            E1_4= E1_4+1;
            s=s+1;
            time_cc(z)=time_cc(z-1)+dt;
            cp(n,:)=EI_rad(cp(n-1,:),drp,(Cr*1.2096e+08)/(Arrh_p),J_LCO);
            [Vp_cal]=volt_cathode((Ip),kp,csmaxp,cp(n,end),U_LCO(cp(n,end)/csmaxp)+du_OCV_T_pos*(T-298.15),T,R_cei);
            Vp(n-1)=real(Vp_cal);
            ce(n,:)=EI_el(ce(n-1,:),Lt,dt,Dl0,Itot,epln,epsep,eplp,Ln,Ls,L);
            Ve(n-1,1)=(6.2500e-05*(Iel)/(1+(lamda_el*(T-T_ref))))+(0.0184*(T/298)*(log(ce(n,end))-log(ce(n,1))));
            cn(n,:)=EI_rad(cn(n-1,:),drn,(-Cr*9.9154e+07)/(Arrh_n),J_C);
             if k>1
                 [icra]=film(nr,rneg,Vm,cn(n,:),Rn,drn,cn_level(1,:),del_s,del,E_sei,dpl,E_pl,v_sei,v_pl,sigUTS_sei,gamma_n,W_cr_m);
                 icra_n(s-1,k)=icra;
             end
            [Ipl,Vn_p,Isei]=volt_cccv_chg(In,kr,csmaxn,cn(n,end),Un(cn(n,end)/csmaxn)+du_OCV_T_neg*(T-298.15),cec,k_sei,k_pl,R_film,icra,T);
            Ipl_1(s,k)=real(Ipl);
            Ipl_2(s-1)=real(Ipl);
            Isei_1(s,1)=real(Isei);
            Vn_p_1(s-1,1)=real(Vn_p);
            Vn(n-1,1)=real(Vn_p);
              if Ipl_1(s,1) ~= 0
                 q1=q1+1;
                 time_pl(q1)=time_pl(q1-1)+dt;         
              end         
           zita=Zita_c*exp(Ipl*0.56681);  
           Q2=Q2+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q3=Q3+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC=QC+((abs(In))*an*Ln*dt)*Area;


           Q21=Q21+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q31=Q31+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC1=QC1+((abs(In))*an*Ln*dt)*Area;
           
            [del_s,R_film,del,cec,dpl_irrev,dpl_rev]=d_sei(Isei,dt,Ipl,K_sei,K_pl,cec0,zita,dpl_rev,del_s,dpl_irrev); 
            Ic_cv(s,1)=Ic_cv(s-1,1)+((abs(Itot)*Area)*dt);

            Vp_cc(n-1,1)=real(Vp(n-1));
            Vn_cc(n-1,1)=real(Vn(n-1));
            Ve_cc(n-1,1)=real(Ve(n-1));

            Volt_cccv_cell(n-1)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            Volt_cell(n-1,k)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            Volt_cell_output(s-1)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            soc_p=soc_p+(abs(Ip)*ap*Lp*dt*Area);
            SOC(s-1)=soc_p/(nomCap*3600);
            if (Volt_cccv_cell(n-1))>=Max_Voltage || SOC(s-1)>=soc_4th
            var=4;   
            Charge_4(1,1)=Charge_3+abs((Itot*Area)*time_cc(E1_4+1));
            Ir_drop=Cr*(iapp*Area)*(0.15*Resis); 
            Iv=iapp*Cr;
            Itot=-Iv;
            Iel=((((abs(Itot))/2)*Ln)+((abs(Itot))*Ls)+(((abs(Itot))/2)*Lp))/(Ln+Ls+Lp);
            Ip=-Itot/(ap*Lp); 
            In=Itot/(an*Ln); 

            end

           case 4
           
            z=z+1;   
            time_cccv(z)=time_cccv(z-1)+dt;
            E1=E1+1;
            E1_10= E1_10+1;
            s=s+1;
            time_cc(z)=time_cc(z-1)+dt;
            cp(n,:)=EI_rad(cp(n-1,:),drp,(Cr*1.2096e+08)/(Arrh_p),J_LCO);
            [Vp_cal]=volt_cathode((Ip),kp,csmaxp,cp(n,end),U_LCO(cp(n,end)/csmaxp)+du_OCV_T_pos*(T-298.15),T,R_cei);
            Vp(n-1)=real(Vp_cal);
            ce(n,:)=EI_el(ce(n-1,:),Lt,dt,Dl0,Itot,epln,epsep,eplp,Ln,Ls,L);
            Ve(n-1,1)=(6.2500e-05*(Iel)/(1+(lamda_el*(T-T_ref))))+(0.0184*(T/298)*(log(ce(n,end))-log(ce(n,1))));
            cn(n,:)=EI_rad(cn(n-1,:),drn,(-Cr*9.9154e+07)/(Arrh_n),J_C);
             if k>1
                 [icra]=film(nr,rneg,Vm,cn(n,:),Rn,drn,cn_level(1,:),del_s,del,E_sei,dpl,E_pl,v_sei,v_pl,sigUTS_sei,gamma_n,W_cr_m);
                 icra_n(s-1,k)=icra;
             end
            [Ipl,Vn_p,Isei]=volt_cccv_chg(In,kr,csmaxn,cn(n,end),Un(cn(n,end)/csmaxn)+du_OCV_T_neg*(T-298.15),cec,k_sei,k_pl,R_film,icra,T);
            Ipl_1(s,k)=real(Ipl);
            Ipl_2(s-1)=real(Ipl);
            Isei_1(s,1)=real(Isei);
            Vn_p_1(s-1,1)=real(Vn_p);
            Vn(n-1,1)=real(Vn_p);
              if Ipl_1(s,1) ~= 0
                 q1=q1+1;
                 time_pl(q1)=time_pl(q1-1)+dt;         
              end         
           zita=Zita_c*exp(Ipl*0.56681);  
           Q2=Q2+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q3=Q3+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC=QC+((abs(In))*an*Ln*dt)*Area;

           Q21=Q21+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q31=Q31+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC1=QC1+((abs(In))*an*Ln*dt)*Area;
           
            [del_s,R_film,del,cec,dpl_irrev,dpl_rev]=d_sei(Isei,dt,Ipl,K_sei,K_pl,cec0,zita,dpl_rev,del_s,dpl_irrev); 
            Ic_cv(s,1)=Ic_cv(s-1,1)+((abs(Itot)*Area)*dt);

            Vp_cc(n-1,1)=real(Vp(n-1));
            Vn_cc(n-1,1)=real(Vn(n-1));
            Ve_cc(n-1,1)=real(Ve(n-1));

            Volt_cccv_cell(n-1)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            Volt_cell(n-1,k)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            Volt_cell_output(s-1)=Vp_cc(n-1)-Vn_cc(n-1)+Ve_cc(n-1)+Ir_drop+fitv;
            soc_p=soc_p+(abs(Ip)*ap*Lp*dt*Area);
            SOC(s-1)=soc_p/(nomCap*3600);
            if (Volt_cccv_cell(n-1))>=4.2
            var=5;   
            Charge(1,1)=Charge_4+abs((Itot*Area)*time_cc(E1_10+1));
            Ir_drop=Cr*(iapp*Area)*(0.15*Resis); 
            Iv=iapp*Cr;
            Itot=-Iv;
            Iel=((((abs(Itot))/2)*Ln)+((abs(Itot))*Ls)+(((abs(Itot))/2)*Lp))/(Ln+Ls+Lp);
            Ip=-Itot/(ap*Lp); 
            In=Itot/(an*Ln); 
            end
         case 5
            p=p+1;
            time_cccv(z)=time_cccv(z-1)+dt;
            E2=E2+1;
            s=s+1;
            [cp(n,:),Ip_cv(p,1)]=EI_rad_gal(cp(n-1,:),drp,Dsp,J_LCO_CV);
            [Vp_cal]=volt_cathode((Ip_cv(p,1)),kp,csmaxp,cp(n,end),U_LCO(cp(n,end)/csmaxp)+du_OCV_T_pos*(T-298.15),T,R_cei);
            Vp(n-1)=real(Vp_cal);
            [cn(n,:),In_cv(p,1)]=EI_rad_gal(cn(n-1,:),drn,Dsn,J_C_CV);
            Iel3=((((abs(In_cv(p,1))*Ln*an))/2)*Ln)+(((abs(In_cv(p,1))*Ln*an))*Ls)+((((abs(Ip_cv(p,1)*Lp*3*epsp/Rp))/2)*Lp))/(Ln+Ls+Lp);          
           if k>1
          [icra]=film(nr,rneg,Vm,cn(n,:),Rn,drn,cn_level(1,:),del_s,del,E_sei,dpl,E_pl,v_sei,v_pl,sigUTS_sei,gamma_n,W_cr_m);
          icra_n(s-1,k)=icra;
           end
            ce(n,:)=EI_el(ce(n-1,:),Lt,dt,Dl0,(In_cv(p,1)*Ln*an),epln,epsep,eplp,Ln,Ls,L);
            Ve(n-1,1)=(6.2500e-05*(Iel3/(1+(lamda_el*(T-T_ref)))))+(0.0184*(T/298)*(log(ce(n,end))-log(ce(n,1))));
            [Ipl,Vn_p,Isei]=volt_cccv_chg(In_cv(p,1),kr,csmaxn,cn(n,end),Un(cn(n,end)/csmaxn)+du_OCV_T_neg*(T-298.15),cec,k_sei,k_pl,R_film,icra,T);      
            Isei_1(s,1)=real(Isei);
            Ipl_1(s,k)=real(Ipl);
            Ipl_2(s-1)=real(Ipl);
            Vn_p_1(s-1,1)=real(Vn_p);
            Vn(n-1)=real(Vn_p);
            zita=Zita_c*exp(Ipl*0.56681); 
             Q2=Q2+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;
             Q3=Q3+((abs(Ipl_1(s,k)))*an*Ln*dt)*Area*zita;
             QC=QC+((abs(In))*an*Ln*dt)*Area;


           Q21=Q21+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;      
           Q31=Q31+((abs(Ipl_1(s,k)))*an*Ln*dt)*zita*Area;
           QC1=QC1+((abs(In))*an*Ln*dt)*Area;
            if Ipl_1(s,1) ~= 0
            q1=q1+1;
            time_pl(q1)=time_pl(q1-1)+dt;
            end
            [del_s,R_film,del,cec,dpl_irrev,dpl_rev]=d_sei(Isei,dt,Ipl,K_sei,K_pl,cec0,zita,dpl_rev,del_s,dpl_irrev);           
             Volt_cccv_cell(n-1)=Vp(n-1)-Vn(n-1)+Ve(n-1)+Ir_drop+fitv;
            Volt_cell(n-1,k)=Vp(n-1)-Vn(n-1)+Ve(n-1)+Ir_drop+fitv;
            Ic_cv(s,1)=Ic_cv(s-1,1)+((abs(In_cv(p,1))*Ln*an)*Area*dt);               
            Charge(p,1)=Charge(p-1,1)+((abs(In_cv(p,1))*Ln*an)*Area*dt);
            z=z+1; 
             soc_p=soc_p+(abs(Ip_cv(p,1))*ap*Lp*dt*Area);
            SOC(s-1)=soc_p/(nomCap*3600);
           if Charge(p,1)>=1*(nomCap*3600) ||(abs(In_cv(p,1))*Ln*an)<=((abs(Itot))/50) 
                mass_pl=4*pi*(Rn^2)*dpl_rev*rho_pl; 
                mass_irrevpl=4*pi*(Rn^2)*dpl_irrev*rho_pl;
                mpl_r(1,1)=mass_pl;
                Qs_irPl(k)=mass_irrevpl;
                Chg_cap=Charge(p,1);
                Itot=0;
                var=6;  
           end
           
           %Rest After Charging                  
        case 6
           s=s+1;
           E3=E3+1;
           m=m+1;
           time_rest_c(m)=time_rest_c(m-1)+dt;
          
          cp(n,:)=EI_rad(cp(n-1,:),drp,0,J_LCO);
           [Vp_cei]=volt_cathode((0),kp,csmaxp,cp(n,end),U_LCO(cp(n,end)/csmaxp)+du_OCV_T_pos*(T-298.15),T,R_cei);
            Vp(n-1)=real(Vp_cei);
            cn(n,:)=EI_rad(cn(n-1,:),drn,0,J_C);
           ce(n,:)=EI_el(ce(n-1,:),Lt,dt,Dl0,Itot,epln,epsep,eplp,Ln,Ls,L);
            Ve(n-1)= (T*6.1627e-05)*(log(ce(n,end))-log(ce(n,1)));
            [Isei2,Vn_s,Istr]=volt_rest(0,kr,csmaxn,cn(n,end),Un(cn(n,end)/csmaxn)+du_OCV_T_neg*(T-298.15),cec,k_sei,k_pl,R_film,mpl_r(m-1,1),mass_pl,T);           
            Isei_1(s,1)=real(Isei2);
            Vn_sp_1(s-1,1)=real(Vn_s);
            Vn(n-1)=real(Vn_s);
            Q2=Q2+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;
            Q21=Q21+((abs(Isei_1(s,1)))*an*Ln*dt)*Area;
            [del_s,dpl,R_film,del,cec,mpl,dpl_rev]=d_rest(Isei2,dt,Istr,K_sei,K_pl,cec0,D_ec,Rn,del_s,dpl_rev,dpl_irrev); 
            Vp_r(n-1)=real(Vp(n-1));
            Ve_r(n-1)=real(Ve(n-1));
            mpl_r(m,1)=mpl;
            Volt_rest_cell(n-1)=Vp_r(n-1)-Vn(n-1)+Ve_r(n-1);
            Volt_cell(n-1,k)=Vp(n-1)-Vn(n-1)+Ve(n-1);
            
           if time_rest_c(m)==1800
               var=7;
               df=0.3/Cr;
               Ir_drop=Cr*(iapp*Area)*(0.15*Resis); 
                Iv=iapp*Cr;
                Itot=+Iv;  
                Iel2=((((abs(Itot)*df)/2)*Ln)+((abs(Itot)*df)*Ls)+(((abs(Itot)*df)/2)*Lp))/(Ln+Ls+Lp);
                 Ip=-Itot/(ap*Lp); 
                 In=Itot/(an*Ln); 
           end
        %Discharge Constant Current

         case 7       
            m1=m1+1;
            E4=E4+1;
            time_dccv(m1)=time_dccv(m1-1)+dt;
            time_dc(m1)=time_dc(m1-1)+dt;
            cp(n,:)=EI_rad(cp(n-1,:),drp,(-Cr*df*1.2096e+08)/(Arrh_p),J_LCO);
            Vp(n-1,1)=volt_1((Ip*df),kp,csmaxp,cp(n,end),U_LCO(cp(n,end)/csmaxp)+du_OCV_T_pos*(T-298.15),T,R_cei);
            cn(n,:)=EI_rad(cn(n-1,:),drn,(Cr*df*9.9154e+07)/(Arrh_n),J_C);
            Vn(n-1)=volt_1((In*df),kr,csmaxn,cn(n,end),Un(cn(n,end)/csmaxn)+du_OCV_T_neg*(T-298.15),T,R_film);
            ce(n,:)=EI_el(ce(n-1,:),Lt,dt,Dl0,(Itot*df),epln,epsep,eplp,Ln,Ls,L);
            Ve(n-1)=(6.2500e-05*(Iel2)/(1+(lamda_el*(T-T_ref))))+(0.0184*(T/298)*(log(ce(n,end))-log(ce(n,1))));
            Vp_d(n-1)=real(Vp(n-1));
            Vn_d(n-1)=real(Vn(n-1));
            Ve_d(n-1)=real(Ve(n-1));
            Volt_dis_cell(n-1)=Vp_d(n-1)-Vn_d(n-1)+Ve_d(n-1)+Ir_drop*df;
            Volt_cell(n-1,k)=Vp_d(n-1)-Vn_d(n-1)+Ve_d(n-1)+Ir_drop*df;
            if (Volt_dis_cell(n-1))<=3
                Discharge(1,1)=(Itot*df*Area)*time_dc(E4); 
                discap(k)=Itot*df*time_dc(E4+1)*Area;
                var=8; 
            end
%Discharge Constant Voltage

    case 8
            m1=m1+1;
            y=y+1; 
            q=q+1;
            time_dccv(m1)=time_dccv(m1-1)+dt;
            time_dcv(y)=time_dcv(y-1)+dt;
            E5=E5+1;
            s1=s1+1;
            p1=p1+1;
            [cp(n,:),Ip_cv_d(q)]=EI_rad_gal(cp(n-1,:),drp,Dsp,J_LCO_CV);
            [Vp_cal]=volt_1(Ip_cv_d(q),kp,csmaxp,cp(n,end),U_LCO(cp(n,end)/csmaxp)+du_OCV_T_pos*(T-298.15),T,R_cei);
             Vp(n-1)=real(Vp_cal);
            [cn(n,:),In_cv_d(q)]=EI_rad_gal(cn(n-1,:),drn,Dsn,J_C_CV);
            Vn(n-1)=volt_1(In_cv_d(q),kr,csmaxn,cn(n,end),Un(cn(n,end)/csmaxn)+du_OCV_T_neg*(T-298.15),T,R_film);
            Iel3=((((abs(In_cv_d(q))*Ln*an))/2)*Ln)+(((abs(In_cv_d(q))*Ln*an))*Ls)+((((abs(Ip_cv_d(q)*Lp*3*epsp/Rp))/2)*Lp))/(Ln+Ls+Lp); %Average Current in Electroloyte
           ce(n,:)=EI_el(ce(n-1,:),Lt,dt,Dl0,(In_cv_d(q)*Ln*an),epln,epsep,eplp,Ln,Ls,L);
            Ve(n-1)=(6.2500e-05*(Iel3)/(1+(lamda_el*(T-T_ref))))+(0.0184*(T/298)*(log(ce(n,end))-log(ce(n,1))));
           Volt_cell(n-1,k)=Vp(n-1)-Vn(n-1)+Ve(n-1)+Ir_drop*df;
           Discharge(q,1)=Discharge(q-1,1)+((abs(In_cv_d(q))*Ln*an)*Area*dt);
          
           if  (abs(In_cv_d(q))*Ln*an)<=((abs(Itot))/25) || Discharge(q,1)>=Chg_cap
               discap(k)=Discharge(q,1);
               Itot=0;
                var=9; 
            end
%Rest after Discharge
         case 9
           
           E6=E6+1;
           m2=m2+1;
           time_rest_d(m2)=time_rest_d(m2-1)+dt;
           cp(n,:)=EI_rad(cp(n-1,:),drp,0,J_LCO);
           Vp(n-1)=volt_1((0),kp,csmaxp,cp(n,end),U_LCO(cp(n,end)/csmaxp)+du_OCV_T_pos*(T-298.15),T,R_cei);
           cn(n,:)=EI_rad(cn(n-1,:),drn,0,J_C);
            Vn(n-1)=volt_1((0),kr,csmaxn,cn(n,end),Un(cn(n,end)/csmaxn)+du_OCV_T_neg*(T-298.15),T,R_film);     
            ce(n,:)=EI_el(ce(n-1,:),Lt,dt,Dl0,(0),epln,epsep,eplp,Ln,Ls,L);
            Ve(n-1)=(0.0184*(T/298)*(log(ce(n,end))-log(ce(n,1))));
            Vp_r(n-1)=real(Vp(n-1));
            Vn_r(n-1)=real(Vn(n-1));
            Ve_r(n-1)=real(Ve(n-1));
            Volt_rest_cell(n-1)=Vp_r(n-1)-Vn_r(n-1)+Ve_r(n-1);
              Volt_cell(n-1,k)=Vp_r(n-1)-Vn_r(n-1)+Ve_r(n-1);      
           if  time_rest_d(m2)==1800
               var=10;
               Itot=-Iv;
               dt=4;
               cp2(k)=mean(cp(2,:));
               cp1=mean(cp(n-1,:));
             % eff(k)=(Q2+Q3+Q4)*100/QC;
               if k==1 && time_rest_d(m2)==1800
               cn_level(1,:)=cn(n,:);             
               end
           end
       end
    end

Cap=abs(discap(1));
Qs_cy(k)=Cap-abs(Q2)-abs(Q3)-abs(Q4);
Qs_rel(k)=Qs_cy(k)/Qs_cy(1)*100;
eff(k)=(Q21+Q31+Q4)*100/QC1;

end
    
    %Qs_rel_damage=(SOH*100)*
    % Combine the matrices if they have the same number of rows
    % combinedData = [SOC', Vn_p_1, Ipl_2'];
    
    % Write all data to a .mat file
    try
        % Save the combined data and Qs_rel to a .mat file
        % save('Anodic_potential_Charging.mat', 'combinedData'); 
        cycleName = sprintf('Cycle%d', k);
        
        % Create a structure for the cycle data.
        cycleData.(cycleName).SOC    = SOC';       % Transposed if needed
        cycleData.(cycleName).Vn_p_1 = Vn_p_1;
        cycleData.(cycleName).Ipl_2  = Ipl_2';
        
        % Save the cycle data to a MAT-file (using -v7.3 for large data support)
        save('Anodic_potential_Charging.mat', '-struct', 'cycleData', '-v7.3');


    catch ME
        fprintf('Error writing to .mat file: %s\n', ME.message);
    end
    
    % Write all data to a .mat file
    try
        % Save the combined data and Qs_rel to a .mat file
        save('SOH.mat', 'Qs_rel'); 
    catch ME
        fprintf('Error writing to .mat file: %s\n', ME.message);
    end
    
    
    toc
