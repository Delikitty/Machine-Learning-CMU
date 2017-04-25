function e = comE(E,T,C)


if T==0 && C==0
   if E == 1
       e = 0.2;
   else
       e = 0.8;
   end    
end


if T==0 && C==1
    if E == 1
        e = 0.7;
    else
        e = 0.3;
    end            
end


if T==1 && C==0
    e = 0.5;  
end

if T==1 && C==1
    if E == 1
        e = 0.9;
    else
        e = 0.1;
    end    
end



end
