function G = comG(E,H)

if E==0 && H==0
    G = 0.05;
end

if E==0 && H==1
    G = 0.3;
    
end

if E==1 && H==0
    G = 0.3;
    
end

if E==1 && H==1
    G = 0.2;
    
end



end
