function h = comH(H,C)

if C==1
    if H == 0
        h = 0.2;
    else
        h = 0.8;
    end    
else
    if H == 0
        h = 0.75;
    else
        h = 0.25;
    end    
end


end
