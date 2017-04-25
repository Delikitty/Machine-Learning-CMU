function E = bruteE(t,c,o)

e = rand(1);

if t==0 && c==0 && o==0
    if e <= 0.15
        E = 1;
    else 
        E = 0;
    end
end

if t==0 && c==0 && o==1
    if e <= 0.2
        E = 1;
    else
        E = 0;
    end
end

if t==0 && c==1 && o==0
    if e <= 0.6
        E = 1;
    else
        E = 0;
    end
    
end

if t==0 && c==1 && o==1
    if e <=0.7
        E = 1;
    else
        E = 0;
    end
    
end

if t==1 && c==0 && o==0
    if e <= 0.4
        E = 1;
    else 
        E = 0;
    end
    
end

if t==1 && c==0 && o==1
    if e <= 0.5
        E = 1;
    else
        E = 0;
    end  
end

if t==1 && c==1 && o==0
    if e <= 0.85
        E = 1;
    else
        E = 0;
    end
    
end

if t==1 && c==1 && o==1
    if e <= 0.9
        E = 1;
    else
        E = 0;
    end
    
end



end