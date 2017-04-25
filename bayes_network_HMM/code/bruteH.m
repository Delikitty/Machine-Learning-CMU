function H = bruteH(c)

h = rand(1);

if c==0
    if h <= 0.25
        H = 1;
    else
        H = 0;
    end
end

if c==1
    if h <=0.8
        H = 1;
    else
        H = 0;
    end
    
end


end