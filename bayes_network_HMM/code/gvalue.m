function G = gvalue(E,H)

G = zeros(size(E));

for i=1:length(E)
    
    g = rand(1);
    
    if E(i)==0 && H(i)==0
        if g >= 0.95 && g <= 1
            G(i) = 1;
        end
        if g >= 0.9 && g < 0.95
            G(i) = 2;
        end
        if g >= 0.8 && g < 0.9
            G(i) = 3;
        end
        if g >= 0.6 && g < 0.8
            G(i) = 4;
            
        end
        if g < 0.6
            G(i) = 5;
        end
        
    end
    
    if E(i)==0 && H(i)==1
        
        if g >= 0.9 && g <= 1
            G(i) = 1;
        end
        if g >= 0.6 && g < 0.9
            G(i) = 2;
        end
        if g >= 0.3 && g < 0.6
            G(i) = 3;
        end
        if g >= 0.1 && g < 0.3
            G(i) = 4;
            
        end
        if g < 0.1
            G(i) = 5;
        end
        
    end
    
    if E(i)==1 && H(i)==0
        
        if g >= 0.9 && g <= 1
            G(i) = 1;
        end
        if g >= 0.6 && g < 0.9
            G(i) = 2;
        end
        if g >= 0.3 && g < 0.6
            G(i) = 3;
        end
        if g >= 0.1 && g < 0.3
            G(i) = 4;
            
        end
        if g < 0.1
            G(i) = 5;
        end
        
    end
    
    if E(i)==1 && H(i)==1
        
        if g >= 0.4 && g <= 1
            G(i) = 1;
        end
        if g >= 0.2 && g < 0.4
            G(i) = 2;
        end
        if g >= 0.1 && g < 0.2
            G(i) = 3;
        end
        if g >= 0.05 && g < 0.1
            G(i) = 4;
            
        end
        if g < 0.05
            G(i) = 5;
        end
        
    end
    
    
    
    
end

end
