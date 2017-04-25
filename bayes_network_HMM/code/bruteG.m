function G = bruteG(E,H)
    
    g = rand(1);
    
    if E==0 && H==0
        if g >= 0.95 && g <= 1
            G = 1;
        end
        if g >= 0.9 && g < 0.95
            G = 2;
        end
        if g >= 0.8 && g < 0.9
            G = 3;
        end
        if g >= 0.6 && g < 0.8
            G = 4;    
        end
        if g < 0.6
            G = 5;
        end
        
    end
    
    if E==0 && H==1
        
        if g >= 0.9 && g <= 1
            G = 1;
        end
        if g >= 0.6 && g < 0.9
            G = 2;
        end
        if g >= 0.3 && g < 0.6
            G = 3;
        end
        if g >= 0.1 && g < 0.3
            G = 4;
            
        end
        if g < 0.1
            G = 5;
        end
        
    end
    
    if E==1 && H==0
        
        if g >= 0.9 && g <= 1
            G = 1;
        end
        if g >= 0.6 && g < 0.9
            G = 2;
        end
        if g >= 0.3 && g < 0.6
            G = 3;
        end
        if g >= 0.1 && g < 0.3
            G = 4;
            
        end
        if g < 0.1
            G = 5;
        end
        
    end
    
    if E==1 && H==1
        
        if g >= 0.4 && g <= 1
            G = 1;
        end
        if g >= 0.2 && g < 0.4
            G = 2;
        end
        if g >= 0.1 && g < 0.2
            G = 3;
        end
        if g >= 0.05 && g < 0.1
            G = 4;
            
        end
        if g < 0.05
            G = 5;
        end
        
    end
    
    
    
    
end
