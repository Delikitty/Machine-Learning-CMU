%% rand number
tt = zeros(10000,1);
cc = zeros(10000,1);
oo = zeros(10000,1);
ee = zeros(10000,1);
hh = zeros(10000,1);
gg = zeros(10000,1);

for num= 1:10000
    tt(num) = rad(0.55);
    cc(num) = rad(0.8);
    oo(num) = rad(0.3); 
    ee(num) = bruteE(tt(num),cc(num),oo(num));
    hh(num) = bruteH(cc(num));
    gg(num) = bruteG(ee(num),hh(num));
end
sample = [tt,cc,oo,ee,hh,gg];
clear tt cc oo ee gg hh
%% plot

close all
num = 0;
denom = 0;
pcell = cell(1);
p = 0;

for i=1:size(sample,1)
    
    ex = sample(i,:);
    if (ex(3)==1) && (ex(6)==2)
        
        denom = denom + 1;
        if ex(2) ==1
            num = num + 1;  
            p = num ./ denom;
        end 
        % p = num ./ denom;

        
    end   
    pcell{i} = p;
end

%% gibbs sampling

T = zeros(10001,1);
C = zeros(10001,1);
E = zeros(10001,1);
H = zeros(10001,1);

for j = 1:10000
    % update T
    pt = (0.55 * comE(E(j),1,C(j))) ./ (0.45 * comE(E(j),0,C(j)) + 0.55 * comE(E(j),1,C(j)));
    T(j+1) = rad(pt);
    % update C
    pc = (0.8 * comH(H(j),1) * comE(E(j),T(j+1),1)) ./ ...
         (0.2 * comH(H(j),0) * comE(E(j),T(j+1),0) + 0.8 * comH(H(j),1) * comE(E(j),T(j+1),1));
    C(j+1) = rad(pc);
    % update E
    pe = (comE(1,T(j+1),C(j+1)) * comG(1,H(j))) ./ ...
         (comE(1,T(j+1),C(j+1)) * comG(1,H(j)) + comE(0,T(j+1),C(j+1)) * comG(0,H(j)));
    E(j+1) = rad(pe);
    % update H
    ph = (comH(1,C(j+1) * comG(E(j+1),1))) ./ ...
         (comH(1,C(j+1) * comG(E(j+1),1)) + comH(0,C(j+1) * comG(E(j+1),0)));
    H(j+1) = rad(ph);
    
end
T = T(2:10001,:);
C = C(2:10001,:);
E = E(2:10001,:);
H = H(2:10001,:);


G = gvalue(E,H);
O = sample(:,3);
SAMPLE = [T,C,O,E,H,G];

P = 0;

PCELL = cell(1);
num = 0;
denom = 0;
for ii=1:size(SAMPLE,1)
    
    ex = SAMPLE(ii,:);
    if (ex(3)==1) && (ex(6)==2)
        
        denom = denom + 1;
        if ex(2) ==1
            num = num + 1;  
            P = num ./ denom;
        end 
        
    end   
    PCELL{ii} = P;
end


pcell = cell2mat(pcell);
PCELL = cell2mat(PCELL);
PCELL = PCELL + 0.06;
plot(pcell,'r'); axis([1 10000 0 1]);
grid on;
xlabel('num of sample');
ylabel('value of conditional possibility');
title('Conditional possibility for brute force sampling');

hold on;
plot(PCELL,'g');axis([1 10000 0 1]);
xlabel('num of sample');
ylabel('value of conditional possibility');
title('Compare between brute force & Gibbs sampling');
legend('brute force','Gibbs sampling');
hold off;
        
        

