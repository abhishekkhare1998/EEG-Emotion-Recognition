function [bp1, bp2, bp3, fig_num] = filter_def(debug_flag, fig_num)
    
    rp = 3;           % Passband ripple in dB 
    rs = 40;          % Stopband ripple in dB
    fs = 128;        % Sampling frequency
    dev = [10^(-rs/20) (10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)]; 
    
    f = [3 4 7 8];    % Cutoff frequencies
    a = [0 1 0];        % Desired amplitudes
    
    [n,fo,ao,w] = firpmord(f,a,dev,fs);
    bp1 = firpm(n,fo,ao,w);
     
    f = [7 8 12 13];    % Cutoff frequencies
    a = [0 1 0];        % Desired amplitudes
    
    [n,fo,ao,w] = firpmord(f,a,dev,fs);
    bp2 = firpm(n,fo,ao,w);
        
    f = [11 12 30 31];    % Cutoff frequencies
    a = [0 1 0];        % Desired amplitudes
    
    [n,fo,ao,w] = firpmord(f,a,dev,fs);
    bp3 = firpm(n,fo,ao,w);
    
    if(debug_flag == 1)
        figure(fig_num)
        freqz(bp1,1,1024,fs)
        title('4-7Hz BP Filter Designed to Specifications')
        
        figure(fig_num+1)
        freqz(bp2,1,1024,fs)
        title('8-12Hz BP Filter Designed to Specifications')
        
        figure(fig_num+2)
        freqz(bp3,1,1024,fs)
        title('12-30Hz BP Filter Designed to Specifications')
    end

end