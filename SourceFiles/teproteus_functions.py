import numpy as np
import math
import csv
import os
from numpy import genfromtxt

def get_cpatured_header(printHeader=False,N=1,buf=[]):
    header_size=72
    number_of_frames = N
    num_bytes = number_of_frames * header_size
    Proteus_header = []

    
    class header(object):
        def __init__(self):
            self.TriggerPos = 0
            self.GateLength = 0
            self.minVpp = 0
            self.maxVpp = 0
            self.TimeStamp = 0
            self.real1_dec = 0
            self.im1_dec = 0
            self.real2_dec = 0
            self.im2_dec = 0
            self.state1 = 0
            self.state2 = 0

    # create sets of header classes
    for i in range(number_of_frames):
        Proteus_header.append(header())

    for i in range(number_of_frames):
        idx = i* header_size
        Proteus_header[i].TriggerPos = buf[idx+0]*(1 << 0) + buf[idx+1]*(1 << 8) + buf[idx+2]*(1 << 16) + buf[idx+3]*(1 << 24)
        Proteus_header[i].GateLength = buf[idx+4]*(1 << 0) + buf[idx+5]*(1 << 8) + buf[idx+6]*(1 << 16) + buf[idx+7]*(1 << 24)
        Proteus_header[i].minVpp     = buf[idx+8]*(1 << 0) + buf[idx+9]*(1 << 8)
        Proteus_header[i].maxVpp     = buf[idx+10]*(1 << 0) + buf[idx+11]*(1 << 8)
        
        timeStamp1 = buf[idx+12]*(1 << 0)
        timeStamp2 = buf[idx+13]*(1 << 8)  
        timeStamp3 = buf[idx+14]*(1 << 16) 
        timeStamp4 = buf[idx+15]*(1 << 32) / 256
        timeStamp5 = buf[idx+16]*(1 << 32) 
        timeStamp6 = buf[idx+17]*(1 << 40) 
        timeStamp7 = buf[idx+18]*(1 << 48) 
        timeStamp8 = buf[idx+19]*(1 << 56)

        Proteus_header[i].TimeStamp = timeStamp1 + timeStamp2 + timeStamp3 + timeStamp4 + timeStamp5 + timeStamp6 + timeStamp7 + timeStamp8
        
        decisionRe1 = buf[idx+20]*(1 << 0) + buf[idx+21]*(1 << 8) + buf[idx+22]*(1 << 16) + buf[idx+23]*(1 << 24)
        decisionIm1 = buf[idx+24]*(1 << 0) + buf[idx+25]*(1 << 8) + buf[idx+26]*(1 << 16) + buf[idx+27]*(1 << 24)
        decisionRe2 = buf[idx+28]*(1 << 0) + buf[idx+29]*(1 << 8) + buf[idx+30]*(1 << 16) + buf[idx+31]*(1 << 24)
        decisionIm2 = buf[idx+32]*(1 << 0) + buf[idx+33]*(1 << 8) + buf[idx+34]*(1 << 16) + buf[idx+35]*(1 << 24)
        
        Proteus_header[i].real1_dec = decisionRe1
        Proteus_header[i].im1_dec = decisionIm1
        Proteus_header[i].real2_dec = decisionRe2
        Proteus_header[i].im2_dec = decisionIm2
        
        state1 = buf[idx+36]*(1 << 0)
        state2 = buf[idx+37]*(1 << 0)
        
        Proteus_header[i].state1 = state1
        Proteus_header[i].state2 = state2
    
    if(printHeader==True):
        outprint = 'header# {0}\n'.format(i)
        outprint += 'TriggerPos: {0}\n'.format(Proteus_header[i].TriggerPos)
        outprint += 'GateLength: {0}\n'.format(Proteus_header[i].GateLength )
        outprint += 'Min Amp: {0}\n'.format(Proteus_header[i].minVpp)
        outprint += 'Max Amp: {0}\n'.format(Proteus_header[i].maxVpp)
        outprint += 'TimeStamp: {0}\n'.format(Proteus_header[i].TimeStamp)
        outprint += 'Decision1: {0} + j* {1}\n'.format(Proteus_header[i].real1_dec,Proteus_header[i].im1_dec)
        outprint += 'Decision2: {0} + j* {1}\n'.format(Proteus_header[i].real2_dec,Proteus_header[i].im2_dec)
        outprint += 'STATE1: {0}\n'.format(Proteus_header[i].state1)
        outprint += 'STATE2: {0}\n'.format(Proteus_header[i].state2)
        print(outprint)
        
    return Proteus_header
	
def gauss_env(pw=50e-9,pl=100e-9,fs=2500e6,fc=10e6,interp=1,phase=0,direct=False,direct_lo=400e6,mode=8,SQP=False,NP=1,PG=50e6):
    
    if mode==8:
        res = 64
    else:
        res = 32
        
    pi = math.pi
    fs = fs / interp
    sigma = pw / 6
    variance = sigma**2
    pg = PG
    wavelength = pl * fs
    wavelength = res * math.ceil(wavelength / res)
    N = wavelength
    ts = 1 / fs
    t = np.linspace(-N*ts/2, N*ts/2, N, endpoint=False)
    tns = t * 1e9
    
    phase = phase * pi / 180
    
    fc_v = np.linspace(fc, fc+NP*pg, NP, endpoint=False)
    sinWave_m = [[np.sin(phase + 2 * pi * fc_v[y] * t[x]) for x in range(N)] for y in range(NP)] 
    cosWave_m = [[np.cos(phase + 2 * pi * fc_v[y] * t[x]) for x in range(N)] for y in range(NP)] 
    
    gauss_sq_pulse = (1/math.sqrt(2*pi*variance))*(np.exp(-t**2/(2*variance)))
    gauss_sq_pulse = np.clip(gauss_sq_pulse, a_min = 0, a_max = 1)
    gauss_e = np.exp(-t**2/(2*variance))
    
    if SQP==False:
        gauss_i_m = [[cosWave_m[y][x] * gauss_e[x] for x in range(N)] for y in range(NP)]
        gauss_q_m = [[sinWave_m[y][x] * gauss_e[x] for x in range(N)] for y in range(NP)] 
    else:
        gauss_i_m = [[cosWave_m[y][x] * gauss_sq_pulse[x] for x in range(N)] for y in range(NP)]
        gauss_q_m = [[sinWave_m[y][x] * gauss_sq_pulse[x] for x in range(N)] for y in range(NP)] 
    
    flo = direct_lo
    lo_sinWave = (np.sin(2 * pi * flo * t))
    lo_cosWave = (np.cos(2 * pi * flo * t))
    
    mod_m = [[(gauss_i_m[y][x] * lo_cosWave[x] - gauss_q_m[y][x] * lo_sinWave[x]) for x in range(N)] for y in range(NP)]
    mod = np.matrix(mod_m)
    mod = np.sum(mod,axis=0)
    
    env = np.zeros(N)
    gauss_i = np.zeros(N)
    gauss_q = np.zeros(N)
    
    if direct==True:
        env = np.squeeze(np.asarray(mod)) 
    else:
        if SQP==False:
            env = gauss_e
        else:
            env = gauss_sq_pulse
    
    
    gauss_i = np.matrix(gauss_i_m)
    gauss_i = np.sum(gauss_i,axis=0)
    gauss_q = np.matrix(gauss_q_m)
    gauss_q = np.sum(gauss_q,axis=0)
    
    gauss_i_A = np.squeeze(np.asarray(gauss_i))
    gauss_q_A = np.squeeze(np.asarray(gauss_q))
    
    
    return (env,gauss_i_A,gauss_q_A)

def chirp_pulse(WL=100e-9,PW=50e-9,fs=2500e6,Fstart=1e6,Fstop=10e6,interp=1,PHASE=0):
    res = 64 * interp
    wavelength = WL * fs
    wavelength = res * math.ceil(wavelength / res)
    pulselength = PW * fs
    pulselength = res * math.ceil(pulselength / res)
    Np = pulselength
    Nw = wavelength
    F1 = Fstart
    F2 = Fstop
    t = np.linspace(0, PW, Np, endpoint=False)
    pi = math.pi
    c = (F2-F1)/PW
    phase = PHASE
    p = np.sin(phase + 2 * pi * (c * t**2 / 2 + F1 * t))
    w = np.zeros(Nw)
    w[int(Nw/2-Np/2):int(Nw/2+Np/2)] = p
    w = w[::interp]
    return(w)

def iq_kernel(fs=1350e6,flo=400e6,kl=10240,coe_file_path='sfir_81_tap.csv'):
    # load coe data for the FIR filter
    data = genfromtxt(coe_file_path, delimiter=',')
    coe = data[1::1]
    TAP = coe.size
    print('loaded {0} TAP filter from {1}'.format(TAP,coe_file_path))
    res = 10
    L = res * math.ceil(kl / res)
    k = np.ones(L+TAP)
    
    pi = math.pi
    ts = 1 / fs
    t = np.linspace(0, L*ts, L, endpoint=False)
    
    loi = np.cos(2 * pi * flo * t)
    loq = -(np.sin(2 * pi * flo * t))
    
    k_i = np.zeros(L)
    k_q = np.zeros(L)
    
    for l in range(L):
        b = 0
        for n in range(TAP):
            b += k[l+n]*coe[n]
        k_q[l] = loq[l] * b
        k_i[l] = loi[l] * b
    
    print('sigma bn = {0}'.format(b))	
    return(k_i,k_q)

def pack_kernel_data(ki,kq,EXPORT=False,PATH=''):
    out_i = []
    out_q = []
    L = int(ki.size/5)
    
    b_ki = np.zeros(ki.size)
    b_kq = np.zeros(ki.size)
    kernel_data = np.zeros(L*4)
    
    b_ki = b_ki.astype(np.uint16)
    b_kq = b_kq.astype(np.uint16)
    kernel_data = kernel_data.astype(np.uint32)
    
    
	
    #print('ki 0:9 = ',ki[:10])
    #print('kq 0:9 = ',kq[:10])
    #print('ki[0] = ',ki[:100:10])
    #print('ki[9] = ',ki[9:100:10])
	
    # convert the signed number into 12bit FIX1_11 presentation
    b_ki,b_kq = convert_IQ_to_sample(ki,kq,12)
    
    
    #print('b_ki[0] = ',b_ki[:100:10])
    #print('b_ki[9] = ',b_ki[9:100:10])
    
	# convert 12bit to 15bit because of FPGA memory structure
    for i in range(L):
        s1 = (b_ki[i*5+1]&0x7) * 4096 + ( b_ki[i*5]               )
        s2 = (b_ki[i*5+2]&0x3F) * 512 + ((b_ki[i*5+1]&0xFF8) >> 3 )
        s3 = (b_ki[i*5+3]&0x1FF) * 64 + ((b_ki[i*5+2]&0xFC0) >> 6 )
        s4 = (b_ki[i*5+4]&0xFFF) *  8 + ((b_ki[i*5+3]&0xE00) >> 9 )
        out_i.append(s1)
        out_i.append(s2)
        out_i.append(s3)
        out_i.append(s4)
     
    out_i = np.array(out_i)
    
    for i in range(L):
        s1 = (b_kq[i*5+1]&0x7) * 4096 + ( b_kq[i*5]               )
        s2 = (b_kq[i*5+2]&0x3F) * 512 + ((b_kq[i*5+1]&0xFF8) >> 3 )
        s3 = (b_kq[i*5+3]&0x1FF) * 64 + ((b_kq[i*5+2]&0xFC0) >> 6 )
        s4 = (b_kq[i*5+4]&0xFFF) *  8 + ((b_kq[i*5+3]&0xE00) >> 9 )
        out_q.append(s1)
        out_q.append(s2)
        out_q.append(s3)
        out_q.append(s4)

    out_q = np.array(out_q)
	
    #print('out_i = ',out_i[:80:10])
    #print('out_q = ',out_q[:80:10])
	
    fout_i = np.zeros(out_i.size)
    fout_q = np.zeros(out_q.size)

    for i in range(out_i.size):
        if(out_i[i] >16383):
            fout_i[i] = out_i[i] - 32768
        else:
            fout_i[i] = out_i[i]

    for i in range(out_q.size):
        if(out_q[i] >16383):
            fout_q[i] = out_q[i] - 32768
        else:
            fout_q[i] = out_q[i]

    #print('fout_i = ',fout_i[:80:10])
	
    for i in range(L*4):
        kernel_data[i] = out_q[i]*(1 << 16) + out_i[i]

    #print('kernel_data = ',kernel_data[:80:10])


    sim_kernel_data = []

    for i in range(kernel_data.size):
        sim_kernel_data.append(hex(kernel_data[i])[2:])

    #print('sim_kernel_data = ',sim_kernel_data[:80:10])
    
    if(EXPORT==True):
        if not os.path.exists(PATH):
            os.mkdir(PATH)

        np.savetxt(PATH+"/kernel_filt.csv"    , list(zip(fout_i,fout_q)), delimiter=',', fmt='%d')
        np.savetxt(PATH+"/mem_data.csv"       , kernel_data             , delimiter=',', fmt='%d')
        np.savetxt(PATH+"/sim_mem_data.csv"   , sim_kernel_data         , delimiter=',', fmt='%s')

    return kernel_data

# convert signed number in the range of -1 to 1 to a signed FIX(SIZE_0) representation
def convert_to_sample(inp,size):
    out = np.zeros(inp.size)
    out = out.astype(np.uint32)

    
    M = 2**(size-1)
    A = 2**(size)
    
    for i in range(inp.size):
        if(inp[i] < 0):
            out[i] = int(inp[i]*M) + A
        else:
            out[i] = int(inp[i]*(M-1))

    return out
	
def convert_IQ_to_sample(inp_i,inp_q,size):
    out_i = np.zeros(inp_i.size)
    out_i = out_i.astype(np.uint32)
	
    out_q = np.zeros(inp_q.size)
    out_q = out_q.astype(np.uint32)

    max_i = np.amax(abs(inp_i))
    max_q = np.amax(abs(inp_q))
	
    max = np.maximum(max_i,max_q)
	
    if max < 1:
        max = 1
	
    inp_i = inp_i / max
    inp_q = inp_q / max
    
    M = 2**(size-1)
    A = 2**(size)
    
    for i in range(inp_i.size):
        if(inp_i[i] < 0):
            out_i[i] = int(inp_i[i]*M) + A
        else:
            out_i[i] = int(inp_i[i]*(M-1))
			
    for i in range(inp_q.size):
        if(inp_q[i] < 0):
            out_q[i] = int(inp_q[i]*M) + A
        else:
            out_q[i] = int(inp_q[i]*(M-1))

    return out_i , out_q
	
def convert_sample_to_signed(inp,size):
    out = np.zeros(inp.size)
    
    M = 2**(size-1)
    A = 2**(size)
    
    for i in range(inp.size):
        if(inp[i] > (M-1)):
            out[i] = math.floor(inp[i]) - A
        else:
            out[i] = math.floor(inp[i])
            
    return out / M

def convert_to_sized_decimal(inp,size):
    out = np.zeros(inp.size)
    out = out.astype(np.int64)

    
    M = 2**(size-1)
    
    for i in range(inp.size):
        if(inp[i] < 0):
            out[i] = int(inp[i]*M)
        else:
            out[i] = int(inp[i]*(M-1))

    return out
	
def convertFftRawDataTodBm(i,q,adcfs=1000,adc_clk=2700e6,decimation=16):
    maxadc = 2**15 - 1
    sampRateMHz = adc_clk / 1e6
    iSample = np.linspace(start=0, stop=1024, num=1024, endpoint=False)
    iPt = np.zeros(1024, dtype=np.double)
    qPt = np.zeros(1024, dtype=np.double)
    
    # convert the bynary offset presentation to a 0 to ADCFS presentation
    iPt = adcfs * i / maxadc;
    qPt = adcfs * q / maxadc;

    # convert to signed presentation
    qPt = qPt - adcfs / 2;
    iPt = iPt - adcfs / 2;
    
    # calculate VRMS FFT
    fft = np.sqrt(iPt**2 + qPt**2);
    
    # change FFT to power presentation
    p = fft**2;
    
    fs = ( sampRateMHz / ( decimation * 1024 ) ) * iSample;
    
    Pdbm = 10 * np.log10( p / ( 50 * 1000 ) );
    
    return Pdbm,fs
	
def convertTimeRawDataTomV(i,q,adcfs=1000):
    maxadc = 2**15 - 1
    
    iPt = np.zeros(i.size, dtype=np.double)
    qPt = np.zeros(q.size, dtype=np.double)
    
    # convert the bynary offset presentation to a 0 to ADCFS presentation
    iPt = adcfs * i / maxadc;
    qPt = adcfs * q / maxadc;

    # convert to signed presentation
    qPt = qPt - adcfs / 2;
    iPt = iPt - adcfs / 2;
    
    return iPt,qPt
	
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    """

    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')

    if x.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size.')


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def pack_fir_data(coe_file_path='sfir_81_tap.csv'):
    # load coe data for the FIR filter
    data = genfromtxt(coe_file_path, delimiter=',')
    coe = data[1::1]
    TAP = coe.size
    print('loaded {0} TAP filter from {1}'.format(TAP,coe_file_path))
    
    #limit coeficient to the range -1 to 1
    max_coe = np.amax(abs(coe))
    if max_coe < 1:
        max_coe = 1
        
    coe = coe / max_coe
    
    # convert to FIX24_23
    fir_data = np.zeros(coe.size)
    fir_data = fir_data.astype(np.double)
    fir_data = coe

    return fir_data
	
def iq_debug_kernel(fs=1350e6,flo=400e6,kl=10240,delay=20):
    ts = 1 / fs
    L = kl
    offset = delay * ts
    t = np.linspace(0, L*ts, L, endpoint=False)
    t[delay:] = t[:L-delay]
    t[0:delay] = 0
    pi = math.pi

    loi = (np.cos(2 * pi * flo * t))
    loq = -(np.sin(2 * pi * flo * t))
    
    return loi,loq