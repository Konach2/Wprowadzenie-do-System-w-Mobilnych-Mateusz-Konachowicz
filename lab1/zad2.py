import math
import time
import matplotlib.pyplot as plt

class GeneratorU:
    def __init__(self, seed=None):
        if seed is None:
            self.x = int(time.time() * 1000) % 2147483647
        else:
            self.x = seed
        self.a = 16807
        self.m = 2147483647
        
    def gen(self):
        self.x = (self.a * self.x) % self.m
        return self.x / self.m

def gen_poisson(gen, lam):
    x = -1
    s = 1.0
    q = math.exp(-lam)
    while s > q:
        u = gen.gen()
        s *= u
        x += 1
    return max(0, x)

def gen_gauss(gen, mu, sigma):
    u1 = gen.gen()
    while u1 == 0: 
        u1 = gen.gen()
    u2 = gen.gen()
    
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z0

def main():
    
    ile_liczb = 10000     
    ziarno = 42           
    
    lam = 5.0             
    
    mu = 0.0              
    sigma = 1.0           
   
    gen_p = GeneratorU(seed=ziarno)
    gen_g = GeneratorU(seed=ziarno)
    
 
    dane_poisson = [gen_poisson(gen_p, lam) for _ in range(ile_liczb)]
    dane_gauss = [gen_gauss(gen_g, mu, sigma) for _ in range(ile_liczb)]
    
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bins_p = range(max(dane_poisson) + 2)
    ax1.hist(dane_poisson, bins=bins_p, edgecolor='black', align='left', color='skyblue')
    ax1.set_title(f'Rozkład Poissona (lambda={lam})')
    ax1.set_xlabel('Wartość k')
    ax1.set_ylabel('Ilość wystąpień')
    
    
    ax2.hist(dane_gauss, bins=50, edgecolor='black', color='orange')
    ax2.set_title(f'Rozkład Normalny / Gaussa (mu={mu}, sigma={sigma})')
    ax2.set_xlabel('Wartość x')
    ax2.set_ylabel('Ilość wystąpień')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()