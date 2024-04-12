import numpy as np
import matplotlib.pyplot as plt

class AS:
    def __init__(self, s0=100, T=1, sigma=2, dt=0.005, q0=0, gamma=0.1, k=1.5, A=140, sim_length=1000):
        self.s0 = s0
        self.T = T
        self.sigma = sigma
        self.dt = dt
        self.q0 = q0
        self.gamma = gamma
        self.k = k
        self.A = A
        self.sim_length = sim_length
        self.calc_spread()

    def initialize(self):
        self.inventory_s1 = [self.q0] * self.sim_length
        self.pnl_s1 = [0] * self.sim_length
        self.inventory_s2 = [self.q0] * self.sim_length
        self.pnl_s2 = [0] * self.sim_length
        self.price_a = [0] * (int(self.T / self.dt) + 1)
        self.price_b = [0] * (int(self.T / self.dt) + 1)
        self.midprice = [0] * (int(self.T / self.dt) + 1)
        
    def calc_spread(self):
        self.sym_spread = 0
        for i in np.arange(0, self.T, self.dt):
            self.sym_spread += self.gamma * self.sigma**2 * (self.T - i) + (2/self.gamma) * np.log(1 + (self.gamma / self.k))
        self.av_sym_spread = (self.sym_spread / (self.T / self.dt))
        self.prob = self.A * np.exp(- self.k * self.av_sym_spread / 2) * self.dt

    def simulate(self, plotting=True):
        self.initialize()
        
        for i in range(self.sim_length):
            white_noise = self.sigma * np.sqrt(self.dt) * np.random.choice([1, -1], int(self.T / self.dt))
            price_process = self.s0 + np.cumsum(white_noise)
            price_process = np.insert(price_process, 0, self.s0)
            for step, s in enumerate(price_process):
        
                """
                Inventory strategy
                """
        
                reservation_price = s - self.inventory_s1[i] * self.gamma * \
                                        self.sigma**2 * (self.T - step * self.dt)
                spread = self.gamma * self.sigma**2 * (self.T - step * self.dt) + \
                         (2 / self.gamma) * np.log(1 + (self.gamma / self.k))
                spread /= 2
                if reservation_price >= s:
                    ask_spread = spread + (reservation_price - s)
                    bid_spread = spread - (reservation_price - s)
                else:
                    ask_spread = spread - (s - reservation_price)
                    bid_spread = spread + (s - reservation_price)
        
                ask_prob = self.A * np.exp(- self.k * ask_spread) * self.dt
                bid_prob = self.A * np.exp(- self.k * bid_spread) * self.dt
                ask_prob = max(0, min(ask_prob, 1))
                bid_prob = max(0, min(bid_prob, 1))
                ask_action_s1 = np.random.choice([1, 0],
                                                 p=[ask_prob, 1 - ask_prob])
                bid_action_s1 = np.random.choice([1, 0],
                                                 p=[bid_prob, 1 - bid_prob])
        
                self.inventory_s1[i] -= ask_action_s1
                self.pnl_s1[i] += ask_action_s1 * (s + ask_spread)
                self.inventory_s1[i] += bid_action_s1
                self.pnl_s1[i] -= bid_action_s1 * (s - bid_spread)
        
                if i == 0:
                    self.price_a[step] = s + ask_spread
                    self.price_b[step] = s - bid_spread
                    self.midprice[step] = s
        
                """
                Symmetric strategy
                """
        
                ask_action_s2 = np.random.choice([1, 0], p=[self.prob, 1 - self.prob])
                bid_action_s2 = np.random.choice([1, 0], p=[self.prob, 1 - self.prob])
                self.inventory_s2[i] -= ask_action_s2
                self.pnl_s2[i] += ask_action_s2 * (s + self.av_sym_spread / 2)
                self.inventory_s2[i] += bid_action_s2
                self.pnl_s2[i] -= bid_action_s2 * (s - self.av_sym_spread / 2)
            self.pnl_s1[i] += self.inventory_s1[i] * s
            self.pnl_s2[i] += self.inventory_s2[i] * s
        if plotting:
            x_range = [-50, 150]
            y_range = [0, 250]
            plt.figure(figsize=(16, 12), dpi=100)
            bins = np.arange(x_range[0], x_range[1] + 1, 4)
            plt.hist(self.pnl_s1, bins=bins, alpha=0.25, label="Inventory strategy")
            plt.hist(self.pnl_s2, bins=bins, alpha=0.25, label="Symmetric strategy")
            plt.ylabel('PnL')
            plt.legend()
            plt.axis(x_range + y_range)
            plt.title("The PnL histogram of the two strategies")
            plt.show()
    
            x = np.arange(0, self.T + self.dt, self.dt)
            plt.figure(figsize=(16, 12), dpi=100)
            plt.plot(x, self.price_a, linewidth=1.0, linestyle="-",label="ASK")
            plt.plot(x, self.price_b, linewidth=1.0, linestyle="-",label="BID")
            plt.plot(x, self.midprice, linewidth=1.0, linestyle="-",label="MID-PRICE")
            plt.legend()
            plt.title("Mid-price, optimal bid and ask quotes")
            plt.show()
        self.out_string()
        
    def out_string(self):            
        print("PnL\nMean of the inventory strategy v.s. symmetric strategy: {} v.s. {}".format(round(np.array(self.pnl_s1).mean(),2), round(np.array(self.pnl_s2).mean(),2)))
        print("Standard deviation of the inventory strategy v.s. symmetric strategy: {} v.s. {}".format(round(np.sqrt(np.array(self.pnl_s1).var()),2),round(np.sqrt(np.array(self.pnl_s2).var()),2)))
        print("INV\nMean of the inventory strategy v.s. symmetric strategy: {} v.s. {}".format(round(np.array(self.inventory_s1).mean(),2), round(np.array(self.inventory_s2).mean(),2)))
        print("Standard deviation of the inventory strategy v.s. symmetric strategy: {} v.s. {}".format(round(np.sqrt(np.array(self.inventory_s1).var()),2), round(np.sqrt(np.array(self.inventory_s2).var()),2)))