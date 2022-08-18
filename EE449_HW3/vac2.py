from vaccination import Vaccination 
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# Resource is:
#https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
class Fuzzy:
    #Initialize constructors
    def __init__(self):
        #Call that class 
        self.vacc=Vaccination()
        #Initially update the percentages so that we can create membership functions 
        self.updateVaccPerc()
        # Generate universe variables
        #   * Vaccinated people set is [0.0,0.01,0.02,0.03,...,0.99,1.0]
        self.vacc_set =  np.arange(0,1.01, 0.01)
        #   * Vaccinated people set is [0.0,0.01,0.02,0.03,...,0.99,1.0]
        self.failure_set =  np.arange(-1,1, 0.01)
        #   * Control has a range of [-0.2, 0.2] in units of percentage points
        self.control_set = np.arange(-0.20,0.21, 0.02)
        
        
        # Generate fuzzy membership functions
        #https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
        #Vacc Rates
        self.vacc_low = fuzz.trapmf(self.vacc_set, [0, 0, 0.4, 0.6])
        self.vacc_mid = fuzz.trimf(self.vacc_set,[0.4, 0.6, 0.8])
        self.vacc_high = fuzz.trapmf(self.vacc_set, [0.6, 0.8, 1, 1])
        #Faiure Rates
        self.failure_low = fuzz.trapmf(self.failure_set, [-1, -1, -0.5, 0.0])
        self.failure_mid = fuzz.trimf(self.failure_set,[-0.5, 0.0, 0.5])
        self.failure_high = fuzz.trapmf(self.failure_set, [0.0, 0.5, 1.0, 1.0])        
        #Control Rates
        # self.control_low = fuzz.trimf(self.control_set, [-0.2, -0.2, 0])
        # self.control_mid = fuzz.trimf(self.control_set, [-0.2, 0, 0.2])
        # self.control_high = fuzz.trimf(self.control_set,[0, 0.2, 0.2])
        self.control_very_low= fuzz.trapmf(self.control_set, [-0.2, -0.2, -0.12,-0.06])
        self.control_low = fuzz.trimf(self.control_set, [-0.12,-0.06,0])
        self.control_mid = fuzz.trimf(self.control_set, [-0.06, 0, 0.06])
        self.control_high = fuzz.trimf(self.control_set,[0,0.06, 0.12])
        self.control_very_high= fuzz.trapmf(self.control_set,[0.06, 0.12, 0.2,0.2])
        #Plot sets partitioning
        self.plot()
    #Plotter for sets partitioning
    #Directly taken from :
    #https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
    def plot(self):
        # Visualize these universes and membership functions
        fig, (ax0, ax1,ax2) = plt.subplots(nrows=3, figsize=(8, 9))
        
        ax0.plot(self.vacc_set, self.vacc_low, 'b', linewidth=1.5, label='Low')
        ax0.plot(self.vacc_set, self.vacc_mid, 'g', linewidth=1.5, label='Mid')
        ax0.plot(self.vacc_set, self.vacc_high, 'r', linewidth=1.5, label='High')
        ax0.set_title('Vaccination Set')
        ax0.legend()

        ax1.plot(self.failure_set, self.failure_low, 'b', linewidth=1.5, label='Low')
        ax1.plot(self.failure_set, self.failure_mid, 'g', linewidth=1.5, label='Mid')
        ax1.plot(self.failure_set, self.failure_high, 'r',linewidth=1.5, label='High')
        ax1.set_title('Failure Set')
        ax1.legend()
        
        ax2.plot(self.control_set, self.control_very_low, 'y', linewidth=1.5, label='Very Low')
        ax2.plot(self.control_set, self.control_low, 'b', linewidth=1.5, label='Low')
        ax2.plot(self.control_set, self.control_mid, 'g', linewidth=1.5, label='Mid')
        ax2.plot(self.control_set, self.control_high, 'r',linewidth=1.5, label='High')
        ax2.plot(self.control_set, self.control_very_high, 'm',linewidth=1.5, label='Very High')

        ax2.set_title('Control Set')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("setsForVacc1.png")
    #This method is to find out control value
    #And apply it
    def apply(self):
        # We need the activation of our fuzzy membership functions at these values.
        # This is what fuzz.interp_membership exists for!
        self.vacc_level_low = fuzz.interp_membership(self.vacc_set , self.vacc_low, self.vacc_perc)
        self.vacc_level_mid = fuzz.interp_membership(self.vacc_set , self.vacc_mid, self.vacc_perc)
        self.vacc_level_high = fuzz.interp_membership(self.vacc_set , self.vacc_high, self.vacc_perc)
        # For rule 1 if we have low vacc level, then push it to high control
        self.rule1 =np.fmin(self.vacc_level_low, self.control_high)
        # For rule 2 if we have medium vacc level, then push it to medium control
        self.rule2 = np.fmin(self.vacc_level_mid, self.control_mid)
        # For rule 3 if we have high vacc level, then push it to low control
        self.rule3 = np.fmin(self.vacc_level_high, self.control_low)
        
        # Aggregate all three output membership functions together
        self.aggregated = np.fmax(self.rule1,np.fmax(self.rule2, self.rule3))
        # Calculate defuzzified result with centroid method
        control_out = fuzz.defuzz(self.control_set, self.aggregated, 'centroid')
        self.vacc.vaccinatePeople(control_out)#Apply controller to public
        
    #Updating vacc perc and cost values 
    def updateVaccPerc(self):
        self.vacc_perc,_ = self.vacc.checkVaccinationStatus()
        self.cost=self.vacc.vaccination_rate_curve_[-1]
        
system = Fuzzy()
#Take the necessa
flag=True #Flag will be True until equilibrium point
cost=0
stopDiff=0.0005
# for slot in range(200):
#     system.apply()#Apply the control by calling the method
#     system.updateVaccPerc()#Update variables so that we can check vacc percentage and calculate costs
#     if(flag==True):
#         cost+=system.cost#Sum all costs until equilibrium point
#     currentDiff=abs(system.vacc_perc-0.6) #Calculate the diff between percentage and 60 
#     if(currentDiff< stopDiff ) and flag==True: #Check if the difference is enough small
#         flag=False #Flag will be Flase when equilibrium point is reached
#         point_ss=slot #record the time when there is a equibilirium

        
system.vacc.viewVaccination(point_ss = point_ss, vaccination_cost = cost, filename='vaccination1')