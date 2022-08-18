import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from vaccination import Vaccination 

#Inspiried from these websites:
#https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html
#https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_control_system_advanced.html#example-plot-control-system-advanced-py
# New Antecedent/Consequent objects hold universe variables and membership
# functions
vacc = ctrl.Antecedent(np.arange(0,1.01, 0.01), 'vacc_rates')
fail = ctrl.Antecedent( np.arange(-1,1, 0.01), 'fail_rates')
control = ctrl.Consequent(np.arange(-0.20,0.21, 0.02), 'control_rates')


# Custom membership functions can be built interactively with a familiar,
# Pythonic API
vacc['low'] = fuzz.trapmf(vacc.universe, [0, 0, 0.4, 0.6])
vacc['medium'] = fuzz.trimf(vacc.universe,[0.4, 0.6, 0.8])
vacc['high'] =  fuzz.trapmf(vacc.universe, [0.6, 0.8, 1, 1])

vacc.view()
# Custom membership functions can be built interactively with a familiar,
# Pythonic API
fail['low'] = fuzz.trapmf(fail.universe, [-1, -1, -0.5, 0.0])
fail['medium'] = fuzz.trimf(fail.universe,[-0.5, 0.0, 0.5])
fail['high'] =  fuzz.trapmf(fail.universe, [0.0, 0.5, 1.0, 1.0]) 
fail.view()
# Custom membership functions can be built interactively with a familiar,
# Pythonic API
control['verylow'] = fuzz.trapmf(control.universe, [-0.2, -0.2, -0.12,-0.06])
control['low'] = fuzz.trimf(control.universe, [-0.12,-0.06,0])
control['medium'] = fuzz.trimf(control.universe, [-0.06, 0, 0.06]) 
control['high'] = fuzz.trimf(control.universe,[0,0.06, 0.12])
control['veryhigh'] = fuzz.trapmf(control.universe,[0.06, 0.12, 0.2,0.2])
control.view()
rule0 = ctrl.Rule(antecedent=(vacc['low'] & fail['low']),
                  consequent=control['veryhigh'], label='rule very high')

rule1 = ctrl.Rule(antecedent=( (vacc['low'] & fail['medium']) |
                              (vacc['medium'] & fail['low']) ),
                  consequent=control['high'], label='rule high')

rule2 = ctrl.Rule(antecedent=( (vacc['medium'] & fail['medium']) |
                              (vacc['high'] & fail['low']) |
                              (vacc['low'] & fail['high']) ),
                  consequent=control['medium'], label='rule medium')

rule3 =  ctrl.Rule(antecedent=( (vacc['high'] & fail['medium']) |
                              (vacc['medium'] & fail['high']) ),
                  consequent=control['low'], label='rule low')

rule4 = ctrl.Rule(antecedent=(vacc['high'] & fail['high']),
                  consequent=control['verylow'], label='rule very low')


# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
system=Vaccination()
flag=True
cost=0
stopDiff=0.0001
for slot in range(200):
    
    controlRules = ctrl.ControlSystem(rules=[rule0, rule1, rule2, rule3, rule4])
    fuzz = ctrl.ControlSystemSimulation(controlRules)
    vacc_rate, fail_rate = system.checkVaccinationStatus()
    fuzz.input['vacc_rates'] = vacc_rate
    fuzz.input['fail_rates'] = fail_rate
    
    fuzz.compute()
    outputControl=fuzz.output['control_rates']
    system.vaccinatePeople(outputControl)
    vacc_rate, fail_rate = system.checkVaccinationStatus()
    if(flag==True):
        cost+=system.vaccination_rate_curve_[-1]#Sum all costs until equilibrium point
    currentDiff=abs(vacc_rate-0.6) #Calculate the diff between percentage and 60 
    if(currentDiff< stopDiff ) and flag==True: #Check if the difference is enough small
        flag=False #Flag will be Flase when equilibrium point is reached
        point_ss=slot #record the time when there is a equibilirium
        
system.viewVaccination(point_ss = point_ss, vaccination_cost = cost, filename='vaccination2')