#!/usr/bin/env python
# coding: utf-8

# In[1]:


from QuantLib import *
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import namedtuple
import math

today = Date(2, January, 2023)
settlement = Date(2, January, 2024)
Settings.instance().evaluationDate = today

# input yield curve data as on 02/01/2024
yield_curve_rates = [0.055, 0.048, 0.0433, 0.0409, 0.0401, 0.0393, 0.0394, 0.0395, 0.0395]
dates = [settlement + Period(i, Years) for i in range(len(yield_curve_rates))]
yield_curve = YieldTermStructureHandle(ZeroCurve(dates, yield_curve_rates, Actual360()))
num_dates = [Actual360().yearFraction(today, date) for date in dates]

term_structure = yield_curve
index = Euribor1Y(term_structure)

#add hypothetical swaption data
CalibrationData = namedtuple("CalibrationData", "start, length, volatility")
data = [
    CalibrationData(1, 5, 0.1148),
    CalibrationData(2, 4, 0.1108),
    CalibrationData(3, 3, 0.1070),
    CalibrationData(4, 2, 0.1021),
    CalibrationData(5, 1, 0.1000)
]
#creating swaption helpers
def create_swaption_helpers(data, index, term_structure, engine):
    swaptions = []
    fixed_leg_tenor = Period(1, Years)
    fixed_leg_daycounter = Actual360()
    floating_leg_daycounter = Actual360()
    for d in data:
        vol_handle = QuoteHandle(SimpleQuote(d.volatility))
        helper = SwaptionHelper(
            Period(d.start, Years),
            Period(d.length, Years),
            vol_handle,
            index,
            fixed_leg_tenor,
            fixed_leg_daycounter,
            floating_leg_daycounter,
            term_structure
        )
        helper.setPricingEngine(engine)
        swaptions.append(helper)
    return swaptions
    
#caliberation report
def calibration_report(swaptions, data):
    columns = ["Model Price", "Market Price", "Implied Vol", "Market Vol", "Rel Error Price", "Rel Error Vols"]
    report_data = []
    cum_err = 0.0
    cum_err2 = 0.0
    for i, s in enumerate(swaptions):
        model_price = s.modelValue()
        market_vol = data[i].volatility
        black_price = s.blackPrice(market_vol)
        rel_error = model_price / black_price - 1.0
        implied_vol = s.impliedVolatility(model_price, 1e-5, 50, 0.0, 0.50)
        rel_error2 = implied_vol / market_vol - 1.0
        cum_err += rel_error * rel_error
        cum_err2 += rel_error2 * rel_error2
        report_data.append((model_price, black_price, implied_vol, market_vol, rel_error, rel_error2))
    
    print("Cumulative Error Price: %7.5f" % math.sqrt(cum_err))
    print("Cumulative Error Vols : %7.5f" % math.sqrt(cum_err2))
    
    return DataFrame(report_data, columns=columns, index=[''] * len(report_data))

model = HullWhite(term_structure)
engine = JamshidianSwaptionEngine(model)
swaptions = create_swaption_helpers(data, index, term_structure, engine)

optimization_method = LevenbergMarquardt(1.0e-8, 1.0e-8, 1.0e-8)
end_criteria = EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)

# Calibrate the Hull-White model
model.calibrate(swaptions, optimization_method, end_criteria)


# Plot both actual and calibrated yield curves
plt.figure(figsize=(10, 6))
plt.plot(num_dates, yield_curve_rates, marker='o', linestyle='-', color='b', label='Actual Yield Curve')
plt.title('Actual and Calibrated Yield Curves')
plt.xlabel('Time to Maturity (Years)')
plt.ylabel('Yield')
plt.plot(num_dates, calibrated_rates, marker='o', linestyle='--', color='r', label='Calibrated Yield Curve')
plt.legend()
plt.show()



# Print the calibrated parameters
a, sigma = model.params()
print(f"Hull-White Model Parameters: a = {a:.6f}, sigma = {sigma:.6f}")

# Print the calibration report
calibration_report(swaptions, data)


# In[ ]:




