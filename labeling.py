import numpy as np
from tradingUtils import *

def labelData(data, max_variation):
    label = np.zeros((data.shape[0], 1))
    
    for row in range(data.shape[0] - 1):
        try:
            
            close_current = data.iloc[row]["Adj Close"]
            close_next = data.iloc[row + 1]["Adj Close"]

            # Check for upward variation
            if close_next > close_current and (close_next - close_current) *100 / close_current >= max_variation:
                label[row] = 1
            # Check for downward variation
            elif close_next < close_current and abs((close_next - close_current) / close_current) >= max_variation:
                label[row] = -1
           
        except:
            pass

    return label

label_tsla_data = labelData(tsla_data, 0.1)
            

        

