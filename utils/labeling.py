import numpy as np

def labelData(data, max_variation):
    label = np.zeros((data.shape[0], 1))
    
    for row in range(data.shape[0] - 1):
        try:
            close_current = data[row]["Close"]
            close_next = data[row + 1]["Close"]
            
            # Check for upward variation
            if close_next > close_current and (close_next - close_current) / close_current > max_variation:
                label[row] = 1
            # Check for downward variation
            elif close_next < close_current and abs((close_next - close_current) / close_current) > max_variation:
                label[row] = -1
           
        except:
            pass

    return label

            

        

