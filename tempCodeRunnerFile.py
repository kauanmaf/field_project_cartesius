
def parabolic_sar(data, acceleration=0.02, max_acceleration=0.2):
    high = data['High'].values
    low = data['Low'].values

    # Initial settings
    sar = []
    af = acceleration
    ep = high[0]  # Initial Extreme Point (assumed high for the first period)
    trend = 1  # Assume starting with an uptrend

    for i in range(len(data)):
        if i == 0:
            sar.append(low[i])  # Initial SAR value is the first low price
            continue

        prev_sar = sar[-1]  # Previous SAR value

        if trend == 1:  # Uptrend
            new_sar = prev_sar + af * (ep - prev_sar)
            sar.append(min(new_sar, low[i-1], low[i]))  # Ensure SAR doesn't cross the lowest points
            if high[i] > ep:
                ep = high[i]  # Update extreme point
                af = min(af + acceleration, max_acceleration)  # Increase AF
            if low[i] < new_sar:  # Trend reversal condition
                trend = -1  # Switch to downtrend
                sar[-1] = ep  # On reversal, set SAR to the last extreme point
                ep = low[i]  # Reset extreme point for the new trend
                af = acceleration  # Reset AF

        else:  # Downtrend
            new_sar = prev_sar + af * (ep - prev_sar)
            sar.append(max(new_sar, high[i-1], high[i]))  # Ensure SAR doesn't cross the highest points
            if low[i] < ep:
                ep = low[i]  # Update extreme point
                af = min(af + acceleration, max_acceleration)  # Increase AF
            if high[i] > new_sar:  # Trend reversal condition
                trend = 1  # Switch to uptrend
                sar[-1] = ep  # On reversal, set SAR to the last extreme point
                ep = high[i]  # Reset extreme point for the new trend
                af = acceleration  # Reset AF

    return pd.Series(sar, index=data.index)
