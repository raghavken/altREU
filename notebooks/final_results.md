# Final results

All numbers below are copied directly from the executed notebooks (final.ipynb, final2.ipynb, finalResults.ipynb). Errors are in basis points (bp). Directional accuracy (DA) skips unwinnable ties and shows the base rate to beat; the random walk shows nan because it predicts zero change. Weighted DA trades every EWA predicted move of at least 1bp with a 0.5bp cost per trade, 0.5 is break even.

## final.ipynb (main research file, 21 rolling windows, level targets)

```
==============================
Model:  Random Walk
==============================
Horizon: 1 day, rmse: 4.440564684656077, MAE: 3.259740259740264, Directional Accuracy: 12.554112554112551 
Horizon: 5 day, rmse: 5.591041071190615, MAE: 3.891774891774897, Directional Accuracy: 12.554112554112555 
Horizon: 20 day, rmse: 9.769862662812438, MAE: 7.943722943722947, Directional Accuracy: 3.463203463203463 
==============================
Model:  VARX
==============================
Horizon: 1 day, rmse: 7.992125580872543, MAE: 6.322842969126873, Directional Accuracy: 44.15584415584414 
Horizon: 5 day, rmse: 8.68972424321125, MAE: 6.937809667835283, Directional Accuracy: 44.58874458874458 
Horizon: 20 day, rmse: 12.006148343204101, MAE: 8.890764876485349, Directional Accuracy: 59.3073593073593 
==============================
Model:  XGBoost
==============================
Horizon: 1 day, rmse: 13.752478847559573, MAE: 11.281875030845733, Directional Accuracy: 48.48484848484848 
Horizon: 5 day, rmse: 18.570333020919826, MAE: 13.96543480295051, Directional Accuracy: 47.61904761904761 
Horizon: 20 day, rmse: 31.818400679299256, MAE: 26.58935773050582, Directional Accuracy: 32.46753246753246 
```

## finalResults.ipynb (final calculations: holdout and forward test)

```
holdout windows:  1149  ( 2021-09-09  to  2026-04-16 )
forward windows:  19  ( 2026-05-15  to  2026-06-11 )
##############################
Dataset:  holdout (2021-09 onward)
##############################
==============================
Model:  Random Walk
==============================
Horizon: 1 day, rmse: 6.111764656832587, MAE: 4.145343777197563, Directional Accuracy: nan %, base rate: 51.37664207306101 % 
Horizon: 5 day, rmse: 13.194960041173733, MAE: 9.1994619827518, Directional Accuracy: nan %, base rate: 55.74819054031308 % 
Horizon: 20 day, rmse: 28.028830115927725, MAE: 20.43674341324472, Directional Accuracy: nan %, base rate: 57.039505771419286 % 

==============================
Model:  VARX
==============================
Horizon: 1 day, rmse: 6.115001736331455, MAE: 4.176150198076738, Directional Accuracy: 52.82526543098795 %, base rate: 51.37664207306101 % 
Horizon: 5 day, rmse: 13.092947022264083, MAE: 9.17592679234941, Directional Accuracy: 56.3457330415755 %, base rate: 55.74819054031308 % 
Horizon: 20 day, rmse: 27.035527572421802, MAE: 19.78421597115533, Directional Accuracy: 58.29946350186962 %, base rate: 57.039505771419286 % 

==============================
Model:  EWA Ensemble
==============================
Horizon: 1 day, rmse: 6.105249345806767, MAE: 4.155626064391398, Directional Accuracy: 52.82526543098795 %, base rate: 51.37664207306101 % 
Horizon: 5 day, rmse: 13.077081826542454, MAE: 9.112997561146324, Directional Accuracy: 56.3457330415755 %, base rate: 55.74819054031308 % 
Horizon: 20 day, rmse: 27.063825464399134, MAE: 19.743808058391576, Directional Accuracy: 58.29946350186962 %, base rate: 57.039505771419286 % 

weighted directional accuracy, EWA Ensemble ( 1.0 bp threshold,  0.5 bp cost per trade)
Horizon: 1 day, weighted DA: 0.5485, capture: 0.097, trades: 224, net pnl: 113.0 bp
Horizon: 5 day, weighted DA: 0.5945, capture: 0.1891, trades: 3757, net pnl: 7063.5 bp
Horizon: 20 day, weighted DA: 0.6408, capture: 0.2816, trades: 8776, net pnl: 54804.0 bp

##############################
Dataset:  forward test (after 2026-05-14, frozen pipeline)
##############################
==============================
Model:  Random Walk
==============================
Horizon: 1 day, rmse: 4.003586908507335, MAE: 2.8038277511961742, Directional Accuracy: nan %, base rate: 55.02958579881657 % 
Horizon: 5 day, rmse: 7.250804278107762, MAE: 5.751196172248804, Directional Accuracy: nan %, base rate: 53.608247422680414 % 
Horizon: 20 day, rmse: 11.224759031311311, MAE: 8.866028708133976, Directional Accuracy: nan %, base rate: 63.86138613861386 % 

==============================
Model:  VARX
==============================
Horizon: 1 day, rmse: 3.9985123084971117, MAE: 2.8478341823673454, Directional Accuracy: 48.5207100591716 %, base rate: 55.02958579881657 % 
Horizon: 5 day, rmse: 7.058428200085527, MAE: 5.595658731123731, Directional Accuracy: 63.4020618556701 %, base rate: 53.608247422680414 % 
Horizon: 20 day, rmse: 11.232374137395789, MAE: 9.16482980618467, Directional Accuracy: 58.91089108910891 %, base rate: 63.86138613861386 % 

==============================
Model:  EWA Ensemble
==============================
Horizon: 1 day, rmse: 3.9973826187693153, MAE: 2.827257913895278, Directional Accuracy: 48.5207100591716 %, base rate: 55.02958579881657 % 
Horizon: 5 day, rmse: 7.1314019109179165, MAE: 5.651114936251911, Directional Accuracy: 63.4020618556701 %, base rate: 53.608247422680414 % 
Horizon: 20 day, rmse: 10.977498677128843, MAE: 8.903529153942818, Directional Accuracy: 58.91089108910891 %, base rate: 63.86138613861386 % 

weighted directional accuracy, EWA Ensemble ( 1.0 bp threshold,  0.5 bp cost per trade)
Horizon: 1 day, no trades cleared the threshold
Horizon: 5 day, weighted DA: 0.5268, capture: 0.0537, trades: 44, net pnl: 16.0 bp
Horizon: 20 day, weighted DA: 0.6921, capture: 0.3843, trades: 177, net pnl: 564.5 bp
```

## final2.ipynb (statistical tests, stress periods, forward test detail)

```
==============================
Model:  Random Walk
==============================
Horizon: 1 day, rmse: 6.1117646568325865, MAE: 4.145343777197564, Directional Accuracy: nan %, base rate: 51.37664207306101 % 
Horizon: 5 day, rmse: 13.194960041173735, MAE: 9.1994619827518, Directional Accuracy: nan %, base rate: 55.74819054031308 % 
Horizon: 20 day, rmse: 28.02883011592773, MAE: 20.436743413244717, Directional Accuracy: nan %, base rate: 57.039505771419286 % 
==============================
Model:  VARX
==============================
Horizon: 1 day, rmse: 6.115001736331455, MAE: 4.176150198076738, Directional Accuracy: 52.82526543098795 %, base rate: 51.37664207306101 % 
Horizon: 5 day, rmse: 13.092947022264084, MAE: 9.175926792349411, Directional Accuracy: 56.3457330415755 %, base rate: 55.74819054031308 % 
Horizon: 20 day, rmse: 27.035527572421802, MAE: 19.78421597115533, Directional Accuracy: 58.29946350186962 %, base rate: 57.039505771419286 % 
==============================
Model:  EWA Ensemble
==============================
Horizon: 1 day, rmse: 6.105249345806767, MAE: 4.155626064391398, Directional Accuracy: 52.82526543098795 %, base rate: 51.37664207306101 % 
Horizon: 5 day, rmse: 13.077081826542452, MAE: 9.112997561146324, Directional Accuracy: 56.3457330415755 %, base rate: 55.74819054031308 % 
Horizon: 20 day, rmse: 27.063825464399134, MAE: 19.743808058391583, Directional Accuracy: 58.29946350186962 %, base rate: 57.039505771419286 % 
negative statistic favors the first model
h= 1   EWA Ensemble  vs  Random Walk  DM:  -1.09  p:  0.2745  BH q:  0.4118
h= 1   VARX  vs  Random Walk  DM:  0.28  p:  0.7814  BH q:  0.7814
h= 1   EWA Ensemble  vs  VARX  DM:  -1.68  p:  0.0942  BH q:  0.2119
h= 5   EWA Ensemble  vs  Random Walk  DM:  -2.06  p:  0.04  BH q:  0.2119
h= 5   VARX  vs  Random Walk  DM:  -1.35  p:  0.1774  BH q:  0.3193
h= 5   EWA Ensemble  vs  VARX  DM:  -0.7  p:  0.4862  BH q:  0.6252
h= 20   EWA Ensemble  vs  Random Walk  DM:  -1.74  p:  0.0823  BH q:  0.2119
h= 20   VARX  vs  Random Walk  DM:  -1.7  p:  0.0889  BH q:  0.2119
h= 20   EWA Ensemble  vs  VARX  DM:  0.42  p:  0.6746  BH q:  0.7589
h= 1  reality check p (model search priced in):  0.1475
h= 5  reality check p (model search priced in):  0.0305
h= 20  reality check p (model search priced in):  0.064
Horizon:  1  day, weighted DA:  0.5485  capture:  0.097  profitable chunks:  46.3 % of  41
Horizon:  5  day, weighted DA:  0.5945  capture:  0.1891  profitable chunks:  65.3 % of  49
Horizon:  20  day, weighted DA:  0.6408  capture:  0.2816  profitable chunks:  75.5 % of  49
==============================
2007-2009 crisis :  500  windows
==============================
Model:  Random Walk
Horizon: 1 day, rmse: 10.111559541615545, MAE: 6.713818181818182, Directional Accuracy: nan %, base rate: 53.70442963543708 % 
Horizon: 5 day, rmse: 21.694716575072544, MAE: 14.766909090909092, Directional Accuracy: nan %, base rate: 57.83041925173905 % 
Horizon: 20 day, rmse: 41.457899784545944, MAE: 30.43927272727273, Directional Accuracy: nan %, base rate: 60.003691399040235 % 
Model:  VARX
Horizon: 1 day, rmse: 10.029257869453785, MAE: 6.759096782428429, Directional Accuracy: 51.47001176009408 %, base rate: 53.70442963543708 % 
Horizon: 5 day, rmse: 21.78201112219068, MAE: 15.0770769012398, Directional Accuracy: 50.23500658018425 %, base rate: 57.83041925173905 % 
Horizon: 20 day, rmse: 40.6286530541594, MAE: 30.789913323833602, Directional Accuracy: 52.805463270579544 %, base rate: 60.003691399040235 % 
Model:  EWA Ensemble
Horizon: 1 day, rmse: 10.042666444219801, MAE: 6.720472908006448, Directional Accuracy: 51.47001176009408 %, base rate: 53.70442963543708 % 
Horizon: 5 day, rmse: 21.660361788757196, MAE: 14.867962385610786, Directional Accuracy: 50.23500658018425 %, base rate: 57.83041925173905 % 
Horizon: 20 day, rmse: 40.75060349074645, MAE: 30.43972551939985, Directional Accuracy: 52.805463270579544 %, base rate: 60.003691399040235 % 
h= 1  EWA vs Random Walk DM:  -2.35  p:  0.019
h= 5  EWA vs Random Walk DM:  -0.22  p:  0.8263
h= 20  EWA vs Random Walk DM:  -1.1  p:  0.2726
==============================
2020 covid :  230  windows
==============================
Model:  Random Walk
Horizon: 1 day, rmse: 4.624090552609796, MAE: 2.569565217391305, Directional Accuracy: nan %, base rate: 54.103502352326196 % 
Horizon: 5 day, rmse: 11.673506370758629, MAE: 5.862055335968379, Directional Accuracy: nan %, base rate: 59.3984962406015 % 
Horizon: 20 day, rmse: 29.320122189673615, MAE: 13.645849802371544, Directional Accuracy: nan %, base rate: 56.8920105355575 % 
Model:  VARX
Horizon: 1 day, rmse: 4.616346863680828, MAE: 2.6781162704803605, Directional Accuracy: 50.601150026136956 %, base rate: 54.103502352326196 % 
Horizon: 5 day, rmse: 11.879600802050817, MAE: 6.763831509253801, Directional Accuracy: 53.712406015037594 %, base rate: 59.3984962406015 % 
Horizon: 20 day, rmse: 30.33133133960215, MAE: 18.44215736018516, Directional Accuracy: 51.40474100087796 %, base rate: 56.8920105355575 % 
Model:  EWA Ensemble
Horizon: 1 day, rmse: 4.609968900566026, MAE: 2.6180766854611552, Directional Accuracy: 50.601150026136956 %, base rate: 54.103502352326196 % 
Horizon: 5 day, rmse: 11.6778155142392, MAE: 5.984268449013264, Directional Accuracy: 53.712406015037594 %, base rate: 59.3984962406015 % 
Horizon: 20 day, rmse: 29.294374559085785, MAE: 14.473445157166509, Directional Accuracy: 51.40474100087796 %, base rate: 56.8920105355575 % 
h= 1  EWA vs Random Walk DM:  -1.05  p:  0.2949
h= 5  EWA vs Random Walk DM:  0.16  p:  0.8762
h= 20  EWA vs Random Walk DM:  -0.12  p:  0.9077
==============================
2022 hiking :  249  windows
==============================
Model:  Random Walk
Horizon: 1 day, rmse: 7.3384511996396435, MAE: 5.428623585250092, Directional Accuracy: nan %, base rate: 57.69080234833659 % 
Horizon: 5 day, rmse: 16.83492648955515, MAE: 13.148959474260677, Directional Accuracy: nan %, base rate: 69.47169811320755 % 
Horizon: 20 day, rmse: 40.483933683066304, MAE: 33.445418035779475, Directional Accuracy: nan %, base rate: 77.52023546725533 % 
Model:  VARX
Horizon: 1 day, rmse: 7.296469081301339, MAE: 5.430444872062687, Directional Accuracy: 52.64187866927593 %, base rate: 57.69080234833659 % 
Horizon: 5 day, rmse: 16.352625339352365, MAE: 12.670271600457644, Directional Accuracy: 63.698113207547166 %, base rate: 69.47169811320755 % 
Horizon: 20 day, rmse: 37.29846312625571, MAE: 30.53623868647804, Directional Accuracy: 64.532744665195 %, base rate: 77.52023546725533 % 
Model:  EWA Ensemble
Horizon: 1 day, rmse: 7.305662766768439, MAE: 5.421713331661201, Directional Accuracy: 52.64187866927593 %, base rate: 57.69080234833659 % 
Horizon: 5 day, rmse: 16.423812001080996, MAE: 12.733030370792086, Directional Accuracy: 63.698113207547166 %, base rate: 69.47169811320755 % 
Horizon: 20 day, rmse: 37.60654022062111, MAE: 30.86901403316521, Directional Accuracy: 64.532744665195 %, base rate: 77.52023546725533 % 
h= 1  EWA vs Random Walk DM:  -2.34  p:  0.0198
h= 5  EWA vs Random Walk DM:  -2.89  p:  0.0042
h= 20  EWA vs Random Walk DM:  -2.34  p:  0.0201
==============================
FORWARD TEST ERRORS ( 19  windows, frozen pipeline )
==============================
Model:  Random Walk
Horizon: 1 day, rmse: 4.003586908507335, MAE: 2.8038277511961742, Directional Accuracy: nan %, base rate: 55.02958579881657 % 
Horizon: 5 day, rmse: 7.250804278107763, MAE: 5.751196172248804, Directional Accuracy: nan %, base rate: 53.608247422680414 % 
Horizon: 20 day, rmse: 11.224759031311311, MAE: 8.866028708133976, Directional Accuracy: nan %, base rate: 63.86138613861386 % 
Model:  VARX
Horizon: 1 day, rmse: 3.998512308497113, MAE: 2.847834182367347, Directional Accuracy: 48.5207100591716 %, base rate: 55.02958579881657 % 
Horizon: 5 day, rmse: 7.058428200085527, MAE: 5.59565873112373, Directional Accuracy: 63.4020618556701 %, base rate: 53.608247422680414 % 
Horizon: 20 day, rmse: 11.232374137395794, MAE: 9.164829806184674, Directional Accuracy: 58.91089108910891 %, base rate: 63.86138613861386 % 
Model:  EWA Ensemble
Horizon: 1 day, rmse: 3.997382618769315, MAE: 2.827257913895279, Directional Accuracy: 48.5207100591716 %, base rate: 55.02958579881657 % 
Horizon: 5 day, rmse: 7.131401910917916, MAE: 5.651114936251908, Directional Accuracy: 63.4020618556701 %, base rate: 53.608247422680414 % 
Horizon: 20 day, rmse: 10.977498677128843, MAE: 8.903529153942822, Directional Accuracy: 58.91089108910891 %, base rate: 63.86138613861386 % 
h= 1  DM:  -0.21  p:  0.8361
h= 5  DM:  -0.49  p:  0.6315
h= 20  DM:  -0.0  p:  1.0
Horizon:  1  day, no trades cleared the threshold
Horizon:  5  day, weighted DA:  0.5268  capture:  0.0537  trades:  44  net pnl bp:  16.0
Horizon:  20  day, weighted DA:  0.6921  capture:  0.3843  trades:  177  net pnl bp:  564.5
```

## Figures

Predictions
- figures/holdout_predicted_vs_actual_2Y.png and _10Y.png (20 day ahead EWA vs actual, full holdout)
- figures/forward_test_predicted_vs_actual_2Y.png and _10Y.png (forward test, predictions at target dates)
- figures/forward_test_predicted_vs_actual_moves.png (predicted vs actual move scatter, diagonal = perfect)
- figures/holdout_rmse_by_model.png (model comparison bar chart)

Regimes
- figures/hmm_decoded_regimes.png (HMM regime path and smoothed probabilities)
- figures/hsmm_decoded_regimes.png (HSMM regime path and smoothed probabilities)
- figures/ewa_weight_path.png (EWA weight on the VARX per horizon, 0 = pure random walk)

Spillover network
- figures/spillover_network_heatmap.png (latest 11x11 network)
- figures/spillover_total_connectedness.png (connectedness over time)
- figures/spillover_connectedness_stress_periods.png (connectedness with 2007-09, 2020, 2022 shaded)

Factors
- figures/dns_factors_timeseries.png (level, slope, curvature over time, holdout start marked)


Caveats to carry into the paper: the holdout was examined during development (all p values conditional on that), the forward test is small but untouched, 2007-2009 and 2020 stress windows are quasi out-of-sample because design choices were made on overlapping data.
