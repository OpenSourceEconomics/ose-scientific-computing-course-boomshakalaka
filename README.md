# Dynamic Bayesian Forecasting of Presidential Elections in the States (Linzer, 2013)

This is the replication project of OSE Scientific Computing for Economists in Winter Semester 2021/2022 by [Başak Çelebi](https://github.com/Basakclb), [Gözde Özden](https://github.com/ozdengo)

## Project overview
In this project we studied the Bayesian Forecasting model in order to predict the upcoming U.S presidential election outcomes at the state level which is a replication of the following paper:
>[Dynamic Bayesian forecasting of presidential elections in the states. Journal of the American Statistical Association, 108(501), 124-134.](https://www.tandfonline.com/doi/abs/10.1080/01621459.2012.737735)

This model mainly considers the large number of state-level opinion surveys as well as information from the historical forecasting models. The end result is a set of data that has an increasing accuracy as the election day approaches. To borrow strength both across state and through the use of random-walk priors, here we employed hierarchical specification. This is done for the purpose of overcoming the limitation that not every state is polled regularly. Moreover, in order to handle the daily tracking of voter preferences towards the presidential candidates, this model sifts through day-to-day variations in the polls caused by sampling error and national campaign effects. We also used simulation techniques to estimate the candidates’ winning chance and thus the winning chance of Electoral College. When this model is applied to 2016 presidential campaign, the victory of Hillary Clinton was never, in fact, in doubt. However, as with many forecasts that year, our model also failed to predict the 2016 US election. Even though this fail clearly demonstrates how easily data can be misinterpreted when viewed in isolation without reading in full context, recognizing this error and investigating the root causes is essential as we seek to better understand the world around us through data.

<a href="https://nbviewer.org/github/OpenSourceEconomics/ose-scientific-computing-course-boomshakalaka/blob/master/project%20notebook.ipynb"
   target="_parent">
   <img align="center"
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png"
      width="109" height="20">
</a>
<a href="https://mybinder.org/v2/gh/OpenSourceEconomics/ose-scientific-computing-course-boomshakalaka/blob/master/project%20notebook.ipynb/HEAD"
    target="_parent">
    <img align="center"
       src="https://mybinder.org/badge_logo.svg"
       width="109" height="20">
</a>
## Reproducibility

We set up Continuous Integration Workflow and share an environment.yml file to ensure full reproducibility of our project.

![Continuous Integration](https://github.com/OpenSourceEconomics/ose-template-course-project/workflows/Continuous%20Integration/badge.svg)

## Acknowledgement

We want to thank Morris, Kremp and Huffingtonpost. This replication project could only be conducted because they provide their data and their R and STAN codes for most of the results. 
