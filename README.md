# OSE data science project template

This is a template for course projects. We use [GitHub Classroom](https://classroom.github.com) to administrate our student projects and so you need to sign up for a [GitHub Account](http://github.com).

## Project overview
In this project we studied the Bayesian Forecasting model in order to predict the upcoming U.S presidential election outcomes at the state level which is a replication of the following paper:
>[Dynamic Bayesian forecasting of presidential elections in the states. Journal of the American Statistical Association, 108(501), 124-134.](https://www.tandfonline.com/doi/abs/10.1080/01621459.2012.737735)

This model mainly considers the large number of state-level opinion surveys as well as information from the historical forecasting models. The end result is a set of data that has an increasing accuracy as the election day approaches. To borrow strength both across state and through the use of random-walk priors, here we employed hierarchical specification. This is done for the purpose of overcoming the limitation that not every state is polled regularly. Moreover, in order to handle the daily tracking of voter preferences towards the presidential candidates, this model sifts through day-to-day variations in the polls caused by sampling error and national campaign effects. We also used simulation techniques to estimate the candidates’ winning chance and thus the winning chance of Electoral College. When this model is applied to 2008 presidential campaign, the victory of Barrack Obama was never, in fact, in doubt.

<a href="https://nbviewer.jupyter.org/github/OpenSourceEconomics/ose-template-course-project/blob/master/example_project.ipynb"
   target="_parent">
   <img align="center"
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png"
      width="109" height="20">
</a>
<a href="https://mybinder.org/v2/gh/OpenSourceEconomics/ose-template-course-project/master?filepath=example_project.ipynb"
    target="_parent">
    <img align="center"
       src="https://mybinder.org/badge_logo.svg"
       width="109" height="20">
</a>
