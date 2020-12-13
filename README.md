# Simulation of some SPDEs & related processes


- We use a Galerkin approximation to simulate the stochastic heat equation with
  additive noise:
  <p align="center"><img src="/tex/3f3b3e0b508fd688fbee13006e8daad7.svg?invert_in_darkmode&sanitize=true" align=middle width=96.99100785pt height=14.611878599999999pt/></p>
  for space-time white noise <img src="/tex/85e60dfc14844168fd12baa5bfd2517d.svg?invert_in_darkmode&sanitize=true" align=middle width=7.94809454999999pt height=22.831056599999986pt/>:
  
  ![alt text](2d_she.gif)

- We simulate a sample path of a Brownian sheet:
  ![alt text](brownian_sheet.gif)

- We simulate to solutions <img src="/tex/b0084c5cb64e36b653b726facedd2f08.svg?invert_in_darkmode&sanitize=true" align=middle width=40.17511904999999pt height=22.831056599999986pt/> to the KPZ equation (via the Cole-Hopf
  transform and an implicit finite difference scheme):
  <p align="center"><img src="/tex/ed433bc08dfe54743ca512815b07c059.svg?invert_in_darkmode&sanitize=true" align=middle width=180.25673325pt height=18.312383099999998pt/></p>
  for space time white noise <img src="/tex/85e60dfc14844168fd12baa5bfd2517d.svg?invert_in_darkmode&sanitize=true" align=middle width=7.94809454999999pt height=22.831056599999986pt/> and two initial conditions <img src="/tex/14149e7a18f6b1c55af4d61165f677e3.svg?invert_in_darkmode&sanitize=true" align=middle width=83.00631734999999pt height=24.65753399999998pt/>
  and we can observe that the solutions synchronize (see
  https://arxiv.org/abs/1907.06278):

  ![alt text](kpz_synchronization.gif)

- Finally, we simulate the ballistic deposition growth process (linked to a famous
  conjecture regarding the KPZ equation: https://arxiv.org/pdf/1106.1596.pdf):

  ![alt text](ballistic.gif)


