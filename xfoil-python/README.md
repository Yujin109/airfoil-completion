> **IMPORTANT NOTE: major changes and upgrades are currently taking place. This readme file might not be up to date. Please contact me if you need any details on current usage. (July 2020)**

General
-------
This is a stripped down version of XFOIL, presented in the form of a Python module. What's unique about this package
 w.r.t. many others out there allowing an interface to XFOIL, is the fact that the Python code talks directly to a 
 compiled Fortran library. This approach avoids having to read/write in-/output files to the disk and communicating with
 the XFOIl executable. Eliminating the need for constant disk I/O operations can significantly speed up parallel 
 frameworks in particular, giving this approach a clear advantage.

Building and Installing the Python Module
-----------------------------------------
See https://github.com/KikeM/xfoil-python/issues/7#issuecomment-659533863 for installation instructions. 

Using the Module
----------------
All XFoil operations are performed using the `XFoil` class. So the first step when using this module is to create an
instance of this class:

```pycon
>>>  from xfoil import XFoil
>>>  xf = XFoil()
```

If this does not produce any errors, the installation should be functioning properly. 


The symmetric NACA 0012 airfoil is included as a test case. It can be loaded into the XFoil library like this:

```pycon
>>>  from xfoil.test import naca0012
>>>  xf.airfoil = naca0012

 Number of input coordinate points: 160
 Counterclockwise ordering
 Max thickness =     0.120008  at x =   0.308
 Max camber    =     0.000000  at x =   0.033

  LE  x,y  =  -0.00000   0.00000  |   Chord =   1.00000
  TE  x,y  =   1.00000   0.00000  |

 Current airfoil nodes set from buffer airfoil nodes ( 160 )
```

Once the airfoil has been loaded successfully it can be analyzed. Let's analyze it for a range of angles of attack, at a
Reynolds number of one million. Let's limit the maximum number of iterations to 40 (the default is 20) as well. 
For the range of angles of attack, we will go from -20 degrees to 20 degrees with steps of 0.5 degrees:

```pycon
>>>  xf.Re = 1e6
>>>  xf.max_iter = 40
>>>  a, cl, cd, cm = xf.aseq(-20, 20, 0.5)
```

The XFOIL library should produce a lot of output, which should be familiar to those who have used the original XFOIL 
application before. The final result are lists of angles of attack, `a`, and the corresponding lift coefficients, `cl`, 
drag coefficients, `cd`, and moment coefficients, `cm`. We can now, for example, plot the lift curve for this airfoil:

```pycon
>>>  import matplotlib.pyplot as plt
>>>  plt.plot(a, cl)
>>>  plt.show()
```

This should produce the following figure:

![NACA 0012 Lift Curve](https://github.com/daniel-de-vries/xfoil-python/raw/master/naca0012-lift-curve.png)

Just like in the original XFOIL application, an airfoil can also analyzed for a single angle of attack, single lift
coefficient, or a range of lift coefficients. The commands for these operations are

```pycon
>>>  cl, cd, cm = xf.a(10)
>>>  a, cd, cm = xf.cl(1)
>>>  a, cl, cd, cm = xf.cseq(-0.5, 0.5, 0.05)
```

to analyze for an angle of attack of 10 degrees, a lift coefficient of 1.0, and for a range of lift coefficients from
-0.5 to 0.5 with steps of 0.05.

For other features and specifics, see the documentation in the Python source files.
 
