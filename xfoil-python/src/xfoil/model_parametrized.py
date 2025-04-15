"""
+------------------+------------------------------------------------+
| PARSEC parameter | Definition                                     |
+------------------+------------------------------------------------+
| p1               | Leading edge radius                            |
+------------------+------------------------------------------------+
| p2               | Upper crest position in horizontal coordinates |
+------------------+------------------------------------------------+
| p3               | Upper crest position in vertical coordinates   |
+------------------+------------------------------------------------+
| p4               | Upper crest curvature                          |
+------------------+------------------------------------------------+
| p5               | Lower crest position in horizontal coordinates |
+------------------+------------------------------------------------+
| p6               | Lower crest position in vertical coordinates   |
+------------------+------------------------------------------------+
| p7               | Lower crest curvature                          |
+------------------+------------------------------------------------+
| p8               | Trailing edge offset in vertical sense         |
+------------------+------------------------------------------------+
| p9               | Trailing edge thickness                        |
+------------------+------------------------------------------------+
| p10              | Trailing edge direction                        |
+------------------+------------------------------------------------+
| p11              | Trailing edge wedge angle                      |
+------------------+------------------------------------------------+
(Generated with http://www.tablesgenerator.com/)

Source
------    
Sobieczky H. (1999) Parametric Airfoils and Wings. 
In: Fujii K., Dulikravich G.S. (eds) Recent Development of Aerodynamic Design Methodologies. 
Notes on Numerical Fluid Mechanics (NNFM), vol 65. Vieweg+Teubner Verlag
"""


from copy import deepcopy

from scipy.optimize import minimize

import numpy as np
from numpy.linalg import solve

from xfoil import Airfoil
from dotmap import DotMap

class ParametrizedAirfoil():

    def __init__(self, *args, **kwargs):
        pass

    def fit(airfoil:Airfoil):
        
        # Code to find the best fitting coefficients.
        
        
        return design_vector

class Parsec(ParametrizedAirfoil):
    """
    Parsec airfoil parametrization.

    Parameters
    ----------
    design_vector: dict
    """

    def __init__(self, design_vector: dict = None, N = 160):
        super().__init__()

        # Save design vector
        if design_vector is None:
            self.dv = self._basic_design_vector()
        else:
            self.dv = design_vector

        self.N = N

        # Create specific PARSEC coefficient vectors
        self._q1 = [n + 1/2 for n in range(0,6)]
        self._q2 = [n**2 - 1/4 for n in range(0,6)]
        
        # Display attributes
        self._rhs_upper    = None
        self._rhs_lower    = None
        self._matrix_lower = None
        self._matrix_upper = None
        self._coeff_lower  = None
        self._coeff_upper  = None

        self.x       = None
        self.y_upper = None
        self.y_lower = None

    def _basic_design_vector(self):
        """
        Load a basic design vector.

        Parameters
        ----------
        None

        Returns
        -------
        dict
        """
        coeff = [
                0.05,    # p1  | Leading edge radius

                0.3,      # p2  | Upper crest position in horizontal coordinates
                0.1,      # p3  | Upper crest position in vertical coordinates
                1.0,      # p4  | Upper crest curvature
                   
                0.3,      # p5  | Lower crest position in horizontal coordinates
                -0.1,     # p6  | Lower crest position in vertical coordinates
                0.1,      # p7  | Lower crest curvature

                0.1,      # p8  | Trailing edge offset in vertical sense
                0.1,      # p9  | Trailing edge thickness
                1.0,      # p10 | Trailing edge direction
                20.0      # p11 | Trailing edge wedge angle
                ]

        return {'p{}'.format(q):value for q, value in zip(range(1,12), coeff)}

    def create_airfoil(self, p:dict = None, x: np.array = None):
        """
        Create airfoil coordinates compatible with xfoil.

        Parameters
        ----------
        p: dict
            Default: None
            New design vector

        Returns
        -------
        Airfoil
        """
        if p is not None:
            self.dv = deepcopy(p)
        else:
            pass

        if x is not None:
            self.N = len(x)

        # Prepare linear system
        self._create_matrices()
        self._create_rhs()

        # Solve linear system
        self._compute_coefficients()

        # Compute airfoil
        self._compute_coordinates(x)

        # Prepare output
        x = np.concatenate([np.flip(self.x), self.x[1:]])
        y = np.concatenate([np.flip(self.y_upper), self.y_lower[1:]])

        return Airfoil(x = x, y = y)

    def _create_rhs(self):
        """
        Assemble the RHS vectors for the upper and lower surface.
        """
        self._rhs_upper = self._fill_in_rhs(sign = '+')
        self._rhs_lower = self._fill_in_rhs(sign = '-')

    def _compute_coefficients(self):
        """
        Solve linear system to obtain PARSEC coefficients. 
        """

        self._coeff_upper = solve(self._matrix_upper, self._rhs_upper)
        self._coeff_lower = solve(self._matrix_lower, self._rhs_lower)

    def _compute_coordinates(self, x: np.array = None):
        """
        Compute the airfoil coordinates for the current 
        design vector.

        Parameters
        ----------
        x: np.array

        Returns
        -------
        """

        if x is None:
            x = self._cosine_distribution()

        # Make sure we are mapping the [0,1] domain
        x = np.sort(x)

        x_q = np.array([x**q for q in self._q1])
        
        self.y_upper = np.dot(np.transpose(self._coeff_upper), x_q)[0]
        self.y_lower = np.dot(np.transpose(self._coeff_lower), x_q)[0]

        self.x = x

    def _cosine_distribution(self, N:int = None):
        """
        Compute a cosine distribution in [0,1]. 

        Returns
        -------
        x: np.array
        """

        from math import cos, pi

        if N is None:
            N = self.N

        x = np.linspace(start = 0, stop = 1, num = N)
        x = list(map(cos, x * pi))

        # Sort from negative to positive
        x = np.sort(np.array(x))

        # Center and scale
        x += 1.0
        x /= 2.0

        return x

    def _fill_in_rhs(self, sign = '+'):
        """
        Fill-in vector according to airfoil surface.

        Parameters
        ---------
        sign: str
            '+': Upper surface
            '-': Lower surface

        Returns
        -------
        vec: np.array
        """

        from math import tan, sqrt, pi

        deg_to_rad = pi / 180.0
        vec = np.ndarray(shape=(6,1))

        if sign == '+':
            vec[0] = self.dv['p8'] + self.dv['p9'] / 2.0
            vec[1] = self.dv['p3']
            vec[2] = tan(deg_to_rad * (self.dv['p10'] - self.dv['p11'] / 2.0))
            vec[3] = 0.0
            vec[4] = self.dv['p4']
            vec[5] = sqrt(2.0 * self.dv['p1'])
        elif sign == '-':
            vec[0] = self.dv['p8'] - self.dv['p9'] / 2.0
            vec[1] = self.dv['p6']
            vec[2] = tan(deg_to_rad * (self.dv['p10'] + self.dv['p11'] / 2.0))
            vec[3] = 0.0
            vec[4] = self.dv['p7']
            vec[5] = -sqrt(2.0 * self.dv['p1'])
        else:
            raise ValueError('Which section do you want to create?')

        return vec

    def _create_matrices(self):

        self._matrix_upper = self._fill_in_matrix(p = self.dv['p2'])
        self._matrix_lower = self._fill_in_matrix(p = self.dv['p5'])

    def _fill_in_matrix(self, p:float):
        """
        Fill-in matrix according to parameter p2/p5.

        Parameters
        ----------
        p: float
            Crest position in horizontal coordinates
                - Upper: p2
                - Lower: p5

        Returns
        -------
        matrix: np.ndarray(shape=(6,6))

        Notes
        -----
        Row definitions:
            0: TE height
            1: Max. location
            2: TE direction
            3: Max. condition
            4: Curvature
            5: LE radius

        """

        matrix = np.ndarray(shape=(6,6))

        matrix[0]   = np.ones(6,                                   dtype=float) 
        matrix[1]   = np.array([p**q for q in self._q1],           dtype=float)
        matrix[2]   = np.array([q for q in self._q1],              dtype=float)
        matrix[3]   = np.array([q * p**(q-1.0) for q in self._q1], dtype=float)
        matrix[4]   = np.array([q2 * p**(q1-2.0) for q1, q2 in zip(self._q1, self._q2)], dtype=float)
        matrix[5]   = np.zeros(6,                                  dtype = float)
        matrix[5,0] = 1.0

        return matrix
