import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil

_global_xf = None


def get_xf_instance():
    global _global_xf
    if _global_xf is None:
        _global_xf = XFoil()
        _global_xf.print = False
    return _global_xf


def get_cl(coord, xf=None, angle=5):
    if xf is None:
        xf = get_xf_instance()
    xf.Re = 3e6
    xf.max_iter = 100
    datax, datay = coord.reshape(2, -1)
    xf.airfoil = Airfoil(x=datax, y=datay)
    c = xf.a(angle)
    return np.round(c[0], 10)


def get_cd(coord, xf=None, angle=5):
    if xf is None:
        xf = get_xf_instance()
    xf.Re = 3e6
    xf.max_iter = 100
    datax, datay = coord.reshape(2, -1)
    xf.airfoil = Airfoil(x=datax, y=datay)
    c = xf.a(angle)
    return np.round(c[1], 10)