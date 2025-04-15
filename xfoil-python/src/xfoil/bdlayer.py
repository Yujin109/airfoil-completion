import pandas as pd
import numpy as np

from dotmap import DotMap

class BoundaryLayer():
    """Read and process the boundary layer file created by Xfoil. 

    The boundary layer properties are created as a DotMap with
    the following levels:
        - upper
        - lower
        - wake

    Note
    ----
    It assumes the file already exists when it is instantiated.
    """

    def __init__(self, filename):

        self.filename = filename

        # Read the file
        self._read_file()

        # Postprocess
        self._find_airfoil()
        self._trim_coord()
        self._trim_boundary_layer()
    
    def _read_file(self):
        """Read the boundary layer file and 
        extract the data in the right format.
        """
        # Read the boundary data file
        df_bl = pd.read_csv(self.filename, delim_whitespace=True)
        
        # Get all the variables
        self.variables = df_bl.columns

        # Remove the columns that do not contain data
        df_bl.dropna(axis=1, how='all', inplace=True)

        # Rename the columns appropriately
        df_bl.columns = self.variables[1:len(df_bl.columns)+1]

        # Extract all the variables
        self.s_raw = df_bl['s'].values
        self.x_raw = df_bl['x'].values
        self.y_raw = df_bl['y'].values
        
        self.ue_vinfty_raw = df_bl['Ue/Vinf'].values
        self.d_star_raw    = df_bl['Dstar'].values
        self.theta_raw     = df_bl['Theta'].values

        self.cf_raw = df_bl['Cf'].values

        self.H_raw      = df_bl['H'].values
        self.H_star_raw = df_bl['H*'].values
        
        self.P_raw = df_bl['P'].values
        self.m_raw = df_bl['m'].values

        # Memory clean-up
        self.df_bl = df_bl

    def _find_airfoil(self):
        """Find the array index at the end of the profile.  
        
        Notes
        -----
        Creates attribute last_index.
        """
        coord_LE = np.min(self.x_raw)

        self.idx_LE    = np.where(self.x_raw == coord_LE)[0][0]
        _, self.idx_TE = np.where(self.x_raw == 1.0)[0]

    def _get_trimmed_values(self, property):
        """Trim the boundary layer property according to the three sections present:
            - Upper surface
            - Lower surface
            - Wake

        Parameters
        ----------
        property: numpy array

        Returns
        -------
        DotMap
            - Keys: 
                - "upper"
                - "lower"
                - "wake"
            - Values: numpy arrays

        Notes
        -----
        The length of upper and lower is not necessarily the same due to curvature.
        """
        data = DotMap()
        data.upper = property[0            :self.idx_LE+1]
        data.lower = property[self.idx_LE  :self.idx_TE+1]
        data.wake  = property[self.idx_TE+1:             ]

        return data        

    def _trim_boundary_layer(self):
        """
        Cut the boundary layer properties between airfoil upper, lower sections and wake. 
        """
        self.cf     = self._get_trimmed_values(self.cf_raw)
        self.H      = self._get_trimmed_values(self.H_raw)
        self.H_star = self._get_trimmed_values(self.H_star_raw)
        self.theta  = self._get_trimmed_values(self.theta_raw)
        self.d_star = self._get_trimmed_values(self.d_star_raw)

    def _trim_coord(self):
        """Cut the coordinates (x,y,s) between airfoil upper, lower sections and wake. 
        """
        self.x = self._get_trimmed_values(self.x_raw)
        self.y = self._get_trimmed_values(self.y_raw)
        self.s = self._get_trimmed_values(self.s_raw)

    def detect_bubbles(self):
        """
        Detect any recirculation bubble via the Cf coefficient.

        Returns
        -------
        dict:
            - Keys: 
                - "upper"
                    - exists
                    - indices
                    - length
                - "lower"
                    - ...
        """

        bubble = DotMap()

        # Upper bubble
        bubble.upper.exists, bubble.upper.indices = self._detect_bubble(self.cf['upper'])

        # Lower bubble
        bubble.lower.exists, bubble.lower.indices = self._detect_bubble(self.cf['lower'])

        # Compute lengths (if any)
        if bubble.upper.exists is True:
            bubble.upper.length = self.s.upper[bubble.upper.indices[-1]] - self.s.upper[bubble.upper.indices[0]]
        else:
            bubble.upper.length = 0.0

        if bubble.lower.exists is True:
            bubble.lower.length = self.s.lower[bubble.lower.indices[-1]] - self.s.lower[bubble.lower.indices[0]]
        else:
            bubble.lower.length = 0.0

        return bubble

    def _detect_bubble(self, cf):
        """
        Detect bubble in cf section.

        Parameters
        ----------
        cf: np.array

        Returns
        -------
        tuple: bool, list
            bool: True when there is a bubble
            list: [start index, last index] of the bubble
        """

        # Look for the the indices with negative cf
        idx = np.where(cf<0)[0]

        # Assume no bubble
        flag_bubble = False

        # If there are no indices, there is no bubble
        if len(idx) == 0:
            return flag_bubble, []
        else:
            flag_bubble = True
            return flag_bubble, [idx[0], idx[-1]]




