import pandas as pd
import numpy as np

from dotmap import DotMap

class CpAnalysis():
    """Read and process the boundary layer file created by Xfoil. 

    Note
    ----
    It assumes the file already exists when it is instantiated.
    """

    def __init__(self, filename):

        self.filename = filename

        # Read the file
        self._read_file()

        self._find_airfoil()
        self._trim_variables()

    def _read_file(self):
        """Read the boundary layer file and 
        extract the data in the right format.
        """
        # Read the boundary data file
        df_cp = pd.read_csv(self.filename, delim_whitespace=True)
        
        # Get all the variables
        self.variables = df_cp.columns

        # Remove the columns that do not contain data
        df_cp.dropna(axis=1, how='all', inplace=True)

        # Rename the columns appropriately
        df_cp.columns = self.variables[1:len(df_cp.columns)+1]

        # Extract all the variables
        self.df_cp = df_cp

        # Memory clean-up
        del df_cp

    def _find_airfoil(self):
        """Find the array index at the end of the profile.  
        
        Notes
        -----
        Creates attribute last_index.
        """
        coord_LE = np.min(self.df_cp['x'].values)

        self.idx_LE    = np.where(self.df_cp['x'].values == coord_LE)[0][0]
        _, self.idx_TE = np.where(self.df_cp['x'].values == 1.0)[0]

    def _trim_variables(self):

        self.cp = DotMap()
        self.x  = DotMap()

        self.cp.upper = self.df_cp['Cp'].values[0          :self.idx_LE+1]
        self.cp.lower = self.df_cp['Cp'].values[self.idx_LE:             ]

        self.x.upper = self.df_cp['x'].values[0          :self.idx_LE+1]
        self.x.lower = self.df_cp['x'].values[self.idx_LE:             ]
        