# %%
# import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# %%
def create_properties_dict(item, caminho_tabela="Composite material properties.csv"):
    '''
    Creates a dict containing the properties of the item with the same index in the csv.
    - All values are metric units with no multipliers (Pa instead of MPa)
    '''
    # Abrindo tabelas para extrair dados
    CompositeTable = pd.DataFrame()
    CompositeTable = pd.read_csv(caminho_tabela,sep=';')

    # # check if item is a string
    # if type(item) == str:
    #     # find the number of the row of the item in the column 'Material'
    #     item = CompositeTable[CompositeTable['Material'] == item].index[0]

    v23 = CompositeTable.loc[item,"nu_23"] # [Adimensional]
    E2 = CompositeTable.loc[item,"E_2 [Gpa]"] # [GPa]
    G23 = E2 / (2*(1+v23))

    properties_dict = {
        'E1':CompositeTable.iloc[item]["E_1 [Gpa]"],  # [GPa]
        'E2': E2, # CompositeTable.iloc[item]["E_2 [Gpa]"], # [GPa]
        'v12':CompositeTable.iloc[item]["nu_12"], # [Adimensional]
        'v21':v23,
        'v23': v23, # CompositeTable.iloc[item]["nu_23"], # [Adimensional]
        'G12':CompositeTable.iloc[item]["G_12 [Gpa]"],  # [GPa]
        'G13':CompositeTable.iloc[item]["G_12 [Gpa]"],  # [GPa]
        'G23': G23, # [GPa]
        'F1t':None,
        'F1c':None,
        'F2t':None,
        'F2c':None,
        'F6':None,
        'Material name':CompositeTable.iloc[item]["Material"],
        'Material type':CompositeTable.iloc[item]["Material type"],
        'Density per area':None,
        'Cost per area':None
        }
    
    # properties_dict  = {'Mechanical':MECHANICAL,'Strenght':STRENGH,'Physical':PHYSICAL}

    return properties_dict


# %%
# Definindo as funções básicas para os cálculos de compósitos

def compute_A(theta, unit = 'graus'):
    '''
    PURPOSE
    - Compute the matrix a whitch Relates laminate xy coordinates to 12
    - It is eq.5.40 on the Barbero book
    
    INPUTS
    - theta         : angle between laminate and lamina

    OUTPUTS
    - T : Transformation Matrix a (eq.5.26)
    '''
    if unit == 'graus':
        theta = np.deg2rad(theta)

    m = np.cos(theta)
    n = np.sin(theta)

    A = np.array([
        [m,n],
        [-n,m]
    ])

    return A


def compute_T(theta, unit = 'graus'):
    '''
    PURPOSE
    - Compute the matrix T whitch Relates laminate deformation to the lamina
      deformation
    - It is eq.5.40 on the Barbero book
    - Deformation_x = T * Deformation_1
    - Stress_x = T^-1 * Stress_1
    
    INPUTS
    - theta : angle between laminate and lamina in degrees
        OUTPUTS
    - T     : Transformation Matrix T (eq.5.40)'''

    if unit == 'graus':
        theta = np.deg2rad(theta)

    m = np.cos(theta)
    n = np.sin(theta)

    T = np.array([
        [m**2  ,  n**2 , 2*m*n],
        [n**2  ,  m**2 , -2*m*n],
        [-m*n , m*n  , m**2 - n**2]
    ])

    return T


def compute_Q(properties_dict):
    '''PURPOSE
    - Compute the matrix Q whitch is the inverse of the compliance matrix
    - It is eq.5.21 on the Barbero book
    - The matrix S is Q^-1

    INPUTS
    - E1         : Youngs modulus on 1 direction
    - E2         : Youngs modulus on 2 direction
    - poison12   : Poison Ration on the 12 direction 
    - G12        : Shear modulus 1 direction
    OUTPUTS
    - Q : Matrix Q (eq.5.21)'''
            
    delta = (
        1 - (properties_dict['v12']**2) * properties_dict['E2']/
        properties_dict['E1']
    )
    Q = np.zeros((3,3))
    Q[0,0] = properties_dict['E1'] / delta
    Q[1,1] = properties_dict['E2'] / delta
    Q[2,2] = properties_dict['G12']
    Q[0,1] = properties_dict['v12'] * properties_dict['E2'] / delta
    Q[1,0] = Q[0,1]
    
    return Q


def compute_QI(properties_dict):
    '''      
    PURPOSE
    - Compute the matrix QI whitch is the inverse of the inatralaminar
      compliance matrix
    - It is eq.5.22 on the Barbero book
    - The matrix SI is QI^-1
    - Sigma 4 e 5 = QI * gama 4 e 5

    INPUTS
    - G13        : Intralaminar shear modulus on 13 plane
    - E2         : Youngs modulus on 2 direction
    - poison23   : Poison Ration on the 12 direction
    OUTPUTS
    - Q* : Matrix Q* (eq.5.21)
    '''

    QI = np.zeros((2,2))
    QI[0,0] = properties_dict['G23']
    QI[1,1] = properties_dict['G13']

    return QI


# %%
# Criando classes para armazenar as propriedades dos laminados e lâminas
reuter_matrix = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,2]
])

class Ply:
    def __init__(self,material,orientation):
        
        self.material_id = material
        self.orientaton = orientation
        self.units = {'Orientation unit':'graus'}
        
        self.properties = create_properties_dict(self.material_id)
        self.T_matrix = compute_T(self.orientaton,self.units['Orientation unit'])
        self.A_matrix = compute_A(self.orientaton,self.units['Orientation unit'])

        self.Q = compute_Q(self.properties)
        self.QI = compute_QI(self.properties)
        # QT(:,:,i)  = inv(T(:,:,i))*Q(:,:,i)*inv(T(:,:,i))'
        # self.QT = np.dot(
        #     np.dot(np.linalg.inv(self.T_matrix),self.Q),
        #     np.linalg.inv(self.T_matrix).T
        # )
        self.QT = np.dot(
            np.dot(compute_T(-orientation),self.Q ),
            compute_T(-orientation).T
            )
        self.QIT = np.dot(
            np.dot(self.A_matrix,self.QI),
            self.A_matrix.T
        )

        def get_transformed_properties(self):
            return None


class Lamina(Ply):
    def __init__(self,material,orientation,thickness,lamina_coordinate):
        self.Zk = lamina_coordinate # Coordenada do centro da lâmina em realção ao plano central do lamminado
        self.thickness = thickness # Espessura da lâmina
        if self.Zk >= 0:
            self.Zn = self.Zk + (self.thickness)/2 # Possição do topo da lamina para calcular deformação máxima
        else:
            self.Zn = self.Zk - (self.thickness)/2

        Ply.__init__(self,material,orientation)

        # Partial Extensional stiffness matrix A0:
        self.A0 = self.QT * self.thickness
        # Partial Bending extension coupling stiffness matrix B:
        self.B0 = -1*self.QT * self.thickness * self.Zk
        # Partial Bending stiffness matrix D:
        self.D0 = self.QT*(self.thickness*self.Zk**2 + (self.thickness**3)/12)
        # Partial Transverse shear stiffness:
        self.H0 = -(5/4) * self.QIT * (self.thickness - (4/self.thickness**2)
                                       * (self.thickness*self.Zk**2 + (self.thickness**3)/12))

    def get_A0(self):
        return self.A0
    def get_B0(self):
        return self.B0
    def get_D0(self):
        return self.D0
    def get_H0(self):
        return self.H0
    
    def compute_deformation_and_stress(self,laminate_deformation): 
        # Cuidado deve ser tomado em laminados de lâmina única 
        # pois estou considerando zk como sendo o meio da lamina
        self.deformation = laminate_deformation[:3] - self.Zn*laminate_deformation[3:] # epsilon_x, epsilon_y, gamma_xy
        self.stress = np.dot(np.linalg.inv(self.Q), self.deformation) # sigma_x, sigma_y, gamma_xy
        pass

    def print_lamina_deformation(self):
        print("Deformation:")
        print(self.deformation)
        print("Estresse:")
        print(self.stress)
        

class Laminate:
    def __init__(self,laminate_list):
        self.laminate_stack = []
        self.t = 0
        self.create_laminate_list_with_coordinates(laminate_list)

        for lamina in self.laminate_list:
            self.laminate_stack.append(Lamina(lamina[0],lamina[1],lamina[2],lamina[3]))

        self.compute_ABD()

    def create_laminate_list_with_coordinates(self, laminate_list):
        # Add the coordinate to the laminate matrix
        L = laminate_list
        tt = sum(i[2] for i in L)
        for i in range(len(L)):
            if i == 0:
                L[i].append(-((tt/2) - L[i][2]/2))
            else:
                L[i].append(L[i-1][3] + (L[i-1][2] + L[i][2]) / 2)
        self.laminate_list=L

    def compute_ABD(self):
        self.A = np.zeros((3,3))
        self.B = np.zeros((3,3))
        self.D = np.zeros((3,3))
        self.H = np.zeros((2,2))

        for lamina in self.laminate_stack:
            self.A += lamina.get_A0()
            self.B += lamina.get_B0()
            self.D += lamina.get_D0()
            self.H += lamina.get_H0()

        self.ABD = np.zeros((6,6))

        self.ABD[0:3,0:3] = self.A
        self.ABD[3:6,3:6] = self.D
        self.ABD[3:6,0:3] = self.B
        self.ABD[3:6,0:3] = self.B

        self.abd = np.linalg.inv(self.ABD)

    def aply_load(self, load_vector):
        self.load_vector = load_vector
        self.compute_deformation()

    
    def aply_deformation(self, deformation_vector):
        self.deformation_vector = deformation_vector
        self.compute_loading()
        
        for i in range(len(self.laminate_stack)):
            self.laminate_stack[i].compute_deformation_and_stress(self.deformation_vector)

    def compute_deformation(self):
        # h = np.linalg.inv(self.H)
        # self.lamiate_transversal_deformations = 
        abd = np.linalg.inv(self.ABD)
        self.deformation_vector = np.dot(abd,self.load_vector)

        for i in range(len(self.laminate_stack)):
            self.laminate_stack[i].compute_deformation_and_stress(self.deformation_vector)
    
    def compute_loading(self):
        self.load_vector = np.dot(self.ABD,self.load_vector)

    def print_load_vector(self):
        print('Load vector:')
        print(' N_x:',self.load_vector[0])
        print(' N_y:',self.load_vector[1])
        print('N_xy:',self.load_vector[2])
        print(' M_x:',self.load_vector[3])
        print(' M_y:',self.load_vector[4])
        print('M_xy:',self.load_vector[5])

    def print_deformation_vector(self):
        print('Deformation vector:')
        print('      u_x:',self.deformation_vector[0])
        print('      u_y:',self.deformation_vector[1])
        print(' gamma_xy:',self.deformation_vector[2])
        print('      k_x:',self.deformation_vector[3])
        print('      k_y:',self.deformation_vector[4])
        print('     k_xy:',self.deformation_vector[5])



        

# %% 
if __name__ == '__main__':
    pass