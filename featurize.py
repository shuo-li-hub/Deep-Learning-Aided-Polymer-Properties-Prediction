import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool

from rdkit import Chem 


def symbol_vector(sym):
  elements=['Br','C','Cl','F','Li','N','O','P','S','Si','*']
  vector=np.zeros(len(elements))
  if sym in elements:
    vector[elements.index(sym)]=1
  return vector
def h_v_vector(num):
  nums=[0,1,2,3]
  vector=np.zeros(len(nums))
  if num in nums:
    vector[nums.index(num)]=1
  return vector
def degree_vector(num):
  nums=[1,2,3,4]
  vector=np.zeros(len(nums))
  if num in nums:
    vector[nums.index(num)]=1
  return vector
def aroma_vector(is_aromatic):
  if is_aromatic:
    return np.array([1,0])
  else:
    return np.array([0,1])
#featurize function
def smiles_to_graph(smiles):
    mol=Chem.MolFromSmiles(smiles)
    feature_vector=[]
    edge_index = []
    for atom in mol.GetAtoms():
        feature=np.concatenate(
            [symbol_vector(atom.GetSymbol()),
             h_v_vector(atom.GetTotalNumHs()),
             h_v_vector(atom.GetTotalValence()),
             degree_vector(atom.GetDegree()),
             aroma_vector(atom.GetIsAromatic())  
            ]
        )
        feature_vector.append(feature)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    x = torch.tensor(feature_vector, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return x, edge_index
def load_data(df, property, is_test=False):
    data_list = []
    for _, row in df.iterrows():
        smiles = row['SMILES']
        try:
            x,edge_index=smiles_to_graph(smiles)
        except:
            continue
        if x is None or edge_index is None:
            continue
        if is_test:
            data = Data(x=x, edge_index=edge_index, id=row['id'])
        else:
            if property not in row or np.isnan(row[property]):
                continue
            y = torch.tensor([np.float32(row[property])], dtype=torch.float32)  
            data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list
