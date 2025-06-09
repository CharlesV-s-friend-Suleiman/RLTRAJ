import pickle
from modulefinder import Module

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self,input_channel, output_dim):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 2, 3, 1,1)
        self.conv2 = nn.Conv2d(2, 1, 3, 1,1)
        self.fc1 = nn.Linear(1*5*5, output_dim)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0),-1)
        x = self.fc1(x)

        return x


def mapdata_to_modelmatrix(mapdata: dict, n_row, n_col) -> dict[str: list[list:int]]:
    """
    Convert the mapdata to a matrix that can be used as input to the lower_model
    :param mapdata: dict, the mapdata
    :return: dict, the matrix that can be used as input to the lower_model
    """
    modelmatrix = {"TG": [[0 for _ in range(n_row)] for _ in range(n_col)],
                     "GG": [[0 for _ in range(n_row)] for _ in range(n_col)],
                     "GSD": [[0 for _ in range(n_row)] for _ in range(n_col)],
                     "TS": [[0 for _ in range(n_row)] for _ in range(n_col)]
                   }
    for k,v in mapdata.items():
        try:
            if v[4] & 1 == 1 or v[4] >>1 &1 == 1:
                modelmatrix['TG'][k[0]][k[1]] = 1
            if v[4] & 1 == 1 or v[4] >>6 &1 == 1 or v[4] >>1 &1 == 1:
                modelmatrix['TS'][k[0]][k[1]] = 1
            if v[4] >>3 & 1 == 1:
                modelmatrix['GG'][k[0]][k[1]] = 1
            if v[4] >>2 & 1 == 1 or v[4] >>5 & 1 == 1:
                modelmatrix['GSD'][k[0]][k[1]] = 1
        except:

            print('Inout Data Out of Range: ',k,v, 'Map Size: ', n_row, n_col)
    return modelmatrix


def get_neighbor(modelmatrix, x, y,size=3)->list:
    """
    Get the neighbor of the grid (x, y)
    :param modelmatrix: list[list], the modelmatrix
    :param x: int, the x coordinate of the grid
    :param y: int, the y coordinate of the grid
    :return: list, the neighbor of the grid (x, y) from (1,0) to (1,-1)
    """
    try:
        xmax = len(modelmatrix)
        ymax = len(modelmatrix[0])
    except:
        print('Input Data Out of Range When Getting Neighbor: ',type(modelmatrix), x, y)
        return [0 for _ in range(size*size)]

    neighbors = []
    if size==3:
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        for dx, dy in directions:
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < xmax and 0 <= ny < ymax:
                neighbors.append(modelmatrix[nx][ny])
            else:
                neighbors.append(0)  # or some other value indicating out of bounds

    if size==5:
        directions = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]

        for dx, dy in directions:
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < xmax and 0 <= ny < ymax:
                neighbors.append(modelmatrix[nx][ny])
            else:
                neighbors.append(0)

    return neighbors



def sense_map(mapdata, position_tensor, grid=5):
    """
    Sense the mapdata at the grid (x, y)
    :param mapdata: list[list], the mapdata
    :param position_tensor: torch.tensor, the position tensor, [[x1, y1], [x2, y2], ...[xn, yn]] x m
    :param grid: int, the size of the grid (default is 5)
    :return: tensor size (n, grid, grid)
    """

    def _inner(mapdata,x,y):
        xmax = len(mapdata)
        ymax = len(mapdata[0])
        sensed_data = [[0 for _ in range(grid)] for _ in range(grid)]
        half_grid = grid // 2

        for i in range(grid):
            for j in range(grid):
                nx, ny = x - half_grid + i, y - half_grid + j
                if 0 <= nx < xmax and 0 <= ny < ymax:
                    sensed_data[i][j] = mapdata[nx][ny]
                else:
                    sensed_data[i][j] = 0  # or some other value indicating out of bounds
        sensed_data = torch.tensor(sensed_data).unsqueeze(0).float().to(torch.device("cuda"))
        return sensed_data
    position_tensor = list(position_tensor)
    return torch.cat([_inner(mapdata, int(x), int(y)) for x, y in position_tensor], dim=0)


### test the function ###
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    with open('../data/GridModesAdjacentRealworld.pkl','rb') as f:
        mapdata = pickle.load(f)
    matrice = mapdata_to_modelmatrix(mapdata, 529, 564)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    modes = ['TG', 'GG', 'GSD', 'TS']
    back_imgs = ['../figur/js.jpg',
                 '../figur/js.jpg',
                 '../figur/js.jpg',
                 '../figur/js.jpg']

    for i, mode in enumerate(modes):
        ax = axs[i // 2, i % 2]
        matrix = matrice[mode]
        x, y= zip(*[(i, j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j] == 1])
        ax.imshow(mpimg.imread(back_imgs[i]), extent=[0, len(matrix), 0, len(matrix[0])], aspect='equal', alpha=0.5)

        ax.scatter(x, y, s=0.5)
        ax.set_title(mode)
        ax.set_xlim(0, len(matrix))
        ax.set_ylim(0, len(matrix[0]))
    plt.tight_layout()
    plt.show()


    for i, mode in enumerate(modes):
        fig, ax = plt.subplots(figsize=(20, 20))
        matrix = matrice[mode]
        x, y = zip(*[(i, j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j] == 1])
        ax.imshow(mpimg.imread(back_imgs[i]), extent=[0, len(matrix), 0, len(matrix[0])], aspect='equal', alpha=1)
        ax.scatter(x, y, s=1/100,alpha=1, c='red',marker='o')
        ax.set_xlim(0, len(matrix))
        ax.set_ylim(0, len(matrix[0]))
        ax.axis('off')  # Turn off the axis

        plt.tight_layout()
        plt.savefig(f'../figur/{mode}_plot_js.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
### test the function ###

