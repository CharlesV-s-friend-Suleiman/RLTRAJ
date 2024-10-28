import pickle
import PIL
from sympy.abc import alpha


def mapdata_to_modelmatrix(mapdata: dict, n_row, n_col) -> dict[str: list[list:int]]:
    """
    Convert the mapdata to a matrix that can be used as input to the model
    :param mapdata: dict, the mapdata
    :return: dict, the matrix that can be used as input to the model
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
            print('Inout Data Out of Range: ',k,v)
    return modelmatrix


def get_neighbor(modelmatrix, x, y):
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
        print('Input Data Out of Range: ',type(modelmatrix), x, y)
        return [0,0,0,0,0,0,0,0]

    neighbors = []
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    for dx, dy in directions:
        nx, ny = int(x) + dx, int(y) + dy
        if 0 <= nx < xmax and 0 <= ny < ymax:
            neighbors.append(modelmatrix[nx][ny])
        else:
            neighbors.append(0)  # or some other value indicating out of bounds

    return neighbors



### test the function ###
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    with open('../data/GridModesAdjacentRes.pkl','rb') as f:
        mapdata = pickle.load(f)
    matrice = mapdata_to_modelmatrix(mapdata, 326, 364)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    modes = ['TG', 'GG', 'GSD', 'TS']
    back_imgs = ['../figur/realmap.jpg',
                 '../figur/realmap.jpg',
                 '../figur/realmap.jpg',
                 '../figur/realmap.jpg']

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
        plt.savefig(f'../figur/{mode}_plot_with_mapdata.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
### test the function ###

