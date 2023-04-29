import numpy as np
from numpy.linalg import norm
import cv2
import argparse
from itertools import product
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from numpy.linalg import LinAlgError
from scipy.stats import multivariate_normal
import igraph as ig

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel
n_components = 5
epsilon = 0.0001

class Gmm:
    def __init__(self, array) -> None:
        kmeans = KMeans(n_clusters=n_components)
        kmeans.fit(array)

        self.pixels = array
        self.labels = kmeans.labels_

        self.means = kmeans.cluster_centers_
        self.cov = None
        self.inv_cov = None
        self.det = None
        self.weights = None

        self.calc_weights()
        self.calc_cov()
        self.calc_inv_cov()
        self.calc_det()

    def update(self, array):
        self.pixels = array
        self.allocate_pixels()
        self.calc_means()
        self.calc_weights()
        self.calc_cov()
        self.calc_inv_cov()
        self.calc_det()

    def calc_weights(self):
        n_pixels = len(self.labels)
        weights = np.zeros(shape=(n_components))
        
        # Count how many pixels in each component\
        for i in range(n_components):
            weights[i] = len(self.labels[self.labels == i])

        assert np.sum(weights) == n_pixels

        # Normalize
        self.weights = weights / n_pixels
        assert(1.0 - epsilon <= sum(self.weights) <= 1.0 + epsilon)

    # TODO check this
    def calc_means(self):
        self.means = np.zeros(shape=(n_components,3))

        for i in range(n_components):
            temp = self.pixels[np.where(self.labels == i)]
            self.means[i] = np.sum(temp, axis=0)
            count = len(temp) if len(temp) > 0 else 1
            self.means[i] = self.means[i]/count

    def calc_cov(self):
        self.cov = np.zeros(shape=(n_components, 3, 3))

        for i in range(n_components):
            pixels = self.pixels[np.where(self.labels == i)]
            cov = np.cov(pixels, rowvar=False)

            assert(cov.shape == (3,3))
            self.cov[i] = np.copy(cov)
        
        assert(self.cov.shape == (n_components, 3, 3))

    def calc_inv_cov(self):
        assert(self.cov is not None)
        try:
            self.inv_cov = np.copy(np.linalg.inv(self.cov))
        except LinAlgError:
            self.cov += epsilon * np.eye(self.cov.shape[1])
            self.inv_cov = np.copy(np.linalg.inv(self.cov))

    def calc_det(self):
        self.det = np.linalg.det(self.cov)

    def calc_score(self, pixels, component):
        score = np.zeros(pixels.shape[0])

        if self.weights[component] > 0:
            diff = pixels - self.means[component]
            # mult = np.einsum('ij,ij->i', diff, np.dot(np.linalg.inv(self.cov[component]), diff.T).T)
            mult = np.einsum('ij,ij->i', diff, np.dot(self.inv_cov[component], diff.T).T)
            score = np.exp(-.5 * mult) / np.sqrt(2 * np.pi) / np.sqrt(np.linalg.det(self.cov[component]))
        return score
    
    def calc_prob(self, pixels):
        prob = [self.calc_score(pixels, component) for component in range(n_components)]
        return np.dot(self.weights, prob)

    def allocate_pixels(self):
        prob = np.array([self.calc_score(self.pixels, component) for component in range(n_components)]).T
        self.labels = np.argmax(prob, axis=1)

class Graph:
    def __init__(self) -> None:
        # igraph
        self.n = 0
        self.bg_id = 0
        self.fg_id = 0
        self.beta = 0
        self.T_weights = []  
        self.T_edges = []
        self.N_edges = []
        self.N_weights = []
        self.max_N_edge = 0
        self.left_V = None
        self.upleft_V = None
        self.up_V = None
        self.upright_V = None

        self.convergence = 0

        # matricies
        self.img = None
        self.img_idx = None
        self.mask = None
        self.bg_idx = None
        self.fg_idx = None
        self.uknown_idx = None
        self.rows = 0
        self.cols = 0

    def init_graph(self, img):
        self.img = np.copy(img)
        x,y = self.img.shape[:2]
        self.img_idx = np.arange(x*y).reshape((x,y))
        
        self.rows = x
        self.cols = y

        # +2 for bg and fg verticies
        self.n = x*y + 2
        self.bg_id = self.n - 1
        self.fg_id = self.n - 2

        self.calc_beta()
        self.calc_V()
        self.init_N_edges()
        self.max_N_edge = np.max(self.N_weights)

    def calc_beta(self):
        w = self.cols
        h = self.rows

        # this 'catches' all the neighbors distances in img
        left_diff = self.img[:, 1:] - self.img[:, :-1]
        upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1]
        up_diff = self.img[1:, :] - self.img[:-1, :]
        upright_diff = self.img[1:, :-1] - self.img[:-1, 1:]

        excep = np.sum(np.square(left_diff)) + np.sum(np.square(upleft_diff)) + np.sum(np.square(up_diff)) + np.sum(np.square(upright_diff))
        
        # this is the amount of edges
        beta = 2 * excep / (4*w*h - 3*w - 3*h + 2)
        beta = 1 / beta

        self.beta = beta

    def calc_V(self):
        left_diff = self.img[:, 1:] - self.img[:, :-1]
        upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1]
        up_diff = self.img[1:, :] - self.img[:-1, :]
        upright_diff = self.img[1:, :-1] - self.img[:-1, 1:]

        self.left_V = 50 * np.exp(-self.beta * np.sum(np.square(left_diff), axis=2))
        self.upleft_V = 50 / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(upleft_diff), axis=2))
        self.up_V = 50 * np.exp(-self.beta * np.sum(np.square(up_diff), axis=2))
        self.upright_V = 50 / np.sqrt(2) * np.exp(-self.beta * np.sum(np.square(upright_diff), axis=2))

    def add_neighbors_edges(self, x, y):
        n_rows = self.rows
        n_culls = self.cols
        
        idx = self.index(x, y)

        # LEFT
        if y > 0:
            neighbor = self.index(x, y-1)
            self.N_edges.append([idx, neighbor])
            self.N_weights.append(self.calc_N_weight((x,y), (x,y-1)))

        # UP-LEFT
        if y > 0 and x > 0:
            neighbor = self.index(x-1, y-1)
            self.N_edges.append([idx, neighbor])
            diag = 1/np.sqrt(2)
            self.N_weights.append(self.calc_N_weight((x,y), (x-1,y-1)) * diag)

        # UP
        if x > 0:
            neighbor = self.index(x-1, y)
            self.N_edges.append([idx, neighbor])
            self.N_weights.append(self.calc_N_weight((x,y), (x-1,y)))

        # UP-RIGHT
        if y < n_culls - 1 and x > 0:
            neighbor = self.index(x-1, y+1)
            self.N_edges.append([idx, neighbor])
            diag = 1/np.sqrt(2)
            self.N_weights.append(self.calc_N_weight((x,y), (x-1,y+1)) * diag)

    def calc_N_weight(self, pixel_1, pixel_2):
        dist = np.sum(np.square(self.img[pixel_1] - self.img[pixel_2]))
        weight = 50 * np.exp(-self.beta*dist)
        return weight

    def init_N_edges(self):
        img_indexes = np.arange(self.rows * self.cols, dtype=np.uint32).reshape(self.rows, self.cols)
        self.N_edges = []
        self.N_weights = []

        mask1 = img_indexes[:, 1:].reshape(-1)
        mask2 = img_indexes[:, :-1].reshape(-1)
        self.N_edges.extend(list(zip(mask1, mask2)))
        self.N_weights.extend(self.left_V.reshape(-1).tolist())

        mask1 = img_indexes[1:, 1:].reshape(-1)
        mask2 = img_indexes[:-1, :-1].reshape(-1)
        self.N_edges.extend(list(zip(mask1, mask2)))
        self.N_weights.extend(self.upleft_V.reshape(-1).tolist())

        mask1 = img_indexes[1:, :].reshape(-1)
        mask2 = img_indexes[:-1, :].reshape(-1)
        self.N_edges.extend(list(zip(mask1, mask2)))
        self.N_weights.extend(self.up_V.reshape(-1).tolist())

        mask1 = img_indexes[1:, :-1].reshape(-1)
        mask2 = img_indexes[:-1, 1:].reshape(-1)
        self.N_edges.extend(list(zip(mask1, mask2)))
        self.N_weights.extend(self.upright_V.reshape(-1).tolist())

        assert len(self.N_edges) == len(self.N_weights)

    def index(self,x, y) -> int:
        # return x*self.culls + y
        return self.img_idx[x,y]
    
    def update_T_edges(self, mask, bg_gmm: Gmm, fg_gmm: Gmm):
        self.mask = np.copy(mask)

        self.bg_idx = np.where(self.mask.reshape(-1) == GC_BGD)
        self.fg_idx = np.where(self.mask.reshape(-1) == GC_FGD)
        self.uknown_idx = np.where(np.logical_or(self.mask.reshape(-1) == GC_PR_FGD, self.mask.reshape(-1) == GC_PR_BGD))
        
        self.T_edges = [] 
        self.T_weights = []

        # unknown to foreground vertix
        self.T_edges.extend(list(zip([self.fg_id] * self.uknown_idx[0].size, self.uknown_idx[0])))
        with np.errstate(divide='ignore'):
            score = -np.log(bg_gmm.calc_prob(self.img.reshape(-1, 3)[self.uknown_idx]))

        assert not np.any(np.isinf(score))
        # if np.any(np.isinf(score)):
        #     score[np.isinf(score)] = -np.log(bg_gmm.calc_prob(self.img.reshape(-1, 3)[self.uknown_idx]) + epsilon)[np.isinf(score)]
        self.T_weights.extend(score.tolist())
        
        # unknown to background vertix
        self.T_edges.extend(list(zip([self.bg_id] * self.uknown_idx[0].size, self.uknown_idx[0])))
        score = -np.log(fg_gmm.calc_prob(self.img.reshape(-1, 3)[self.uknown_idx]))
        self.T_weights.extend(score.tolist())
        
        # foreground vertix to foreground pixels is biggest
        self.T_edges.extend(list(zip([self.fg_id] * self.fg_idx[0].size, self.fg_idx[0])))
        self.T_weights.extend([self.max_N_edge * 10] * self.fg_idx[0].size)
        
        # background vertix to foreground pixels is 0
        self.T_edges.extend(list(zip([self.bg_id] * self.fg_idx[0].size, self.fg_idx[0])))
        self.T_weights.extend([0] * self.fg_idx[0].size)
        
        # foreground vertix to background pixels is 0
        self.T_edges.extend(list(zip([self.fg_id] * self.bg_idx[0].size, self.bg_idx[0])))
        self.T_weights.extend([0] * self.bg_idx[0].size)
        
        # background vertix to background pixels is biggest
        self.T_edges.extend(list(zip([self.bg_id] * self.bg_idx[0].size, self.bg_idx[0])))
        self.T_weights.extend([self.max_N_edge * 10] * self.bg_idx[0].size)

        assert len(self.T_edges) == len(self.T_weights)

    def min_cut(self, img, mask, bg_gmm, fg_gmm):
        self.update_T_edges(mask, bg_gmm, fg_gmm)
        
        edges = np.concatenate((self.N_edges, self.T_edges))
        capacities = np.concatenate((self.N_weights, self.T_weights))

        assert len(edges) == 4 * self.cols * self.rows - 3 * (self.cols + self.rows) + 2 + 2 * self.cols * self.rows

        graph = ig.Graph(self.n, edges)
        mincut = graph.st_mincut(self.fg_id, self.bg_id, capacity=list(capacities))

        return mincut.partition, mincut.value

G = Graph()

#------------------------------------------------tools-------------------------------------------------#

def reshape(img, mask):
    x,y,z = img.shape
    img_vector = np.reshape(img, (x*y, z))

    x,y = mask.shape
    mask_vector = np.reshape(mask, (x*y))
    return img_vector, mask_vector

def seperate(img, mask):
    img_vector, mask_vector = reshape(img, mask)

    bg_pixels = img_vector[mask_vector == GC_BGD]
    fg_pixels = img_vector[mask_vector > GC_BGD]
    
    return bg_pixels, fg_pixels

#--------------------------------------------algo by order-------------------------------------------------#

def initalize_GMMs(img, mask):
    bgGMM = None
    fgGMM = None

    bg_pixels, fg_pixels = seperate(img, mask)

    bgGMM = Gmm(bg_pixels)
    fgGMM = Gmm(fg_pixels)

    G.init_graph(img)

    return bgGMM, fgGMM

def update_GMMs(img, mask, bgGMM: Gmm, fgGMM: Gmm):
    x,y,z = img.shape

    bg_pixels = img.reshape(x*y, z)[np.logical_or(mask.reshape(-1) == GC_BGD, mask.reshape(-1) == GC_PR_BGD)]
    fg_pixels = img.reshape(x*y, z)[np.logical_or(mask.reshape(-1) == GC_FGD, mask.reshape(-1) == GC_PR_FGD)]

    bgGMM.update(bg_pixels)
    fgGMM.update(fg_pixels)

    return bgGMM, fgGMM

def calculate_mincut(img, mask, bgGMM, fgGMM):
    return G.min_cut(img, mask, bgGMM, fgGMM)

# def update_mask(mincut_sets, mask):
#     x,y = mask.shape
#     img_idx = np.arange(x*y).reshape((x,y))
#     new_mask = np.copy(mask)
#     unknown_idx = np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD))
#     new_mask[unknown_idx] = np.where(np.isin(img_idx[unknown_idx], mincut_sets[0]), GC_PR_FGD, GC_PR_BGD)
#     return new_mask

def update_mask(mincut_sets, mask):
    new_mask = np.copy(mask)
    count = 0

    if (G.fg_id in mincut_sets[0]):
        fg_set = 0
    else:
        fg_set = 1
    ##check if mincut_set[0] is the nodes of source (source = FG

    for i in mincut_sets[fg_set]:
        if (i < G.n - 2):
            row = i // new_mask.shape[1]
            col = i % new_mask.shape[1]
            # if the mask pixcel is hard background dont change
            if (mask[row][col] != GC_BGD and mask[row][col] != GC_PR_FGD):
                new_mask[row][col] = GC_PR_FGD
                count += 1
    
    for i in mincut_sets[1 - fg_set]:
        if (i < G.n - 2):
            row = i // mask.shape[1]
            col = i % mask.shape[1]
            if (mask[row][col] != GC_BGD and mask[row][col] != GC_PR_BGD):
                new_mask[row][col] = GC_PR_BGD
                count += 1
    if True:
        print("in iter ", i, ": changed: ", G.convergence)
    return new_mask

# TODO !!!!
def check_convergence(energy):
    return False

def cal_metric(predicted_mask, gt_mask):
    inter = np.sum(np.logical_and(predicted_mask, gt_mask))
    union = np.sum(np.logical_or(predicted_mask, gt_mask))
    accuracy = np.sum(predicted_mask == gt_mask) / gt_mask.size
    jaccard = inter / union

    return accuracy, jaccard

#-------------------------------------------------main-------------------------------------------------#

def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD # soft fg (3)
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD # hard fg (1)

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in tqdm(range(1)):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()