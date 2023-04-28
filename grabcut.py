import numpy as np
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

    def update(self):
        self.calc_means()
        self.calc_weights()
        self.calc_cov()
        self.calc_inv_cov()
        self.calc_det()

    def calc_weights(self):
        n_pixels = len(self.labels)
        weights = np.zeros(shape=(n_components))
        
        # Count how many pixels in each component
        for label in self.labels:
            weights[label] += 1

        # Normalize
        for i, weight in enumerate(weights):
            weights[i] = weight/n_pixels

        self.weights = np.copy(weights)
        assert(1.0 - epsilon <= sum(weights) <= 1.0)

    def calc_means(self):
        means = np.zeros(shape=(n_components,3))
        counts = np.zeros(shape=(n_components))

        for i, com in enumerate(self.labels):
            means[com] += self.pixels[i]
            counts[com] += 1
        
        for i, count in enumerate(counts):
            if count > 0:
                means[i] = means[i]/count
            else:
                means[i] = 0

        self.means = np.copy(means)

    def calc_cov(self):
        self.cov = np.zeros(shape=(n_components, 3, 3))

        for i in range(n_components):
            pixels = self.pixels[self.labels == i]
            cov = np.cov(pixels, rowvar=False)

            assert(cov.shape == (3,3))
            np.append(self.cov, np.copy(cov))
        
        assert(self.cov.shape == (n_components, 3, 3))

    def calc_inv_cov(self):
        assert(self.cov is not None)
        try:
            self.inv_cov = np.copy(np.linalg.inv(self.cov))
        except LinAlgError:
            self.cov += epsilon * np.eye(self.cov.shape[1])
            self.inv_cov = np.copy(np.linalg.inv(self.cov))
        return

    def calc_det(self):
        self.det = np.linalg.det(self.cov)

    def calc_prob(self, x, k):
        return multivariate_normal.pdf(x, self.means[k], self.cov[k])

    def allocate_pixels(self, pixels):
        # Label each pixel to his most liklyhood component
        self.labels = np.zeros(shape=(len(pixels)))
        for i, pixel in enumerate(pixels):
            argmax_k, max_k = -1, 0
            for k in range(n_components):
                prob = self.calc_prob(pixel, k)
                if prob > max_k:
                    max_k = prob
                    argmax_k = k
            self.labels[i] = argmax_k

class Graph:
    def __init__(self) -> None:
        self.graph = None
        self.img = None
        self.edges = []
        self.weights = []
    
    def set_graph(self):
        x,y = self.img.shape[:2]
        
        # +2 for bg and fg verticies
        n = x*y + 2

        self.graph = ig.Graph(n, self.edges)
        self.graph.es['weight'] = self.weights
        self.graph.vs['color'] = self.img
    
    def add_img(self, img):
        self.img = np.copy(img)

    def add_N_edge(self, x, y):
        weight = self.calc_N_weight(x,y)

        self.edges.append([x,y])
        self.weights.append(weight)

    # TODO: this
    def calc_N_weight(self, x, y):
        beta = 0

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
    bg_pixels = img_vector[(mask_vector == GC_BGD) | (mask_vector == GC_PR_BGD)]
    fg_pixels = img_vector[(mask_vector == GC_FGD) | (mask_vector == GC_PR_FGD)]
    return bg_pixels, fg_pixels

def init_graph(img, mask=None, bgGMM=None, fgGMM=None) -> Graph:
    n_rows, n_cull = img.shape[:2]
    
    G.add_img(img)
    
    for idx in range(n_cull*n_rows):
        if idx%n_rows < n_cull - 1:
            G.add_N_edge(idx, idx+1)
        if idx//n_rows < n_rows - 1:
            G.add_N_edge(idx, idx+n_cull)
        if idx%n_rows < n_cull - 1 and idx//n_rows < n_rows - 1:
            G.add_N_edge(idx, idx+n_cull+1)
        if idx%n_rows < n_cull - 1 and 0 < idx//n_rows:
            G.add_N_edge(idx, idx+n_cull-1)

    G.set_graph()
    return G

def index(pixel: tuple, width: int) -> int:
    return pixel[0]*width + pixel[1]

    #### weight = 0
    G.add_edge(pixel_1, pixel_2)
    # G.add_edge(idx_1, idx_2, weight=0)

def initalize_GMMs(img, mask):
    bgGMM = None
    fgGMM = None

    bg_pixels, fg_pixels = seperate(img, mask)

    bgGMM = Gmm(bg_pixels)
    fgGMM = Gmm(fg_pixels)

    return bgGMM, fgGMM

def update_GMMs(img, mask, bgGMM: Gmm, fgGMM: Gmm):
    bg_pixels, fg_pixels = seperate(img, mask)

    bgGMM.allocate_pixels(bg_pixels)
    fgGMM.allocate_pixels(fg_pixels)

    bgGMM.update()
    fgGMM.update()

    return bgGMM, fgGMM

def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]
    energy = 0
    init_graph(img, mask, bgGMM, fgGMM)
    return min_cut, energy

def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask

def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence

def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100

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
    for i in range(num_iters):
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

    init_graph(img)


# if __name__ == '__main__':
def main():
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