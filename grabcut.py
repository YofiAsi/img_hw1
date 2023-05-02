import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
import igraph as ig

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

epsilon = 0.000001
energy_threshold = 10000

class GMM:
    def __init__(self, pixels_array, n_components=5):
        kmeans = KMeans(n_clusters=n_components, n_init=1).fit(pixels_array)
        self.labels = kmeans.labels_
        self.means = kmeans.cluster_centers_
        self.n_components = n_components

        self.dim = pixels_array.shape[1]
        self.pixels = pixels_array
        self.counter = None
        self.weights = None
        self.covs = None

        self.calc_counter()
        self.calc_weights()
        self.calc_covs()

    def update_gmm(self, pixels_array):
        self.pixels = np.copy(pixels_array)
        self.allocate_pixels()
        self.calc_means()
        self.calc_weights()
        self.calc_covs()
    
    def allocate_pixels(self):
        prob = np.array([self.calc_score(self.pixels, component) for component in range(self.n_components)]).T
        self.labels = np.argmax(prob, axis=1)
    
    def calc_counter(self):
        uni_labels, count = np.unique(self.labels, return_counts=True)
        
        self.counter = np.zeros(self.n_components)
        self.counter[uni_labels] = count
    
    def calc_means(self):
        self.means = np.zeros(shape=(self.n_components, self.dim))
        for component in range(self.n_components):
            self.means[component] = np.mean(self.pixels[self.labels == component], axis=0)

    def calc_weights(self):
        self.weights = np.zeros(self.n_components)
        s = np.sum(self.counter)

        for component in range(self.n_components):
            self.weights[component] = self.counter[component] / s

    def calc_covs(self):
        self.covs = np.zeros(shape=(self.n_components, self.dim, self.dim))

        for component in range(self.n_components):
            self.covs[component] = 0 if self.counter[component] <= 1 else np.cov(self.pixels[self.labels == component].T)

            # making sure the inv will be ok
            det = np.linalg.det(self.covs[component])
            if det <= 0:
                self.covs[component] += np.eye(self.dim) * epsilon

    def calc_score(self, pixels, component):
        score = np.zeros(pixels.shape[0])
        if self.weights[component] > 0:
            diff = pixels - self.means[component]
            mult = np.einsum('ij,ij->i', diff, np.dot(np.linalg.inv(self.covs[component]), diff.T).T)
            score = np.exp(-.5 * mult) / np.sqrt(2 * np.pi) / np.sqrt(np.linalg.det(self.covs[component]))
        return score

    def calc_prob(self, pixels):
        prob = [self.calc_score(pixels, component) for component in range(self.n_components)]
        return np.dot(self.weights, prob)

class GrabCut:
    def __init__(self) -> None:
        # edges & weights
        self.T_weights = []  
        self.T_edges = []
        self.N_edges = []
        self.N_weights = []
        self.max_N_edge = 0
        
        self.left_V = None
        self.upleft_V = None
        self.up_V = None
        self.upright_V = None

        self.n = 0 # num of verticies
        self.source_id = 0 # will be foreground
        self.sink_id = 0 # will be background
        self.beta = 0

        self.last_energy = 0

        self.img = None
        self.mask = None
        self.img_idx = None # a mat with the graph indexes
        
        # self.comp_idxs = np.empty((self.rows, self.cols), dtype=np.uint32) # every pixel which components it belongs
        self.rows = 0
        self.cols = 0

    def init_GrabCut(self, img):
        self.img = np.copy(img)
        x,y,z = self.img.shape
        self.img_idx = np.arange(x*y).reshape((x,y))
        self.rows = x
        self.cols = y

        # +2 for bg and fg verticies
        self.n = x*y + 2
        self.source_id = self.n - 1
        self.sink_id = self.n - 2

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

    def update_T_edges(self, mask, bg_gmm: GMM, fg_gmm: GMM):
        self.left_V = np.empty((self.rows, self.cols - 1))
        self.upleft_V = np.empty((self.rows - 1, self.cols - 1))
        self.up_V = np.empty((self.rows - 1, self.cols))
        self.upright_V = np.empty((self.rows - 1, self.cols - 1))

        self.mask = np.copy(mask)
        bg_idx = np.where(self.mask.reshape(-1) == GC_BGD)
        fg_idx = np.where(self.mask.reshape(-1) == GC_FGD)
        uknown_idx = np.where(np.logical_or(self.mask.reshape(-1) == GC_PR_FGD, self.mask.reshape(-1) == GC_PR_BGD))
        
        self.T_edges = [] 
        self.T_weights = []

        # unknown to foreground vertix
        self.T_edges.extend(list(zip([self.source_id] * uknown_idx[0].size, uknown_idx[0])))
        with np.errstate(divide='ignore'):
            _score = -np.log(bg_gmm.calc_prob(self.img.reshape(-1, 3)[uknown_idx]))
        if np.any(np.isinf(_score)):
            _score[np.isinf(_score)] = -np.log(bg_gmm.calc_prob(self.img.reshape(-1, 3)[uknown_idx]) + epsilon)[np.isinf(_score)]
        self.T_weights.extend(_score.tolist())
        
        # unknown to background vertix
        self.T_edges.extend(list(zip([self.sink_id] * uknown_idx[0].size, uknown_idx[0])))
        with np.errstate(divide='ignore'):
            _score = -np.log(fg_gmm.calc_prob(self.img.reshape(-1, 3)[uknown_idx]))
        if np.any(np.isinf(_score)):
            _score[np.isinf(_score)] = -np.log(fg_gmm.calc_prob(self.img.reshape(-1, 3)[uknown_idx]) + epsilon)[np.isinf(_score)]
        self.T_weights.extend(_score.tolist())
        
        # foreground vertix to foreground pixels is biggest
        self.T_edges.extend(list(zip([self.source_id] * fg_idx[0].size, fg_idx[0])))
        self.T_weights.extend([self.max_N_edge * 9] * fg_idx[0].size)
        
        # background vertix to foreground pixels is 0
        self.T_edges.extend(list(zip([self.sink_id] * fg_idx[0].size, fg_idx[0])))
        self.T_weights.extend([0] * fg_idx[0].size)
        
        # foreground vertix to background pixels is 0
        self.T_edges.extend(list(zip([self.source_id] * bg_idx[0].size, bg_idx[0])))
        self.T_weights.extend([0] * bg_idx[0].size)
        
        # background vertix to background pixels is biggest
        self.T_edges.extend(list(zip([self.sink_id] * bg_idx[0].size, bg_idx[0])))
        self.T_weights.extend([self.max_N_edge * 9] * bg_idx[0].size)

        assert len(self.T_edges) == len(self.T_weights)

    def min_cut(self, img, mask, bg_gmm: GMM, fg_gmm: GMM):
        self.update_T_edges(mask, bg_gmm=bg_gmm, fg_gmm=fg_gmm)
        
        edges = list( np.concatenate((self.N_edges, self.T_edges)) )
        capacities = list( np.concatenate((self.N_weights, self.T_weights)) )

        assert len(edges) == 4 * self.cols * self.rows - 3 * (self.cols + self.rows) + 2 + 2 * self.cols * self.rows

        graph = ig.Graph(self.n, edges)
        mincut = graph.st_mincut(self.source_id, self.sink_id, capacity=capacities)

        return mincut.partition, mincut.value

    def update_energy(self, energy):
        self.last_energy = energy

G = GrabCut()

#--------------------------------------------------------tools----------------------------------------------------------#

def initalize_GMMs(img, mask, n_components=5):

    G.init_GrabCut(img)

    bg_pixels = np.where(mask == GC_BGD)
    fg_pixels = np.where(mask != GC_BGD) # we start like that as the paper say

    bgGMM = GMM(img[bg_pixels], n_components=n_components)
    fgGMM = GMM(img[fg_pixels], n_components=n_components)

    return bgGMM, fgGMM

def update_GMMs(img, mask, bgGMM: GMM, fgGMM: GMM):
    bg_pixels = np.where(np.logical_or(mask == GC_BGD, mask == GC_PR_BGD))
    fg_pixels = np.where(np.logical_or(mask == GC_FGD, mask == GC_PR_FGD))

    bgGMM.update_gmm(img[bg_pixels])
    fgGMM.update_gmm(img[fg_pixels])

    return bgGMM, fgGMM

def calculate_mincut(img, mask, bgGMM, fgGMM):
    return G.min_cut(img, mask, bg_gmm=bgGMM, fg_gmm=fgGMM)

def update_mask(mincut_sets, mask):
    x,y = mask.shape
    img_idx = np.arange(x*y).reshape((x,y))
    new_mask = np.copy(mask)
    unknown_idx = np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD))
    if G.source_id in mincut_sets[0]:
        new_mask[unknown_idx] = np.where(np.isin(img_idx[unknown_idx], mincut_sets[0]), GC_PR_FGD, GC_PR_BGD)
    else:
        new_mask[unknown_idx] = np.where(np.isin(img_idx[unknown_idx], mincut_sets[0]), GC_PR_BGD, GC_PR_FGD)
        
    return new_mask

def check_convergence(energy):
    convergence = np.abs(energy-G.last_energy) <= energy_threshold
    G.update_energy(energy)
    return convergence

def cal_metric(predicted_mask, gt_mask):
    inter = np.sum(np.logical_and(predicted_mask, gt_mask))
    union = np.sum(np.logical_or(predicted_mask, gt_mask))
    accuracy = np.sum(predicted_mask == gt_mask) / gt_mask.size
    jaccard = inter / union

    return accuracy, jaccard

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='book', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

#--------------------------------------------------------main----------------------------------------------------------#

def grabcut(img, rect, n_components=5, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    w -= x
    h -= y

    # Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD
    
    bgGMM, fgGMM = initalize_GMMs(img, mask, n_components)

    num_iters = 1000
    for i in range(num_iters):
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

"""def test():
    import os
    import csv
    import time
    from tqdm import tqdm
    input_names = [file.split('.')[0] for file in os.listdir("data/imgs")[1:]]
    results = []
    for input_name in input_names:
        print(f'working on {input_name}...')

        input_path = f'data/imgs/{input_name}.jpg'
        rect = tuple(map(int, open(f"data/bboxes/{input_name}.txt", "r").read().split(' ')))

        img = cv2.imread(input_path)

        gt_mask = cv2.imread(f'data/seg_GT/{input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]

        for n_components in tqdm(range(2,6)):
            
            start = time.time()
            mask, bgGMM, fgGMM, iter = grabcut(img, rect, n_components)
            end = time.time()

            for i in range(G.n-2):
                row = i // img.shape[1]
                col = i % img.shape[1]
                if (mask[row][col] == GC_PR_BGD):
                    mask[row][col] = GC_BGD

            mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
            
            gt_mask = cv2.imread(f'data/seg_GT/{input_name}.bmp', cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
            acc, jac = cal_metric(mask, gt_mask)
            
            t = end-start
            results.append({
                'name': input_name,
                'n components': n_components,
                'accuracy': acc,
                'jac': jac,
                'time': t,
                'iter': iter,
            })

            dir = f'test/{input_name}/{n_components}_comps'
            os.makedirs(dir)

            img_cut = img * (mask[:, :, np.newaxis])
            cv2.imwrite(f'{dir}/result.jpg', img_cut)
            cv2.imwrite(f'{dir}/mask.jpg', mask*255)
            cv2.imwrite(f'{dir}/gt.jpg', gt_mask*255)
            cv2.imwrite(f'{dir}/original.jpg', img)

    with open('results.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, ['name', 'n components', 'accuracy', 'jac', 'time', 'iter'])
        writer.writeheader()
        writer.writerows(results)

def test_blur():
    import os
    import csv
    import time
    from tqdm import tqdm
    input_names = ['flower', 'cross', 'stone2', 'teddy']
    results = []
    for input_name in input_names:
        print(f'working on {input_name}...')

        input_path = f'data/imgs/{input_name}.jpg'
        rect = tuple(map(int, open(f"data/bboxes/{input_name}.txt", "r").read().split(' ')))

        img = cv2.imread(input_path)

        gt_mask = cv2.imread(f'data/seg_GT/{input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]

        for n_components in tqdm(range(2,6)):
            img_low = cv2.blur(img, (5,5))
            img_high = cv2.blur(img, (30,30))

            start = time.time()
            mask, bgGMM, fgGMM, iter = grabcut(img_low, rect, n_components)
            end = time.time()

            for i in range(G.n-2):
                row = i // img.shape[1]
                col = i % img.shape[1]
                if (mask[row][col] == GC_PR_BGD):
                    mask[row][col] = GC_BGD

            mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
            
            gt_mask = cv2.imread(f'data/seg_GT/{input_name}.bmp', cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
            acc, jac = cal_metric(mask, gt_mask)
            
            t = end-start
            results.append({
                'name': input_name+"_low_blur",
                'n components': n_components,
                'accuracy': acc,
                'jac': jac,
                'time': t,
                'iter': iter,
            })

            dir = f'test/{input_name}_low_blur/{n_components}_comps'
            os.makedirs(dir)

            img_cut = img_low * (mask[:, :, np.newaxis])
            cv2.imwrite(f'{dir}/result.jpg', img_cut)
            cv2.imwrite(f'{dir}/mask.jpg', mask*255)
            cv2.imwrite(f'{dir}/gt.jpg', gt_mask*255)
            cv2.imwrite(f'{dir}/original.jpg', img_low)

            start = time.time()
            mask, bgGMM, fgGMM, iter = grabcut(img_high, rect, n_components)
            end = time.time()

            for i in range(G.n-2):
                row = i // img.shape[1]
                col = i % img.shape[1]
                if (mask[row][col] == GC_PR_BGD):
                    mask[row][col] = GC_BGD

            mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
            
            gt_mask = cv2.imread(f'data/seg_GT/{input_name}.bmp', cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
            acc, jac = cal_metric(mask, gt_mask)
            
            t = end-start
            results.append({
                'name': input_name+"_high_blur",
                'n components': n_components,
                'accuracy': acc,
                'jac': jac,
                'time': t,
                'iter': iter,
            })

            dir = f'test/{input_name}_high_blur/{n_components}_comps'
            os.makedirs(dir)

            img_cut = img_high * (mask[:, :, np.newaxis])
            cv2.imwrite(f'{dir}/result.jpg', img_cut)
            cv2.imwrite(f'{dir}/mask.jpg', mask*255)
            cv2.imwrite(f'{dir}/gt.jpg', gt_mask*255)
            cv2.imwrite(f'{dir}/original.jpg', img_high)

    with open('results_blur.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, ['name', 'n components', 'accuracy', 'jac', 'time', 'iter'])
        writer.writeheader()
        writer.writerows(results)

def test_rect():
    import os
    import csv
    import time
    from tqdm import tqdm
    input_names = ['llama', 'grave']
    results = []
    for input_name in input_names:
        print(f'working on {input_name}...')

        input_path = f'data/imgs/{input_name}.jpg'
        rect = tuple(map(int, open(f"data/bboxes/{input_name}.txt", "r").read().split(' ')))

        img = cv2.imread(input_path)
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask.fill(0)
        x, y, w, h = rect
        w -= x
        h -= y
        mask[y:y + h, x:x + w] = 1
        dir = f'test/{input_name}_rec/rec1.jpg'
        cv2.imwrite(dir, img * (mask[:, :, np.newaxis]))

        if input_name == 'llama':
            x = 10
            y = 10
            w = 500
            h = 350
        else:
            x = 10
            y = 10
            w = 430
            h = 680

        mask.fill(0)
        mask[y:y + h, x:x + w] = 1
        dir = f'test/{input_name}_rec/rec2.jpg'
        cv2.imwrite(dir, img * (mask[:, :, np.newaxis]))

        mask[y:y + h, x:x + w] = GC_PR_FGD

        gt_mask = cv2.imread(f'data/seg_GT/{input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]

        for n_components in tqdm(range(2,6)):

            start = time.time()
            mask, bgGMM, fgGMM, iter = grabcut(img, rect, n_components)
            end = time.time()

            for i in range(G.n-2):
                row = i // img.shape[1]
                col = i % img.shape[1]
                if (mask[row][col] == GC_PR_BGD):
                    mask[row][col] = GC_BGD

            mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
            
            gt_mask = cv2.imread(f'data/seg_GT/{input_name}.bmp', cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
            acc, jac = cal_metric(mask, gt_mask)
            
            t = end-start
            results.append({
                'name': input_name+"_new_rec",
                'n components': n_components,
                'accuracy': acc,
                'jac': jac,
                'time': t,
                'iter': iter,
            })

            dir = f'test/{input_name}_rec/{n_components}_comps'
            os.makedirs(dir)

            img_cut = img * (mask[:, :, np.newaxis])
            cv2.imwrite(f'{dir}/result.jpg', img_cut)
            cv2.imwrite(f'{dir}/mask.jpg', mask*255)
            cv2.imwrite(f'{dir}/gt.jpg', gt_mask*255)
            cv2.imwrite(f'{dir}/original.jpg', img)

    with open('results_rec.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, ['name', 'n components', 'accuracy', 'jac', 'time', 'iter'])
        writer.writeheader()
        writer.writerows(results)"""

if __name__ == '__main__':
    # test_rect()
    # test_blur()
    # test()

    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)

    # why wasn't this a part of the main?
    for i in range(G.n-2):
        row = i // img.shape[1]
        col = i % img.shape[1]
        if (mask[row][col] == GC_PR_BGD):
            mask[row][col] = GC_BGD

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
