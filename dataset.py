import numpy as np
import utils

class Dataset(object):
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode

    def __str__(self):
        return '{}({}, {}+{}+{})'.format(self.name, self.mode, \
                                         len(self.tr_xs), len(self.te_xs), len(self.va_xs))

    @property
    def train_count(self):
        return len(self.tr_xs)

def dataset_get_train_data(self, batch_size, nth):
    from_idx = nth * batch_size
    to_idx = (nth + 1) * batch_size

    tr_X = self.tr_xs[self.indices[from_idx:to_idx]]
    tr_Y = self.tr_ys[self.indices[from_idx:to_idx]]

    return tr_X, tr_Y


def dataset_shuffle_train_data(self, size):
    self.indices = np.arange(size)
    np.random.shuffle(self.indices)

Dataset.get_train_data = dataset_get_train_data
Dataset.shuffle_train_data = dataset_shuffle_train_data

def dataset_get_test_data(self):
    return self.te_xs, self.te_ys

Dataset.get_test_data = dataset_get_test_data

def dataset_get_validate_data(self, count):
    self.va_indices = np.arange(len(self.va_xs))
    np.random.shuffle(self.va_indices)

    va_X = self.va_xs[self.va_indices[0:count]]
    va_Y = self.va_ys[self.va_indices[0:count]]

    return va_X, va_Y

Dataset.get_validate_data = dataset_get_validate_data
Dataset.get_visualize_data = dataset_get_validate_data


def dataset_shuffle_data(self, xs, ys, tr_ratio=0.8, va_ratio=0.05):
    data_count = len(xs)

    tr_cnt = int(data_count * tr_ratio / 10) * 10
    va_cnt = int(data_count * va_ratio)
    te_cnt = data_count - (tr_cnt + va_cnt)

    tr_from, tr_to = 0, tr_cnt
    va_from, va_to = tr_cnt, tr_cnt + va_cnt
    te_from, te_to = tr_cnt + va_cnt, data_count

    indices = np.arange(data_count)
    np.random.shuffle(indices)

    self.tr_xs = xs[indices[tr_from:tr_to]]
    self.tr_ys = ys[indices[tr_from:tr_to]]
    self.va_xs = xs[indices[va_from:va_to]]
    self.va_ys = ys[indices[va_from:va_to]]
    self.te_xs = xs[indices[te_from:te_to]]
    self.te_ys = ys[indices[te_from:te_to]]

    self.input_shape = xs[0].shape
    self.output_shape = ys[0].shape

    return indices[tr_from:tr_to], indices[va_from:va_to], indices[te_from:te_to]


Dataset.shuffle_data = dataset_shuffle_data


def dataset_forward_postproc(self, output, y, mode=None):
    if mode is None: mode = self.mode

    if mode == 'regression':
        diff = output - y
        square = np.square(diff)
        loss = np.mean(square)
        aux = diff
    elif mode == 'binary':
        entropy = utils.sigmoid_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        aux = [y, output]
    elif mode == 'select':
        entropy = utils.softmax_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        aux = [output, y, entropy]

    return loss, aux


Dataset.forward_postproc = dataset_forward_postproc


def dataset_backprop_postproc(self, G_loss, aux, mode=None):
    if mode is None: mode = self.mode

    if mode == 'regression':
        diff = aux
        shape = diff.shape

        g_loss_square = np.ones(shape) / np.prod(shape)
        g_square_diff = 2 * diff
        g_diff_output = 1

        G_square = g_loss_square * G_loss
        G_diff = g_square_diff * G_square
        G_output = g_diff_output * G_diff
    elif mode == 'binary':
        y, output = aux
        shape = output.shape

        g_loss_entropy = np.ones(shape) / np.prod(shape)
        g_entropy_output = utils.sigmoid_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy
    elif mode == 'select':
        output, y, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = utils.softmax_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

    return G_output


Dataset.backprop_postproc = dataset_backprop_postproc


def dataset_eval_accuracy(self, x, y, output, mode=None):
    if mode is None: mode = self.mode

    if mode == 'regression':
        mse = np.mean(np.square(output - y))
        accuracy = 1 - np.sqrt(mse) / np.mean(y)
    elif mode == 'binary':
        estimate = np.greater(output, 0)
        answer = np.equal(y, 1.0)
        correct = np.equal(estimate, answer)
        accuracy = np.mean(correct)
    elif mode == 'select':
        estimate = np.argmax(output, axis=1)
        answer = np.argmax(y, axis=1)
        correct = np.equal(estimate, answer)
        accuracy = np.mean(correct)

    return accuracy


Dataset.eval_accuracy = dataset_eval_accuracy


def dataset_get_estimate(self, output, mode=None):
    if mode is None: mode = self.mode

    if mode == 'regression':
        estimate = output
    elif mode == 'binary':
        estimate = utils.sigmoid(output)
    elif mode == 'select':
        estimate = utils.softmax(output)

    return estimate


Dataset.get_estimate = dataset_get_estimate

def dataset_train_prt_result(self, epoch, costs, accs, acc, time1, time2):
    print('    Epoch {}: cost={:5.3f}, accuracy={:5.3f}/{:5.3f} ({}/{} secs)'. \
          format(epoch, np.mean(costs), np.mean(accs), acc, time1, time2))

def dataset_test_prt_result(self, name, acc, time):
    print('Model {} test report: accuracy = {:5.3f}, ({} secs)\n'. \
          format(name, acc, time))

Dataset.train_prt_result = dataset_train_prt_result
Dataset.test_prt_result = dataset_test_prt_result

class FlowersDataset(Dataset):
    pass


def flowers_init(self, resolution=[100, 100], input_shape=[-1]):
    super(FlowersDataset, self).__init__('flowers', 'select')

    path = 'datasets/flowers'
    self.target_names = utils.list_dir(path)

    images = []
    idxs = []

    for dx, dname in enumerate(self.target_names):
        subpath = path + '/' + dname
        filenames = utils.list_dir(subpath)
        for fname in filenames:
            if fname[-4:] != '.jpg':
                continue
            imagepath = utils.os.path.join(subpath, fname)
            pixels = utils.load_image_pixels(imagepath, resolution, input_shape)
            images.append(pixels)
            idxs.append(dx)

    self.image_shape = resolution + [3]

    xs = np.asarray(images, np.float32)
    ys = utils.onehot(idxs, len(self.target_names))

    self.shuffle_data(xs, ys, 0.8)


FlowersDataset.__init__ = flowers_init

def flowers_visualize(self, xs, estimates, answers):
    utils.draw_images_horz(xs, self.image_shape)
    utils.show_select_results(estimates, answers, self.target_names)

FlowersDataset.visualize = flowers_visualize


class AbaloneDataset(Dataset):
    def __init__(self):
        super(AbaloneDataset, self).__init__('abalone', 'regression')

        rows, _ = utils.load_csv('datasets/abalone.csv')

        xs = np.zeros([len(rows), 10])
        ys = np.zeros([len(rows), 1])

        for n, row in enumerate(rows):
            if row[0] == 'I': xs[n, 0] = 1
            if row[0] == 'M': xs[n, 1] = 1
            if row[0] == 'F': xs[n, 2] = 1
            xs[n, 3:] = row[1:-1]
            ys[n, :] = row[-1:]

        self.shuffle_data(xs, ys, 0.8)

    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = utils.vector_to_str(x, '%4.2f')
            print('{} => 추정 {:4.1f} : 정답 {:4.1f}'.
                  format(xstr, est[0], ans[0]))


class PulsarDataset(Dataset):
    def __init__(self):
        super(PulsarDataset, self).__init__('pulsar', 'binary')

        rows, _ = utils.load_csv('datasets/pulsar_stars.csv')

        data = np.asarray(rows, dtype='float32')
        self.shuffle_data(data[:, :-1], data[:, -1:], 0.8)
        self.target_names = ['별', '펄서']

    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = utils.vector_to_str(x, '%5.1f', 3)
            estr = self.target_names[int(round(est[0]))]
            astr = self.target_names[int(round(ans[0]))]
            rstr = 'O'
            if estr != astr: rstr = 'X'
            print('{} => 추정 {}(확률 {:4.2f}) : 정답 {} => {}'. \
                  format(xstr, estr, est[0], astr, rstr))


class SteelDataset(Dataset):
    def __init__(self):
        super(SteelDataset, self).__init__('steel', 'select')

        rows, headers = utils.load_csv('datasets/faults.csv')

        data = np.asarray(rows, dtype='float32')
        self.shuffle_data(data[:, :-7], data[:, -7:], 0.8)

        self.target_names = headers[-7:]

    def visualize(self, xs, estimates, answers):
        utils.show_select_results(estimates, answers, self.target_names)


class PulsarSelectDataset(Dataset):
    def __init__(self):
        super(PulsarSelectDataset, self).__init__('pulsarselect', 'select')

        rows, _ = utils.load_csv('datasets/pulsar_stars.csv')

        data = np.asarray(rows, dtype='float32')
        self.shuffle_data(data[:, :-1], utils.onehot(data[:, -1], 2), 0.8)
        self.target_names = ['별', '펄서']

    def visualize(self, xs, estimates, answers):
        utils.show_select_results(estimates, answers, self.target_names)