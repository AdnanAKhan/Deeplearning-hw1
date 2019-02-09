import torch


def to_2d_tensor(inp):
    """
        This function converts 1-D vector to  2-D Tensor
        Example:
            x = torch.tensor([1,2,3,1,3,4])
            print(x.size()) # torch.Size([6])
            o= torch.unsqueeze(x, 0)
            print(o.size()) # torch.Size([1, 6])
    """
    inp = torch.Tensor(inp)
    if len(inp.size()) < 2:  # if one D vector
        inp = inp.unsqueeze(0)
    return inp


def convert_xywh_to_x1y1x2y2(boxes):
    """
        This function convert x_1, y_1, W, H => x_1,y_1, x_2, y_2 (top left and bottom right corner). This is a vector
        operation.

        :param boxes: a list of box(x,y,w,h)
        :return: box in (x1,y1,x2,y2) format (representing two corners)
    """
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] += boxes[:, 0] - 1
    boxes[:, 3] += boxes[:, 1] - 1
    return boxes


def convert_x1y1x2y2_to_xywh(boxes):
    """
        This function inverts the operation of  'convert_xywh_to_x1y1x2y2'  function.
        :param boxes: a list of box x1,y1,x2,y2)
        :return: box in (x1,y1,w,h) format.
    """
    boxes = to_2d_tensor(boxes)
    boxes[:, 2] -= boxes[:, 0] - 1
    boxes[:, 3] -= boxes[:, 1] - 1
    return boxes


def box_transform(boxes, im_sizes):
    """
    Normalize the box value.
    :param boxes: a np array of box in (x, y, w, h) format
    :param im_sizes: size of the images.
    :return: normalize (x,y,w,h) in the range of [-1..1]
    """
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes[:, 0] = 2 * boxes[:, 0] / im_sizes[:, 0] - 1
    boxes[:, 1] = 2 * boxes[:, 1] / im_sizes[:, 1] - 1
    boxes[:, 2] = 2 * boxes[:, 2] / im_sizes[:, 0] - 1
    boxes[:, 3] = 2 * boxes[:, 3] / im_sizes[:, 1] - 1
    return boxes


def box_transform_inv(boxes, im_sizes):
    """
    inverse the normalization process.
    :param boxes: normalize (x,y,w,h) in the range of [-1..1]
    :param im_sizes: size of the images.
    :return: np array of box in (x, y, w, h) format
    """
    boxes = to_2d_tensor(boxes)
    im_sizes = to_2d_tensor(im_sizes)
    boxes[:, 0] = (boxes[:, 0] + 1) / 2 * im_sizes[:, 0]
    boxes[:, 1] = (boxes[:, 1] + 1) / 2 * im_sizes[:, 1]
    boxes[:, 2] = (boxes[:, 2] + 1) / 2 * im_sizes[:, 0]
    boxes[:, 3] = (boxes[:, 3] + 1) / 2 * im_sizes[:, 1]
    return boxes


def compute_IoU(boxes1, boxes2):
    """
    Calculates aIntersection over Union (IoU) between two shapes.
    :param boxes1:
    :param boxes2:
    :return:
    """
    boxes1 = to_2d_tensor(boxes1)
    boxes1 = convert_xywh_to_x1y1x2y2(boxes1)
    boxes2 = to_2d_tensor(boxes2)
    boxes2 = convert_xywh_to_x1y1x2y2(boxes2)

    intersec = boxes1.clone()
    intersec[:, 0] = torch.max(boxes1[:, 0], boxes2[:, 0])
    intersec[:, 1] = torch.max(boxes1[:, 1], boxes2[:, 1])
    intersec[:, 2] = torch.min(boxes1[:, 2], boxes2[:, 2])
    intersec[:, 3] = torch.min(boxes1[:, 3], boxes2[:, 3])

    def compute_area(boxes):
        # in (x1, y1, x2, y2) format
        dx = boxes[:, 2] - boxes[:, 0]
        dx[dx < 0] = 0
        dy = boxes[:, 3] - boxes[:, 1]
        dy[dy < 0] = 0
        return dx * dy

    a1 = compute_area(boxes1)
    a2 = compute_area(boxes2)
    ia = compute_area(intersec)
    assert ((a1 + a2 - ia <= 0).sum() == 0)

    return ia / (a1 + a2 - ia)


def compute_acc(preds, targets, im_sizes, theta=0.4):
    """
    Computes accuracy between predictions and targets
    :param preds:
    :param targets:
    :param im_sizes:
    :param theta:
    :return:
    """
    preds = box_transform_inv(preds.clone(), im_sizes)
    targets = box_transform_inv(targets.clone(), im_sizes)
    IoU = compute_IoU(preds, targets)
    return IoU.sum()/preds.size(0)

