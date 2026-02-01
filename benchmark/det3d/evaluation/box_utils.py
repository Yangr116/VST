import torch
import numpy as np
from scipy.spatial.transform import Rotation


def boxconverter_xyxy2xywh(box):
    original_type = type(box)
    is_numpy = isinstance(box, np.ndarray)
    single_box = isinstance(box, (list, tuple))
    if single_box:
        assert len(box) == 4 or len(box) == 5, (
            "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
            " where k == 4 or 5"
        )
        arr = torch.tensor(box)[None, :]
    else:
        # avoid modifying the input box
        if is_numpy:
            arr = torch.from_numpy(np.asarray(box)).clone()
        else:
            arr = box.clone()

    # xyxy2xywh
    arr[:, 2] -= arr[:, 0]
    arr[:, 3] -= arr[:, 1]

    if single_box:
        return original_type(arr.flatten().tolist())
    if is_numpy:
        return arr.numpy()
    else:
        return arr


def compute_box3d_from_answer(answer, K=None):
    """
    obj: [x, y, z, x_size, y_size, z_size, pitch, yaw, roll]
    K: camera intrinsics
    """
    obj = extract_3dbbox_list(answer)
    return compute_box3d(obj, K=K)


def compute_box2d_from_answer(answer, K=None):
    """
    obj: [x, y, z, x_size, y_size, z_size, pitch, yaw, roll]
    K: camera intrinsics
    """
    obj = extract_3dbbox_list(answer)
    return compute_box2d(obj, K=K)


def compute_box3d(obj, K=None):
    """
    Args:
        obj (dict): {'centroid_cam': xxx, 'dimensions': xxx, 'R_bbox2cam': xxx}
        K: camera intrinsics
    Return:
        bbox3d verts:
                    v4_____________________v5
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
              v0|    |                 |v1 |
                |    |                 |   |
                |    |                 |   |
                |    |                 |   |
                |    |_________________|___|
                |   / v7               |   /v6
                |  /                   |  /
                | /                    | /
                |/_____________________|/
                v3                     v2
    """
    # TODO: support other format
    box3d = obj.get('centroid_cam') + obj.get('dimensions')
    R = obj.get('R_bbox2cam')

    box3d = to_float_tensor(box3d)
    R = to_float_tensor(R)
    
    if K is not None:
        if isinstance(K, dict):
            K = create_intrinsics_from_dict(K)
        K = to_float_tensor(K)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)
    
    n = len(box3d)

    x3d = box3d[:, 0].unsqueeze(1)
    y3d = box3d[:, 1].unsqueeze(1)
    z3d = box3d[:, 2].unsqueeze(1)
    xsize3d = box3d[:, 3].unsqueeze(1)
    ysize3d = box3d[:, 4].unsqueeze(1)
    zsize3d = box3d[:, 5].unsqueeze(1)
    verts = to_float_tensor(torch.zeros([n, 3, 8], device=box3d.device)) # shape=(3, 8), verts[0, :] 表示 8 个点的 x 坐标

    # 这里的 setup 和他们的坐标系是一致的 X right, Y down, Z toword screen
    # setup X
    verts[:, 0, [0, 3, 4, 7]] = -xsize3d / 2
    verts[:, 0, [1, 2, 5, 6]] = xsize3d / 2

    # setup Y
    verts[:, 1, [0, 1, 4, 5]] = -ysize3d / 2
    verts[:, 1, [2, 3, 6, 7]] = ysize3d / 2

    # setup Z
    verts[:, 2, [0, 1, 2, 3]] = -zsize3d / 2
    verts[:, 2, [4, 5, 6, 7]] = zsize3d / 2
    if R is not None:
        verts = R @ verts
    # translate
    verts[:, 0, :] += x3d
    verts[:, 1, :] += y3d
    verts[:, 2, :] += z3d

    verts = verts.transpose(1, 2)

    if squeeze:
        verts = verts.squeeze()
    
    if K is not None:
        verts2d = (K @ verts.T).T
        verts2d = verts2d / verts2d[:, -1][:, None]
        return verts2d[:, :2]

    return verts


def compute_box2d(obj, K):
    bbox2d = compute_box3d(obj, K)
    bbox2d = bbox2d.numpy()
    x1 = np.min(bbox2d[:, 0], axis=0)
    y1 = np.min(bbox2d[:, 1], axis=0)
    x2 = np.max(bbox2d[:, 0], axis=0)
    y2 = np.max(bbox2d[:, 1], axis=0)
    bbox2D_proj = np.stack([x1, y1, x2, y2], axis=0)
    return bbox2D_proj


def extract_3dbbox_list(sequence: list, category: str = None):
    is_single = not isinstance(sequence[0], list)
    if is_single:
        sequence = [sequence]
    box_anns_new = []
    for box_ann in sequence:
        euler_angle_pyr = box_ann[6:]
        rotation = Rotation.from_euler(angles=euler_angle_pyr, seq='xyz', degrees=True)
        R_bbox2cam = rotation.as_matrix()
        box_anns_new.append(
            {
                'category': category,
                'centroid_cam': box_ann[0:3],
                'dimensions': box_ann[3:6],
                'R_bbox2cam': R_bbox2cam.tolist(),
                'bbox2D_tight': None
            }
        )
    if is_single:
        return box_anns_new[0]
    return box_anns_new


def to_float_tensor(input):

    data_type = type(input)

    if data_type != torch.Tensor:
        input = torch.tensor(input)
    
    return input.float()


def create_intrinsics_from_dict(input_dict):
    return [
        [input_dict['fx'], 0, input_dict['cx']], 
        [0, input_dict['fy'], input_dict['cy']], 
        [0, 0, 1]]
