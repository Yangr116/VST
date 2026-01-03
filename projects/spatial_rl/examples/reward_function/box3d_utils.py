import re
import json
import math
from loguru import logger
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import torch
try:
    from pytorch3d import _C
except:
    print("please install pytorch3d")
import torch.nn.functional as F
from pytorch3d.ops.iou_box3d import _box_planes, _box_triangles


def extract_bbox(input_string, from_range: str='180', to_range: str = '180'):
    def _extract_bbox(input_string, from_range: str='180', to_range: str = '180'):
        """Extract 3D bounding boxes from the input string."""
        results = []
        matches = re.findall(r'<3dbbox>(.*?)</3dbbox>', input_string)
        for bbox3d in matches:
            bbox3d = bbox3d.split(' ')
            bbox3d = list(map(float, bbox3d))
            if len(bbox3d) != 9: continue
            bbox3d = convert_degree_range(bbox3d, from_range=from_range, to_range=to_range)
            results.append(bbox3d)
        return results
    try:
        bboxes3d = _extract_bbox(input_string, from_range=from_range, to_range=to_range)
    except Exception as e:
        logger.error(f"Error processing sample: {input_string}, Error: {e}")
        bboxes3d = []
    return bboxes3d


def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def extract_bbox3d_from_json(input_string, from_range: str='180', to_range: str = '180'):
    def _extract_bbox(input_string, from_range: str='180', to_range: str = '180'):
        output_string = parse_json(input_string)
        results = json.loads(output_string)
        bboxes3d = [x.get("bbox_3d", None) for x in results]
        bboxes3d = [convert_degree_range(x, from_range=from_range, to_range=to_range) for x in bboxes3d if x is not None and len(x)==9]
        return bboxes3d
    try:
        bboxes3d = _extract_bbox(input_string, from_range=from_range, to_range=to_range)
    except:
        bboxes3d = []
    return bboxes3d


def convert_degree_range(bbox3d: list, from_range: str='180', to_range: str = '180'):
    """
        from_range: support '1', 'pi', '180'
        to_range: support '1', 'pi', '180'
    """
    if from_range == to_range:
        return bbox3d
    if from_range == '1' and to_range == '180':
        bbox3d[-3:] = map(lambda x: x*180, bbox3d[-3:])
    elif from_range == 'pi' and to_range == '180':
        bbox3d[-3:] = map(lambda x: x*180 / math.pi, bbox3d[-3:])
    elif from_range == '180' and to_range == '1':
        bbox3d[-3:] = map(lambda x: x / 180, bbox3d[-3:])
    elif from_range == '180' and to_range == 'pi':
        bbox3d[-3:] = map(lambda x: x*math.pi / 180, bbox3d[-3:])
    return bbox3d


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


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> torch.BoolTensor:
    """
    Checks that plane vertices are coplanar.
    Returns a bool tensor of size B, where True indicates a box is coplanar.
    """
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    
    return (mat1.bmm(mat2).abs() < eps).view(B)


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-8) -> torch.BoolTensor:
    """
    Checks that the sides of the box have a non zero area.
    Returns a bool tensor of size B, where True indicates a box is nonzero.
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    return (face_areas > eps).all(1).view(B)


def box3d_overlap(
    boxes_dt: torch.Tensor, boxes_gt: torch.Tensor, 
    eps_coplanar: float = 1e-4, eps_nonzero: float = 1e-8
) -> torch.Tensor:
    """
    Computes the intersection of 3D boxes_dt and boxes_gt.

    Inputs boxes_dt, boxes_gt are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes_dt and boxes_gt),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        boxes_dt: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes_gt: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    # Make sure predictions are coplanar and nonzero 
    invalid_coplanar = ~_check_coplanar(boxes_dt, eps=eps_coplanar)
    invalid_nonzero  = ~_check_nonzero(boxes_dt, eps=eps_nonzero)

    ious = _C.iou_box3d(boxes_dt, boxes_gt)[1]

    # Offending boxes are set to zero IoU
    if invalid_coplanar.any():
        ious[invalid_coplanar] = 0
        print('Warning: skipping {:d} non-coplanar boxes at eval.'.format(int(invalid_coplanar.float().sum())))
    
    if invalid_nonzero.any():
        ious[invalid_nonzero] = 0
        print('Warning: skipping {:d} zero volume boxes at eval.'.format(int(invalid_nonzero.float().sum())))

    return ious
