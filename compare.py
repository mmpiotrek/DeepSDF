import argparse
import numpy as np
import open3d as o3d
import os
from loss import metrics

metrics_names = ('chamfer', 'sinkhorn')
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Path of the input object.')
parser.add_argument('-r', '--reference', help='Path of the input object.')
parser.add_argument('-d', '--display', help='Renders the input and the reference objects.', action='store_true')
parser.add_argument('-m', '--metrics', help=f'Calculated loss for the given objects. Default are: {metrics_names}',
                    action='append')

args = parser.parse_args()

if args.input is None:
    print('Input not provided!')
elif args.reference is None:
    print('Reference not provided!')


def load_object(filename):
    ext = os.path.splitext(filename)[-1].lower()
    if ext in ('.ply', '.stl', '.obj'):
        obj = o3d.io.read_triangle_mesh(filename)
        obj.compute_vertex_normals()
        return obj
    else:
        obj = o3d.io.read_point_cloud(filename)
        return obj


def extract_data(obj):
    data = {}
    if isinstance(obj, o3d.geometry.TriangleMesh):
        data['points'] = np.array(obj.vertices)
    elif isinstance(obj, o3d.geometry.PointCloud):
        data['points'] = np.array(obj.points)

    return data


def print_metrics(data):
    for k, v in data.items():
        print(f'{k}: {v}')

source_path = '/home/piotr/Desktop/ProRoc/DeepSDF/PPRAI_Result_noth/Meshes/Dataset_PPRAI'#  '/home/piotr/Desktop/ProRoc/DeepSDF/examples/PPRAI/Reconstructions/2000/Meshes/Dataset_PPRAI'  #  /home/piotr/Desktop/ProRoc/DeepSDF/data_PPRAI/SdfSamples/Dataset_PPRAI
reference_path = '/home/piotr/Desktop/ProRoc/DeepSDF/others/DatasetObjectsTest'
classess = os.listdir(source_path)
text_to_file = []
mean_result = []
for c in classess:
    class_path = os.path.join(source_path, c)
    names_list = sorted([x for x in os.listdir(class_path) if '_' in x])
    chamfer = 0
    hausdorff = 0
    for n in names_list:
        input_object = load_object(os.path.join(class_path, n))
        reference_object = load_object(os.path.join(reference_path, c, n[:-6], 'models/model_normalized.obj'))

        input_data = extract_data(input_object)
        reference_data = extract_data(reference_object)

        losses = {}

        for m in ['chamfer', 'hausdorff']:
            if metrics.exists(m):
                losses[m] = metrics.calculate(input_data, reference_data, m)

        print_metrics(losses)
        comparison_result = f"{n[:-6]}, {c.title()}, {n}, chamfer: {losses['chamfer']}, hausdorff: {losses['hausdorff']}"
        text_to_file.append(comparison_result)
        chamfer += losses['chamfer']
        hausdorff += losses['hausdorff']
        # print(comparison_result)
        # if args.display:
        #     vis = o3d.visualization.Visualizer()
        #     visualize = list()
        #     visualize.append(input_object)
        #     visualize.append(reference_object)
        #     for x in visualize:
        #         vis.add_geometry(x)

        #     o3d.visualization.draw_geometries(visualize, mesh_show_back_face=True)
        #     vis.clear_geometries()
        #     vis.destroy_window()
    mean_chamfer = chamfer/len(names_list)
    mean_hausdorff = hausdorff/len(names_list)
    mean_values = f"for class {c} mean chamfer: {mean_chamfer}, mean hausdorff: {mean_hausdorff}"
    mean_result.append(mean_values)

print(mean_result)
with open('DeepSDF_metrics_5mln.txt', 'w') as f:
    for line in text_to_file:
        f.write(f"{line}\n")
    for line in mean_result:
        f.write(f"{line}\n")


# input_object = load_object(args.input)
# reference_object = load_object(args.reference)

# input_data = extract_data(input_object)
# reference_data = extract_data(reference_object)

# losses = {}

# for m in args.metrics:
#     if metrics.exists(m):
#         losses[m] = metrics.calculate(input_data, reference_data, m)

# print_metrics(losses)

# if args.display:
#     vis = o3d.visualization.Visualizer()
#     visualize = list()
#     visualize.append(input_object)
#     visualize.append(reference_object)
#     for x in visualize:
#         vis.add_geometry(x)

#     o3d.visualization.draw_geometries(visualize, mesh_show_back_face=True)
#     vis.clear_geometries()
#     vis.destroy_window()