import cv2
import numpy as np
from shapely.geometry import Polygon
from typing import Collection, Tuple
from numba import jit
from .ortho_tools import lon_lat_to_pixel


@jit(nopython=True)
def fracture_polygon_north_up(poly: Polygon, factor_x: int, factor_y) -> Collection[Polygon]:
    """
    Breaks a Shapely polygon into small north up rectangles, attempts to fill as much area as possible
    with smaller squares. The factors control the number of rectangles per dimension (resolution)
    :param poly:
    :param factor_x: width of the resulting grid
    :param factor_y: height of the resulting grid
    :return: Collection of small north up rectangles (as Shapely Polygons)
    """
    env = poly.envelope.exterior
    min_x = min([coord[0] for coord in env.coords])
    min_y = min([coord[1] for coord in env.coords])
    max_x = max([coord[0] for coord in env.coords])
    max_y = max([coord[1] for coord in env.coords])
    xs = np.linspace(min_x, max_x, factor_x)
    ys = np.linspace(min_y, max_y, factor_y)
    polygons = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cell = Polygon(
                [
                    [xs[i], ys[j]],
                    [xs[i + 1], ys[j]],
                    [xs[i + 1], ys[j + 1]],
                    [xs[i], ys[j + 1]],
                    [xs[i], ys[j]],
                ]
            )
            if cell.within(poly):
                polygons.append(cell)
    return polygons


@jit(nopython=True)
def fracture_parallelogram(poly: Polygon, factor: int) -> Collection[Polygon]:
    """
    Fractures a parallelogram into smaller parallelogram with a non-regular orientation.
    The orientation of each individual small parallelogram with respect to the original
    parallelogram is maintained.
    :param poly: Shapely polygon that must be a parallelogram (quadrilateral with 2 pairs of parallel sides)
    :param factor: Number of times per dimension to fracture the parallelogram
    :return: Collection of small parallelograms as Shapely polygons. No verification of parallelogram membership
    is performed on the input polygon, so expect absolute junk if you provide a non-parallelogram polygon
    """
    quadrilateral = poly.exterior
    v1 = quadrilateral.coords[0]
    v_across = quadrilateral.coords[1]
    v_down = quadrilateral.coords[3]
    step_across = (v_across - v1) / factor
    step_down = (v_down - v1) / factor

    polygons = []
    for i in range(factor):
        for j in range(factor):
            ul = v1 + (i * step_across) + (j * step_down)
            ur = v1 + ((i + 1) * step_across) + (j * step_down)
            ll = v1 + ((i + 1) * step_across) + ((j + 1) * step_down)
            lr = v1 + (i * step_across) + ((j + 1) * step_down)
            cell = Polygon([ul, ur, lr, ll, ul])
            polygons.append(cell)
    return polygons


def rescale_elevation_data(elevation_data: np.ndarray) -> np.ndarray:
    """
    Maps elevation data into a 0-255 8-bit represenation that is suitable for viewing
    :param elevation_data: elevation data input image
    :return: rescaled elevation image
    """
    rescaled_elevation_data = elevation_data - elevation_data.min()
    rescaled_elevation_data = (
        rescaled_elevation_data / rescaled_elevation_data.max() * 255
    )
    return rescaled_elevation_data


def reproject_with_affine(
    coords: Collection[Tuple[float, float]],
    geo_transform: np.array,
    resize_factor: float = 1.0,
) -> Collection[Tuple[float, float]]:
    """
    Reprojects a coordinate array according to a provided affine transform and resizing factor
    :param coords: Coordinate array that will be reprojected
    :param geo_transform: Affine transform as the basis of the reprojection  (length 6 np.array of float32/64)
    :param resize_factor: Optional resizing factor, useful if you want to display reprojected points on a downsampled
    image
    :return: reprojected coordinate array
    """
    reprojected_points = [
        lon_lat_to_pixel(point[0], point[1], geo_transform) for point in coords
    ]
    reprojected_points = [
        (int(point[0] / resize_factor), int(point[1] / resize_factor))
        for point in reprojected_points
    ]
    return reprojected_points


def overlay_polygon(
    img: np.ndarray,
    polygon: Polygon,
    color: Tuple[int, int, int, int] = (0, 255, 0, 0),
    opacity: float = 0.2
) -> np.ndarray:
    """
    Takes an input image and a polygon (which must exist in pixel coordinate space)
    and returns an image with the polygon transparently overlaid
    NOTE: only utilizes the exterior ring of the provided polygon
    :param img: The image on which the transparent polygon will be overlaid
    :param polygon: The shapely Polygon to overlay
    :param color: The color to make the polygon
    :param opacity: The alpha level of the polygon (ignore the alpha channel in color!!)
    :return: a modified image as an np.ndarray
    """
    polygon_coords = polygon.exterior.coords
    # needed for opencv /shrug
    polygon_array = np.array(polygon_coords, dtype=np.int32).reshape(-1, 1, 2)
    overlay = np.zeros((img.shape[0], img.shape[1]), dtype=np.ubyte)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay = cv2.fillConvexPoly(overlay, polygon_array, color, lineType=cv2.LINE_AA)
    result = cv2.addWeighted(overlay, opacity, result, 1 - opacity, 0, result)

    return result


@jit(nopython=True)
def generate_triangle_mesh(elevation_data: np.ndarray, reach: int = 10) -> Tuple[np.array, np.array, np.array]:
    """
    Generates a triangle mesh from elevation data using a very simple and non-optimal but extremely fast procedure
    Produces a 3D vertex list as a Tuple of np.array
    :param elevation_data:
    :param reach: The resulting mesh will be centered on the origin (0,0,0), reach determines how wide in the (x and y)
    dimension the mesh will go. Does not have an effect on height (z).
    :return: a Tuple of vertex arrays (Tuple[X, Y, Z])
    """
    shape_x = elevation_data.shape[0]
    shape_y = elevation_data.shape[1]
    x = np.linspace(-reach, reach, int(shape_x))
    y = np.linspace(-reach, reach, int(shape_y))

    display_e_matrix = elevation_data / 1000
    new_x = np.zeros(x.shape[0] * y.shape[0] * 6)
    new_y = np.zeros(x.shape[0] * y.shape[0] * 6)
    new_elev = np.zeros(x.shape[0] * y.shape[0] * 6)
    idx = 0
    for i in range(1, x.shape[0] - 1):
        for j in range(1, y.shape[0] - 1):
            new_x[idx] = x[i-1]
            new_y[idx] = y[j-1]
            new_elev[idx] = display_e_matrix[i-1, j-1]
            new_x[idx + 1] = x[i]
            new_y[idx + 1] = y[j-1]
            new_elev[idx + 1] = display_e_matrix[i, j-1]
            new_x[idx + 2] = x[i]
            new_y[idx + 2] = y[j]
            new_elev[idx + 2] = display_e_matrix[i, j]
            new_x[idx + 3] = x[i-1]
            new_y[idx + 3] = y[j]
            new_elev[idx + 3] = display_e_matrix[i-1, j]
            new_x[idx + 4] = x[i-1]
            new_y[idx + 4] = y[j-1]
            new_elev[idx + 4] = display_e_matrix[i-1, j-1]
            new_x[idx + 5] = x[i]
            new_y[idx + 5] = y[j]
            new_elev[idx + 5] = display_e_matrix[i, j]
            idx = idx + 6
    return new_x, new_y, new_elev
