import cppyy
import numpy as np
from .ortho_tools import RPCCoeffs

cpp_source = """
#include <cmath>


template <class T> constexpr float interpolate(float x, float y, T source, int source_height) {
    int x_floor = static_cast<int>(std::floor(x));
    int x_ceil =  static_cast<int>(std::ceil(x));
    int y_floor =  static_cast<int>(std::floor(y));
    int y_ceil =  static_cast<int>(std::ceil(y));

    float x_frac = x - x_floor;
    float y_frac = y - y_floor;
    
    long ul_index = x_floor + source_height * y_floor;
    long ur_index = x_ceil + source_height * y_floor;
    long lr_index = x_ceil + source_height * y_ceil;
    long ll_index = x_floor + source_height * y_ceil;
    
    float ul = static_cast<float>(source[ul_index]);
    float ur = static_cast<float>(source[ur_index]);
    float lr = static_cast<float>(source[lr_index]);
    float ll = static_cast<float>(source[ll_index]);
    
    float upper_x = (1 - x_frac) * ul + x_frac * ur;
    float lower_x = (1 - x_frac) * ll + x_frac * lr;
    float upper_y = (1 - y_frac) * ul + y_frac * ll;
    float lower_y = (1 - y_frac) * ur + y_frac * lr;
    float final_value = (upper_x + lower_x + upper_y + lower_y) / 4;
    return final_value;
}

void lon_lat_alt_to_xy_cpp(
    float lon,
    float lat,
    float alt,
    float height_off,
    float height_scale,
    float lat_off,
    float lat_scale,
    float* line_den_coeff,
    float* line_num_coeff,
    float line_off,
    float line_scale,
    float long_off,
    float long_scale,
    float max_lat,
    float max_long,
    float min_lat,
    float min_long,
    float* samp_den_coeff,
    float* samp_num_coeff,
    float samp_off,
    float samp_scale,
    float* output
) {
    
    float norm_lon = (lon - long_off) / long_scale;
    float norm_lat = (lat - lat_off) / lat_scale;
    float norm_alt = (alt - height_off) / height_scale;
    
    float formula[20] = {
        1,
        norm_lon,
        norm_lat,
        norm_alt,
        norm_lon * norm_lat,
        norm_lon * norm_alt,
        norm_lat * norm_alt,
        static_cast<float>(std::pow(norm_lon, 2)),
        static_cast<float>(std::pow(norm_lat, 2)),
        static_cast<float>(std::pow(norm_alt, 2)),
        norm_lat * norm_lon * norm_alt,
        static_cast<float>(std::pow(norm_lon, 3)),
        norm_lon * static_cast<float>(std::pow(norm_lat, 2)),
        norm_lon * static_cast<float>(std::pow(norm_alt, 2)),
        static_cast<float>(std::pow(norm_lon, 2)) * norm_lat,
        static_cast<float>(std::pow(norm_lat, 3)),
        norm_lat * static_cast<float>(std::pow(norm_alt, 2)),
        static_cast<float>(std::pow(norm_lon, 2)) * norm_alt,
        static_cast<float>(std::pow(norm_lat, 2)) * norm_alt,
        static_cast<float>(std::pow(norm_alt, 3))
    };
    
    float f1 = 0.0;
    for (int i = 0; i < 20; i++) 
        f1 += samp_num_coeff[i] * formula[i];
    
    float f2 = 0.0;
    for (int i = 0; i < 20; i++) 
        f2 += samp_den_coeff[i] * formula[i];
       
    float f3 = 0.0;
    for (int i = 0; i < 20; i++) 
        f3 += line_num_coeff[i] * formula[i];
    
    float f4 = 0.0;
    for (int i = 0; i < 20; i++) 
        f4 += line_den_coeff[i] * formula[i];   

    float samp_number_normed = f1 / f2;
    float line_number_normed = f3 / f4;
    
    output[0] = samp_number_normed * samp_scale + samp_off;
    output[1] = line_number_normed * line_scale + line_off;

}

void make_ortho_cpp(
    float x1,
    float x2,
    float y1,
    float y2,
    int width,
    int height, 
    uint8_t* source,
    int source_width,
    int source_height,
    float height_off,
    float height_scale,
    float lat_off,
    float lat_scale,
    float* line_den_coeff,
    float* line_num_coeff,
    float line_off,
    float line_scale,
    float long_off,
    float long_scale,
    float max_lat,
    float max_long,
    float min_lat,
    float min_long,
    float* samp_den_coeff,
    float* samp_num_coeff,
    float samp_off,
    float samp_scale,
    float* dem,
    int dem_width,
    int dem_height,
    float* dem_geot,
    uint8_t* output
) {

    float gsd_x = abs(x2 - x1) / width;
    float gsd_y = abs(y2 - y1) / height;
    for (int i = 0; i < width; i++) {
        float lon = x1 + ((float) i * gsd_x);
        for (int j = 0; j < height; j++) {
            float lat = y1 + ((float) j * gsd_y);
            float approx_dem_x = ((lon - dem_geot[0]) / dem_geot[1]);
            float approx_dem_y = ((lat - dem_geot[3]) / dem_geot[5]);    
            float altitude = interpolate(approx_dem_x, approx_dem_y, dem, dem_height);
            
            float* xy_loc = (float*) malloc(sizeof(float) * 2);
            lon_lat_alt_to_xy_cpp(
                lon,
                lat,
                altitude,
                height_off,
                height_scale,
                lat_off,
                lat_scale,
                line_den_coeff,
                line_num_coeff,
                line_off,
                line_scale,
                long_off,
                long_scale,
                max_lat,
                max_long,
                min_lat,
                min_long,
                samp_den_coeff,
                samp_num_coeff,
                samp_off,
                samp_scale,
                xy_loc
            );
            
            float x = xy_loc[0];
            float y = xy_loc[1];
            free(xy_loc);

            if (x > 1 && x < source_height - 1 && y > 1 && y < source_width - 1) {
                uint8_t result = static_cast<uint8_t>(
                    std::round(interpolate(x, y, source, source_height))
                );
                output[i * height + j] = result;
            }
        }
    }
}
"""


def ortho_cpp(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    width: int,
    rpcs: RPCCoeffs,
    source: np.ndarray,
    dem: np.ndarray,
    dem_geot: np.array,
):
    cppyy.cppdef(cpp_source)
    from cppyy.gbl import make_ortho_cpp
    gsd = abs(x1 - x2) / width
    height = int(abs(y1 - y2) / gsd)
    output = np.zeros((width * height), dtype=np.ubyte)
    make_ortho_cpp(
        x1,
        x2,
        y1,
        y2,
        width,
        height,
        source.reshape(-1),
        source.shape[0],
        source.shape[1],
        *rpcs,
        dem.astype(np.float32).reshape(-1),
        dem.shape[0],
        dem.shape[1],
        dem_geot.astype(np.float32),
        output
    )
    return output.reshape(width, height).transpose(), gsd, x1, y1

